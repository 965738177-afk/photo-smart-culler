"""Experimental context-conditioned person-set delivery ranking, not detection.

Keeps each observed person's measurements together before nonlinear pooling.
Deep Sets: Zaheer et al., NeurIPS 2017 (https://arxiv.org/abs/1703.06114).
This is a small original implementation on the existing PyTorch runtime, not
downloaded source or a pretrained wedding selector. Missing observations are
explicit; no identities, roles, completed actions or precise gaze are claimed.
"""
from __future__ import annotations

import threading
import time
import numpy as np

from src.core.image_features.human_analysis import (FACE_DIMENSION,
                                                  MAX_POSES, validate_observation)

PERSON_SET_PIPELINE = "observed-face-context-deepset-ranker-v1"
MAX_FACES = 12
FACE_INPUT_DIMENSION = FACE_DIMENSION + 4 + 1
BODY_INPUT_DIMENSION = 33*4 + 4 + 1
JOINT_SET_PIPELINE = "observed-face-joint-context-deepset-ranker-v1"
TRAINING_EPOCHS = 60
TRAINING_BATCH = 128
PREDICTION_BATCH = 256
# torch RNG/thread configuration is process-global. Serialize this experimental
# trainer and restore caller state; no competing training runtime in the GUI.
_TRAINING_LOCK = threading.RLock()


def person_sets(results, *, maximum_faces=MAX_FACES):
    """Read shared observed face records once; never scan, decode or detect.

    Reliability is a separate channel rather than turning uncertain eyes into
    closed eyes. Sorting only bounds the retained set; inference is invariant
    to permutation of the selected people. Full observed count remains in the
    shared global descriptor when a crowded frame exceeds the set cap.
    """
    if not 1 <= maximum_faces <= MAX_FACES:
        raise ValueError("Invalid person-set limit")
    values = np.zeros((len(results), maximum_faces, FACE_INPUT_DIMENSION), np.float32)
    masks = np.zeros((len(results), maximum_faces), bool)
    for index, item in enumerate(results):
        person = item.person_evidence
        validate_observation(person)
        if not isinstance(person.get("faces"), (list, tuple)):
            raise ValueError("Incomplete observed face set")
        observed = []
        for face in person.get("faces", ()):
            if not isinstance(face, dict):
                raise ValueError("Malformed observed face record")
            features = np.asarray(face.get("features", ()), np.float32)
            bounds = np.asarray(face.get("bbox", ()), np.float32)
            if (features.shape != (FACE_DIMENSION,) or bounds.shape != (4,)
                    or not np.isfinite(features).all() or not np.isfinite(bounds).all()
                    or not isinstance(face.get("reliable_eyes"), (bool, np.bool_))):
                raise ValueError("Malformed observed face record")
            bounds = np.clip(bounds, 0, 1)
            area = (bounds[2]-bounds[0]) * (bounds[3]-bounds[1])
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError("Invalid observed face geometry")
            observed.append((float(area), np.concatenate((features, bounds, [face["reliable_eyes"]]))))
        observed.sort(key=lambda row: row[0], reverse=True)
        for face_index, (_, record) in enumerate(observed[:maximum_faces]):
            values[index, face_index] = record
            masks[index, face_index] = True
    return values, masks


def body_sets(results):
    """Reuse individual observed joints; averaged skeletons are not recoverable."""
    values = np.zeros((len(results), MAX_POSES, BODY_INPUT_DIMENSION), np.float32)
    masks = np.zeros((len(results), MAX_POSES), bool)
    for index, item in enumerate(results):
        validate_observation(item.person_evidence, require_details=True)
        for body_index, pose in enumerate(item.person_evidence["poses"]):
            joints = np.asarray(pose["joints"], np.float32)
            values[index, body_index] = np.concatenate((
                joints.ravel(), pose["bbox"], [(joints[:, 3] >= .5).mean()]))
            masks[index, body_index] = True
    return values, masks


def _make_network(global_dimension, face_dimension, *, width=64, body_dimension=None):
    # Imported only when this training/inference strategy is explicitly used.
    import torch
    from torch import nn

    class ContextPersonSet(nn.Module):
        def __init__(self):
            super().__init__()
            self.context = nn.Sequential(nn.Linear(global_dimension, width), nn.GELU())
            self.person = nn.Sequential(nn.Linear(face_dimension, width), nn.GELU(),
                                        nn.Linear(width, width), nn.GELU())
            self.attention = nn.Sequential(nn.Linear(width*2, width), nn.Tanh(), nn.Linear(width, 1))
            if body_dimension is not None:
                self.body = nn.Sequential(nn.Linear(body_dimension, width), nn.GELU(),
                                          nn.Linear(width, width), nn.GELU())
                self.body_attention = nn.Sequential(nn.Linear(width*2, width), nn.Tanh(), nn.Linear(width, 1))
            head_dimension = width*4+1 if body_dimension is None else width*7+2
            self.feature_dimension = head_dimension
            self.score = nn.Sequential(nn.Linear(head_dimension, width), nn.GELU(), nn.Dropout(.15),
                                       nn.Linear(width, 1))

        def pool(self, context, people, mask, attention):
            valid = mask.unsqueeze(-1)
            count = mask.sum(dim=1, keepdim=True)
            mean = (people * valid).sum(dim=1) / count.clamp_min(1)
            maximum = people.masked_fill(~valid, -1e4).amax(dim=1)
            maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
            logits = attention(torch.cat((people, context[:, None].expand_as(people)), dim=-1))
            # No observed faces must not produce NaNs from an all -inf softmax
            # or a dummy padded person. Unknown set features are exactly zero.
            weights = torch.softmax(logits.masked_fill(~valid, -1e4), dim=1) * valid
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            attended = (people * weights).sum(dim=1)
            return torch.cat((mean, maximum, attended, torch.log1p(count)), dim=1)

        def forward_features(self, global_values, face_values, mask, body_values=None, body_mask=None):
            """Shared observed-person encoder for unary and burst ranking."""
            context = self.context(global_values)
            parts = [context, self.pool(context, self.person(face_values), mask, self.attention)]
            if body_dimension is not None:
                parts.append(self.pool(context, self.body(body_values), body_mask, self.body_attention))
            return torch.cat(parts, dim=1)

        def forward(self, global_values, face_values, mask, body_values=None, body_mask=None):
            return self.score(self.forward_features(global_values, face_values, mask, body_values, body_mask)).squeeze(-1)

    return ContextPersonSet()


def _validate_inputs(features, faces, mask):
    features = np.asarray(features, np.float32)
    faces = np.asarray(faces, np.float32)
    mask = np.asarray(mask)
    if (features.ndim != 2 or not len(features) or faces.ndim != 3
            or len(faces) != len(features) or mask.shape != faces.shape[:2] or mask.dtype != bool
            or faces.shape[2] != FACE_INPUT_DIMENSION or not 1 <= faces.shape[1] <= MAX_FACES
            or not np.isfinite(features).all() or not np.isfinite(faces).all()):
        raise ValueError("Incomplete/nonfinite person-set features")
    return features, faces, mask


def _validate_bodies(features, bodies, mask):
    bodies, mask = np.asarray(bodies, np.float32), np.asarray(mask)
    if (bodies.shape != (len(features), MAX_POSES, BODY_INPUT_DIMENSION)
            or mask.shape != bodies.shape[:2] or mask.dtype != bool or not np.isfinite(bodies).all()):
        raise ValueError("Incomplete/nonfinite joint-set features")
    return bodies, mask


def fit_input_normalizers(features, faces, mask, *, bodies=None, body_mask=None):
    """Normalize training observations only; padding is never an observation."""
    features,faces,mask = _validate_inputs(features,faces,mask)
    def observed(values):
        dimension=values.shape[-1]
        if not len(values):
            return np.zeros(dimension,np.float32),np.ones(dimension,np.float32)
        return values.mean(0),np.maximum(values.std(0),1e-4)
    normalizers=(*observed(features),*observed(faces[mask]))
    if bodies is not None:
        bodies,body_mask=_validate_bodies(features,bodies,body_mask)
        normalizers+=observed(bodies[body_mask])
    return normalizers


def transform_person_inputs(features, faces, mask, normalizers, *, bodies=None, body_mask=None):
    """Reuse the same masked face/body scaling for all ranking strategies."""
    mean,scale,face_mean,face_scale=normalizers[:4]
    x=((features-mean)/scale).astype(np.float32)
    f=np.where(mask[...,None],(faces-face_mean)/face_scale,0).astype(np.float32)
    inputs=[x,f,mask]
    if len(normalizers)==6:
        bodies,body_mask=_validate_bodies(features,bodies,body_mask)
        body_mean,body_scale=normalizers[4:]
        inputs.extend((np.where(body_mask[...,None],(bodies-body_mean)/body_scale,0).astype(np.float32),body_mask))
    return inputs


class PersonSetRanker:
    """Train-only normalization, bounded pair supervision and owned lifecycle.

    Scores are rankings, not calibrated keep/quality probabilities. No model is
    activated, uploaded or written to disk by this class. Refit is transactional.
    """
    def __init__(self, *, checkpoint=lambda: None, progress=None, epochs=TRAINING_EPOCHS,
                 device="cpu", width=64, with_bodies=False):
        if not 1 <= epochs <= TRAINING_EPOCHS or not 4 <= width <= 128:
            raise ValueError("Invalid bounded person-set training configuration")
        # Training changes RNG state. CPU state is explicitly owned/restored;
        # do not promise GPU isolation before that backend has been validated.
        if device != "cpu":
            raise ValueError("Experimental person-set training currently requires CPU")
        self._checkpoint, self._progress = checkpoint, progress
        self.epochs, self.device, self.width = epochs, device, width
        self.with_bodies = bool(with_bodies)
        self._model = self._normalization = None
        self._lock = threading.RLock()
        self.positive_fraction = None
        self.training_pairs = 0
        self.training_seconds = None

    def fit(self, features, labels, faces, mask, *, groups=None, bodies=None, body_mask=None):
        self._checkpoint()
        features, faces, mask = _validate_inputs(features, faces, mask)
        labels = np.asarray(labels)
        if (labels.shape != (len(features),) or not np.isin(labels, (0, 1)).all()
                or labels.sum() < 12 or (1-labels).sum() < 12):
            raise ValueError("Insufficient valid person-set supervision")
        # Normalizers see ONLY the training samples and observed people. Padding
        # is excluded and explicitly restored to zero after transformation.
        body_arguments={}
        if self.with_bodies:
            bodies, body_mask = _validate_bodies(features, bodies, body_mask)
            body_arguments={"bodies":bodies,"body_mask":body_mask}
        normalizers=fit_input_normalizers(features,faces,mask,**body_arguments)
        normalized_inputs=transform_person_inputs(features,faces,mask,normalizers,**body_arguments)
        left = right = np.asarray([], dtype=int)
        if groups is not None:
            if len(groups) != len(features):
                raise ValueError("Person-set burst identities differ")
            # Shared comparison owner; never form cross-scene/opposite-label
            # pairs from re-numbered held-out PhotoResults.
            from .retention_ranker import same_group_pairs
            left, right = same_group_pairs(groups)
            different = labels[left] != labels[right]
            left, right = left[different], right[different]
        import torch
        import torch.nn.functional as functional
        started = time.monotonic()
        with self._lock, _TRAINING_LOCK, torch.random.fork_rng(devices=[]):
            old_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(min(4, old_threads))
                # Initialization is on CPU. Do not seed every GPU as a side
                # effect of a CPU-only experiment; fork_rng restores this RNG.
                torch.random.default_generator.manual_seed(0)
                model = _make_network(features.shape[1], faces.shape[2], width=self.width,
                                      body_dimension=BODY_INPUT_DIMENSION if self.with_bodies else None).to(self.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05)
                inputs = [torch.as_tensor(values,device=self.device) for values in normalized_inputs]
                y = torch.as_tensor(labels.astype(np.float32), device=self.device)
                positive_weight = torch.as_tensor(float((1-labels).sum()/labels.sum()), device=self.device)
                generator = np.random.default_rng(0)
                model.train()
                for epoch in range(self.epochs):
                    indexes = generator.permutation(len(features))
                    for start in range(0, len(indexes), TRAINING_BATCH):
                        self._checkpoint()
                        take = torch.as_tensor(indexes[start:start+TRAINING_BATCH], device=self.device)
                        logits = model(*(v[take] for v in inputs))
                        loss = functional.binary_cross_entropy_with_logits(logits, y[take], pos_weight=positive_weight)
                        if len(left):
                            # At most one small pair batch per unary batch. Pair
                            # sampling is label-disagreement within training
                            # bursts, never a held-out quota or filename lookup.
                            chosen = generator.integers(0, len(left), size=min(TRAINING_BATCH, len(left)))
                            a = torch.as_tensor(left[chosen], device=self.device)
                            b = torch.as_tensor(right[chosen], device=self.device)
                            preference = model(*(v[a] for v in inputs)) - model(*(v[b] for v in inputs))
                            loss = loss + .5*functional.binary_cross_entropy_with_logits(preference, y[a])
                        if not torch.isfinite(loss).item():
                            raise ValueError("Nonfinite person-set training loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                    if self._progress and (epoch % 10 == 9 or epoch+1 == self.epochs):
                        self._progress(epoch+1, self.epochs)
                self._checkpoint()
                model.eval()
                self._model, self._normalization = model, normalizers
                self.positive_fraction = float(labels.mean())
                self.training_pairs = len(left)
                self.training_seconds = time.monotonic()-started
            finally:
                torch.set_num_threads(old_threads)
        return self

    @staticmethod
    def _transform(features, faces, mask, normalizers):
        return tuple(transform_person_inputs(features,faces,mask,normalizers[:4])[:2])

    def predict(self, features, faces, mask, *, bodies=None, body_mask=None):
        self._checkpoint()
        features, faces, mask = _validate_inputs(features, faces, mask)
        import torch
        with self._lock:
            if self._model is None:
                raise ValueError("Person-set ranker has not been trained")
            if features.shape[1] != len(self._normalization[0]):
                raise ValueError("Person-set feature dimension changed")
            body_arguments={}
            if self.with_bodies:
                bodies, body_mask = _validate_bodies(features, bodies, body_mask)
                body_arguments={"bodies":bodies,"body_mask":body_mask}
            all_inputs=transform_person_inputs(features,faces,mask,self._normalization,**body_arguments)
            scores = []
            with torch.inference_mode():
                for start in range(0, len(features), PREDICTION_BATCH):
                    self._checkpoint()
                    inputs = [torch.as_tensor(v[start:start+PREDICTION_BATCH], device=self.device) for v in all_inputs]
                    scores.append(torch.sigmoid(self._model(*inputs)).cpu().numpy())
            self._checkpoint()
            values = np.concatenate(scores)
            if not np.isfinite(values).all():
                raise ValueError("Invalid person-set prediction")
            return values

    def close(self):
        with self._lock:
            self._model = self._normalization = None
