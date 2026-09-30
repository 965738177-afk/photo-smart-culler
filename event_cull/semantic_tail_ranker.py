"""Experimental task-trained visual tail using cached frozen prefix tokens.

The legacy SigLIP experiment updates two blocks; the optional small image tower
updates all ten transformer blocks, keeping patch embeddings frozen. Both use
the same context-person head and training owner. Prefixes see images, never labels. No image decoder,
detector, downloads, snapshot mutations, active-profile writes or uploads here.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np

from src.core.image_features.semantic_regions import (TOKEN_PIPELINE_ID,TOKEN_COUNT,EMBEDDING_DIMENSION,
                                                      TAIL_LAYERS,BATCH_SIZE,SIGLIP_SPATIAL_SPEC,TINY_SPATIAL_SPEC)
from .person_set_ranker import _make_network, _TRAINING_LOCK
from .retention_ranker import same_group_pairs

PIPELINE_ID = "siglip2-two-block-task-trained-retention-v1"
IMAGE_BATCH = 8
# Fixed before any real held-out predictions. Bound the first expensive visual
# pilot to three passes rather than choosing an epoch from the user's test set.
TRAINING_EPOCHS = 3
MAX_TRAINING_EPOCHS = 12
SCALED_PIPELINE_ID = "siglip2-two-block-nested-scaled-retention-v2"
TINY_TRAINED_PIPELINE_ID = "tinyclip-uncropped-ten-block-nested-scaled-retention-v2"
KIND_IDS = {"whole":0,"context":1,"face":2}


class SpatialTokenCache:
    """Validated read-only ragged token mmap; never re-decode training images."""
    def __init__(self, folder, *, expected_paths=None, allow_smoke=False,expected_model=None,
                 spatial_spec=SIGLIP_SPATIAL_SPEC):
        if spatial_spec not in (SIGLIP_SPATIAL_SPEC,TINY_SPATIAL_SPEC):
            raise ValueError("Known spatial token contract required")
        folder = Path(folder)
        report = json.loads((folder/"tokens.json").read_text(encoding="utf-8"))
        records = report.get("records")
        if (report.get("complete") is not True or report.get("pipeline") != spatial_spec.pipeline
                or report.get("tail_layers") != spatial_spec.trainable_layers or report.get("token_count") != spatial_spec.token_count
                or report.get("dimension") != spatial_spec.dimension or report.get("dtype") != "float16"
                or not isinstance(records,list) or len(records) != report.get("images")
                or not records or (report.get("smoke_subset") and not allow_smoke)
                or any(not isinstance(row,dict) or not isinstance(row.get("path"),str) or not row["path"] for row in records)):
            raise ValueError("Incomplete/incompatible spatial token cache")
        if expected_model is not None and (report.get("model"),report.get("revision")) != tuple(expected_model):
            raise ValueError("Spatial tokens belong to another model/revision")
        paths = [row["path"] for row in records]
        if len(set(paths)) != len(paths) or (expected_paths is not None and paths != list(map(str,expected_paths))):
            raise ValueError("Token source identities/order differ")
        cursor = 0
        for record in records:
            kinds = record.get("kinds")
            try:
                masks = np.asarray(record.get("masks"),np.float32)
            except (ValueError,TypeError) as exc:
                raise ValueError("Invalid spatial token observation masks") from exc
            count = record.get("count")
            if (record.get("start") != cursor or type(count) is not int or not 1 <= count <= 6
                    or not isinstance(kinds,list) or len(kinds) != count or kinds[0] != "whole"
                    or kinds.count("whole") != 1 or kinds.count("context") > 1 or kinds.count("face") > 4
                    or any(kind not in KIND_IDS for kind in kinds) or masks.shape != (4,)
                    or not np.isfinite(masks).all() or (masks < 0).any()
                    or masks[1] != kinds.count("face") or masks[0] < masks[1]
                    or masks[2] != ("context" in kinds) or masks[3] != bool(kinds.count("face"))):
                raise ValueError("Invalid spatial token region offsets/kinds")
            cursor += count
        token_file = folder/"tokens.f16"
        if report.get("regions") != cursor or token_file.stat().st_size != cursor*spatial_spec.token_count*spatial_spec.dimension*2:
            raise ValueError("Truncated/extra spatial token bytes")
        self.records, self.report, self.spatial_spec = records, report, spatial_spec
        self.tokens = np.memmap(token_file,dtype=np.float16,mode="r",shape=(cursor,spatial_spec.token_count,spatial_spec.dimension))

    def batch(self,indexes):
        if self.tokens is None:
            raise ValueError("Spatial token cache is closed")
        indexes = list(indexes)
        if not 1 <= len(indexes) <= IMAGE_BATCH+4:
            raise ValueError("Unbounded spatial training image batch")
        values, entries = [], []
        for index in indexes:
            if not isinstance(index,(int,np.integer)) or not 0 <= index < len(self.records):
                raise ValueError("Invalid spatial cache row index")
            record = self.records[int(index)]
            tokens = self.tokens[record["start"]:record["start"]+record["count"]]
            if not np.isfinite(tokens).all():
                raise ValueError("Nonfinite frozen spatial tokens; no skipped image")
            values.append(tokens)
            entries.append(record)
        return np.concatenate(values).astype(np.float32), entries

    def close(self):
        # Views must not escape this owner. batch() always copies before return.
        if self.tokens is not None:
            self.tokens._mmap.close()
            self.tokens = None


def build_tail_network(tail, feature_dimension, *, semantic_scale=1.,spatial_spec=SIGLIP_SPATIAL_SPEC):
    import torch
    from torch import nn
    from torch.nn import functional
    layers,norm,pooler = tail
    if len(layers) != spatial_spec.trainable_layers:
        raise ValueError("Pretrained layer count differs from spatial contract")
    dimension = spatial_spec.dimension

    class TailNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers, self.norm, self.pooler = layers,norm,pooler
            # Reuse the nonlinear person/context head; image-valued faces stay
            # together rather than coordinatewise extrema masquerading as a person.
            self.ranker = _make_network(dimension*2+feature_dimension+4,dimension)

        def forward(self,tokens,entries,features):
            pooled = []
            for start in range(0,len(tokens),BATCH_SIZE):
                value = tokens[start:start+BATCH_SIZE]
                for layer in self.layers:
                    value = layer(value,None)
                # Unit-normalized image coordinates are about 1/sqrt(d), while
                # standardized native measurements are order one. An explicit,
                # label-free scale can balance modalities without test fitting.
                pooled.append(functional.normalize(self.pooler(self.norm(value)).float(),dim=-1)*semantic_scale)
            values = torch.cat(pooled)
            whole,context,faces,masks = [],[],[],[]
            cursor = 0
            for entry in entries:
                kinds = entry["kinds"]
                whole.append(cursor)
                context.append(cursor+kinds.index("context") if "context" in kinds else cursor)
                observed = [cursor+i for i,kind in enumerate(kinds) if kind == "face"]
                faces.append(observed+[len(values)]*(4-len(observed)))
                masks.append([True]*len(observed)+[False]*(4-len(observed)))
                cursor += len(kinds)
            values_with_unknown = torch.cat((values,values.new_zeros(1,dimension)))
            face_values = values_with_unknown[torch.as_tensor(faces,device=values.device)]
            face_mask = torch.as_tensor(masks,dtype=torch.bool,device=values.device)
            coverage = torch.log1p(torch.as_tensor([e["masks"] for e in entries],dtype=values.dtype,device=values.device))
            global_values = torch.cat((values[whole],values[context],features,coverage),dim=1)
            return self.ranker(global_values,face_values,face_mask)

    return TailNetwork()


@contextmanager
def training_runtime(device):
    """Restore CPU/MPS RNG and CPU threading even after cancellation/error."""
    import torch
    with _TRAINING_LOCK,torch.random.fork_rng(devices=[]):
        old_threads = torch.get_num_threads()
        mps_state = None
        try:
            if device == "mps":
                mps_state = torch.mps.get_rng_state()
                torch.mps.manual_seed(0)
            torch.random.default_generator.manual_seed(0)
            torch.set_num_threads(min(4,old_threads))
            yield
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
            torch.set_num_threads(old_threads)


class SemanticTailRanker:
    def __init__(self,tail_factory,*,device="cpu",checkpoint=lambda:None,progress=None,
                 batch_progress=None,epochs=TRAINING_EPOCHS,mixed_precision=False,
                 semantic_scale=1.,validation_progress=None,spatial_spec=SIGLIP_SPATIAL_SPEC):
        if (device not in {"cpu","mps"} or type(epochs) is not int or not 1 <= epochs <= MAX_TRAINING_EPOCHS
                or spatial_spec not in (SIGLIP_SPATIAL_SPEC,TINY_SPATIAL_SPEC)
                or not np.isfinite(semantic_scale) or not 0 < semantic_scale <= spatial_spec.dimension**.5):
            raise ValueError("Unsupported bounded semantic-tail training configuration")
        self.tail_factory,self.device,self.checkpoint = tail_factory,device,checkpoint
        self.progress,self.epochs = progress,epochs
        self.batch_progress = batch_progress
        self.semantic_scale,self.validation_progress = float(semantic_scale),validation_progress
        self.spatial_spec = spatial_spec
        self.mixed_precision = bool(mixed_precision)
        if self.mixed_precision and device != "mps":
            raise ValueError("Experimental mixed precision is validated only for MPS")
        self.model,self.normalization,self.positive_fraction = None,None,None
        self.selected_epoch,self.validation_history = None,[]

    def fit(self,cache,features,labels,training,groups,*,validation=None,patience=3):
        self.checkpoint()
        if getattr(cache,"spatial_spec",SIGLIP_SPATIAL_SPEC) != self.spatial_spec:
            raise ValueError("Training cache/model spatial contracts differ")
        features,labels,training = np.asarray(features,np.float32),np.asarray(labels),np.asarray(training,bool)
        if (features.ndim != 2 or len(features) != len(cache.records) or labels.shape != training.shape
                or labels.shape != (len(features),) or not np.isfinite(features).all()
                or not np.isin(labels,(0,1)).all() or labels[training].sum() < 12
                or (1-labels[training]).sum() < 12 or len(groups) != len(features)):
            raise ValueError("Incomplete bounded training data")
        validation = np.zeros(len(features),bool) if validation is None else np.asarray(validation,bool)
        if (validation.shape != training.shape or (validation & training).any()
                or type(patience) is not int or not 1 <= patience <= MAX_TRAINING_EPOCHS
                or (validation.any() and (labels[validation].sum() < 4 or (1-labels[validation]).sum() < 4))):
            raise ValueError("Validation must be disjoint with sufficient class coverage")
        # Held-out rows never affect normalization or pair/quantity supervision.
        mean = features[training].mean(0)
        scale = np.maximum(features[training].std(0),1e-4)
        standardized = (features-mean)/scale
        indexes = np.flatnonzero(training)
        left,right = same_group_pairs(np.asarray(groups)[training])
        disagreement = labels[indexes[left]] != labels[indexes[right]]
        left,right = indexes[left[disagreement]], indexes[right[disagreement]]
        import torch
        from torch.nn import functional
        generator = np.random.default_rng(0)
        started = time.monotonic()
        last_progress = started
        with training_runtime(self.device):
            model = build_tail_network(self.tail_factory(),features.shape[1],semantic_scale=self.semantic_scale,
                                       spatial_spec=self.spatial_spec).to(self.device)
            optimizer = torch.optim.AdamW([
                {"params":model.layers.parameters(),"lr":1e-5},
                {"params":list(model.norm.parameters())+list(model.pooler.parameters()),"lr":1e-5},
                {"params":model.ranker.parameters(),"lr":.001}],weight_decay=.05,
                fused=True if self.device == "mps" else None)
            weight = torch.tensor(float((1-labels[training]).sum()/labels[training].sum()),device=self.device)
            model.train()
            history,best_state,best_epoch,best_score,stale = [],None,None,-np.inf,0
            for epoch in range(self.epochs):
                order = generator.permutation(indexes)
                losses = []
                for start in range(0,len(order),IMAGE_BATCH):
                    self.checkpoint()
                    take = order[start:start+IMAGE_BATCH]
                    pair_count = min(2,len(left))
                    chosen = generator.integers(0,len(left),size=pair_count) if pair_count else []
                    combined = np.concatenate((take,left[chosen],right[chosen])) if pair_count else take
                    tokens,entries = cache.batch(combined)
                    value = torch.as_tensor(tokens,device=self.device)
                    x = torch.as_tensor(standardized[combined],device=self.device)
                    target = torch.as_tensor(labels[take].astype(np.float32),device=self.device)
                    with torch.autocast(device_type=self.device,dtype=torch.bfloat16,enabled=self.mixed_precision):
                        logits = model(value,entries,x).float()
                    loss = functional.binary_cross_entropy_with_logits(logits[:len(take)],target,pos_weight=weight)
                    if pair_count:
                        offset = len(take)
                        relative = logits[offset:offset+pair_count]-logits[offset+pair_count:]
                        pair_target = torch.as_tensor(labels[left[chosen]].astype(np.float32),device=self.device)
                        loss = loss+.25*functional.binary_cross_entropy_with_logits(relative,pair_target)
                    if not torch.isfinite(loss).item():
                        raise ValueError("Nonfinite task-trained vision loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    now = time.monotonic()
                    if self.batch_progress and now-last_progress >= 20:
                        self.batch_progress(epoch+1,min(start+IMAGE_BATCH,len(order)),len(order),now-started)
                        last_progress = now
                if self.progress:
                    self.progress(epoch+1,self.epochs,float(np.mean(losses)),time.monotonic()-started)
                if validation.any():
                    # This mask must be inside the caller's outer training side.
                    # Never select a checkpoint against the final held-out fold.
                    from sklearn.metrics import average_precision_score,roc_auc_score
                    model.eval()
                    val_scores = self._predict_model(model,cache,standardized,np.flatnonzero(validation))
                    average_precision = float(average_precision_score(labels[validation],val_scores))
                    row = {"epoch":epoch+1,"training_loss":float(np.mean(losses)),
                           "validation_examples":int(validation.sum()),
                           "validation_average_precision":average_precision,
                           "validation_auc":float(roc_auc_score(labels[validation],val_scores))}
                    history.append(row)
                    if self.validation_progress:
                        self.validation_progress(row)
                    if average_precision > best_score+1e-4:
                        best_score,best_epoch,stale = average_precision,epoch+1,0
                        best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
                    else:
                        stale += 1
                    if stale >= patience:
                        break
                    model.train()
            if best_state is not None:
                model.load_state_dict(best_state)
            self.checkpoint()
            model.eval()
            self.model,self.normalization = model,(mean,scale)
            self.positive_fraction = float(labels[training].mean())
            self.selected_epoch = best_epoch if best_epoch is not None else self.epochs
            self.validation_history = history
        return self

    def _predict_model(self,model,cache,standardized,indexes):
        """Bounded inference shared by train-side validation and final prediction."""
        import torch
        scores = []
        with torch.inference_mode():
            for start in range(0,len(indexes),IMAGE_BATCH):
                self.checkpoint()
                take = indexes[start:start+IMAGE_BATCH]
                tokens,entries = cache.batch(take)
                with torch.autocast(device_type=self.device,dtype=torch.bfloat16,enabled=self.mixed_precision):
                    logits = model(torch.as_tensor(tokens,device=self.device),entries,
                                   torch.as_tensor(standardized[take],device=self.device))
                scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        self.checkpoint()
        values = np.concatenate(scores) if scores else np.empty(0,np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite task-trained visual prediction")
        return values

    def predict(self,cache,features,indexes):
        self.checkpoint()
        if getattr(cache,"spatial_spec",SIGLIP_SPATIAL_SPEC) != self.spatial_spec:
            raise ValueError("Inference cache/model spatial contracts differ")
        if self.model is None:
            raise ValueError("Semantic-tail ranker is not trained")
        features = np.asarray(features,np.float32)
        raw_indexes = np.asarray(indexes)
        if (raw_indexes.ndim != 1 or (len(raw_indexes) and raw_indexes.dtype.kind not in "iu")
                or (raw_indexes < 0).any() or (raw_indexes >= len(cache.records)).any()):
            raise ValueError("Invalid semantic-tail inference indices")
        indexes = raw_indexes.astype(int)
        if (features.ndim != 2 or len(features) != len(cache.records)
                or features.shape[1] != len(self.normalization[0]) or not np.isfinite(features).all()):
            raise ValueError("Incomplete semantic-tail inference features")
        mean,scale = self.normalization
        return self._predict_model(self.model,cache,(features-mean)/scale,indexes)

    def close(self):
        self.model,self.normalization = None,None
