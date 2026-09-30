"""Experimental order-free within-burst attention, not a released selector.

Shared observed-person encoding supplies each frame. Set attention compares
the frames together without filenames, positions, clocks or positional
embeddings. Retention loss includes all-unretained bursts; listwise preference loss
is only applied where training positives AND negatives are observed.

Original implementation informed by SetRank (arxiv.org/abs/1912.05891) and
allRank's listwise design (github.com/allegro/allRank). No third-party source
or weights are copied, and no decoder/detector/model-download is introduced.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import threading
import time
import numpy as np

from .person_set_ranker import (_make_network, _validate_inputs, _validate_bodies,
                               BODY_INPUT_DIMENSION, fit_input_normalizers, transform_person_inputs)
from .semantic_tail_ranker import training_runtime

PIPELINE_ID="observed-person-burst-listwise-attention-v1"
MAX_GROUP_FRAMES=64
GROUP_BATCH=24
MAX_EPOCHS=40


def group_indexes(groups, count):
    """Group IDs delimit sets only. No silent split/truncation of large sets."""
    groups=list(groups)
    if len(groups)!=count:
        raise ValueError("Burst identities differ from feature rows")
    members=defaultdict(list)
    for index,key in enumerate(groups):
        if not isinstance(key,(str,int,np.integer)):
            raise ValueError("Invalid burst identity")
        members[key].append(index)
    if any(len(rows)>MAX_GROUP_FRAMES for rows in members.values()):
        raise ValueError("Burst exceeds bounded attention limit; no silent partial evaluation")
    return [np.asarray(rows,dtype=np.int64) for rows in members.values()]


def _network(global_dimension, face_dimension, *, width, with_bodies):
    import torch
    from torch import nn

    class BurstSet(nn.Module):
        def __init__(self):
            super().__init__()
            self.person=_make_network(global_dimension,face_dimension,width=width,
                                      body_dimension=BODY_INPUT_DIMENSION if with_bodies else None)
            self.project=nn.Sequential(nn.Linear(self.person.feature_dimension,width),nn.GELU(),nn.LayerNorm(width))
            self.attention=nn.ModuleList([nn.MultiheadAttention(width,4,dropout=.1,batch_first=True) for _ in range(2)])
            self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
            self.feedforward=nn.ModuleList([nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Dropout(.1),
                                                         nn.Linear(width*2,width)) for _ in range(2)])
            self.output=nn.Sequential(nn.Linear(width*4+1,width),nn.GELU(),nn.Dropout(.15),nn.Linear(width,1))

        def forward(self,values,valid):
            # Gather observed frames first. A padded face/body/frame is never
            # sent to the shared encoder as a fictitious observed person.
            shape=valid.shape
            encoded=self.project(self.person.forward_features(*(v[valid] for v in values)))
            initial=encoded.new_zeros((*shape,encoded.shape[-1]))
            initial[valid]=encoded
            context=initial
            for attention,norm,feedforward in zip(self.attention,self.norms,self.feedforward):
                query=norm(context)
                compared,_=attention(query,query,query,key_padding_mask=~valid,need_weights=False)
                context=context+compared
                context=context+feedforward(norm(context))
                context=context.masked_fill(~valid[...,None],0)
            count=valid.sum(1,keepdim=True).clamp_min(1)
            mean=context.sum(1)/count
            head=torch.cat((initial,context,initial-context,mean[:,None].expand_as(context),
                            torch.log1p(count).to(context.dtype)[:,None].expand(*shape,1)),dim=-1)
            return self.output(head).squeeze(-1).masked_fill(~valid,0)

    return BurstSet()


def _loss(logits, labels, valid, positive_weight):
    import torch
    import torch.nn.functional as functional
    unary=functional.binary_cross_entropy_with_logits(logits[valid],labels[valid],pos_weight=positive_weight)
    positives=(labels*valid).sum(1)
    mixed=(positives>0)&(positives<valid.sum(1))
    if mixed.any():
        target=labels[mixed]*valid[mixed]/positives[mixed,None]
        logprob=torch.log_softmax(logits[mixed].masked_fill(~valid[mixed],-1e4),dim=1)
        unary=unary+.5*(-(target*logprob).sum(1)).mean()
    return unary


class BurstSetRanker:
    """Train-side early stopping, transactional refit, bounded shared inputs.

    Group caps are explicit experimental limits, not a new default UI failure.
    Scores rank retained originals and are not calibrated quality probabilities.
    """
    def __init__(self, *, epochs=MAX_EPOCHS, width=32, with_bodies=True,
                 checkpoint=lambda:None, progress=None):
        if type(epochs) is not int or not 1<=epochs<=MAX_EPOCHS or type(width) is not int or width not in {8,16,32,64}:
            raise ValueError("Invalid bounded burst training configuration")
        self.epochs,self.width,self.with_bodies=epochs,width,bool(with_bodies)
        self.checkpoint,self.progress=checkpoint,progress
        self.model=self.normalizers=self.positive_fraction=None
        self.selected_epoch=self.training_seconds=None
        self.validation_history=[]
        self._lock=threading.RLock()

    def _inputs(self,features,faces,mask,normalizers,*,bodies=None,body_mask=None):
        values=transform_person_inputs(features,faces,mask,normalizers,
                                      bodies=bodies if self.with_bodies else None,body_mask=body_mask)
        # A constant observed measurement in training can have tiny std.
        # Bound numerical extrapolation without fitting held-out statistics.
        return [np.clip(v,-20,20) if v.dtype!=bool else v for v in values]

    @staticmethod
    def _batch(values, groups):
        import torch
        longest=max(len(g) for g in groups)
        valid=np.zeros((len(groups),longest),bool)
        indexes=np.zeros_like(valid,dtype=np.int64)
        for i,rows in enumerate(groups):
            valid[i,:len(rows)]=True
            indexes[i,:len(rows)]=rows
        # Index placeholders are masked before the person encoder, not fake
        # extra copies of the first photo in short bursts.
        return [torch.as_tensor(v[indexes]) for v in values],torch.as_tensor(valid),indexes

    def fit(self,features,labels,groups,faces,mask,*,training=None,validation=None,
            bodies=None,body_mask=None,patience=5):
        self.checkpoint()
        features,faces,mask=_validate_inputs(features,faces,mask)
        if self.with_bodies:
            bodies,body_mask=_validate_bodies(features,bodies,body_mask)
        labels=np.asarray(labels)
        count=len(features)
        training=np.ones(count,bool) if training is None else np.asarray(training)
        validation=np.zeros(count,bool) if validation is None else np.asarray(validation)
        if (labels.shape!=(count,) or not np.isin(labels,(0,1)).all()
                or training.shape!=(count,) or validation.shape!=(count,)
                or training.dtype!=bool or validation.dtype!=bool or (training&validation).any()
                or labels[training].sum()<12 or (1-labels[training]).sum()<12
                or type(patience) is not int or not 1<=patience<=MAX_EPOCHS
                or (validation.any() and (labels[validation].sum()<4 or (1-labels[validation]).sum()<4))):
            raise ValueError("Invalid disjoint burst supervision")
        sets=group_indexes(groups,count)
        if any((training[rows].any() and not training[rows].all()) or
               (validation[rows].any() and not validation[rows].all()) for rows in sets):
            raise ValueError("Whole bursts must stay inside one training/validation partition")
        train_sets=[rows for rows in sets if training[rows].all()]
        val_sets=[rows for rows in sets if validation[rows].all()]
        body_training={} if not self.with_bodies else {"bodies":bodies[training],"body_mask":body_mask[training]}
        normalizers=fit_input_normalizers(features[training],faces[training],mask[training],**body_training)
        values=self._inputs(features,faces,mask,normalizers,bodies=bodies,body_mask=body_mask)
        import torch
        from sklearn.metrics import average_precision_score
        started=time.monotonic()
        history=[]
        best,best_epoch,best_state,stale=-float("inf"),None,None,0
        with self._lock,training_runtime("cpu"):
            model=_network(features.shape[1],faces.shape[2],width=self.width,with_bodies=self.with_bodies)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.05)
            positive_weight=torch.tensor(float((1-labels[training]).sum()/labels[training].sum()))
            generator=np.random.default_rng(0)
            for epoch in range(self.epochs):
                model.train()
                order=generator.permutation(len(train_sets))
                losses=[]
                for start in range(0,len(order),GROUP_BATCH):
                    self.checkpoint()
                    batch=[train_sets[i] for i in order[start:start+GROUP_BATCH]]
                    inputs,valid,indexes=self._batch(values,batch)
                    logits=model(inputs,valid)
                    loss=_loss(logits,torch.as_tensor(labels[indexes],dtype=torch.float32),valid,positive_weight)
                    if not torch.isfinite(loss).item():
                        raise ValueError("Nonfinite burst training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(loss.item())
                model.eval()
                row={"epoch":epoch+1,"training_loss":float(np.mean(losses))}
                if val_sets:
                    scores=self._predict(model,values,val_sets,count)
                    ap=float(average_precision_score(labels[validation],scores[validation]))
                    row["validation_average_precision"]=ap
                    if ap>best+1e-4:
                        best,best_epoch,best_state,stale=ap,epoch+1,copy.deepcopy(model.state_dict()),0
                    else:
                        stale+=1
                history.append(row)
                if self.progress:
                    self.progress(row,time.monotonic()-started)
                if val_sets and stale>=patience:
                    break
            if best_state is not None:
                model.load_state_dict(best_state)
            self.checkpoint()
            # Publish only after complete fit and successful cancellation check.
            self.model,self.normalizers=model.eval(),normalizers
            self.positive_fraction=float(labels[training].mean())
            self.selected_epoch=best_epoch or len(history)
            self.validation_history=history
            self.training_seconds=time.monotonic()-started
        return self

    def _predict(self,model,values,sets,count):
        import torch
        scores=np.full(count,np.nan,dtype=np.float32)
        with torch.inference_mode():
            for start in range(0,len(sets),GROUP_BATCH):
                self.checkpoint()
                inputs,valid,indexes=self._batch(values,sets[start:start+GROUP_BATCH])
                logits=model(inputs,valid)
                scores[indexes[valid.numpy()]]=torch.sigmoid(logits[valid]).numpy()
        return scores

    def predict(self,features,groups,faces,mask,*,bodies=None,body_mask=None):
        self.checkpoint()
        with self._lock:
            if self.model is None:
                raise ValueError("Burst ranker has not been trained")
            if len(features)==0:
                if np.shape(features)!=(0,len(self.normalizers[0])) or len(list(groups)):
                    raise ValueError("Empty burst feature dimension changed")
                return np.zeros(0,np.float32)
            features,faces,mask=_validate_inputs(features,faces,mask)
            if features.shape[1]!=len(self.normalizers[0]):
                raise ValueError("Burst feature dimension changed")
            if self.with_bodies:
                bodies,body_mask=_validate_bodies(features,bodies,body_mask)
            sets=group_indexes(groups,len(features))
            values=self._inputs(features,faces,mask,self.normalizers,bodies=bodies,body_mask=body_mask)
            # Runtime owner restores CPU thread configuration even on cancel.
            with training_runtime("cpu"):
                scores=self._predict(self.model,values,sets,len(features))
            self.checkpoint()
            if not np.isfinite(scores).all():
                raise ValueError("Incomplete/nonfinite burst prediction")
            return scores

    def close(self):
        with self._lock:
            self.model=self.normalizers=None
