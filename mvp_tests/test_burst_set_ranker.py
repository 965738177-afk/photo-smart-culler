"""Set symmetry, supervision isolation and bounded lifecycle, not 75% proof."""
import numpy as np
import pytest
import torch

from event_cull.burst_set_ranker import (BurstSetRanker,group_indexes,_network,_loss,
                                       MAX_GROUP_FRAMES,GROUP_BATCH)
from event_cull.person_set_ranker import FACE_INPUT_DIMENSION,BODY_INPUT_DIMENSION
from src.core.image_features.human_analysis import MAX_POSES


def samples():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(160,3)).astype(np.float32)
    y=np.tile([0,1,0,1],40)
    faces=rng.normal(size=(160,3,FACE_INPUT_DIMENSION)).astype(np.float32)
    face_mask=np.tile([True,True,False],(160,1))
    bodies=rng.normal(size=(160,MAX_POSES,BODY_INPUT_DIMENSION)).astype(np.float32)
    body_mask=np.tile([True]+[False]*(MAX_POSES-1),(160,1))
    return x,y,np.arange(160)//4,faces,face_mask,bodies,body_mask


def test_shared_person_encoder_and_set_attention_are_frame_order_equivariant():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model=_network(3,FACE_INPUT_DIMENSION,width=8,with_bodies=False).eval()
        values=[torch.randn(2,5,3),torch.randn(2,5,3,FACE_INPUT_DIMENSION),torch.ones(2,5,3,dtype=torch.bool)]
        valid=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
        order=[2,4,0,3,1]
        with torch.inference_mode():
            scores=model(values,valid)
            reordered=model([v[:,order] for v in values],valid[:,order])
            values[0][~valid]=float("nan")
            values[1][~valid]=float("nan")
            padded=model(values,valid)
        assert torch.allclose(scores[:,order],reordered,atol=1e-6)
        assert torch.allclose(scores,padded,atol=1e-6)
        assert torch.isfinite(scores).all()


def test_all_rejected_group_still_supplies_absolute_training_gradient():
    logits=torch.randn(2,4,requires_grad=True)
    mask=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
    loss=_loss(logits,torch.zeros_like(logits),mask,torch.tensor(1.))
    loss.backward()
    assert torch.isfinite(loss) and (logits.grad[mask]>0).all()
    assert (logits.grad[~mask]==0).all()


def test_train_only_scaling_and_cpu_state_restore_without_decoding(monkeypatch):
    from event_cull import engine
    monkeypatch.setattr(engine,"_load_preview",lambda *a,**k:pytest.fail("set ranking must reuse shared observations"))
    x,y,g,f,m,b,bm=samples()
    train=np.arange(160)<96
    val=(np.arange(160)>=96)&(np.arange(160)<128)
    x[~train]+=1000
    owner=BurstSetRanker(epochs=2,width=8)
    state,threads=torch.random.get_rng_state(),torch.get_num_threads()
    owner.fit(x,y,g,f,m,training=train,validation=val,bodies=b,body_mask=bm)
    assert torch.equal(state,torch.random.get_rng_state()) and threads==torch.get_num_threads()
    assert np.allclose(owner.normalizers[0],x[train].mean(0))
    assert owner.selected_epoch in {1,2} and len(owner.validation_history)==2
    normalizers=[v.copy() for v in owner.normalizers]
    scores=owner.predict(x,g,f,m,bodies=b,body_mask=bm)
    assert np.isfinite(scores).all() and len(scores)==160
    assert all(np.array_equal(v,before) for v,before in zip(owner.normalizers,normalizers))
    owner.close()


def test_partitioned_burst_or_changed_group_identity_is_refused():
    x,y,g,f,m,b,bm=samples()
    train=np.arange(160)<97
    val=~train
    with pytest.raises(ValueError,match="Whole bursts"):
        BurstSetRanker(epochs=1).fit(x,y,g,f,m,training=train,validation=val,bodies=b,body_mask=bm)
    with pytest.raises(ValueError,match="identities"):
        group_indexes(g[:-1],len(g))
    with pytest.raises(ValueError,match="bounded"):
        group_indexes([1]*(MAX_GROUP_FRAMES+1),MAX_GROUP_FRAMES+1)
    with pytest.raises(ValueError,match="identity"):
        group_indexes([None],1)


def test_refit_cancel_preserves_prior_model_and_masked_unknowns():
    x,y,g,f,m,b,bm=samples()
    owner=BurstSetRanker(epochs=1,width=8).fit(x,y,g,f,m,bodies=b,body_mask=bm)
    before=owner.predict(x,g,f,m,bodies=b,body_mask=bm)
    model,density,normalizers=owner.model,owner.positive_fraction,owner.normalizers
    count=[0]
    def stop():
        count[0]+=1
        if count[0]>1:
            raise InterruptedError()
    owner.checkpoint=stop
    with pytest.raises(InterruptedError):
        owner.fit(x,y,g,f,m,bodies=b,body_mask=bm)
    owner.checkpoint=lambda:None
    assert owner.model is model and owner.normalizers is normalizers and owner.positive_fraction==density
    f[~m]=100000
    b[~bm]=100000
    assert np.allclose(before,owner.predict(x,g,f,m,bodies=b,body_mask=bm))
    assert owner.predict(np.zeros((0,3)),[],f[:0],m[:0],bodies=b[:0],body_mask=bm[:0]).shape==(0,)
    owner.close()
    with pytest.raises(ValueError,match="not been trained"):
        owner.predict(x,g,f,m,bodies=b,body_mask=bm)


def test_batched_prediction_is_bounded_and_group_names_are_not_features():
    x,y,g,f,m,b,bm=samples()
    owner=BurstSetRanker(epochs=1,width=8).fit(x,y,g,f,m,bodies=b,body_mask=bm)
    observed=[]
    hook=owner.model.register_forward_pre_hook(lambda model,inputs:observed.append(inputs[1].shape))
    before=owner.predict(x,g,f,m,bodies=b,body_mask=bm)
    hook.remove()
    assert all(size[0]<=GROUP_BATCH and size[1]<=MAX_GROUP_FRAMES for size in observed)
    order=np.random.default_rng(0).permutation(len(x))
    renamed=[f"another-case-burst-{i}" for i in g[order]]
    after=owner.predict(x[order],renamed,f[order],m[order],bodies=b[order],body_mask=bm[order])
    assert np.allclose(before[order],after,atol=1e-6)
    owner.close()


def test_curated_positive_control_refused_before_reading_any_labels(tmp_path):
    from scripts.evaluate_burst_sets import load_case
    for case in ({"source_preculled":True,"retention_labels_complete":True},
                 {"role":"curated_positive_control_not_full_selection_validation","retention_labels_complete":True},
                 {"retention_labels_complete":False}):
        with pytest.raises(ValueError,match="Preculled/partial"):
            load_case(case,tmp_path/"does-not-exist.json",tmp_path/"does-not-exist.npz")
