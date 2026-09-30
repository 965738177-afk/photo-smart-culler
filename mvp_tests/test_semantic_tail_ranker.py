"""Visual-tail training/validation ownership, not wedding acceptance evidence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from event_cull.semantic_tail_ranker import (SpatialTokenCache,SemanticTailRanker,build_tail_network,
                                            training_runtime)
from src.core.image_features.semantic_regions import TOKEN_PIPELINE_ID


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(768))
    def forward(self,x,mask):
        return x*self.weights


class Pool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project = torch.nn.Linear(768,768,bias=False)
    def forward(self,x):
        return self.project(x.mean(1))


def tail():
    return torch.nn.ModuleList([Layer(),Layer()]),torch.nn.Identity(),Pool()


class Cache:
    def __init__(self,n=48):
        self.records = [dict(kinds=["whole"],masks=[0,0,False,False]) for _ in range(n)]
        self.calls = []
        self.tokens = np.random.default_rng(42).normal(size=(n,196,768)).astype(np.float32)
    def batch(self,indexes):
        self.calls.append(np.asarray(indexes).copy())
        assert len(indexes) <= 12
        return self.tokens[indexes], [self.records[i] for i in indexes]


def test_training_updates_actual_vision_tail_not_just_final_classifier(monkeypatch):
    from event_cull import engine
    monkeypatch.setattr(engine,"_load_preview",lambda *a,**k:pytest.fail("cached training must not decode"))
    frozen = tail()
    before = frozen[0][0].weights.detach().clone()
    cache = Cache()
    features = np.random.default_rng(4).normal(size=(48,3)).astype(np.float32)
    features[-12:] += 10000
    labels = np.tile([0,1],24)
    training = np.asarray([True]*36+[False]*12)
    state, threads = torch.random.get_rng_state(),torch.get_num_threads()
    owner = SemanticTailRanker(lambda:copy.deepcopy(frozen),epochs=1)
    owner.fit(cache,features,labels,training,np.arange(48)//4)
    assert torch.equal(state,torch.random.get_rng_state()) and torch.get_num_threads() == threads
    assert np.allclose(owner.normalization[0],features[training].mean(0))
    assert all(training[indexes].all() for indexes in cache.calls)
    assert not torch.equal(before,owner.model.layers[0].weights.detach())
    assert torch.isfinite(owner.model.layers[0].weights.grad).all()
    assert owner.model.layers[0].weights.grad.abs().sum().item() > 0  # Not weight decay alone.
    assert torch.equal(before,frozen[0][0].weights.detach())
    scores = owner.predict(cache,features,np.arange(36,48))
    assert np.isfinite(scores).all() and len(scores) == 12
    owner.close()


def test_cancelled_refit_keeps_model_density_and_normalization():
    cache = Cache()
    x = np.zeros((48,3),np.float32)
    labels, training = np.tile([0,1],24),np.ones(48,bool)
    owner = SemanticTailRanker(tail,epochs=1).fit(cache,x,labels,training,np.arange(48)//4)
    model,normalization,density = owner.model,owner.normalization,owner.positive_fraction
    calls = [0]
    def stop():
        calls[0] += 1
        if calls[0] > 1:
            raise InterruptedError("cancel mid-fit")
    owner.checkpoint = stop
    with pytest.raises(InterruptedError):
        owner.fit(cache,x,labels,training,np.arange(48)//4)
    assert owner.model is model and owner.normalization is normalization and owner.positive_fraction == density
    owner.close()


def token_fixture(tmp_path):
    values = np.ones((2,196,768),np.float16)
    values.tofile(tmp_path/"tokens.f16")
    report = dict(complete=True,pipeline=TOKEN_PIPELINE_ID,tail_layers=2,token_count=196,dimension=768,
                  dtype="float16",images=2,regions=2,smoke_subset=False,records=[
                      dict(path=str(Path("/photos")/f"{i}.ARW"),start=i,count=1,kinds=["whole"],masks=[0,0,0,0])
                      for i in range(2)])
    (tmp_path/"tokens.json").write_text(json.dumps(report))
    return report


def test_mmap_owner_returns_copy_and_refuses_invalid_indices(tmp_path):
    token_fixture(tmp_path)
    cache = SpatialTokenCache(tmp_path,expected_paths=["/photos/0.ARW","/photos/1.ARW"])
    values,_ = cache.batch([0,1])
    values[:] = 5
    assert np.all(cache.tokens == 1)
    with pytest.raises(ValueError,match="index"):
        cache.batch([-1])
    cache.close()
    assert cache.tokens is None
    with pytest.raises(ValueError,match="closed"):
        cache.batch([0])


def test_spatial_model_revision_must_match_owned_base(tmp_path):
    report = token_fixture(tmp_path)
    report.update(model="pinned/model",revision="first")
    (tmp_path/"tokens.json").write_text(json.dumps(report))
    with pytest.raises(ValueError,match="model/revision"):
        SpatialTokenCache(tmp_path,expected_model=("pinned/model","second"))
    cache = SpatialTokenCache(tmp_path,expected_model=("pinned/model","first"))
    cache.close()


@pytest.mark.parametrize("corrupt",["incomplete","size","identities","offsets","masks","smoke","record","string_mask"])
def test_incomplete_mismatched_token_caches_are_not_valid_experiments(tmp_path,corrupt):
    report = token_fixture(tmp_path)
    if corrupt == "incomplete":
        report["complete"] = False
    elif corrupt == "size":
        (tmp_path/"tokens.f16").write_bytes(b"truncated")
    elif corrupt == "identities":
        report["records"][1]["path"] = report["records"][0]["path"]
    elif corrupt == "offsets":
        report["records"][1]["start"] = 0
    elif corrupt == "masks":
        report["records"][0]["masks"] = [-1,0,0,0]
    elif corrupt == "smoke":
        report["smoke_subset"] = True
    elif corrupt == "record":
        report["records"][0] = "not a record"
    else:
        report["records"][0]["masks"] = ["invalid",0,0,0]
    (tmp_path/"tokens.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        SpatialTokenCache(tmp_path)


def test_training_runtime_restores_cpu_state_after_failure():
    state,threads = torch.random.get_rng_state(),torch.get_num_threads()
    with pytest.raises(InterruptedError):
        with training_runtime("cpu"):
            torch.randn(5)
            raise InterruptedError()
    assert torch.equal(state,torch.random.get_rng_state()) and torch.get_num_threads() == threads


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason="Native MPS RNG isolation")
def test_training_runtime_restores_mps_state_after_failure():
    state = torch.mps.get_rng_state()
    with pytest.raises(InterruptedError):
        with training_runtime("mps"):
            torch.rand(5,device="mps")
            raise InterruptedError()
    assert torch.equal(state,torch.mps.get_rng_state())


def test_label_free_image_scale_balances_unit_embeddings():
    network = build_tail_network(tail(),3,semantic_scale=768**.5)
    captured = []
    hook = network.ranker.register_forward_pre_hook(lambda _owner,args:captured.append(args))
    cache = Cache(2)
    values,entries = cache.batch([0,1])
    network(torch.as_tensor(values),entries,torch.zeros(2,3))
    hook.remove()
    globals,faces,mask = captured[0]
    assert torch.allclose(globals[:,:768].norm(dim=1),torch.full((2,),768**.5),atol=1e-4)
    assert not mask.any() and not faces.any()


def test_checkpoint_selection_uses_only_disjoint_training_side_validation():
    cache = Cache()
    x = np.zeros((48,3),np.float32)
    labels = np.tile([0,1],24)
    training = np.arange(48) < 24
    validation = (np.arange(48) >= 24) & (np.arange(48) < 36)
    owner = SemanticTailRanker(tail,epochs=2).fit(cache,x,labels,training,np.arange(48)//4,validation=validation)
    assert len(owner.validation_history) == 2 and owner.selected_epoch in {1,2}
    assert all((training|validation)[indexes].all() for indexes in cache.calls)
    assert all(row["validation_examples"] == 12 for row in owner.validation_history)
    assert owner.predict(cache,x,[]).shape == (0,)
    for bad in ([1.5],[-1],[48],[[1]]):
        with pytest.raises(ValueError,match="indices"):
            owner.predict(cache,x,bad)
    with pytest.raises(ValueError,match="disjoint"):
        owner.fit(cache,x,labels,training,np.arange(48)//4,validation=training)
    owner.close()


def test_nested_split_never_uses_outer_test_or_neighbors():
    from scripts.evaluate_semantic_tail import train_side_validation
    blocks = np.repeat(np.arange(50),10)
    eligible = ~np.isin(blocks%5,[0,1,4])
    training,validation = train_side_validation(blocks,eligible)
    assert not (training&validation).any() and not ((training|validation)&~eligible).any()
    for block in np.unique(blocks[validation]):
        assert not np.isin(blocks[training],[block-1,block,block+1]).any()
    assert training.sum() > 0 and validation.sum() > 0


def test_early_stop_restores_best_training_side_checkpoint(monkeypatch):
    import sklearn.metrics
    ap = iter([.9,.8,.7,.6])
    monkeypatch.setattr(sklearn.metrics,"average_precision_score",lambda *a,**k:next(ap))
    cache = Cache()
    x,labels = np.zeros((48,3),np.float32),np.tile([0,1],24)
    training = np.arange(48) < 24
    validation = (np.arange(48) >= 24) & (np.arange(48) < 36)
    owner = SemanticTailRanker(tail,epochs=12)
    original = owner._predict_model
    states = []
    def observe(model,*args):
        states.append({key:value.detach().clone() for key,value in model.state_dict().items()})
        return original(model,*args)
    owner._predict_model = observe
    owner.fit(cache,x,labels,training,np.arange(48)//4,validation=validation,patience=3)
    assert owner.selected_epoch == 1 and len(owner.validation_history) == 4
    assert all(torch.equal(value,states[0][key]) for key,value in owner.model.state_dict().items())
    owner.close()
