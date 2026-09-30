"""Tiny task-training ownership and integrity, never selection accuracy."""
import copy
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.core.image_features import semantic_regions as semantic
from src.core.image_features.human_analysis import PIPELINE_ID
from event_cull.semantic_tail_ranker import SemanticTailRanker,SpatialTokenCache


@pytest.fixture
def provisioning(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"src"))
    from src.core import model_provisioning
    return model_provisioning


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(256))
        self.calls = 0
    def forward(self,value,mask):
        assert mask is None
        self.calls += 1
        return value*self.weights


class Embeddings(torch.nn.Module):
    def forward(self,pixels):
        values = torch.arange(197*256,device=pixels.device,dtype=pixels.dtype).reshape(1,197,256)/100
        return values.expand(len(pixels),-1,-1)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = Embeddings()
        self.pre_layrnorm = torch.nn.LayerNorm(256)
        self.post_layernorm = torch.nn.LayerNorm(256)
        self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList([Layer() for _ in range(10)])
    def forward(self,pixel_values):
        value = self.pre_layrnorm(self.embeddings(pixel_values))
        for layer in self.encoder.layers:
            value = layer(value,None)
        return SimpleNamespace(pooler_output=self.post_layernorm(value[:,0]))


def owner():
    model,loads = TinyModel(),[]
    def backend(snapshot,device):
        loads.append(snapshot)
        return lambda images,return_tensors:{"pixel_values":torch.ones((len(images),3,224,224))},model
    return semantic.SemanticRegionEncoder(tiny_vision=True,resolver=lambda:"local",
            device_resolver=lambda:"cpu",backend_factory=backend),model,loads


def test_uncropped_provider_preserves_top_bottom_and_caller_image():
    from transformers import CLIPImageProcessor
    processor = CLIPImageProcessor(size=224,crop_size=224)
    image = Image.new("RGB",(240,640),(0,255,0))
    image.paste((255,0,0),(0,0,240,80))
    image.paste((0,0,255),(0,560,240,640))
    def colors(values):
        return np.moveaxis(values[0],0,-1)*np.asarray(processor.image_std)+np.asarray(processor.image_mean)
    original = colors(processor(images=[image],return_tensors="np")["pixel_values"])
    assert not ((original[:,:,0]>.8)&(original[:,:,1]<.2)).any()
    assert not ((original[:,:,2]>.8)&(original[:,:,1]<.2)).any()
    actual = colors(semantic._uncropped_processor(processor)(images=[image],return_tensors="np")["pixel_values"])
    assert ((actual[:,:,0]>.8)&(actual[:,:,1]<.2)).sum() > 1000
    assert ((actual[:,:,2]>.8)&(actual[:,:,1]<.2)).sum() > 1000
    assert image.size == (240,640) and image.getpixel((0,0)) == (255,0,0)
    image.close()


def test_tiny_prefix_contains_patches_without_running_any_trainable_block():
    service,model,loads = owner()
    image = Image.new("RGB",(100,100))
    observed = {"pipeline":PIPELINE_ID,"status":"ok","faces":[]}
    tokens,entries = service.encode_prefix([image],[observed])
    assert tokens[0].shape == (1,197,256) and all(layer.calls == 0 for layer in model.encoder.layers)
    service.encode_prefix([image],[observed],verify_boundary=True)
    assert service.last_prefix_check["passed_numerical_boundary"]
    assert all(layer.calls == 2 for layer in model.encoder.layers) and loads == ["local"]
    layers,norm,pooler = service.copy_trainable_tail()
    assert len(layers) == 10 and pooler(torch.ones(2,197,256)).shape == (2,256)
    with torch.no_grad():
        layers[0].weights.fill_(9.)
    assert model.encoder.layers[0].weights[0].item() == 1.
    with pytest.raises(ValueError,match="spatial contract"):
        service.encode([image],[observed])
    with pytest.raises(ValueError,match="text tower"):
        service.encode_texts(["not loaded"])
    assert image.getpixel((0,0)) == (0,0,0)
    service.close()
    image.close()


class Cache:
    spatial_spec = semantic.TINY_SPATIAL_SPEC
    def __init__(self):
        self.records = [dict(kinds=["whole"],masks=[0,0,0,0]) for _ in range(48)]
        self.values = np.random.default_rng(3).normal(size=(48,197,256)).astype(np.float32)
        self.calls = []
    def batch(self,indexes):
        self.calls.append(np.asarray(indexes).copy())
        return self.values[indexes],[self.records[index] for index in indexes]


def test_all_ten_blocks_receive_real_gradient_without_held_out_images(monkeypatch):
    from event_cull import engine
    monkeypatch.setattr(engine,"_load_preview",lambda *a,**k:pytest.fail("No epoch-time decoder"))
    service,base,_ = owner()
    frozen = service.copy_trainable_tail()
    cache = Cache()
    x = np.random.default_rng(4).normal(size=(48,3)).astype(np.float32)
    labels,training = np.tile([0,1],24),np.arange(48)<36
    owner_ranker = SemanticTailRanker(lambda:copy.deepcopy(frozen),epochs=1,
        spatial_spec=semantic.TINY_SPATIAL_SPEC,semantic_scale=16.)
    owner_ranker.fit(cache,x,labels,training,np.arange(48)//4)
    assert all(training[indexes].all() for indexes in cache.calls)
    for original,trained in zip(frozen[0],owner_ranker.model.layers):
        assert not torch.equal(original.weights,trained.weights)
        assert torch.isfinite(trained.weights.grad).all() and trained.weights.grad.abs().sum() > 0
    assert all(torch.all(layer.weights == 1.) for layer in base.encoder.layers)
    scores = owner_ranker.predict(cache,x,np.arange(36,48))
    assert scores.shape == (12,) and np.isfinite(scores).all()
    with pytest.raises(ValueError,match="contracts differ"):
        SemanticTailRanker(service.copy_trainable_tail).fit(cache,x,labels,training,np.arange(48)//4)
    owner_ranker.close()
    service.close()


def test_tiny_cache_rejects_siglip_contract_and_model_revision(tmp_path):
    spec = semantic.TINY_SPATIAL_SPEC
    np.ones((2,spec.token_count,spec.dimension),np.float16).tofile(tmp_path/"tokens.f16")
    report = dict(complete=True,pipeline=spec.pipeline,tail_layers=10,token_count=197,dimension=256,
                  dtype="float16",images=2,regions=2,smoke_subset=False,model="pinned",revision="one",
                  records=[dict(path=f"/photos/{i}.ARW",start=i,count=1,kinds=["whole"],masks=[0,0,0,0])
                           for i in range(2)])
    (tmp_path/"tokens.json").write_text(json.dumps(report))
    with pytest.raises(ValueError,match="incompatible"):
        SpatialTokenCache(tmp_path)
    with pytest.raises(ValueError,match="model/revision"):
        SpatialTokenCache(tmp_path,spatial_spec=spec,expected_model=("pinned","two"))
    cache = SpatialTokenCache(tmp_path,spatial_spec=spec,expected_model=("pinned","one"))
    assert cache.batch([0])[0].shape == (1,197,256)
    cache.close()


def checkpoint_fixture(tmp_path):
    vision = dict(hidden_size=256,num_hidden_layers=10,num_attention_heads=4,
                  intermediate_size=1024,image_size=224,patch_size=16,num_channels=3)
    config = dict(model_type="clip",vision_config=vision)
    (tmp_path/"config.json").write_text(json.dumps(config))
    (tmp_path/"preprocessor_config.json").write_text("{}")
    (tmp_path/"README.md").write_text("Test fixture")
    (tmp_path/"model.safetensors").write_bytes(b"synthetic checkpoint, not a real model")
    return config


def test_tiny_download_is_pinned_safe_and_not_a_startup_requirement(tmp_path,monkeypatch,provisioning):
    model = provisioning.TINY_RETENTION_MODEL
    assert model.key not in provisioning.MODEL_REGISTRY
    assert len(model.revision) == 40 and "pytorch_model.bin" not in model.allow_patterns
    checkpoint_fixture(tmp_path)
    with pytest.raises(provisioning.ModelDownloadError,match="SHA-256"):
        model.validate(tmp_path)
    monkeypatch.setattr(provisioning,"TINY_VISION_SHA256",
                        hashlib.sha256((tmp_path/"model.safetensors").read_bytes()).hexdigest())
    monkeypatch.setattr(provisioning,"TINY_CONFIG_SHA256",
                        hashlib.sha256((tmp_path/"config.json").read_bytes()).hexdigest())
    monkeypatch.setattr(provisioning,"TINY_PROCESSOR_SHA256",
                        hashlib.sha256((tmp_path/"preprocessor_config.json").read_bytes()).hexdigest())
    model.validate(tmp_path)
    (tmp_path/"model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(provisioning.ModelDownloadError,match="SHA-256"):
        model.validate(tmp_path)


def test_changed_preprocessor_cannot_silently_reuse_pinned_features(tmp_path,monkeypatch,provisioning):
    checkpoint_fixture(tmp_path)
    for constant,name in [("TINY_VISION_SHA256","model.safetensors"),("TINY_CONFIG_SHA256","config.json"),
                          ("TINY_PROCESSOR_SHA256","preprocessor_config.json")]:
        monkeypatch.setattr(provisioning,constant,hashlib.sha256((tmp_path/name).read_bytes()).hexdigest())
    provisioning.TINY_RETENTION_MODEL.validate(tmp_path)
    (tmp_path/"preprocessor_config.json").write_text('{"image_mean":[0,0,0]}')
    with pytest.raises(provisioning.ModelDownloadError,match="preprocessor_config.json SHA-256"):
        provisioning.TINY_RETENTION_MODEL.validate(tmp_path)


@pytest.mark.parametrize("fault",["large","remote"])
def test_wrong_or_remote_architecture_refused_before_loading(tmp_path,fault,provisioning):
    config = checkpoint_fixture(tmp_path)
    if fault == "large":
        config["vision_config"]["hidden_size"] = 1024
    else:
        config["auto_map"] = {"AutoModel":"remote.module"}
    (tmp_path/"config.json").write_text(json.dumps(config))
    with pytest.raises(provisioning.ModelDownloadError,match="architecture"):
        provisioning.TINY_RETENTION_MODEL.validate(tmp_path)
