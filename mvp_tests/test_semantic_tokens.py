"""Pretrained-prefix ownership and exact separation; no accuracy claim."""
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from src.core.image_features import semantic_regions as semantic
from src.core.image_features.human_analysis import PIPELINE_ID


class Layer(torch.nn.Module):
    def __init__(self,value):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(value))
        self.calls = 0
    def forward(self,hidden,mask):
        assert mask is None
        self.calls += 1
        return hidden+self.offset


class Embeddings(torch.nn.Module):
    def forward(self,pixels):
        return torch.ones((len(pixels),196,768),dtype=pixels.dtype,device=pixels.device)


class PrefixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.ones(1))
        self.embeddings = Embeddings()
        self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList([Layer(.1),Layer(.2),Layer(.3),Layer(.4)])
        self.post_layernorm = torch.nn.LayerNorm(768)
        self.head = torch.nn.Linear(768,768)
    def forward(self,pixel_values):
        hidden = self.embeddings(pixel_values)
        for layer in self.encoder.layers:
            hidden = layer(hidden,None)
        return SimpleNamespace(pooler_output=hidden.mean(1))


def owner(checkpoint=lambda:None):
    model, loads = PrefixModel(), []
    def factory(snapshot,device):
        loads.append(snapshot)
        return lambda images,return_tensors:{"pixel_values":torch.ones((len(images),3,224,224))}, model
    return semantic.SemanticRegionEncoder(resolver=lambda:"local",backend_factory=factory,
                                         device_resolver=lambda:"cpu",checkpoint=checkpoint), model, loads


def evidence():
    return {"pipeline":PIPELINE_ID,"status":"ok","faces":[]}


def test_prefix_stops_before_trainable_layers_and_same_owner_encodes_full():
    service, model, loads = owner()
    image = Image.new("RGB",(100,100))
    tokens, entries = service.encode_prefix([image],[evidence()])
    assert tokens[0].shape == (1,196,768) and tokens[0].dtype == np.float16
    assert np.allclose(tokens[0],1.3,atol=.001)
    assert [layer.calls for layer in model.encoder.layers] == [1,1,0,0]
    assert entries == [{"kinds":["whole"],"masks":[0,0,False,False]}]
    service.encode([image],[evidence()])
    assert loads == ["local"] and [layer.calls for layer in model.encoder.layers] == [2,2,1,1]
    assert image.getpixel((0,0)) == (0,0,0)
    image.close()
    service.close()


def test_tail_is_independent_of_frozen_base_parameters():
    service, model, _ = owner()
    layers,norm,head = service.copy_trainable_tail()
    with torch.no_grad():
        layers[0].offset.fill_(9.)
    assert model.encoder.layers[2].offset.item() == pytest.approx(.3)
    assert layers[0].offset.data_ptr() != model.encoder.layers[2].offset.data_ptr()
    assert norm.weight.data_ptr() != model.post_layernorm.weight.data_ptr()
    assert head.weight.data_ptr() != model.head.weight.data_ptr()
    service.close()


def test_cancelled_prefix_releases_crops_but_not_caller_preview():
    calls = [0]
    def checkpoint():
        calls[0] += 1
        if calls[0] >= 8:
            raise InterruptedError("cancelled between frozen layers")
    service, _, _ = owner(checkpoint)
    image = Image.new("RGB",(100,100))
    with pytest.raises(InterruptedError):
        service.encode_prefix([image],[evidence()])
    assert image.getpixel((0,0)) == (0,0,0)
    image.close()
    service.close()


def test_prefix_refuses_wrong_native_architecture_instead_of_fake_valid_tokens():
    service,model,_ = owner()
    model.encoder.layers = torch.nn.ModuleList([Layer(.1)])
    image = Image.new("RGB",(100,100))
    with pytest.raises(ValueError, match="architecture"):
        service.encode_prefix([image],[evidence()])
    image.close()
    service.close()
