"""CPU experiment isolation; these engineering controls are not quality truth."""
from dataclasses import replace

import numpy as np
import pytest

from event_cull import calibration, deep_features, engine
from scripts.check_deep_backend_consistency import check_backend
from scripts.evaluate_burst_sets import require_comparable_observations


def test_cpu_contract_forces_the_existing_shared_encoder(monkeypatch):
    calls = []
    def encode(images, *, backend):
        calls.append(backend)
        return np.ones((len(images), 576), np.float32), "cpu"
    monkeypatch.setattr(calibration, "encode_images", encode)
    matrix = calibration._feature_matrix([np.zeros((32,32,3), np.uint8)], calibration.CPU_DEEP_PIPELINE_ID)
    assert matrix.shape == (1,832) and calls == ["cpu"]
    assert calibration.CPU_DEEP_PIPELINE_ID != calibration.DEEP_PIPELINE_ID


def test_cpu_contract_rejects_masquerading_provider(monkeypatch):
    monkeypatch.setattr(calibration, "encode_images", lambda images, **kw: (np.ones((1,576)), "coreml"))
    with pytest.raises(calibration.CalibrationError):
        calibration._feature_matrix([np.zeros((32,32,3),np.uint8)], calibration.CPU_DEEP_PIPELINE_ID)


def test_explicit_cpu_does_not_choose_or_mutate_coreml_owner(monkeypatch):
    created = []
    class Fake:
        def __init__(self, backend):
            created.append(backend)
            self.backend = backend
        def encode(self, images):
            return np.zeros((len(images),576), np.float32)
    old = object()
    monkeypatch.setattr(deep_features, "_ENCODERS", {"coreml": old})
    monkeypatch.setattr(deep_features, "DeepVisualEncoder", Fake)
    monkeypatch.setattr(deep_features, "preferred_backend", lambda: "coreml")
    for _ in range(2):
        values, backend = deep_features.encode_images([np.zeros((1,1,3),np.uint8)], backend="cpu")
        assert values.shape == (1,576) and backend == "cpu"
    assert created == ["cpu"] and deep_features._ENCODERS["coreml"] is old


@pytest.mark.parametrize("backend", ["metal", "unavailable", "", True])
def test_explicit_unknown_backend_refused(monkeypatch, backend):
    monkeypatch.setattr(deep_features, "preferred_backend", lambda: "cpu")
    with pytest.raises(ValueError):
        deep_features.encode_images([], backend=backend)


def test_encoder_close_prevents_reloading():
    owner = deep_features.DeepVisualEncoder("cpu")
    owner.close()
    owner.close()
    with pytest.raises(deep_features.DeepFeatureUnavailable, match="closed"):
        owner.encode([np.zeros((32,32,3),np.uint8)])


def test_cross_case_cpu_and_legacy_measurements_cannot_mix():
    options = engine.AnalysisOptions(calibration_feature_pipeline=calibration.CPU_DEEP_PIPELINE_ID)
    require_comparable_observations(options, options)
    with pytest.raises(ValueError, match="visual measurement"):
        require_comparable_observations(options, replace(options,calibration_feature_pipeline=calibration.DEEP_PIPELINE_ID))


def test_cpu_cache_rejects_truncated_descriptors(tmp_path):
    path = tmp_path/"source.jpg"
    path.write_bytes(b"cache signature fixture")
    options = engine.AnalysisOptions(calibration_feature_pipeline=calibration.CPU_DEEP_PIPELINE_ID)
    signature = engine._cache_signature(path, options)
    item = engine.PhotoResult(path, sharpness=100., highlight_clip=0., shadow_clip=0., perceptual_hash=0,
                              visual_features=[0.]*832,
                              visual_feature_pipeline=calibration.CPU_DEEP_PIPELINE_ID)
    record = engine._cache_record(item, signature)
    assert engine._result_from_cache(path, [], record, signature) is not None
    record["visual_features"] = [0.]*16
    assert engine._result_from_cache(path, [], record, signature) is None


def test_cpu_archive_refuses_wrong_dimensions_without_inference():
    from pathlib import Path
    from scripts.benchmark_event_cull import frozen_visual_features
    item = engine.PhotoResult(Path("fixture.ARW"), visual_features=[0.]*16,
                              visual_feature_pipeline=calibration.CPU_DEEP_PIPELINE_ID)
    with pytest.raises(ValueError, match="dimensions"):
        frozen_visual_features([item],calibration.CPU_DEEP_PIPELINE_ID)


def test_control_reports_backend_switch_even_when_values_match():
    owners = []
    class Fake:
        def __init__(self, backend):
            self.backend, self.closed = backend, False
            owners.append(self)
        def encode(self, images):
            if len(images)>1:
                self.backend = "cpu"
            return np.ones((len(images),576),np.float32)
        def close(self):
            self.closed = True
    report = check_backend([np.zeros((2,2,3),np.uint8)]*2, "coreml", factory=Fake)
    assert report["backend_switched"] and not report["consistent_on_control"]
    assert owners[0].closed


def test_control_releases_owner_on_invalid_output():
    owners = []
    class Fake:
        def __init__(self, backend):
            self.backend, self.closed = backend, False
            owners.append(self)
        def encode(self, images):
            return np.full((len(images),576),np.nan)
        def close(self):
            self.closed = True
    with pytest.raises(ValueError, match="nonfinite"):
        check_backend([np.zeros((2,2,3),np.uint8)]*2,"cpu",factory=Fake)
    assert owners[0].closed
