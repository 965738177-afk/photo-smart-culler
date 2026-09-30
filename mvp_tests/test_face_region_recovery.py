"""Shared region ownership/protocol/cache tests, not culling accuracy proof."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from event_cull import engine
from src.core.image_features import human_analysis as human, human_worker
from src.core.image_features.face_analysis import FaceRegionDetector, FACE_REGION_PIPELINE


def detection(x=30, y=40, width=40, height=50):
    return [x, y, width, height, 40, 50, 60, 50, 50, 60, 40, 70, 60, 70, .99]


class Detector:
    def __init__(self, values=None):
        self.values = np.asarray([detection()] if values is None else values, dtype=np.float32)
        self.sizes, self.images, self.closed = [], [], 0
    def setInputSize(self, size):
        self.sizes.append(size)
    def detect(self, image):
        self.images.append(image.copy())
        return None, self.values
    def close(self):
        self.closed += 1


def test_detector_has_one_lazy_backend_and_original_pixel_coordinates():
    backend, created = Detector(), []
    owner = FaceRegionDetector(model_path_resolver=lambda:Path("fixture.onnx"),
                              backend_factory=lambda path:created.append(path) or backend)
    image = np.full((800, 1600, 3), (30, 20, 10), dtype=np.uint8)
    for _ in range(3):
        result = owner.detect(image)
        assert np.allclose(result[0, :14], np.asarray(detection()[:14])/0.4)
        assert result[0, 14] == pytest.approx(.99)
    assert created == ["fixture.onnx"] and backend.sizes == [(640, 320)]*3
    assert np.array_equal(backend.images[0][0, 0], [30, 20, 10])
    owner.close()
    owner.close()
    assert backend.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        owner.detect(image)


@pytest.mark.parametrize("change", ["shape", "nan", "negative-size", "score", "unbounded"])
def test_bad_native_proposals_fail_not_unknown_good_faces(change):
    value = np.asarray([detection()], np.float32)
    if change == "shape": value = value[:, :14]
    elif change == "nan": value[0, 0] = np.nan
    elif change == "negative-size": value[0, 2] = -1
    elif change == "score": value[0, 14] = 2
    else: value = np.repeat(value, 101, axis=0)
    owner = FaceRegionDetector(model_path_resolver=lambda:Path("fixture"), backend_factory=lambda _:Detector(value))
    with pytest.raises(ValueError, match="Invalid face region"):
        owner.detect(np.zeros((100, 100, 3), np.uint8))
    owner.close()


def test_border_proposals_are_clipped_without_changing_confidence_or_landmarks():
    value = [detection(-5, -5, 20, 30), detection(105, 105, 20, 30)]
    owner = FaceRegionDetector(model_path_resolver=lambda:Path("fixture"), backend_factory=lambda _:Detector(value))
    result = owner.detect(np.zeros((100, 100, 3), np.uint8))
    assert len(result) == 1 and list(result[0, :4]) == [0, 0, 15, 25]
    assert np.allclose(result[0, 4:], np.asarray(value[0])[4:])
    owner.close()


def test_invalid_images_or_cancel_do_not_initialize_backend():
    def forbidden(_): pytest.fail("no model initialization")
    owner = FaceRegionDetector(model_path_resolver=lambda:Path("fixture"), backend_factory=forbidden)
    for image, maximum in [(np.zeros((1601, 10, 3), np.uint8), 640),
                           (np.zeros((20, 20, 3), np.float32), 640),
                           (np.zeros((20, 20, 3), np.uint8), 1000),
                           (np.zeros((20, 20, 3), np.uint8), True)]:
        with pytest.raises(ValueError, match="bounded"):
            owner.detect(image, maximum_dimension=maximum)
    def stop(): raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        owner.detect(np.zeros((20, 20, 3), np.uint8), checkpoint=stop)
    owner.close()


class EmptyFace:
    def __init__(self): self.calls, self.closed = 0, 0
    def detect_result(self, image):
        self.calls += 1
        return SimpleNamespace(face_landmarks=[], face_blendshapes=[], facial_transformation_matrixes=[])
    def close(self): self.closed += 1


class EmptyPose:
    def __init__(self): self.calls, self.closed = 0, 0
    def detect_result(self, image):
        self.calls += 1
        return SimpleNamespace(pose_landmarks=[])
    def close(self): self.closed += 1


def test_recovery_reuses_models_and_caps_native_roi_inspections():
    face, pose = EmptyFace(), EmptyPose()
    backend = Detector([detection(10+20*i, 10, 15, 20) for i in range(14)])
    service = human.HumanAnalysisService(face_factory=lambda _:face, pose_factory=lambda _:pose,
        resolver=Path, recover_faces=True, region_factory=lambda _:backend)
    image = np.zeros((320, 640, 3), np.uint8)
    result = service.inspect(image)
    assert result["region_pipeline"] == FACE_REGION_PIPELINE
    assert result["region_proposal_count"] == 14 and result["region_crop_cap"] == 12
    assert face.calls == 13 and pose.calls == 1 and len(backend.images) == 1
    assert result["faces"] == [] and not any(result["features"])
    assert "all_people_found" not in result and "action_complete" not in result
    service.close()
    assert face.closed == pose.closed == backend.closed == 1


def test_disabled_recovery_keeps_original_path_and_never_initializes_detector():
    face, pose = EmptyFace(), EmptyPose()
    service = human.HumanAnalysisService(face_factory=lambda _:face, pose_factory=lambda _:pose,
        resolver=Path, region_factory=lambda _:pytest.fail("no new detector"))
    observed = service.inspect(np.zeros((100, 100, 3), np.uint8))
    assert face.calls == pose.calls == 1 and "region_pipeline" not in observed
    service.close()


def test_recovery_signature_includes_yunet_bytes_and_preserves_original_signature(tmp_path, monkeypatch):
    names = ("face_landmarker.task", "pose_landmarker_lite.task", "face_detection_yunet_2023mar.onnx")
    for name in names: (tmp_path/name).write_bytes(name.encode())
    monkeypatch.setattr(human, "bundled_model", lambda name:tmp_path/name)
    before = human.model_signature()
    expected = hashlib.sha256((human.PIPELINE_ID+human.DETAIL_PIPELINE_ID).encode())
    for name in names[:2]: expected.update(name.encode())
    assert before == expected.hexdigest()
    recovered = human.model_signature(recover_faces=True)
    (tmp_path/names[2]).write_bytes(b"changed model")
    assert human.model_signature() == before
    assert human.model_signature(recover_faces=True) != recovered != before


def test_recovery_evidence_cannot_reuse_unrecovered_person_cache(tmp_path, monkeypatch):
    source = tmp_path/"one.jpg"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(engine, "_human_model_signature", lambda **kw:"recovered" if kw else "original")
    original = engine._cache_signature(source, engine.AnalysisOptions(human_analysis_enabled=True))
    recovered = engine._cache_signature(source, engine.AnalysisOptions(
        human_analysis_enabled=True, human_face_recovery_enabled=True))
    assert "human_face_region_pipeline" not in original and recovered != original
    row = engine.PhotoResult(source, sharpness=100, highlight_clip=0, shadow_clip=0, perceptual_hash=1,
                             person_evidence=human.aggregate_observation([], []))
    record = engine._cache_record(row, recovered)
    assert engine._result_from_cache(source, [source], record, recovered) is None
    row.person_evidence["region_pipeline"] = FACE_REGION_PIPELINE
    record = engine._cache_record(row, recovered)
    assert engine._result_from_cache(source, [source], record, recovered) is not None


def test_recovery_without_human_evidence_is_rejected_before_scanning(monkeypatch):
    monkeypatch.setattr(engine, "_paths_for_selected_files", lambda *_:pytest.fail("no scanning"))
    with pytest.raises(ValueError, match="requires enabled"):
        engine.analyze_paths([], engine.AnalysisOptions(human_face_recovery_enabled=True))


def request(recover):
    header = json.dumps({"shape":[32, 32], "regions":[], "recover_faces":recover}).encode()
    return struct.pack("!I", len(header))+header+bytes(32*32*3)


def test_worker_mode_is_immutable_for_one_owned_runtime(monkeypatch):
    created, closed = [], []
    class Service:
        def __init__(self, *, recover_faces): created.append(recover_faces)
        def inspect(self, *_a, **_k): return human.aggregate_observation([], [])
        def close(self): closed.append(True)
    monkeypatch.setattr(human, "HumanAnalysisService", Service)
    monkeypatch.setattr(human_worker.sys, "stdin", SimpleNamespace(buffer=BytesIO(request(False)+request(True))))
    with pytest.raises(ValueError, match="mode changed"):
        human_worker.main()
    assert created == [False] and closed == [True]


@pytest.mark.parametrize("mode", [1, "yes", None])
def test_worker_invalid_recovery_mode_never_constructs_model(monkeypatch, mode):
    monkeypatch.setattr(human, "HumanAnalysisService", lambda **_:pytest.fail("invalid mode"))
    monkeypatch.setattr(human_worker.sys, "stdin", SimpleNamespace(buffer=BytesIO(request(mode))))
    with pytest.raises(ValueError, match="invalid face recovery"):
        human_worker.main()


def test_training_and_test_must_use_the_same_recovery_observations():
    from scripts.evaluate_burst_sets import require_comparable_observations, validate_recovery_measurements
    before = engine.AnalysisOptions(human_analysis_enabled=True)
    after = engine.AnalysisOptions(human_analysis_enabled=True, human_face_recovery_enabled=True)
    require_comparable_observations(before, before)
    require_comparable_observations(after, after)
    with pytest.raises(ValueError, match="pipelines differ"):
        require_comparable_observations(before, after)
    with pytest.raises(ValueError, match="incomplete"):
        validate_recovery_measurements([{"person_evidence":human.aggregate_observation([], [])}],
                                       {"human_analysis_enabled":True, "human_face_recovery_enabled":True})
    validate_recovery_measurements([{"person_evidence":{"region_pipeline":FACE_REGION_PIPELINE}}],
                                  {"human_analysis_enabled":True, "human_face_recovery_enabled":True})
