from pathlib import Path
import shutil
import sys
import threading
import time

import cv2
import numpy as np
import pytest
from PIL import Image

import event_cull.engine as engine
from event_cull.engine import AnalysisOptions, PhotoResult


def test_jpeg_uses_reduced_decode(monkeypatch, tmp_path: Path):
    photo = tmp_path / "IMG_0001.JPG"
    photo.write_bytes(b"jpeg")
    calls = []

    def fake_read(path, flag):
        calls.append((path, flag))
        return np.zeros((1000, 2000, 3), dtype=np.uint8)

    monkeypatch.setattr(cv2, "imread", fake_read)
    preview = engine._load_preview(photo)

    assert calls == [(str(photo), cv2.IMREAD_REDUCED_COLOR_4)]
    assert preview.shape[:2] == (800, 1600)


def test_preview_respects_requested_display_limit(monkeypatch, tmp_path: Path):
    photo = tmp_path / "IMG_0001.JPG"
    photo.write_bytes(b"jpeg")

    monkeypatch.setattr(
        cv2,
        "imread",
        lambda _path, _flag: np.zeros((1000, 2000, 3), dtype=np.uint8),
    )

    preview = engine._load_preview(photo, maximum_dimension=320)

    assert preview.shape[:2] == (160, 320)


def test_heic_preview_registers_the_bundled_decoder(monkeypatch, tmp_path: Path):
    photo = tmp_path / "IMG_0001.HEIC"
    photo.write_bytes(b"heic")
    registrations = []
    source = Image.new("RGB", (640, 480), color=(30, 40, 50))

    class OpenedImage:
        def __enter__(self):
            return source

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(engine, "register_heif_opener", lambda: registrations.append(True))
    monkeypatch.setattr(engine.Image, "open", lambda _path: OpenedImage())

    preview = engine._load_preview(photo, maximum_dimension=320)

    assert registrations == [True]
    assert preview.shape[:2] == (240, 320)


def test_face_detectors_are_reused_within_worker_thread(monkeypatch):
    created = []

    class FakeCascade:
        def __init__(self):
            created.append(True)

        def read(self, _node):
            return True

        def empty(self):
            return False

        def detectMultiScale(self, *args, **kwargs):
            return []

    monkeypatch.setattr(engine, "_DETECTORS", threading.local())
    monkeypatch.setattr(cv2, "CascadeClassifier", FakeCascade)
    gray = np.zeros((200, 200), dtype=np.uint8)

    engine._face_reasons(gray, AnalysisOptions())
    engine._face_reasons(gray, AnalysisOptions())

    assert len(created) == 3


def test_face_detectors_use_frozen_bundle_when_opencv_source_path_is_stale(monkeypatch, tmp_path):
    model_names = (
        "haarcascade_frontalface_default.xml",
        "haarcascade_eye_tree_eyeglasses.xml",
        "haarcascade_smile.xml",
    )
    bundle_root = tmp_path / "秒选-Windows"
    bundled = bundle_root / "cv2" / "data"
    bundled.mkdir(parents=True)
    for name in model_names:
        shutil.copyfile(Path(cv2.data.haarcascades) / name, bundled / name)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)
    monkeypatch.setattr(cv2.data, "haarcascades", str(tmp_path / "old-build-path"))
    monkeypatch.setattr(engine, "_DETECTORS", threading.local())

    for name in model_names:
        assert engine._cascade_path(name) == bundled / name
    engine._face_reasons(np.zeros((200, 200), dtype=np.uint8), AnalysisOptions())


def test_missing_face_detector_stops_analysis_instead_of_scoring_zero(monkeypatch, tmp_path):
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (64, 64), "#7287a0").save(photo)
    monkeypatch.setattr(engine, "_cascade_path", lambda name: tmp_path / name)

    with pytest.raises(engine.AnalysisSetupError, match="人脸检测模型无法加载"):
        engine.analyze_paths([photo])


def test_people_priority_favors_clear_frontal_faces_and_smiles_without_unbounded_bonus():
    plain = engine._people_priority_bonus(0, 0)
    group = engine._people_priority_bonus(4, 2)
    crowd = engine._people_priority_bonus(40, 40)

    assert plain == 0
    assert group > plain
    assert crowd == engine._people_priority_bonus(6, 6)


def test_analyze_adds_people_priority_to_quality_score(monkeypatch, tmp_path: Path):
    photo = tmp_path / "IMG_0001.JPG"
    photo.write_bytes(b"jpeg")
    image = np.full((240, 320, 3), 127, dtype=np.uint8)
    monkeypatch.setattr(engine, "_load_preview", lambda _path: image)
    monkeypatch.setattr(
        engine,
        "_face_metrics",
        lambda _gray, _options, _sharpness: ([], 100.0, 4, 2, 4, 150.0),
    )

    result = engine._analyze(photo, [photo], AnalysisOptions())

    assert result.frontal_face_count == 4
    assert result.smiling_face_count == 2
    assert result.open_eye_face_count == 4
    assert result.score >= engine._people_priority_bonus(4, 2)
    assert "4 张清晰正面脸，2 张疑似笑脸" in result.reasons[0]


def test_subject_detail_prevents_background_bokeh_from_being_called_blur():
    assert engine._effective_sharpness(39.31, 133.3) > AnalysisOptions().blur_floor


def test_second_analysis_uses_cache_until_source_changes(monkeypatch, tmp_path: Path):
    photo = tmp_path / "IMG_0001.JPG"
    photo.write_bytes(b"first")
    calls = []

    def fake_analyze(path, pairs, options):
        calls.append(path)
        return PhotoResult(
            path,
            pairs,
            score=80,
            reasons=[],
            sharpness=100,
            highlight_clip=0,
            shadow_clip=0,
            perceptual_hash=1,
            frontal_face_count=3,
            smiling_face_count=1,
        )

    monkeypatch.setattr(engine, "_analyze", fake_analyze)
    engine.analyze_paths([photo])
    cached = engine.analyze_paths([photo])
    assert calls == [photo]
    assert cached[0].frontal_face_count == 3
    assert cached[0].smiling_face_count == 1

    photo.write_bytes(b"changed-size")
    engine.analyze_paths([photo])
    assert calls == [photo, photo]


def test_worker_count_scales_conservatively_for_nested_image_processing():
    assert engine._analysis_worker_count(2) == 2
    assert engine._analysis_worker_count(4) == 4
    assert engine._analysis_worker_count(8) == 4
    assert engine._analysis_worker_count(10) == 6
    assert engine._analysis_worker_count(16) == 8


def test_adaptive_workers_run_in_parallel_and_return_original_order(monkeypatch, tmp_path: Path):
    photos = []
    for index in range(8):
        photo = tmp_path / f"IMG_{index:02d}.JPG"
        photo.write_bytes(str(index).encode())
        photos.append(photo)

    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_analyze(path, pairs, options):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return PhotoResult(
            path,
            pairs,
            score=80,
            reasons=[],
            sharpness=100,
            highlight_clip=0,
            shadow_clip=0,
            perceptual_hash=int(path.stem[-2:]),
        )

    monkeypatch.setattr(engine, "_analyze", fake_analyze)
    monkeypatch.setattr(engine, "_same_group", lambda previous, current, options: False)
    monkeypatch.setattr(engine, "_analysis_worker_count", lambda: 4)
    results = engine.analyze_paths(photos)

    assert maximum_active == 4
    assert [result.path for result in results] == photos
