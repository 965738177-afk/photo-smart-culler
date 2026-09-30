from pathlib import Path

import numpy as np

from event_cull.calibration import (
    CalibrationProfile,
    _source_previews,
    apply_profile,
    build_profile,
    exported_selection_stems,
    read_xmp_rating,
    train_profile,
)
from event_cull.engine import PhotoResult


def test_reads_embedded_xmp_rating(tmp_path: Path):
    image = tmp_path / "rated.jpg"
    image.write_bytes(b"prefix xmp:Rating=\"3\" suffix")

    assert read_xmp_rating(image) == 3


def test_final_selection_uses_filenames_without_embedded_xmp(tmp_path: Path):
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "IMG_0001.JPG").write_bytes(b"plain jpeg without XMP")
    (selected / "IMG_0002.ARW").write_bytes(b"raw fixture")

    assert exported_selection_stems(selected) == {"img_0001", "img_0002"}


def test_profile_trains_with_group_held_out_validation():
    samples = []
    for index in range(100):
        label = 3 if index % 5 == 0 else 0
        feature = np.asarray([float(label), float(index % 3)], dtype=np.float32)
        samples.append((feature, label, f"group-{index // 2}"))

    profile = train_profile(samples, featured_rating=3)

    assert profile.training_examples == 100
    assert profile.positive_examples == 20
    assert 0.5 <= profile.validation_auc <= 1.0


def test_profile_only_reorders_featured_tier(monkeypatch):
    import event_cull.calibration as calibration

    profile = CalibrationProfile(
        mean=(0.0,), scale=(1.0,), weights=(1.0,), bias=0.0,
        featured_rating=3, training_examples=100, positive_examples=20, validation_auc=.7,
    )
    monkeypatch.setattr(calibration, "_load_preview", lambda path: np.full((2, 2, 3), int(path.stem[-1]), dtype=np.uint8))
    monkeypatch.setattr(calibration, "_visual_descriptor", lambda image: np.asarray([float(image[0, 0, 0])]))
    first = PhotoResult(Path("first1.jpg"), group_id=1, suggested_rating=3, score=90, pair_paths=[Path("first1.jpg")])
    better_same_group = PhotoResult(Path("second4.jpg"), group_id=1, suggested_rating=1, score=70, pair_paths=[Path("second4.jpg")])
    second_group = PhotoResult(Path("third2.jpg"), group_id=2, suggested_rating=2, score=80, pair_paths=[Path("third2.jpg")])

    chosen = apply_profile([first, better_same_group, second_group], profile)

    assert chosen == 1
    assert first.suggested_rating == 1
    assert better_same_group.suggested_rating == 3
    assert second_group.suggested_rating == 2


def test_build_profile_reads_log_exports_and_originals(tmp_path: Path):
    import cv2

    originals = tmp_path / "originals"
    exports = tmp_path / "exports"
    originals.mkdir()
    exports.mkdir()
    log = tmp_path / "culling.csv"
    lines = ["file,group,rating"]
    for index in range(80):
        name = f"DSC{index:04d}"
        image = np.full((40, 40, 3), 230 if index % 5 == 0 else 30, dtype=np.uint8)
        assert cv2.imwrite(str(originals / f"{name}.jpg"), image)
        lines.append(f"{name}.jpg,group-{index},0")
        if index % 5 == 0:
            export = exports / f"{name}.jpg"
            assert cv2.imwrite(str(export), image)
            with export.open("ab") as stream:
                stream.write(b' xmp:Rating="3"')
    log.write_text("\n".join(lines), encoding="utf-8")

    profile = build_profile(log, exports, [originals])

    assert profile.training_examples == 80
    assert profile.positive_examples == 16
    assert len(profile.weights) >= 256
    assert profile.feature_pipeline in {"handcrafted-visual-v1", "mobilenet-v3-small-visual-v2"}
    assert profile.objective == "retention"
    assert profile.featured_rating == 1  # exported original stars are not retouch truth
    assert profile.validation_precision is not None


def test_calibration_ignores_curated_and_prior_export_copies(tmp_path: Path):
    from event_cull.export_selected import EXPORT_MARKER

    original = tmp_path / "IMG_0001.HEIC"
    original.write_bytes(b"original")
    curated = tmp_path / "选"
    curated.mkdir()
    (curated / original.name).write_bytes(b"curated copy")
    prior_export = tmp_path / "秒选-3"
    prior_export.mkdir()
    (prior_export / EXPORT_MARKER).write_text("", encoding="utf-8")
    (prior_export / original.name).write_bytes(b"export copy")

    assert _source_previews([tmp_path], {"img_0001"}, excluded_folders=[curated]) == {
        "img_0001": original
    }
