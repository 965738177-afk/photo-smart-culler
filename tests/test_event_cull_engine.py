from pathlib import Path

from event_cull.engine import AnalysisOptions, PhotoResult


def test_target_count_keeps_one_lead_per_group_before_filling_global_slots(monkeypatch, tmp_path: Path):
    paths = []
    for number in range(4):
        path = tmp_path / f"IMG_{number}.jpg"
        path.write_bytes(b"fixture")
        paths.append(path)

    values = iter(
        [
            PhotoResult(paths[0], score=90, perceptual_hash=1),
            PhotoResult(paths[1], score=80, perceptual_hash=1),
            PhotoResult(paths[2], score=70, perceptual_hash=10),
            PhotoResult(paths[3], score=60, perceptual_hash=10),
        ]
    )

    import event_cull.engine as engine

    monkeypatch.setattr(engine, "_representatives", lambda folder: [(path, [path]) for path in paths])
    monkeypatch.setattr(engine, "_analyze", lambda path, pairs, options: next(values))
    monkeypatch.setattr(engine, "_same_group", lambda previous, current, options: previous.perceptual_hash == current.perceptual_hash)

    results = engine.analyze_folder(tmp_path, AnalysisOptions(target_keep_count=2))

    assert [item.suggested_rating for item in results] == [3, 2, 3, 2]


def test_single_capture_stays_in_review_instead_of_becoming_an_unrated_reject(monkeypatch, tmp_path: Path):
    path = tmp_path / "only.jpg"
    path.write_bytes(b"fixture")
    result = PhotoResult(path, score=80, perceptual_hash=1)
    import event_cull.engine as engine

    monkeypatch.setattr(engine, "_representatives", lambda folder: [(path, [path])])
    monkeypatch.setattr(engine, "_analyze", lambda path, pairs, options: result)
    results = engine.analyze_folder(tmp_path)

    assert results[0].suggested_rating == 2
    assert "单张画面保留，建议人工确认" in results[0].reasons


def test_relative_subject_darkening_is_a_flash_misfire_and_is_not_selected(tmp_path: Path):
    """A dark face in an otherwise stable burst must not survive scene coverage."""
    import event_cull.engine as engine

    frames = [
        PhotoResult(tmp_path / "normal-left.jpg", score=80, perceptual_hash=42, median_luminance=130, subject_luminance=160),
        PhotoResult(tmp_path / "flash-miss.jpg", score=98, perceptual_hash=42, median_luminance=128, subject_luminance=100),
        PhotoResult(tmp_path / "normal-right.jpg", score=82, perceptual_hash=42, median_luminance=129, subject_luminance=162),
    ]

    engine._assign_tiered_ratings([frames], AnalysisOptions(target_keep_count=3))

    missed = frames[1]
    assert missed.relative_exposure_ev is not None
    assert missed.relative_exposure_ev <= -1.0
    assert missed.suggested_rating == 0
    assert "疑似闪光灯漏闪" in "；".join(missed.reasons)
