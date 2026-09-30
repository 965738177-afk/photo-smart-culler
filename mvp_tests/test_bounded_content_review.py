"""Throughput and bounded inference regressions at the shared engine boundary."""
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from event_cull import engine
from event_cull import bundled_inference as inference
from event_cull.app import EventCullApp


def fixture_batch(tmp_path, monkeypatch, total=6):
    paths = [tmp_path / f"{i}.jpg" for i in range(total)]
    for p in paths:
        p.write_bytes(b"fixture")
    monkeypatch.setattr(engine, "_load_analysis_cache", lambda: {})
    monkeypatch.setattr(engine, "_write_analysis_cache", lambda cache: None)
    calls = []

    def analyze(path, pairs, options, cache):
        assert not options.content_model_enabled
        calls.append(path)
        return engine.PhotoResult(path, pair_paths=pairs, score=95, sharpness=200,
                                  subject_sharpness=200, perceptual_hash=int(path.stem)*10,
                                  median_luminance=100, highlight_clip=0, shadow_clip=0), str(path), None

    monkeypatch.setattr(engine, "_analyze_cached", analyze)
    monkeypatch.setattr(engine, "_load_preview", lambda _: np.zeros((16, 16, 3), np.uint8))
    return paths, calls


def observation():
    return {"status": "ok", "content_type": "documentary", "section": 11,
            "checks": dict.fromkeys(("subject_usable", "expression_appropriate", "action_complete",
                                     "group_complete", "gaze_coherent"), "pass"),
            "observations": [], "rule_ids": []}


def test_companion_expansion_scans_once_per_folder_and_treats_stems_as_literals(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("[1].ARW", "[1].jpg", "1.JPG", "x.PNG", "x.XMP")]
    for p in paths:
        p.write_bytes(b"fixture")
    scanned = []
    scan = engine.os.scandir
    monkeypatch.setattr(engine.os, "scandir", lambda folder: scanned.append(folder) or scan(folder))
    assert engine._paths_for_selected_files([paths[0], paths[3]]) == sorted([paths[0], paths[1], paths[3]])
    assert scanned == [tmp_path]


def test_base_path_never_constructs_content_model(tmp_path, monkeypatch):
    paths, calls = fixture_batch(tmp_path, monkeypatch)
    monkeypatch.setattr(engine, "BundledWeddingContentModel", lambda *args: pytest.fail("slow model started"))
    result = engine.analyze_paths(paths, engine.AnalysisOptions(scene_type="婚礼"))
    assert set(calls) == set(paths)
    assert all(item.content_evidence["status"] == "base_only" for item in result)
    assert all(item.suggested_rating in {0, 1, 3} for item in result)


def test_bounded_review_starts_only_after_all_fast_metrics_and_stops_at_budget(tmp_path, monkeypatch):
    paths, calls = fixture_batch(tmp_path, monkeypatch)
    closed, inspected, phases = [], [], []

    class Model:
        fingerprint = "fixture"
        ready = True
        runtime = SimpleNamespace(deadline=None)

        def __init__(self, *args):
            assert len(calls) == len(paths)

        def inspect(self, *args):
            inspected.append(1)
            if len(inspected) == 2:
                raise inference.InferenceBudgetExpired()
            return observation()

        def close(self):
            closed.append(1)

    monkeypatch.setattr(engine, "BundledWeddingContentModel", Model)
    options = engine.AnalysisOptions(scene_type="婚礼", content_model_enabled=True, content_review_mode="limited")
    result = engine.analyze_paths(paths, options, phase_callback=lambda *args: phases.append(args))
    assert len(inspected) == 2 and closed == [1] and phases
    assert sum(item.content_evidence["status"] == "ok" for item in result) == 1
    assert all(item.suggested_rating != 3 for item in result if item.content_evidence["status"] != "ok")
    assert all(item.suggested_rating != 2 for item in result)


def test_expired_budget_skips_model_but_returns_complete_base_results(tmp_path, monkeypatch):
    paths, calls = fixture_batch(tmp_path, monkeypatch)
    monkeypatch.setattr(engine, "_review_budget_seconds", lambda total: 0.)
    monkeypatch.setattr(engine, "BundledWeddingContentModel", lambda *args: pytest.fail("expired budget started model"))
    result = engine.analyze_paths(paths, engine.AnalysisOptions(scene_type="婚礼", content_model_enabled=True, content_review_mode="limited"))
    assert len(result) == len(paths) and len(calls) == len(paths)
    assert all(item.content_evidence["status"] == "not_reviewed" for item in result)


def test_review_candidates_spread_groups_and_do_not_mutate_ranked_state():
    items = [engine.PhotoResult(Path(f"{i}.jpg"), score=95, sharpness=200, perceptual_hash=i*10)
             for i in range(8)]
    options = engine.AnalysisOptions(scene_type="婚礼", detect_duplicates=False)
    candidates = engine._review_candidates(items, options)
    assert candidates[0] is items[4]
    assert {item.path for item in candidates} == {item.path for item in items}
    assert all(item.suggested_rating == 0 and not item.reasons for item in items)


def test_cancel_during_limited_review_closes_model_and_never_publishes_results(tmp_path, monkeypatch):
    paths, _ = fixture_batch(tmp_path, monkeypatch)
    cancel, closed = threading.Event(), []

    class Model:
        ready = True
        fingerprint = "fixture"
        runtime = SimpleNamespace(deadline=None)

        def __init__(self, *args): pass

        def inspect(self, *args):
            cancel.set()
            raise engine.ContentModelCancelled()

        def close(self): closed.append(1)

    monkeypatch.setattr(engine, "BundledWeddingContentModel", Model)
    with pytest.raises(engine.AnalysisCancelled):
        engine.analyze_paths(paths, engine.AnalysisOptions(scene_type="婚礼", content_model_enabled=True,
                                                          content_review_mode="limited"), cancel_event=cancel)
    assert closed == [1]


def test_inference_deadline_is_checked_before_loading_assets():
    client = inference.BundledInference(SimpleNamespace(verify=lambda _: pytest.fail("assets loaded")))
    client.deadline = 0
    with pytest.raises(inference.InferenceBudgetExpired):
        client.start()


def test_user_budget_scales_only_up_to_ten_minutes():
    assert engine._review_budget_seconds(1000) == 120
    assert engine._review_budget_seconds(5049) == 600
    assert engine._review_budget_seconds(100000) == 600


def test_default_picker_reset_does_not_reenable_slow_model():
    source = Path(engine.__file__).with_name("app.py").read_text(encoding="utf-8")
    assert "self.content_model_enabled = tk.BooleanVar(value=False)" in source
    assert '("content_model_enabled", False)' in source
    assert 'content_review_mode="limited"' in source


def test_missing_source_cannot_crash_grouping(tmp_path):
    first = engine.PhotoResult(tmp_path / "moved.jpg", perceptual_hash=0)
    second = engine.PhotoResult(tmp_path / "missing.jpg", perceptual_hash=0)
    assert not engine._same_group(first, second, engine.AnalysisOptions())


def test_frozen_baseline_survives_later_manual_log_rewrite(tmp_path):
    subject = object.__new__(EventCullApp)
    subject.current_log_path = tmp_path / "analysis.json"
    subject.current_log_path.write_text("{}", encoding="utf-8")
    subject._analysis_log_payload = lambda: {"photos": [{"rating": 3}], "manual_ratings": {}}
    subject._freeze_analysis_baseline()
    baseline = tmp_path / "baselines" / "analysis.json"
    before = baseline.read_bytes()
    subject.current_log_path.write_text('{"rating":0}', encoding="utf-8")
    subject._analysis_log_payload = lambda: {"photos": [{"rating": 0}], "manual_ratings": {"a": 0}}
    subject._freeze_analysis_baseline()
    assert baseline.read_bytes() == before


@pytest.mark.parametrize("faces", [2, 3])
def test_wedding_small_face_with_absolute_focus_pass_is_not_rejected_by_leafy_background(faces):
    # Reproduces the measured DSC04531 discrepancy, not a filename whitelist.
    options = engine.AnalysisOptions(scene_type="婚礼")
    item = engine.PhotoResult(Path("small-subject.jpg"), perceptual_hash=1, score=105,
                              sharpness=496.71, subject_sharpness=81.03,
                              frontal_face_count=faces, open_eye_face_count=0,
                              face_coverage=.0243, eye_detection_mode="scene_small_face")
    assert not engine._subject_motion_fault(item, options)
    engine._assign_tiered_ratings([[item]], options)
    assert item.suggested_rating == 1
    assert item.featured_blocked  # Also blocks a two-person ambiguous frame.


def test_small_subject_still_fails_absolute_focus_floor():
    options = engine.AnalysisOptions(scene_type="婚礼")
    item = engine.PhotoResult(Path("blurred.jpg"), perceptual_hash=1, sharpness=500,
                              subject_sharpness=20, frontal_face_count=3,
                              face_coverage=.02, eye_detection_mode="scene_small_face")
    assert engine._subject_motion_fault(item, options)
