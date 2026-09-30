"""Regressions for worker completion, preference memory and manual decisions."""
from pathlib import Path
import queue
import threading
import weakref

import numpy as np
import pytest

from event_cull import app as app_module, calibration
from event_cull.app import EventCullApp
from event_cull.engine import AnalysisCancelled, AnalysisOptions, PhotoResult
from event_cull.history import load_history, upsert_task


def analysis_subject():
    subject = object.__new__(EventCullApp)
    subject.pause_event = threading.Event()
    subject.cancel_event = threading.Event()
    subject.ui_events = queue.Queue()
    subject.language = "zh"
    subject.calibration_profile = object()
    return subject


def test_manual_ratings_survive_every_reanalysis_stage(monkeypatch):
    subject = analysis_subject()
    results = [PhotoResult(Path(f"photo{i}.jpg"), suggested_rating=3) for i in range(4)]
    original = {item.path: index for index, item in enumerate(results)}
    monkeypatch.setattr(app_module, "analyze_paths", lambda *args, **kwargs: results)

    def preferences(items, profile, **kwargs):
        assert kwargs["protected_paths"] == frozenset(original)
        # Even future scoring stages must not override the saved manual choice.
        for item in items:
            item.suggested_rating = 2

    monkeypatch.setattr(app_module, "apply_profile", preferences)
    subject._analyze([item.path for item in results], AnalysisOptions(), original)
    event, payload = subject.ui_events.get_nowait()
    assert event == "completed"
    assert [item.suggested_rating for item in payload] == [0, 1, 2, 3]


def test_stop_during_preference_stage_never_publishes_completed_results(monkeypatch):
    subject = analysis_subject()
    results = [PhotoResult(Path("photo.jpg"), suggested_rating=3)]
    monkeypatch.setattr(app_module, "analyze_paths", lambda *args, **kwargs: results)

    def preferences(*args, **kwargs):
        subject.cancel_event.set()
        kwargs["checkpoint"]()

    monkeypatch.setattr(app_module, "apply_profile", preferences)
    subject._analyze([results[0].path], AnalysisOptions())
    assert subject.ui_events.get_nowait() == ("cancelled", None)
    assert subject.ui_events.empty()


def test_stop_while_paused_interrupts_checkpoint():
    subject = analysis_subject()
    subject.pause_event.set()
    subject.cancel_event.set()
    with pytest.raises(AnalysisCancelled):
        subject._analysis_checkpoint()


def test_large_preference_job_retains_only_one_image_batch(monkeypatch):
    references = []
    batch_sizes = []

    def decode(path):
        assert sum(ref() is not None for ref in references) < calibration.FEATURE_BATCH_SIZE
        image = np.full((16, 16, 3), int(path.stem), dtype=np.uint8)
        references.append(weakref.ref(image))
        return image

    def features(images, pipeline):
        batch_sizes.append(len(images))
        return np.asarray([[image[0, 0, 0]] for image in images])

    monkeypatch.setattr(calibration, "_load_preview", decode)
    monkeypatch.setattr(calibration, "_feature_matrix", features)
    rows = list(calibration._feature_rows([Path(f"{i}.jpg") for i in range(67)], "test"))
    assert len(rows) == 67
    assert max(batch_sizes) == calibration.FEATURE_BATCH_SIZE
    assert all(ref() is None for ref in references)
    assert [int(feature[0]) for _, feature in rows] == list(range(67))


def test_preference_stop_during_decode_does_not_mutate_ratings(monkeypatch):
    stop = threading.Event()
    results = [PhotoResult(Path("A.jpg"), suggested_rating=3), PhotoResult(Path("B.jpg"), suggested_rating=2)]

    def decode(path):
        stop.set()
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def checkpoint():
        if stop.is_set():
            raise AnalysisCancelled()

    monkeypatch.setattr(calibration, "_load_preview", decode)
    profile = calibration.CalibrationProfile((0.,), (1.,), (1.,), 0., 3, 100, 20, .7)
    with pytest.raises(AnalysisCancelled):
        calibration.apply_profile(results, profile, checkpoint=checkpoint)
    assert [item.suggested_rating for item in results] == [3, 2]


def test_preferences_never_decode_or_reassign_protected_manual_photos(monkeypatch):
    protected = PhotoResult(Path("manual.jpg"), suggested_rating=3)
    original = PhotoResult(Path("original.jpg"), suggested_rating=3, group_id=1)
    better = PhotoResult(Path("better.jpg"), suggested_rating=1, group_id=1)
    decoded = []
    monkeypatch.setattr(calibration, "_load_preview", lambda path: decoded.append(path) or np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(calibration, "_feature_matrix", lambda images, pipeline: np.asarray([[0.], [1.]]))
    profile = calibration.CalibrationProfile((0.,), (1.,), (1.,), 0., 3, 100, 20, .7)
    calibration.apply_profile([protected, original, better], profile, protected_paths=frozenset({protected.path}))
    assert protected.path not in decoded
    assert [item.suggested_rating for item in (protected, original, better)] == [3, 1, 3]


def test_one_ui_callback_error_does_not_disable_later_worker_events(monkeypatch):
    subject = analysis_subject()
    scheduled, errors, finished = [], [], []
    subject.after = lambda delay, callback: scheduled.append(callback)
    subject.report_callback_exception = lambda *details: errors.append(details)
    subject._finish_license_refresh = lambda payload: finished.append(payload)
    subject.ui_events.put(("progress", (1,)))  # Simulate an invalid worker payload.
    subject.ui_events.put(("license_refresh_finished", None))
    subject.process_ui_events()
    assert errors and scheduled
    scheduled.pop(0)()
    assert finished == [None]


def test_ui_event_burst_yields_to_rendering(monkeypatch):
    subject = analysis_subject()
    subject.after = lambda *args: None
    subject.report_callback_exception = lambda *args: pytest.fail("unexpected callback error")
    finished = []
    subject._finish_license_refresh = lambda payload: finished.append(payload)
    monkeypatch.setattr(app_module.time, "monotonic", lambda: 0.)
    for index in range(1000):
        subject.ui_events.put(("license_refresh_finished", index))
    subject.process_ui_events()
    assert len(finished) == 100
    assert subject.ui_events.qsize() == 900


@pytest.mark.parametrize("tasks", [None, 3, "bad", {}])
def test_malformed_history_does_not_break_loading_or_future_saves(tmp_path, tasks):
    import json
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    assert load_history(path) == []
    upsert_task({"id": "new", "status": "已完成"}, path)
    assert load_history(path) == [{"id": "new", "status": "已完成"}]
