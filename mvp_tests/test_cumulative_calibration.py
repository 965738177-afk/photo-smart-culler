"""Cumulative preference history is local, deduplicated and fail-safe."""

import json
from pathlib import Path

import numpy as np

from event_cull import calibration
from event_cull.app import EventCullApp


def task(tmp_path: Path, name: str, scene: str, *, changed_picks=False):
    originals = tmp_path / name / "originals"
    selected = tmp_path / name / "selected"
    originals.mkdir(parents=True, exist_ok=True)
    selected.mkdir()
    rows = []
    for index in range(100):
        stem = f"{name}_{index:04d}"
        (originals / f"{stem}.jpg").write_bytes(b"fixture")
        if index % (4 if changed_picks else 5) == 0:
            (selected / f"{stem}.jpg").write_bytes(b"selected")
        rows.append({"file": f"{stem}.jpg", "group": f"group-{index}", "rating": 0})
    log = tmp_path / name / "log.json"
    log.write_text(json.dumps({"task": {"id": name}, "settings": {"scene_type": scene},
                               "photos": rows}), encoding="utf-8")
    return log, selected, [originals]


def fake_features(paths, _pipeline, **_kwargs):
    for index, path in enumerate(paths):
        # Source index 0,5,10... is the human retention signal.
        frame = int(path.stem.rsplit("_", 1)[1])
        vector = np.full(256, float(frame % 5 == 0), dtype=np.float32)
        yield index, vector


def balanced_holdout(_groups, labels):
    # The fixture has a fixed positive every fifth frame. Hold out complete
    # synthetic groups with the same positive fraction as the training half.
    return np.asarray([index % 10 < 5 for index in range(len(labels))], dtype=bool)


def test_cumulative_updates_replace_same_task_and_reuse_features(monkeypatch, tmp_path):
    monkeypatch.setattr(calibration, "_preferred_feature_pipeline", lambda: calibration.HANDCRAFTED_PIPELINE_ID)
    monkeypatch.setattr(calibration, "_balanced_group_holdout", balanced_holdout)
    decoded = []

    def counted(paths, pipeline, **kwargs):
        decoded.extend(paths)
        yield from fake_features(paths, pipeline, **kwargs)

    monkeypatch.setattr(calibration, "_feature_rows", counted)
    history = tmp_path / "history.sqlite3"
    first = task(tmp_path, "one", "婚礼")
    second = task(tmp_path, "two", "婚礼")
    result = calibration.update_cumulative_profile(*first, path=history)
    assert result.profile is not None and (result.task_count, result.example_count) == (1, 100)
    assert len(decoded) == 100
    again = calibration.update_cumulative_profile(*first, path=history)
    assert again.replaced_task and again.task_count == 1 and again.example_count == 100
    assert len(decoded) == 100  # same images, changed labels still reuse features
    combined = calibration.update_cumulative_profile(*second, path=history)
    assert combined.profile is not None
    assert (combined.task_count, combined.example_count, combined.positive_count) == (2, 200, 40)
    assert calibration.load_cumulative_profiles(history)["婚礼"].task_count == 2
    assert calibration.load_cumulative_history(history)["婚礼"] == (2, 200, 40)


def test_pending_batch_does_not_replace_active_and_scenes_do_not_mix(monkeypatch, tmp_path):
    monkeypatch.setattr(calibration, "_preferred_feature_pipeline", lambda: calibration.HANDCRAFTED_PIPELINE_ID)
    monkeypatch.setattr(calibration, "_balanced_group_holdout", balanced_holdout)
    monkeypatch.setattr(calibration, "_feature_rows", fake_features)
    history = tmp_path / "history.sqlite3"
    wedding = task(tmp_path, "wedding", "婚礼")
    portrait = task(tmp_path, "portrait", "人像写真")
    initial = calibration.update_cumulative_profile(*wedding, path=history)
    assert initial.profile is not None
    portrait_result = calibration.update_cumulative_profile(*portrait, path=history)
    assert portrait_result.profile is not None and portrait_result.task_count == 1
    # Inconsistent human labels fail activation, but the corrected task is
    # retained for later combined learning and never overwrites the good model.
    for photo in portrait[1].iterdir():
        photo.unlink()
    for index in range(0, 100, 4):
        (portrait[1] / f"portrait_{index:04d}.jpg").write_bytes(b"selected")
    pending = calibration.update_cumulative_profile(*portrait, path=history)
    assert pending.profile is None and pending.pending_reason and pending.replaced_task
    active = calibration.load_cumulative_profiles(history)
    assert active["婚礼"].task_count == 1
    assert active["人像写真"].positive_examples == portrait_result.profile.positive_examples
    assert calibration.load_cumulative_history(history)["人像写真"] == (1, 100, 25)


def test_cancelled_update_keeps_previous_history(monkeypatch, tmp_path):
    monkeypatch.setattr(calibration, "_preferred_feature_pipeline", lambda: calibration.HANDCRAFTED_PIPELINE_ID)
    monkeypatch.setattr(calibration, "_balanced_group_holdout", balanced_holdout)
    monkeypatch.setattr(calibration, "_feature_rows", fake_features)
    history = tmp_path / "history.sqlite3"
    first = task(tmp_path, "first", "婚礼")
    second = task(tmp_path, "second", "婚礼")
    calibration.update_cumulative_profile(*first, path=history)
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("cancel")

    try:
        calibration.update_cumulative_profile(*second, path=history, checkpoint=cancel)
    except RuntimeError:
        pass
    else:
        raise AssertionError("checkpoint did not stop the update")
    assert calibration.load_cumulative_profiles(history)["婚礼"].task_count == 1


def test_runtime_uses_scene_profile_then_legacy_fallback():
    app = object.__new__(EventCullApp)
    wedding = calibration.CalibrationProfile((0.,), (1.,), (1.,), 0., 1, 100, 20, .8,
                                             objective="retention", scene_type="婚礼")
    legacy = calibration.CalibrationProfile((0.,), (1.,), (1.,), 0., 3, 100, 20, .8,
                                            objective="featured", scene_type="人像写真")
    app._active_scene_profiles = {"婚礼": wedding}
    app._legacy_profile = legacy
    assert app._profile_for_scene("婚礼") is wedding
    assert app._profile_for_scene("人像写真") is legacy
    assert app._profile_for_scene("活动纪实") is None


def test_pending_history_is_visible_after_restart():
    app = object.__new__(EventCullApp)
    app._active_scene_profiles = {}
    app._calibration_history = {"婚礼": (1, 100, 20)}
    app._calibration_pending_summary = ""
    app.calibration_profile = None
    report = app._calibration_report_text()
    assert "1 个任务" in report and "待验证" in report


def test_pending_feedback_is_not_mislabelled_as_active():
    app = object.__new__(EventCullApp)
    app._active_scene_profiles = {"婚礼": calibration.CalibrationProfile(
        (0.,), (1.,), (1.,), 0., 1, 100, 20, .8,
        objective="retention", scene_type="婚礼", task_count=1,
    )}
    app._calibration_history = {"婚礼": (2, 200, 40)}
    app._calibration_pending_summary = ""
    app.calibration_profile = app._active_scene_profiles["婚礼"]
    assert "新反馈待验证，旧偏好仍启用" in app._calibration_report_text()
