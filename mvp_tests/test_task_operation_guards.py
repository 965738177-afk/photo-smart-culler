"""Busy task state must protect every public entry, not just disabled buttons."""
from pathlib import Path
from types import SimpleNamespace
import queue
import threading

import pytest

from event_cull import app as app_module
from event_cull.app import EventCullApp


@pytest.mark.parametrize("busy", ["analysis_running", "calibration_running", "export_running", "metadata_running", "_picker_dialog"])
@pytest.mark.parametrize("action", [
    "new_task", "choose_files", "choose_folder", "import_calibration", "import_historical_calibration", "start_analysis",
    "confirm_write", "confirm_restore", "repair_legacy_heic_ratings", "set_manual_rating",
    "_load_selected_files", "_start_calibration", "show_picker_settings",
])
def test_busy_state_blocks_menu_shortcut_and_import_entry_points(monkeypatch, busy, action):
    subject = object.__new__(EventCullApp)
    setattr(subject, busy, True)
    monkeypatch.setattr(app_module.messagebox, "showinfo", lambda *_args: None)
    args = {
        "set_manual_rating": (3,),
        "_load_selected_files": ([Path("missing.jpg")],),
        "_start_calibration": (Path("log.json"), Path("picks"), [Path("source")]),
    }
    # No initialized widgets/files are available: any attempt to proceed
    # instead of returning at the shared guard would fail this test.
    result = getattr(subject, action)(*args.get(action, ()))
    if action == "_load_selected_files":
        assert result is False


class Control:
    def __init__(self):
        self.state = "normal"

    def configure(self, **kwargs):
        self.state = kwargs.get("state", self.state)

    def cget(self, _key):
        return self.state

    def set(self, _value):
        pass


@pytest.mark.parametrize("busy", ["analysis_running", "calibration_running", "export_running", "metadata_running"])
def test_periodic_editor_check_does_not_enable_write_during_busy_task(busy):
    subject = object.__new__(EventCullApp)
    setattr(subject, busy, True)
    subject.results = [object()]
    subject.write_button = Control()
    subject.restore_button = Control()
    subject._metadata_editors_running = lambda: ()
    subject.lr_guard_label = SimpleNamespace(pack_forget=lambda: None)
    subject.refresh_lightroom_guard(schedule_next=False)
    assert subject.write_button.state == subject.restore_button.state == "disabled"


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ValueError])
def test_restore_passes_failures_to_shared_background_boundary(monkeypatch, error_type):
    subject = object.__new__(EventCullApp)
    subject.ensure_metadata_editors_closed = lambda: True
    subject.results = [SimpleNamespace(path=Path("image.jpg"))]
    subject.language = "zh"
    subject.t = {"restore_confirm": "还原", "restore_confirm_detail": "确认还原"}
    monkeypatch.setattr(app_module.messagebox, "askyesno", lambda *_args: True)
    operations = []
    subject._begin_metadata_operation = lambda title, operation, done: operations.append(operation)

    def fail(_files, **_options):
        raise error_type("disk unavailable")

    monkeypatch.setattr(app_module, "restore_managed_ratings", fail)
    subject.confirm_restore()
    assert len(operations) == 1
    with pytest.raises(error_type, match="disk unavailable"):
        operations[0]()


def test_unexpected_export_failure_is_reported_and_controls_can_recover(monkeypatch):
    subject = object.__new__(EventCullApp)
    subject.results = [object()]
    subject._choose_export_options = lambda: (frozenset({3}), None)
    subject.export_cancel_event = threading.Event()
    subject.ui_events = queue.Queue()
    subject.manual_rating_buttons = {}
    subject.t = {"export_cancel": "取消", "export_selected": "导出"}
    for name in (
        "new_task_button", "choose_button", "choose_files_button", "analyze_button",
        "calibration_button", "picker_settings_button", "write_button", "restore_button",
        "repair_legacy_heic_button", "export_selected_button", "progress", "progress_text", "status",
    ):
        setattr(subject, name, Control())
    monkeypatch.setattr(app_module, "collect_selected_files",
                        lambda *_args: SimpleNamespace(photo_count=1, file_count=1))

    def fail(*_args, **_kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(app_module, "export_selected_files", fail)
    monkeypatch.setattr(app_module.threading, "Thread",
                        lambda **kwargs: SimpleNamespace(start=kwargs["target"]))
    subject.export_selected_photos()
    event, detail = subject.ui_events.get_nowait()
    assert event == "selected_export_failed"
    assert "原片未改动" in detail
    subject._finish_selected_export()
    assert not subject.export_running
    assert subject.analyze_button.state == "normal"
