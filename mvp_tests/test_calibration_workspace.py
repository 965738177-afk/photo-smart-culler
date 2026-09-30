"""Navigation and learning UI reuse task state and the existing worker."""
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from event_cull import app as app_module
from event_cull.app import EventCullApp, TEXT


class Control:
    def __init__(self):
        self.mapped = True
        self.options = {}
        self.value = None
        self.running = False

    def grid(self):
        self.mapped = True

    def grid_remove(self):
        self.mapped = False

    def configure(self, **options):
        self.options.update(options)

    def set(self, value):
        self.value = value

    def stop(self):
        self.running = False

    def start(self, _interval):
        self.running = True


def subject():
    value = object.__new__(EventCullApp)
    value.t = TEXT["zh"]
    value.calibration_profile = None
    value.current_log_path = None
    value.imported_files = []
    for name in (
        "task_header", "workflow_band", "decision_panel", "workspace_footer_note",
        "grid_panel", "table_panel", "calibration_panel", "calibration_button",
        "calibration_history_button", "calibration_task_status", "calibration_report",
        "calibration_activity", "calibration_progress", "calibration_stop_button",
        "progress", "progress_text", "status", "new_task_button",
    ):
        setattr(value, name, Control())
    value._set_active_nav = lambda _key: None
    value._schedule_grid_render = lambda: None
    return value


@pytest.mark.parametrize("return_page", ["show_culling_workspace", "show_export_workspace"])
def test_learning_hides_chrome_and_restores_it_without_changing_task(return_page):
    value = subject()
    value.results = [object()]
    value.imported_files = [Path("original.heic")]
    value.manual_ratings = {value.imported_files[0]: 3}
    state = value.results, value.imported_files, value.manual_ratings
    value.show_calibration_workspace()
    assert value.calibration_panel.mapped
    assert not value.grid_panel.mapped and not value.table_panel.mapped
    for name in ("task_header", "workflow_band", "decision_panel"):
        assert not getattr(value, name).mapped
    assert value._active_workspace == "calibration"
    getattr(value, return_page)()
    for name in ("task_header", "workflow_band", "decision_panel"):
        assert getattr(value, name).mapped
    assert not value.calibration_panel.mapped
    assert all(before is after for before, after in zip(
        state, (value.results, value.imported_files, value.manual_ratings)
    ))


def test_missing_log_is_not_reported_ready_and_historical_action_remains_available():
    value = subject()
    value.current_log_path = Path("missing-auto-log.json")
    value.imported_files = [Path("photo.jpg")]
    value._refresh_calibration_button()
    assert value.calibration_button.options["state"] == "disabled"
    assert value.calibration_history_button.options["state"] == "normal"
    assert "先完成" in value.calibration_task_status.options["text"]
    assert "最终保留" in value.calibration_button.options["text"]


def test_ready_task_only_asks_for_final_picks(monkeypatch, tmp_path):
    value = subject()
    value.current_log_path = tmp_path / "task.json"
    value.current_log_path.touch()
    value.imported_files = [tmp_path / "one.jpg", tmp_path / "two.heic"]
    value.last_import_directory = tmp_path
    starts = []
    value._start_calibration = lambda *args: starts.append(args)
    monkeypatch.setattr(app_module.filedialog, "askdirectory", lambda **_kwargs: str(tmp_path / "picks"))
    monkeypatch.setattr(app_module.filedialog, "askopenfilename", lambda **_kwargs: pytest.fail("current task must not ask for a log"))
    value.import_calibration()
    assert starts == [(value.current_log_path, tmp_path / "picks", [tmp_path])]
    value._refresh_calibration_button()
    assert value.calibration_button.options["state"] == "normal"


def test_primary_action_never_silently_opens_historical_three_picker_flow(monkeypatch):
    value = subject()
    messages = []
    monkeypatch.setattr(app_module.messagebox, "showinfo", lambda *args: messages.append(args))
    monkeypatch.setattr(app_module.filedialog, "askopenfilename", lambda **_kwargs: pytest.fail("historical workflow must be explicit"))
    value.import_calibration()
    assert len(messages) == 1


def test_report_does_not_label_auc_as_accuracy_or_promise_rollback():
    value = subject()
    value.calibration_profile = SimpleNamespace(training_examples=100, positive_examples=30, validation_auc=0.72)
    report = value._calibration_report_text()
    assert "100" in report and "30" in report and "3 星排序" in report
    assert not any(term in report for term in ("准确度", "一键回退", "更好时替换", "最高星级"))
    assert "最终保留的照片" in TEXT["zh"]["calibration_done"]


def test_hidden_global_progress_is_mirrored_by_same_worker_event():
    value = subject()
    value.workflow_band.grid_remove()
    value.calibration_running = True
    value.update_calibration_progress(5, 10, Path("frame.heic"), apply=False)
    for progress in (value.progress, value.calibration_progress):
        assert progress.options == {"mode": "determinate", "maximum": 10, "value": 5}
    assert "5/10" in value.progress_text.value


def test_cancel_uses_shared_event_and_finish_refreshes_report():
    value = subject()
    value.calibration_running = True
    value.cancel_event = threading.Event()
    value.pause_event = threading.Event()
    value.pause_event.set()
    value._refresh_picker_confirmation_ui = lambda: None
    value.cancel_calibration()
    assert value.cancel_event.is_set() and not value.pause_event.is_set()
    assert value.calibration_stop_button.options["state"] == "disabled"
    value.calibration_profile = SimpleNamespace(training_examples=90, positive_examples=20)
    value._finish_calibration()
    assert not value.calibration_running and not value.calibration_activity.mapped
    assert "90" in value.calibration_report.options["text"]
    assert value.calibration_history_button.options["state"] == "normal"


def test_hidden_canvas_cannot_steal_mousewheel_from_learning_page():
    assert not EventCullApp._pointer_is_over_widget(
        SimpleNamespace(x_root=10, y_root=10), SimpleNamespace(winfo_ismapped=lambda: False)
    )
