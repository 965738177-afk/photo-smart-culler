"""Metadata workers preserve atomic per-file work and recover every UI exit."""
from pathlib import Path
from types import SimpleNamespace
import queue
import threading

import pytest

from event_cull import app as app_module
from event_cull.app import EventCullApp
from event_cull.xmp import MetadataCancelled, restore_managed_ratings, write_ratings


def test_cancel_write_finishes_current_file_without_touching_remaining_originals(tmp_path):
    photos = [tmp_path / f"{index}.ARW" for index in range(3)]
    for photo in photos:
        photo.write_bytes(b"original pixels")
    cancel = threading.Event()
    with pytest.raises(MetadataCancelled) as raised:
        write_ratings(dict.fromkeys(photos, 3), cancel_event=cancel,
                      progress_callback=lambda current, total, path: cancel.set())
    assert raised.value.outcomes == {photos[0]: "created"}
    assert photos[0].with_suffix(".xmp").is_file()
    assert not photos[1].with_suffix(".xmp").exists()
    assert all(photo.read_bytes() == b"original pixels" for photo in photos)
    restore_managed_ratings(photos)
    assert not photos[0].with_suffix(".xmp").exists()


def test_cancel_restore_preserves_unprocessed_ratings(tmp_path):
    photos = [tmp_path / f"{index}.ARW" for index in range(3)]
    for photo in photos:
        photo.write_bytes(b"original")
    write_ratings(dict.fromkeys(photos, 3))
    cancel = threading.Event()
    with pytest.raises(MetadataCancelled):
        restore_managed_ratings(photos, cancel_event=cancel,
                                progress_callback=lambda current, total, path: cancel.set())
    assert not photos[0].with_suffix(".xmp").exists()
    assert all(photo.with_suffix(".xmp").is_file() for photo in photos[1:])


class Widget:
    def __init__(self, state="normal"):
        self.state = state
        self.visible = False
    def cget(self, name):
        return self.state
    def configure(self, **kwargs):
        self.state = kwargs.get("state", self.state)
    def set(self, value):
        pass
    def pack(self, **kwargs):
        self.visible = True
    def pack_forget(self):
        self.visible = False


def subject():
    app = object.__new__(EventCullApp)
    app.metadata_cancel_event = threading.Event()
    app.cancel_event = threading.Event()
    app.pause_event = threading.Event()
    app.export_cancel_event = threading.Event()
    app.export_running = False
    app.metadata_running = False
    app._close_after_operations = False
    app.ui_events = queue.Queue()
    app.manual_rating_buttons = {1: Widget()}
    for name in ("new_task_button", "choose_button", "choose_files_button", "analyze_button",
                 "calibration_button", "picker_settings_button", "write_button", "restore_button",
                 "repair_legacy_heic_button", "export_selected_button", "write_mode_menu",
                 "backup_star_menu", "featured_star_menu",
                 "metadata_cancel_button", "status", "progress", "progress_text"):
        setattr(app, name, Widget())
    app.refresh_lightroom_guard = lambda **kwargs: None
    app._set_xmp_result = lambda text: None
    return app


@pytest.mark.parametrize("outcome", ["success", "cancel", "error"])
def test_metadata_worker_always_releases_controls_on_main_thread(monkeypatch, outcome):
    app = subject()
    workers, completed, errors = [], [], []
    monkeypatch.setattr(app_module.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: workers.append(kwargs["target"])))
    monkeypatch.setattr(app_module.messagebox, "showerror", lambda *args: errors.append(args))
    photo = Path("A.ARW")

    def operation(**kwargs):
        if outcome == "cancel":
            raise MetadataCancelled({photo: "created"})
        if outcome == "error":
            raise OSError("disk full")
        return {photo: "created"}

    app._begin_metadata_operation("星级写入", operation, lambda result: completed.append(result))
    assert app.metadata_running and app.write_button.state == "disabled"
    assert not completed
    workers[0]()
    assert app.metadata_running  # Worker only posts; it never mutates Tk.
    event, payload = app.ui_events.get_nowait()
    assert event == "metadata_finished"
    app._finish_metadata_operation(*payload)
    assert not app.metadata_running
    assert app.write_button.state == "normal"
    assert not app.metadata_cancel_button.visible
    assert bool(completed) == (outcome == "success")
    assert bool(errors) == (outcome == "error")


def test_close_waits_for_active_file_operation_to_finish(monkeypatch):
    app = subject()
    app.metadata_running = True
    closed, pending = [], []
    app.destroy = lambda: closed.append(True)
    app.after = lambda delay, callback: pending.append(callback)
    monkeypatch.setattr(app_module.messagebox, "askyesno", lambda *args: True)
    app._request_close()
    assert app.metadata_cancel_event.is_set()
    assert not closed and pending
    app.metadata_running = False
    pending.pop()()
    assert closed == [True]
