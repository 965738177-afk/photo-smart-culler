"""Native widget smoke tests; deliberately use fixture pixels, not user photos."""
import threading
import time
import tkinter as tk
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from event_cull.duplicate_review import DuplicateReviewSession
from event_cull.duplicate_review_ui import DuplicateReviewDialog
from event_cull.engine import PhotoResult


def wait_for(root, predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(.01)
    pytest.fail("Review widget did not reach expected state")


def _check_confirmation():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Native Tk display unavailable: {exc}")
    root.withdraw()
    release = threading.Event()
    pair = [PhotoResult(Path(f"/fixture/{i}.JPG"), group_id=1, score=70) for i in range(2)]
    session = DuplicateReviewSession(pair, {})
    changed = []
    def load(_path):
        assert release.wait(3)
        return Image.new("RGB", (80, 60), "grey")
    dialog = DuplicateReviewDialog(root, session, load_preview=load,
                                  preview_source=lambda path, _companions: path,
                                  on_change=lambda: changed.append(True))
    try:
        root.update()
        dialog.act("a")
        assert session.decisions == []
        release.set()
        wait_for(root, lambda: len(dialog._loaded_paths) == 2)
        dialog._toggle_zoom()
        assert dialog._zoom
        dialog.act("a")
        assert [p.suggested_rating for p in pair] == [2, 0]
        assert len(changed) == 1
        dialog.undo()
        assert session.manual_ratings == {}
        assert session.pair == tuple(pair)
        dialog.close()
        assert dialog._closed and dialog._warmup._closed
    finally:
        release.set()
        root.destroy()


def _check_failure():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Native Tk display unavailable: {exc}")
    root.withdraw()
    pair = [PhotoResult(Path(f"/fixture/{i}.ARW"), group_id=1, score=70) for i in range(2)]
    session = DuplicateReviewSession(pair, {})
    def failed(_path):
        raise FileNotFoundError("fixture missing")
    dialog = DuplicateReviewDialog(root, session, load_preview=failed,
                                  preview_source=lambda path, _companions: path, on_change=lambda: None)
    try:
        wait_for(root, lambda: len(dialog._images) == 2)
        dialog.act("both")
        assert session.manual_ratings == {}
        assert str(dialog.buttons[0]["state"]) == "disabled"
        dialog.act("skip")
        assert len(session.deferred) == 1
        root.destroy()
        assert dialog._closed and dialog._warmup._closed
    finally:
        if not dialog._closed:
            root.destroy()


@pytest.mark.parametrize("case", ["confirmation", "failure"])
def test_native_review_in_isolated_tk_process(case):
    # Cocoa/Tcl owns process-wide state. Do not leave destroyed native roots
    # for unrelated test workers' GC to release on a background thread.
    completed = subprocess.run([sys.executable, "-m", "mvp_tests.test_duplicate_review_ui", case],
                               capture_output=True, text=True, timeout=15)
    if completed.returncode == 77:
        pytest.skip(completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr


if __name__ == "__main__":
    try:
        {"confirmation": _check_confirmation, "failure": _check_failure}[sys.argv[1]]()
    except pytest.skip.Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(77)
