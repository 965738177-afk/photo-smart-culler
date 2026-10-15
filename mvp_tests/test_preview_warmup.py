"""Previews are prepared on import, bounded, prioritized and shared across views."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import threading

import numpy as np
import pytest

from event_cull import app as app_module
from event_cull.app import EventCullApp
from event_cull.preview_warmup import PreviewWarmup


def test_all_2764_imports_are_prepared_once_without_analysis():
    ready = queue.Queue()
    decoded = []
    pool = PreviewWarmup(lambda path: decoded.append(path), lambda batch, path: ready.put((batch, path)))
    sources = [Path(f"DSC{i}.ARW") for i in range(2764)]
    try:
        pool.submit(sources + sources[:10], 1)
        events = [ready.get(timeout=5) for _ in sources]
        assert {path for batch, path in events} == set(sources)
        assert len(decoded) == len(sources)
        assert all(batch == 1 for batch, path in events)
    finally:
        pool.close()


def test_scrolled_visible_files_jump_ahead_of_pending_queue():
    started, release = threading.Event(), threading.Event()
    ready = queue.Queue()
    sources = [Path(f"DSC{i}.ARW") for i in range(5)]

    def load(path):
        if path == sources[0]:
            started.set()
            assert release.wait(5)

    pool = PreviewWarmup(load, lambda batch, path: ready.put(path), workers=1)
    try:
        pool.submit(sources, 1)
        assert started.wait(5)
        pool.prioritize([sources[4], sources[3]])
        release.set()
        assert [ready.get(timeout=5) for _ in sources] == [sources[0], sources[4], sources[3], sources[1], sources[2]]
    finally:
        release.set()
        pool.close()


def test_new_task_drops_pending_and_stale_preview_events():
    started, release = threading.Event(), threading.Event()
    ready = queue.Queue()
    calls = []
    old, pending, new = Path("old.ARW"), Path("pending.ARW"), Path("new.HEIC")

    def load(path):
        calls.append(path)
        if path == old:
            started.set()
            assert release.wait(5)

    pool = PreviewWarmup(load, lambda batch, path: ready.put((batch, path)), workers=1)
    try:
        pool.submit([old, pending], 1)
        assert started.wait(5)
        pool.cancel(2)
        pool.submit([new], 3)
        release.set()
        assert ready.get(timeout=5) == (3, new)
        assert ready.empty()
        assert calls == [old, new]
    finally:
        release.set()
        pool.close()


def test_pixel_delivery_discards_stale_ab_pair_without_decoding_on_ui_thread():
    started, release = threading.Event(), threading.Event()
    delivered = queue.Queue()
    old, new = Path("old.ARW"), Path("new.ARW")
    def load(path):
        if path == old:
            started.set()
            assert release.wait(5)
        return (path, threading.current_thread().name)
    pool = PreviewWarmup(load, lambda _g, _p: None, workers=1,
                         result_ready=lambda g, p, pixels: delivered.put((g, p, pixels)))
    try:
        pool.submit([old], 1)
        assert started.wait(5)
        pool.submit([new], 2)
        release.set()
        generation, path, pixels = delivered.get(timeout=5)
        assert generation == 2 and path == new
        assert pixels[1].startswith("preview-warmup-")
        assert delivered.empty()
    finally:
        release.set()
        pool.close()


def test_corrupt_preview_reports_failure_and_does_not_stop_other_files():
    events = queue.Queue()

    def load(path):
        if path.name == "bad.HEIC":
            raise ValueError("unsupported")

    pool = PreviewWarmup(load, lambda batch, path: events.put(("ready", path)), workers=1,
                         failed=lambda batch, path, error: events.put(("failed", path)))
    try:
        pool.submit([Path("bad.HEIC"), Path("good.JPG")], 1)
        assert events.get(timeout=5) == ("failed", Path("bad.HEIC"))
        assert events.get(timeout=5) == ("ready", Path("good.JPG"))
    finally:
        pool.close()


def test_worker_count_is_bounded_even_if_more_are_requested():
    pool = PreviewWarmup(lambda path: None, lambda batch, path: None, workers=20)
    try:
        pool.submit([Path("A.JPG")], 1)
        assert len(pool._threads) == 2
    finally:
        pool.close()


def preview_subject():
    subject = object.__new__(EventCullApp)
    subject._preview_memory = OrderedDict()
    subject._preview_lock = threading.RLock()
    subject._preview_load_locks = [threading.Lock() for _ in range(64)]
    return subject


def test_concurrent_preview_consumers_decode_same_file_only_once(tmp_path, monkeypatch):
    source = tmp_path / "camera.ARW"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(app_module, "application_data_dir", lambda: tmp_path / "cache")
    decoded = []
    monkeypatch.setattr(app_module, "_load_preview", lambda path, **kwargs: decoded.append(path) or np.zeros((8, 8, 3), dtype=np.uint8))
    subject = preview_subject()
    with ThreadPoolExecutor(max_workers=8) as workers:
        images = list(workers.map(subject._preview_image, [source] * 8))
    assert len(decoded) == 1
    assert all(image.size == (8, 8) for image in images)


def test_disk_cache_is_reused_after_restart_without_raw_decode(tmp_path, monkeypatch):
    source = tmp_path / "camera.ARW"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(app_module, "application_data_dir", lambda: tmp_path / "cache")
    decoded = []
    monkeypatch.setattr(app_module, "_load_preview", lambda path, **kwargs: decoded.append(path) or np.zeros((8, 8, 3), dtype=np.uint8))
    preview_subject()._preview_image(source)
    preview_subject()._preview_image(source)
    assert len(decoded) == 1


def test_disk_cache_key_changes_if_original_changes(tmp_path, monkeypatch):
    source = tmp_path / "camera.ARW"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(app_module, "application_data_dir", lambda: tmp_path / "cache")
    first = EventCullApp._preview_cache_path(source)
    source.write_bytes(b"changed fixture")
    assert EventCullApp._preview_cache_path(source) != first


@pytest.mark.parametrize("failed,expected", [(False, "正在准备预览"), (True, "预览不可用")])
def test_failed_files_never_claim_to_still_be_loading(failed, expected):
    subject = preview_subject()
    source = Path("broken.HEIC")
    subject._preview_failures = {source: "unsupported"} if failed else {}
    assert subject._preview_placeholder(source) == expected
