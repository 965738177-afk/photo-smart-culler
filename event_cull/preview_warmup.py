"""Bounded, cancellable preview scheduling shared by import and review."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import threading
from typing import Callable, Iterable


class PreviewWarmup:
    """Load all imports, but let visible/selected files jump ahead of the queue.

    Callbacks run on a worker: consumers must deliver them through their UI
    queue. A new generation discards pending old work, never the source files.
    """

    def __init__(self, load: Callable[[Path], object], ready: Callable[[int, Path], None], *, workers: int = 2,
                 failed: Callable[[int, Path, str], None] | None = None):
        self._load = load
        self._ready = ready
        self._failed = failed
        self._condition = threading.Condition()
        self._pending: OrderedDict[Path, None] = OrderedDict()
        self._in_flight: set[Path] = set()
        self._generation = 0
        self._closed = False
        self._worker_count = max(1, min(workers, 2))
        self._threads: list[threading.Thread] = []

    def submit(self, sources: Iterable[Path], generation: int) -> None:
        with self._condition:
            if self._closed:
                return
            self._generation = generation
            self._pending = OrderedDict.fromkeys(sources)
            if not self._threads:
                for number in range(self._worker_count):
                    thread = threading.Thread(target=self._run, name=f"preview-warmup-{number}", daemon=True)
                    self._threads.append(thread)
                    thread.start()
            self._condition.notify_all()

    def prioritize(self, sources: Iterable[Path], *, request: bool = False) -> None:
        with self._condition:
            if self._closed:
                return
            for source in reversed(list(dict.fromkeys(sources))):
                if source not in self._pending and request and source not in self._in_flight:
                    self._pending[source] = None
                if source in self._pending:
                    self._pending.move_to_end(source, last=False)
            self._condition.notify_all()

    def cancel(self, generation: int) -> None:
        with self._condition:
            self._generation = generation
            self._pending.clear()
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                while True:
                    if self._closed:
                        return
                    source = next((path for path in self._pending if path not in self._in_flight), None)
                    if source is not None:
                        self._pending.pop(source)
                        self._in_flight.add(source)
                        generation = self._generation
                        break
                    self._condition.wait()
            try:
                self._load(source)
                with self._condition:
                    if not self._closed and generation == self._generation:
                        self._ready(generation, source)
            except Exception as exc:
                # One unsupported/corrupt file cannot stop other thumbnails.
                with self._condition:
                    if self._failed is not None and not self._closed and generation == self._generation:
                        self._failed(generation, source, str(exc))
            finally:
                with self._condition:
                    self._in_flight.discard(source)
                    self._condition.notify_all()
