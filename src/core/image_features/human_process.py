"""Isolate native aborts from the desktop, with one bounded worker per task."""
from __future__ import annotations
import json
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time

import numpy as np


class HumanProcessService:
    def __init__(self, *, checkpoint=lambda:None, timeout=20., process_factory=subprocess.Popen,
                 recycle_after=256, recover_faces=False):
        if type(recover_faces) is not bool:
            raise ValueError("Face recovery mode must be boolean")
        self._checkpoint,self._timeout,self._factory=checkpoint,timeout,process_factory
        self._lock=threading.RLock()
        self.process=None
        self._responses=None
        self._reader_stop=None
        self._closed=False
        self._restarts=0
        self._calls=0
        self._recycle_after=max(1,int(recycle_after))
        self._recover_faces = recover_faces

    def _start(self):
        if self._closed:
            raise RuntimeError("Person worker is closed")
        if self.process is not None:
            return
        command=([sys.executable,"--person-evidence-worker"] if getattr(sys,"frozen",False)
                 else [sys.executable,"-m","src.core.image_features.human_worker"])
        self.process=self._factory(command,cwd=str(Path(__file__).resolve().parents[3]),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        self._responses=queue.Queue(maxsize=1)
        self._reader_stop=threading.Event()
        self._calls=0
        stream,responses=self.process.stdout,self._responses
        stopped=self._reader_stop
        def offer(value):
            # A reply followed by EOF must not strand a daemon on a full queue
            # when the caller has cancelled or recycled this worker.
            while not stopped.is_set():
                try:
                    responses.put(value,timeout=.05)
                    return
                except queue.Full:
                    continue
        def read():
            try:
                while not stopped.is_set():
                    line=stream.readline(2_000_001)
                    if not line:
                        offer(None)
                        return
                    offer(line)
            except (OSError,ValueError):
                offer(None)
        threading.Thread(target=read,name="person-worker-response",daemon=True).start()

    def _stop(self):
        if self._reader_stop is not None:
            self._reader_stop.set()
        process,self.process=self.process,None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for stream in (process.stdin,process.stdout):
            if stream is not None:
                stream.close()

    def inspect(self,rgb,*,face_regions=()):
        with self._lock:
            self._checkpoint()
            if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8 or max(rgb.shape[:2])>1600:
                raise ValueError("Invalid shared person preview")
            try:
                # Bound lifetime of Metal/native pixel-buffer allocations too,
                # not just Python RAM. OS cleanup avoids long-run allocator
                # failures that don't reproduce on an isolated single frame.
                if self._calls>=self._recycle_after:
                    self._stop()
                self._start()
                header=json.dumps({"shape":list(rgb.shape[:2]),"regions":[list(map(int,r)) for r in face_regions[:12]],
                                   "recover_faces":self._recover_faces}).encode()
                request=struct.pack("!I",len(header))+header+np.ascontiguousarray(rgb).tobytes()
                # Write off-thread too: a hung native worker must not leave a
                # caller permanently blocked on a full OS stdin pipe.
                sent=queue.Queue(maxsize=1)
                process=self.process
                def send():
                    try:
                        process.stdin.write(request)
                        process.stdin.flush()
                        sent.put(None)
                    except (OSError,ValueError) as exc:
                        sent.put(exc)
                threading.Thread(target=send,name="person-worker-input",daemon=True).start()
                deadline=time.monotonic()+self._timeout
                sending=True
                while True:
                    checkpoint_started=time.monotonic()
                    self._checkpoint()
                    # A user pause is not a hung native call. Otherwise a
                    # legitimate pause longer than 20 s consumes a restart and
                    # repeated pauses disable all later person observations.
                    deadline+=time.monotonic()-checkpoint_started
                    if time.monotonic()>=deadline:
                        raise TimeoutError("Person worker exceeded per-image time limit")
                    if sending:
                        try:
                            error=sent.get(timeout=.05)
                        except queue.Empty:
                            continue
                        if error:
                            raise RuntimeError("Person worker closed during request") from error
                        sending=False
                    try:
                        line=self._responses.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if line is None or len(line)>2_000_000:
                        raise RuntimeError("Native person worker exited; desktop protected")
                    value=json.loads(line)
                    if not isinstance(value,dict) or value.get("status") not in {"ok","error"}:
                        raise ValueError("Invalid person worker reply")
                    if value.get("status")=="ok":
                        from .human_analysis import validate_observation
                        validate_observation(value,require_details=True)
                        if self._recover_faces:
                            from .face_analysis import FACE_REGION_PIPELINE
                            if value.get("region_pipeline") != FACE_REGION_PIPELINE:
                                raise ValueError("Missing requested face recovery evidence")
                    self._calls+=1
                    return value
            except BaseException:
                self._stop()
                self._restarts+=1
                # Never repeatedly restart a broken component for 5000 frames.
                if self._restarts>1:
                    self._closed=True
                raise

    def close(self):
        with self._lock:
            self._closed=True
            self._stop()
