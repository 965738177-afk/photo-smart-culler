"""Shared, lazy face-landmark analysis used by photo workflows."""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import threading
from typing import Protocol

import numpy as np

from ..runtime_paths import resolve_face_landmarker_model_path


FACE_DESCRIPTOR_VERSION = "mediapipe-face-landmarks-v1"
FACE_REGION_PIPELINE = "yunet-2023mar-640-score08-regions-v1"


class FaceRegionDetector:
    """Own bounded YuNet proposals for recovery and expression alignment.

    Inputs are shared, decoded BGR pixels. Coordinates (including the five
    alignment landmarks) are returned in the caller's original pixel units.
    Proposals do not prove eye visibility, wedding roles or expression quality.
    """

    def __init__(self, *, model_path_resolver, backend_factory=None, checkpoint=lambda: None):
        self._resolver = model_path_resolver
        self._factory = backend_factory
        self._checkpoint = checkpoint
        self._backend = None
        self._closed = False
        self._lock = threading.RLock()

    def load(self):
        with self._lock:
            self._checkpoint()
            if self._closed:
                raise RuntimeError("Face region detector is closed")
            if self._backend is None:
                path = str(self._resolver())
                if self._factory is None:
                    import cv2
                    backend = cv2.FaceDetectorYN.create(path, "", (320, 320), .8, .3, 100)
                else:
                    backend = self._factory(path)
                self._backend = backend

    def detect(self, bgr, *, maximum_dimension=640, checkpoint=None):
        import cv2
        checkpoint = checkpoint or self._checkpoint
        checkpoint()
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3
                or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 1 or max(bgr.shape[:2]) > 1600
                or type(maximum_dimension) is not int or not 32 <= maximum_dimension <= 640):
            raise ValueError("Invalid bounded face region preview")
        with self._lock:
            self.load()
            checkpoint()
            height, width = bgr.shape[:2]
            scale = min(1., maximum_dimension/max(height, width))
            small = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr
            self._backend.setInputSize((small.shape[1], small.shape[0]))
            _, detected = self._backend.detect(small)
            checkpoint()
            if detected is None:
                return np.empty((0, 15), dtype=np.float32)
            values = np.asarray(detected, dtype=np.float32)
            if (values.ndim != 2 or values.shape[1] != 15 or len(values) > 100
                    or not np.isfinite(values).all() or (values[:, 2:4] <= 0).any()
                    or ((values[:, 14] < 0) | (values[:, 14] > 1)).any()):
                raise ValueError("Invalid face region output")
            values = values.copy()
            values[:, :14] /= scale
            # Border proposals are clipped, not used as out-of-image crops.
            x0, y0 = np.clip(values[:, 0], 0, width), np.clip(values[:, 1], 0, height)
            x1 = np.clip(values[:, 0]+values[:, 2], 0, width)
            y1 = np.clip(values[:, 1]+values[:, 3], 0, height)
            values[:, :4] = np.column_stack((x0, y0, x1-x0, y1-y0))
            return values[(x1-x0 >= 1) & (y1-y0 >= 1)]

    def close(self):
        with self._lock:
            self._closed = True
            backend, self._backend = self._backend, None
            if backend is not None and hasattr(backend, "close"):
                backend.close()


@lru_cache(maxsize=1)
def face_descriptor_signature() -> str:
    """Include the exact bundled model bytes in persistent descriptor validity."""

    model_path = resolve_face_landmarker_model_path()
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{FACE_DESCRIPTOR_VERSION}:{digest.hexdigest()}"


class FaceLandmarkerBackend(Protocol):
    def detect_landmarks(self, rgb_image) -> Sequence[Sequence[object]]:
        """Return normalized landmarks for every face in an RGB image."""

    def close(self) -> None:
        """Release native resources."""


@dataclass(frozen=True, slots=True)
class FaceDescriptor:
    """Serializable normalized geometry for one detected face."""

    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "bbox": list(self.bbox),
            "landmarks": [list(point) for point in self.landmarks],
        }

    @classmethod
    def from_dict(cls, value: object) -> FaceDescriptor | None:
        if not isinstance(value, dict):
            return None
        bbox = value.get("bbox")
        landmarks = value.get("landmarks")
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not isinstance(landmarks, (list, tuple))
        ):
            return None
        try:
            parsed_bbox = tuple(float(item) for item in bbox)
            parsed_landmarks = tuple(
                (float(point[0]), float(point[1]))
                for point in landmarks
                if isinstance(point, (list, tuple)) and len(point) >= 2
            )
        except TypeError, ValueError:
            return None
        if len(parsed_landmarks) != len(landmarks):
            return None
        if not np.isfinite((*parsed_bbox, *np.asarray(parsed_landmarks).ravel())).all():
            return None
        return cls(parsed_bbox, parsed_landmarks)


@dataclass(frozen=True, slots=True)
class SubjectDescriptor:
    """All reusable subject geometry extracted from one analysis image."""

    faces: tuple[FaceDescriptor, ...]
    version: str = FACE_DESCRIPTOR_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "faces": [face.to_dict() for face in self.faces],
        }

    @classmethod
    def from_dict(cls, value: object) -> SubjectDescriptor | None:
        if (
            not isinstance(value, dict)
            or value.get("version") != FACE_DESCRIPTOR_VERSION
        ):
            return None
        values = value.get("faces")
        if not isinstance(values, list):
            return None
        faces = tuple(FaceDescriptor.from_dict(item) for item in values)
        if any(face is None for face in faces):
            return None
        return cls(tuple(face for face in faces if face is not None))


def _default_backend_factory(model_path: Path) -> FaceLandmarkerBackend:
    # Import lazily so workflows that never inspect a near-duplicate do not import
    # MediaPipe or initialize its native runtime.
    from .landmarker_runtime import TasksLandmarker

    return TasksLandmarker(model_path)


class FaceAnalysisService:
    """Own one lazy backend and expose immutable, cacheable face descriptors."""

    def __init__(
        self,
        backend_factory: Callable[[Path], FaceLandmarkerBackend] | None = None,
        model_path_resolver: Callable[[], Path] = resolve_face_landmarker_model_path,
    ) -> None:
        self._backend_factory = backend_factory or _default_backend_factory
        self._model_path_resolver = model_path_resolver
        self._backend: FaceLandmarkerBackend | None = None

    def get_backend(self) -> FaceLandmarkerBackend:
        if self._backend is None:
            self._backend = self._backend_factory(self._model_path_resolver())
        return self._backend

    def detect_landmarks(self, rgb_image: np.ndarray) -> Sequence[Sequence[object]]:
        return self.get_backend().detect_landmarks(rgb_image)

    def describe(self, rgb_image: np.ndarray) -> SubjectDescriptor:
        faces: list[FaceDescriptor] = []
        for landmarks in self.detect_landmarks(rgb_image):
            points = tuple((float(point.x), float(point.y)) for point in landmarks)
            if not points:
                continue
            array = np.asarray(points, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError("Face Landmarker returned non-finite coordinates")
            xs = array[:, 0]
            ys = array[:, 1]
            faces.append(
                FaceDescriptor(
                    bbox=(
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max()),
                        float(ys.max()),
                    ),
                    landmarks=points,
                )
            )
        return SubjectDescriptor(tuple(faces))

    def close(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None:
            with contextlib.suppress(RuntimeError):
                backend.close()
