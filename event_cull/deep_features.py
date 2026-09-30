"""Local deep visual features through a compact ONNX MobileNet encoder.

This module is deliberately optional.  The regular culling rules always run on
CPU, and personal calibration falls back to its compact handcrafted descriptor
when the packaged neural runtime or model is unavailable.  ONNX Runtime uses
Core ML on supported Macs and otherwise falls back to CPU.  No image is uploaded.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import numpy as np


LEGACY_PIPELINE_ID = "mobilenet-v3-small-visual-v1"
PIPELINE_ID = "mobilenet-v3-small-visual-v2"
# An opt-in experiment contract: CPU-only extraction cannot silently mix
# Core ML descriptors with descriptors produced after a batch fallback.
CPU_PIPELINE_ID = "mobilenet-v3-small-cpu-visual-v3"
MODEL_FILENAME = "mobilenet_v3_small_visual.onnx"
FEATURE_DIMENSION = 576
_IMAGE_SIZE = 224
_BATCH_SIZE = 24
_LOCK = threading.RLock()
_ENCODERS: dict[str, DeepVisualEncoder] = {}


class DeepFeatureUnavailable(RuntimeError):
    """Raised when the optional local deep-feature runtime is unavailable."""


def _resource_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "event_cull" / "models"
    return Path(__file__).resolve().parent / "models"


def model_path() -> Path | None:
    """Find the small inference-only model shipped with the application."""
    bundled = _resource_directory() / MODEL_FILENAME
    return bundled if bundled.is_file() else None


def preferred_backend() -> str:
    """Return ``coreml``, ``cpu``, or ``unavailable`` without loading the model."""
    if model_path() is None:
        return "unavailable"
    try:
        import onnxruntime as ort
    except ImportError:
        return "unavailable"
    return "coreml" if "CoreMLExecutionProvider" in ort.get_available_providers() else "cpu"


def status_label(language: str = "zh") -> str:
    backend = preferred_backend()
    if language == "zh":
        return {
            "coreml": "Apple Core ML 深度评分已就绪",
            "cpu": "深度评分将使用 CPU（当前设备没有可用 GPU）",
            "unavailable": "深度评分组件未安装；将使用稳定 CPU 基础筛选",
        }[backend]
    return {
        "coreml": "Apple Core ML deep scoring is ready",
        "cpu": "Deep scoring will use CPU (no GPU is available)",
        "unavailable": "Deep scoring is not installed; stable CPU culling remains active",
    }[backend]


class DeepVisualEncoder:
    """A reusable MobileNet feature extractor for one execution backend."""

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or preferred_backend()
        self._model = None
        self._closed = False
        if self.backend == "unavailable":
            raise DeepFeatureUnavailable("Deep visual model is not available locally.")

    def _load(self) -> None:
        if self._closed:
            raise DeepFeatureUnavailable("Deep visual encoder is closed.")
        if self._model is not None:
            return
        path = model_path()
        if path is None:
            raise DeepFeatureUnavailable("Deep visual model is not installed.")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DeepFeatureUnavailable("Deep visual runtime is not installed.") from exc
        try:
            providers = ["CPUExecutionProvider"]
            if self.backend == "coreml":
                providers.insert(0, "CoreMLExecutionProvider")
            settings = ort.SessionOptions()
            # Avoid nesting a full CPU pool inside the six RAW-analysis workers.
            settings.intra_op_num_threads = 2
            settings.inter_op_num_threads = 1
            self._model = ort.InferenceSession(str(path), sess_options=settings, providers=providers)
        except Exception as exc:
            self._model = None
            raise DeepFeatureUnavailable("Deep visual model could not start on this computer.") from exc

    @staticmethod
    def _batch(images: list[np.ndarray], start: int, end: int) -> np.ndarray:
        prepared: list[np.ndarray] = []
        for image in images[start:end]:
            if image is None or image.size == 0:
                raise DeepFeatureUnavailable("One image could not be prepared for deep scoring.")
            resized = cv2.resize(image, (_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
            normalized = (rgb - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)) / np.asarray(
                (0.229, 0.224, 0.225), dtype=np.float32
            )
            prepared.append(np.transpose(normalized, (2, 0, 1)))
        return np.ascontiguousarray(np.stack(prepared))

    @staticmethod
    def _compress(values: np.ndarray) -> np.ndarray:
        # Channels are learned independently, not spatial neighbours. Averaging
        # arbitrary adjacent channels discards the very pose/content differences
        # needed for selection. Regularize the classifier instead (v1 migration
        # is handled by the shared calibration feature builder).
        if values.shape[1] != 576:
            raise DeepFeatureUnavailable("Deep model returned an unexpected feature shape.")
        return values.astype(np.float32)

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, FEATURE_DIMENSION), dtype=np.float32)
        with _LOCK:
            self._load()
            try:
                chunks: list[np.ndarray] = []
                for start in range(0, len(images), _BATCH_SIZE):
                    batch = self._batch(images, start, start + _BATCH_SIZE)
                    encoded = self._model.run(["features"], {"images": batch})[0]
                    chunks.append(np.asarray(encoded, dtype=np.float32))
                return self._compress(np.vstack(chunks))
            except Exception as exc:
                if self.backend == "coreml":
                    self.backend = "cpu"
                    self._model = None
                    return self.encode(images)
                raise DeepFeatureUnavailable("Deep scoring could not process these images.") from exc

    def close(self) -> None:
        """Release a caller-owned diagnostic encoder without touching others."""
        with _LOCK:
            self._closed = True
            self._model = None


def encode_images(images: list[np.ndarray], *, backend: str | None = None) -> tuple[np.ndarray, str]:
    """Return compressed local deep features and the backend actually used."""
    available = preferred_backend()
    if available == "unavailable":
        raise DeepFeatureUnavailable("Deep visual model is not available locally.")
    if backend not in {None, "cpu", "coreml"}:
        raise ValueError("Unknown deep visual backend")
    backend = backend or available
    if backend == "coreml" and available != "coreml":
        raise DeepFeatureUnavailable("Core ML deep visual backend is not available.")
    with _LOCK:
        encoder = _ENCODERS.get(backend)
        if encoder is None:
            encoder = DeepVisualEncoder(backend)
            _ENCODERS[backend] = encoder
        values = encoder.encode(images)
    return values, encoder.backend
