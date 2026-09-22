"""Deterministic first-pass culling for a folder of event images."""

from __future__ import annotations

import os
import json
import math
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import rawpy
except ImportError:  # pragma: no cover - setup validation covers this
    rawpy = None

RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif", *RAW_EXTENSIONS}
ANALYSIS_CACHE_VERSION = 9
MAX_ANALYSIS_WORKERS = 8
_DETECTORS = threading.local()
_EMOTION_MODEL = Path("event_cull/models/emotion-ferplus-8.onnx")
MESSAGES = {
    "zh": {
        "blur": "明显虚焦或抖动",
        "over": "严重过曝",
        "under": "严重欠曝",
        "relative_over": "相邻连拍中突然过曝约 {ev} EV",
        "relative_under": "相邻连拍中突然欠曝约 {ev} EV",
        "flash_miss": "相邻连拍中人物肤色突然变暗约 {ev} EV，疑似闪光灯漏闪",
        "closed": "未检测到睁眼，可能是低头或侧脸，请复核",
        "people": "多人正面表情优先：{faces} 张清晰正面脸，{smiles} 张疑似笑脸",
        "people_focus": "人物清晰度低于前景，疑似对焦到前方物体",
        "expression": "多人表情不完整，不作为 3★ 首选",
        "soft_scene": "连续场景整体偏柔，可能是头纱或前景遮挡，保留代表画面复核",
        "scene_coverage": "连续场景保底：不依赖人脸，保留代表画面复核",
        "single_scene": "独立场景代表：无相似连拍，保留供复核",
        "best": "本组最佳",
        "backup": "优质备用",
        "review": "初选复核",
        "target": "未进入目标选图数量",
        "failed": "无法分析：{error}",
    },
    "en": {
        "blur": "Clear blur or camera shake",
        "over": "Severe overexposure",
        "under": "Severe underexposure",
        "relative_over": "About {ev} EV brighter than adjacent burst frames",
        "relative_under": "About {ev} EV darker than adjacent burst frames",
        "flash_miss": "Subject skin is about {ev} EV darker than adjacent burst frames; possible flash misfire",
        "closed": "Open eyes not detected; the subject may be looking down or sideways",
        "people": "People priority: {faces} clear frontal faces, {smiles} possible smiles",
        "people_focus": "People appear softer than the foreground; focus may be in front",
        "expression": "Group expressions are incomplete; not a first-choice 3-star frame",
        "soft_scene": "The continuous scene is uniformly soft, possibly through a veil or foreground layer; keep a representative frame for review",
        "scene_coverage": "Continuous-scene safeguard: keep a representative frame regardless of face detection",
        "single_scene": "Independent scene representative: no similar burst, retained for review",
        "best": "Best in group",
        "backup": "Strong backup",
        "review": "Review selection",
        "target": "Outside the target selection count",
        "failed": "Could not analyze: {error}",
    },
}


def _analysis_worker_count(logical_cpus: int | None = None) -> int:
    """Scale conservatively because OpenCV also uses CPU inside each task."""
    cores = logical_cpus if logical_cpus is not None else (os.cpu_count() or 4)
    if cores <= 4:
        return max(1, cores)
    if cores <= 8:
        return 4
    if cores <= 12:
        return 6
    return MAX_ANALYSIS_WORKERS


class AnalysisCancelled(Exception):
    """Raised only between files, so an in-flight RAW decode is never interrupted unsafely."""


@dataclass(frozen=True)
class AnalysisOptions:
    picks_per_group: int = 2
    max_picks_per_group: int | None = None
    target_keep_count: int | None = None
    target_keep_max_count: int | None = None
    burst_gap_seconds: float = 8.0
    similarity_threshold: int = 18
    blur_floor: float = 55.0
    severe_clip_fraction: float = 0.12
    detect_closed_eyes: bool = True
    detect_blur: bool = True
    detect_exposure: bool = True
    detect_duplicates: bool = True
    ai_selection_enabled: bool = True
    prefer_people: bool = True
    max_duplicate_picks_per_group: int | None = None
    fill_with_similar: bool = False
    language: str = "zh"


@dataclass
class PhotoResult:
    path: Path
    pair_paths: list[Path] = field(default_factory=list)
    group_id: int = 0
    score: float = 0.0
    suggested_rating: int = 0
    reasons: list[str] = field(default_factory=list)
    sharpness: float | None = None
    highlight_clip: float | None = None
    shadow_clip: float | None = None
    perceptual_hash: int | None = None
    subject_sharpness: float | None = None
    frontal_face_count: int = 0
    smiling_face_count: int = 0
    open_eye_face_count: int = 0
    median_luminance: float | None = None
    subject_luminance: float | None = None
    relative_exposure_ev: float | None = None
    calibration_confidence: float | None = None
    scene_protected: bool = False
    coverage_protected: bool = False

    @property
    def display_name(self) -> str:
        return self.path.name


def _representatives(paths: list[Path]) -> list[tuple[Path, list[Path]]]:
    """Collapse supplied files into logical RAW+JPG photos, preferring RAW."""
    stems: dict[tuple[Path, str], list[Path]] = {}
    for path in sorted(set(paths)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            # Camera counters commonly restart in each card/folder.  Folder is
            # therefore part of the identity; otherwise DSC00001 from two
            # imported folders is silently collapsed into one logical photo.
            stems.setdefault((path.parent.resolve(), path.stem.lower()), []).append(path)
    result: list[tuple[Path, list[Path]]] = []
    for paths in stems.values():
        paths.sort(key=lambda p: (p.suffix.lower() not in RAW_EXTENSIONS, p.name.lower()))
        result.append((paths[0], paths))
    return sorted(
        result,
        key=lambda entry: (
            str(entry[0].parent.resolve()).casefold(),
            entry[0].stat().st_mtime,
            entry[0].name.casefold(),
        ),
    )


def _paths_for_selected_files(paths: list[Path]) -> list[Path]:
    """Include companions so dropping either half of RAW+JPG keeps the pair."""
    expanded: set[Path] = set()
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        expanded.update(
            sibling
            for sibling in path.parent.glob(f"{path.stem}.*")
            if sibling.is_file() and sibling.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(expanded)


def _resize_preview(image: np.ndarray, maximum_dimension: int) -> np.ndarray:
    """Bound display/analysis preview memory without touching the original file."""
    if image is None or image.size == 0:
        return image
    limit = max(1, int(maximum_dimension))
    largest = max(image.shape[:2])
    if largest <= limit:
        return image
    scale = limit / largest
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _load_preview(path: Path, *, maximum_dimension: int = 1600) -> np.ndarray:
    if path.suffix.lower() in RAW_EXTENSIONS:
        if rawpy is None:
            raise RuntimeError("RAW support requires rawpy; install requirements-mvp.txt")
        try:
            with rawpy.imread(str(path)) as raw:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        array = np.frombuffer(thumb.data, np.uint8)
                        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
                    else:
                        image = cv2.cvtColor(thumb.data, cv2.COLOR_RGB2BGR)
                    if image is not None:
                        return _resize_preview(image, maximum_dimension)
                except (
                    rawpy.LibRawNoThumbnailError,
                    rawpy.LibRawUnsupportedThumbnailError,
                ):
                    # Some camera profiles expose no embedded JPEG.  LibRaw can
                    # still generate a fast half-size display preview without
                    # changing the original RAW file.
                    pass
                rendered = raw.postprocess(
                    half_size=True,
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                )
                if rendered is not None and rendered.size:
                    return _resize_preview(
                        cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR), maximum_dimension
                    )
        except rawpy.LibRawError:
            # Fall through to macOS Quick Look below. It sometimes supports a
            # recently released camera before the bundled LibRaw does.
            pass
        quicklook = _macos_quicklook_preview(path, maximum_dimension)
        if quicklook is not None:
            return quicklook
        raise RuntimeError(
            "无法生成 RAW 缩略图：此机型暂未被 LibRaw 或 macOS Quick Look 支持"
        )
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
        if image is None:
            raise RuntimeError("JPEG could not be decoded")
        return _resize_preview(image, maximum_dimension)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((maximum_dimension, maximum_dimension))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _macos_quicklook_preview(path: Path, maximum_dimension: int = 1600) -> np.ndarray | None:
    """Ask macOS for a read-only RAW thumbnail after LibRaw has exhausted it.

    Quick Look has Apple-maintained camera profiles. It is deliberately a
    fallback because spawning it for every ordinary RAW would be slower than
    extracting the embedded JPEG through LibRaw.
    """
    if sys.platform != "darwin":
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="event-cull-quicklook-") as directory:
            output = Path(directory)
            command = [
                "/usr/bin/qlmanage", "-t", "-s", str(max(1, int(maximum_dimension))),
                "-o", str(output), str(path),
            ]
            # Tk has background preview and network threads by the time this
            # fallback is reached.  ``subprocess`` may fork that threaded
            # process on recent macOS releases, which can terminate the app
            # inside Cocoa/Tk.  posix_spawn creates Quick Look directly.
            process_id = os.posix_spawn(command[0], command, os.environ.copy())
            deadline = time.monotonic() + 20
            while True:
                finished_id, wait_status = os.waitpid(process_id, os.WNOHANG)
                if finished_id:
                    if os.waitstatus_to_exitcode(wait_status) != 0:
                        return None
                    break
                if time.monotonic() >= deadline:
                    os.kill(process_id, 15)
                    os.waitpid(process_id, 0)
                    return None
                time.sleep(0.05)
            if not output.exists():
                return None
            candidates = [
                item
                for item in output.rglob("*")
                if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
            if not candidates:
                return None
            image = cv2.imread(str(candidates[0]), cv2.IMREAD_COLOR)
            return _resize_preview(image, maximum_dimension) if image is not None else None
    except OSError:
        return None


def _dhash(gray: np.ndarray) -> int:
    reduced = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return sum(int(reduced[y, x] > reduced[y, x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _message(options: AnalysisOptions, key: str, **values: str) -> str:
    return MESSAGES.get(options.language, MESSAGES["en"])[key].format(**values)


def _face_metrics(
    gray: np.ndarray,
    options: AnalysisOptions,
    global_sharpness: float | None = None,
) -> tuple[list[str], float | None, int, int, int, float | None]:
    """Return eye hints, subject detail, expression counts, and face luminance."""
    if not hasattr(_DETECTORS, "faces"):
        _DETECTORS.faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _DETECTORS.eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
        _DETECTORS.smiles = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    faces = _DETECTORS.faces
    eyes = _DETECTORS.eyes
    smiles = _DETECTORS.smiles

    def happiness_probability(face_gray: np.ndarray) -> float:
        if not hasattr(_DETECTORS, "emotion"):
            base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
            model_path = base / _EMOTION_MODEL
            try:
                _DETECTORS.emotion = cv2.dnn.readNetFromONNX(str(model_path))
            except (cv2.error, OSError):
                _DETECTORS.emotion = None
        if _DETECTORS.emotion is None or face_gray.size == 0:
            return 0.0
        input_face = cv2.resize(face_gray, (64, 64)).astype(np.float32)
        _DETECTORS.emotion.setInput(input_face.reshape(1, 1, 64, 64))
        scores = _DETECTORS.emotion.forward().reshape(-1)
        shifted = scores - np.max(scores)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        return float(probabilities[1]) if len(probabilities) > 1 else 0.0

    def detect_at(maximum: int, minimum_face: int) -> list[tuple[int, int, int, int]]:
        detection_scale = min(1.0, maximum / max(gray.shape))
        detection_gray = (
            gray
            if detection_scale == 1.0
            else cv2.resize(gray, None, fx=detection_scale, fy=detection_scale, interpolation=cv2.INTER_AREA)
        )
        return [
            tuple(int(round(value / detection_scale)) for value in (x, y, width, height))
            for x, y, width, height in faces.detectMultiScale(
                detection_gray, scaleFactor=1.1, minNeighbors=5, minSize=(minimum_face, minimum_face)
            )
        ]

    detected = detect_at(640, 18)
    # A soft whole-frame score plus no small-preview face is ambiguous. Retry
    # only those frames at higher resolution instead of slowing every image.
    if not detected and global_sharpness is not None and global_sharpness < options.blur_floor:
        detected = detect_at(960, 24)

    # The cascade can report the same face twice at nearly identical bounds.
    # Keep the largest rectangle and discard heavy overlaps before measuring.
    distinct: list[tuple[int, int, int, int]] = []
    for candidate in sorted(detected, key=lambda rect: int(rect[2]) * int(rect[3]), reverse=True):
        x, y, width, height = map(int, candidate)
        area = width * height
        duplicate = False
        for kept_x, kept_y, kept_width, kept_height in distinct:
            overlap_width = max(0, min(x + width, kept_x + kept_width) - max(x, kept_x))
            overlap_height = max(0, min(y + height, kept_y + kept_height) - max(y, kept_y))
            overlap = overlap_width * overlap_height
            union = area + kept_width * kept_height - overlap
            if union and overlap / union >= 0.35:
                duplicate = True
                break
        if not duplicate:
            distinct.append((x, y, width, height))

    sharpness_values = [
        float(cv2.Laplacian(gray[y : y + height, x : x + width], cv2.CV_64F).var())
        for x, y, width, height in distinct
    ]
    face_luminances = [
        float(np.median(gray[y : y + height, x : x + width]))
        for x, y, width, height in distinct
    ]
    subject_sharpness = float(np.median(sharpness_values)) if sharpness_values else None
    subject_luminance = float(np.median(face_luminances)) if face_luminances else None
    clear_faces = [
        (rect, sharpness)
        for rect, sharpness in zip(distinct, sharpness_values)
        if sharpness >= options.blur_floor * 0.7
    ]
    smiling_face_count = 0
    open_eye_face_count = 0
    smile_candidates = sorted(
        clear_faces,
        key=lambda item: item[0][2] * item[0][3],
        reverse=True,
    )[:8]
    for (x, y, width, height), _sharpness in smile_candidates:
        upper_face = gray[y : y + height // 2, x : x + width]
        eye_minimum = (max(10, width // 8), max(10, height // 10))
        if len(eyes.detectMultiScale(upper_face, 1.1, 6, minSize=eye_minimum)) > 0:
            open_eye_face_count += 1
        lower_face = gray[y + height // 3 : y + height, x : x + width]
        scale = min(1.0, 96 / max(lower_face.shape))
        smile_image = (
            lower_face
            if scale == 1.0
            else cv2.resize(lower_face, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        )
        minimum_smile = (
            max(12, int(round(width * scale / 5))),
            max(8, int(round(height * scale / 10))),
        )
        haar_smile = len(
            smiles.detectMultiScale(smile_image, 1.5, 12, minSize=minimum_smile)
        ) > 0
        padding = max(2, int(round(width * 0.12)))
        face_crop = gray[
            max(0, y - padding) : min(gray.shape[0], y + height + padding),
            max(0, x - padding) : min(gray.shape[1], x + width + padding),
        ]
        emotion_smile = not haar_smile and happiness_probability(face_crop) >= 0.85
        if haar_smile or emotion_smile:
            smiling_face_count += 1
    reasons: list[str] = []
    if options.detect_closed_eyes and len(distinct) == 1 and open_eye_face_count == 0:
        reasons.append(_message(options, "closed"))
    if (
        options.detect_closed_eyes
        and len(clear_faces) >= 2
        and 0 < open_eye_face_count < len(clear_faces)
    ):
        reasons.append(_message(options, "expression"))
    return (
        reasons,
        subject_sharpness,
        len(clear_faces),
        smiling_face_count,
        open_eye_face_count,
        subject_luminance,
    )


def _face_reasons(gray: np.ndarray, options: AnalysisOptions) -> list[str]:
    """Compatibility wrapper used by focused detector tests."""
    return _face_metrics(gray, options)[0]


def _people_priority_bonus(frontal_faces: int, smiling_faces: int) -> float:
    """Favor clear group expressions without overwhelming technical quality."""
    return min(max(frontal_faces, 0), 6) * 2.0 + min(max(smiling_faces, 0), 6) * 3.0


def _effective_sharpness(global_sharpness: float, subject_sharpness: float | None) -> float:
    """Prefer subject detail while retaining a small whole-frame contribution."""
    if subject_sharpness is None:
        return global_sharpness
    return subject_sharpness * 0.7 + global_sharpness * 0.3


def _analysis_source(primary: Path, pairs: list[Path]) -> Path:
    """Use a paired JPEG for fast, faithful culling while rating the RAW primary."""
    for path in pairs:
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            return path
    return primary


def _cache_path() -> Path:
    override = os.environ.get("EVENT_CULL_CACHE_PATH")
    if override:
        return Path(override)
    return Path.home() / "Library" / "Caches" / "照片原图快速筛选器" / "analysis-v1.json"


def _load_analysis_cache() -> dict[str, dict]:
    path = _cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") == ANALYSIS_CACHE_VERSION and isinstance(payload.get("items"), dict):
            return payload["items"]
    except (OSError, ValueError, TypeError):
        pass
    return {}


def _write_analysis_cache(items: dict[str, dict]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump({"version": ANALYSIS_CACHE_VERSION, "items": items}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_signature(source: Path, options: AnalysisOptions) -> dict:
    stat = source.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "blur_floor": options.blur_floor,
        "clip_fraction": options.severe_clip_fraction,
        "detect_closed_eyes": options.detect_closed_eyes,
        "detect_blur": options.detect_blur,
        "detect_exposure": options.detect_exposure,
        "language": options.language,
    }


def _result_from_cache(photo: Path, pairs: list[Path], record: dict, signature: dict) -> PhotoResult | None:
    if record.get("signature") != signature:
        return None
    try:
        legacy_eye_hints = {"疑似闭眼，请复核", "Possible closed eyes — review"}
        reasons = [
            _message(AnalysisOptions(language=signature.get("language", "zh")), "closed")
            if reason in legacy_eye_hints
            else reason
            for reason in record["reasons"]
        ]
        return PhotoResult(
            photo,
            pairs,
            score=float(record["score"]),
            reasons=reasons,
            sharpness=float(record["sharpness"]),
            highlight_clip=float(record["highlight_clip"]),
            shadow_clip=float(record["shadow_clip"]),
            perceptual_hash=int(record["perceptual_hash"]),
            subject_sharpness=(
                float(record["subject_sharpness"])
                if record.get("subject_sharpness") is not None
                else None
            ),
            frontal_face_count=int(record.get("frontal_face_count", 0)),
            smiling_face_count=int(record.get("smiling_face_count", 0)),
            open_eye_face_count=int(record.get("open_eye_face_count", 0)),
            median_luminance=(
                float(record["median_luminance"])
                if record.get("median_luminance") is not None
                else None
            ),
            subject_luminance=(
                float(record["subject_luminance"])
                if record.get("subject_luminance") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _cache_record(result: PhotoResult, signature: dict) -> dict:
    return {
        "signature": signature,
        "score": result.score,
        "reasons": result.reasons,
        "sharpness": result.sharpness,
        "highlight_clip": result.highlight_clip,
        "shadow_clip": result.shadow_clip,
        "perceptual_hash": result.perceptual_hash,
        "subject_sharpness": result.subject_sharpness,
        "frontal_face_count": result.frontal_face_count,
        "smiling_face_count": result.smiling_face_count,
        "open_eye_face_count": result.open_eye_face_count,
        "median_luminance": result.median_luminance,
        "subject_luminance": result.subject_luminance,
    }


def _analyze(path: Path, pairs: list[Path], options: AnalysisOptions) -> PhotoResult:
    image = _load_preview(_analysis_source(path, pairs))
    if max(image.shape[:2]) > 1600:
        scale = 1600 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    highlights = float(np.mean(gray >= 250))
    shadows = float(np.mean(gray <= 5))
    (
        face_reasons,
        subject_sharpness,
        frontal_faces,
        smiling_faces,
        open_eye_faces,
        subject_luminance,
    ) = _face_metrics(gray, options, sharpness)
    median_luminance = float(np.median(gray))
    effective_sharpness = _effective_sharpness(sharpness, subject_sharpness)
    reasons: list[str] = []
    if options.detect_blur and effective_sharpness < options.blur_floor:
        reasons.append(_message(options, "blur"))
    severe_overexposure = highlights >= options.severe_clip_fraction and (
        median_luminance >= 205
        or (subject_luminance is not None and subject_luminance >= 220)
    )
    severe_underexposure = shadows >= options.severe_clip_fraction and median_luminance <= 50
    if options.detect_exposure and severe_overexposure:
        reasons.append(_message(options, "over"))
    if options.detect_exposure and severe_underexposure:
        reasons.append(_message(options, "under"))
    reasons.extend(face_reasons)
    if (
        subject_sharpness is not None
        and subject_sharpness < options.blur_floor * 0.8
        and sharpness >= max(options.blur_floor * 1.4, subject_sharpness * 2.0)
    ):
        reasons.append(_message(options, "people_focus"))
    if frontal_faces >= 2 or smiling_faces > 0:
        reasons.append(
            _message(options, "people", faces=str(frontal_faces), smiles=str(smiling_faces))
        )
    quality = min(effective_sharpness / max(options.blur_floor, 1), 3.0) * 25
    quality += (1 - min(highlights + shadows, 0.5) / 0.5) * 25
    if options.prefer_people:
        quality += _people_priority_bonus(frontal_faces, smiling_faces)
    return PhotoResult(path, pairs, score=quality, reasons=reasons, sharpness=sharpness, highlight_clip=highlights, shadow_clip=shadows, perceptual_hash=_dhash(gray), subject_sharpness=subject_sharpness, frontal_face_count=frontal_faces, smiling_face_count=smiling_faces, open_eye_face_count=open_eye_faces, median_luminance=median_luminance, subject_luminance=subject_luminance)


def _analyze_cached(
    photo: Path,
    pairs: list[Path],
    options: AnalysisOptions,
    cache: dict[str, dict],
) -> tuple[PhotoResult, str, dict | None]:
    source = _analysis_source(photo, pairs)
    key = str(source.resolve())
    signature = _cache_signature(source, options)
    cached = _result_from_cache(photo, pairs, cache.get(key, {}), signature)
    if cached is not None:
        return cached, key, None
    result = _analyze(photo, pairs, options)
    return result, key, _cache_record(result, signature)


def _same_group(previous: PhotoResult, current: PhotoResult, options: AnalysisOptions) -> bool:
    if not options.detect_duplicates:
        return False
    if previous.path.parent.resolve() != current.path.parent.resolve():
        return False
    gap = current.path.stat().st_mtime - previous.path.stat().st_mtime
    distance = _hamming(previous.perceptual_hash or 0, current.perceptual_hash or 0)
    return (
        gap <= options.burst_gap_seconds and distance <= options.similarity_threshold
    ) or (
        gap <= min(2.0, options.burst_gap_seconds)
        and distance <= options.similarity_threshold + 4
    )


def _luminance_ev(value: float) -> float:
    """Approximate exposure value from a gamma-encoded preview luminance."""
    linear = max((value / 255.0) ** 2.2, 1e-6)
    return math.log2(linear)


def _mark_relative_exposure_outliers(
    group: list[PhotoResult], options: AnalysisOptions
) -> None:
    """Flag an isolated roughly one-stop jump inside a stable burst.

    Two agreeing neighbours are required.  This avoids calling a genuine scene
    or framing change over/underexposure merely because its overall brightness
    differs.
    """
    for item in group:
        item.relative_exposure_ev = None
    if not options.detect_exposure:
        return
    if len(group) < 3:
        return
    for index, item in enumerate(group):
        if item.perceptual_hash is None:
            continue
        neighbours: list[PhotoResult] = []
        for offset in (1, -1, 2, -2):
            neighbour_index = index + offset
            if not 0 <= neighbour_index < len(group):
                continue
            candidate = group[neighbour_index]
            if candidate.perceptual_hash is None:
                continue
            if _hamming(item.perceptual_hash, candidate.perceptual_hash) <= options.similarity_threshold + 4:
                neighbours.append(candidate)
            if len(neighbours) == 2:
                break
        if len(neighbours) < 2:
            continue
        use_subject_luminance = (
            item.subject_luminance is not None
            and all(candidate.subject_luminance is not None for candidate in neighbours)
        )
        target_luminance = (
            item.subject_luminance
            if use_subject_luminance
            else item.median_luminance
        )
        neighbour_luminances = [
            candidate.subject_luminance if use_subject_luminance else candidate.median_luminance
            for candidate in neighbours
        ]
        if target_luminance is None or any(value is None for value in neighbour_luminances):
            continue
        neighbour_evs = [_luminance_ev(float(value)) for value in neighbour_luminances]
        if max(neighbour_evs) - min(neighbour_evs) > 0.45:
            continue
        baseline_ev = sum(neighbour_evs) / len(neighbour_evs)
        delta = _luminance_ev(float(target_luminance)) - baseline_ev
        if abs(delta) < 1.0:
            continue
        item.relative_exposure_ev = delta
        key = "relative_over" if delta > 0 else (
            "flash_miss" if use_subject_luminance else "relative_under"
        )
        reason = _message(options, key, ev=f"{abs(delta):.1f}")
        if reason not in item.reasons:
            item.reasons.append(reason)


def _people_focus_issue(item: PhotoResult, options: AnalysisOptions) -> bool:
    if not options.detect_blur:
        return False
    return (
        item.subject_sharpness is not None
        and item.sharpness is not None
        and item.subject_sharpness < options.blur_floor * 0.8
        and item.sharpness >= max(options.blur_floor * 1.4, item.subject_sharpness * 2.0)
    )


def _group_expression_issue(item: PhotoResult) -> bool:
    # Treat a confirmed mixed result as risky.  When the eye cascade finds no
    # eyes at all, keep it ambiguous so low-head and atmospheric frames survive.
    # A natural smile is a positive moment even when someone else looks away.
    return (
        item.frontal_face_count >= 2
        and item.smiling_face_count == 0
        and 0 < item.open_eye_face_count < item.frontal_face_count
    )


def _focus_fault(item: PhotoResult, options: AnalysisOptions) -> bool:
    return options.detect_blur and item.sharpness is not None and _effective_sharpness(
        item.sharpness, item.subject_sharpness
    ) < options.blur_floor


def _protect_coherent_soft_scene(
    group: list[PhotoResult], options: AnalysisOptions
) -> None:
    """Keep 3-4 representatives when a long intentional scene is uniformly soft."""
    for item in group:
        item.scene_protected = False
    if len(group) < 10:
        return
    focus_faults = [item for item in group if _focus_fault(item, options)]
    face_evidence = [item for item in group if item.subject_sharpness is not None]
    if len(focus_faults) < math.ceil(len(group) * 0.55) or len(face_evidence) < 3:
        return
    keep_count = min(4, max(3, math.ceil(len(group) / 6)))
    representatives = sorted(
        group,
        key=lambda item: (
            _effective_sharpness(item.sharpness or 0.0, item.subject_sharpness),
            item.score,
        ),
        reverse=True,
    )[:keep_count]
    blur_reason = _message(options, "blur")
    soft_reason = _message(options, "soft_scene")
    for item in representatives:
        item.scene_protected = True
        item.reasons = [reason for reason in item.reasons if reason != blur_reason]
        if soft_reason not in item.reasons:
            item.reasons.append(soft_reason)


def _has_hard_fault(item: PhotoResult, options: AnalysisOptions) -> bool:
    if item.perceptual_hash is None:
        return True
    focus_fault = _focus_fault(item, options) and not (
        item.scene_protected or item.coverage_protected
    )
    return focus_fault or _has_exposure_fault(item, options)


def _has_exposure_fault(item: PhotoResult, options: AnalysisOptions) -> bool:
    if not options.detect_exposure:
        return False
    severe_overexposure = (
        item.highlight_clip is not None
        and item.highlight_clip >= options.severe_clip_fraction
        and (
            (
                item.median_luminance is not None
                and item.median_luminance >= 205
            )
            or (
                item.subject_luminance is not None
                and item.subject_luminance >= 220
            )
        )
    )
    severe_underexposure = (
        item.shadow_clip is not None
        and item.shadow_clip >= options.severe_clip_fraction
        and item.median_luminance is not None
        and item.median_luminance <= 50
    )
    exposure_fault = (
        severe_overexposure
        or severe_underexposure
        or (
            item.relative_exposure_ev is not None
            and abs(item.relative_exposure_ev) >= 1.0
        )
    )
    return exposure_fault


def _protect_scene_minimum(
    group: list[PhotoResult], options: AnalysisOptions
) -> None:
    """Ensure a deliberate 3+ frame scene is not lost solely to face/focus heuristics."""
    for item in group:
        item.coverage_protected = False
    if len(group) < 3:
        return

    minimum = 1 if len(group) <= 5 else 2
    already_sound = [item for item in group if not _has_hard_fault(item, options)]
    missing = minimum - len(already_sound)
    if missing <= 0:
        return

    # Rescue only moderately soft frames. Missing hashes, exposure outliers,
    # and catastrophically blurred captures remain hard rejects.
    rescue_floor = options.blur_floor * 0.55
    rescue_candidates = [
        item
        for item in group
        if item not in already_sound
        and item.perceptual_hash is not None
        and not _has_exposure_fault(item, options)
        and item.sharpness is not None
        and _effective_sharpness(item.sharpness, item.subject_sharpness) >= rescue_floor
    ]
    blur_reason = _message(options, "blur")
    coverage_reason = _message(options, "scene_coverage")
    for item in sorted(
        rescue_candidates,
        key=lambda candidate: (
            _effective_sharpness(candidate.sharpness or 0.0, candidate.subject_sharpness),
            candidate.score,
        ),
        reverse=True,
    )[:missing]:
        item.coverage_protected = True
        item.reasons = [reason for reason in item.reasons if reason != blur_reason]
        if coverage_reason not in item.reasons:
            item.reasons.append(coverage_reason)


def _ranking_score(item: PhotoResult, options: AnalysisOptions) -> float:
    """Rank on image quality; a missing-eye hint is informational only."""
    focus_penalty = 35.0 if _people_focus_issue(item, options) else 0.0
    expression_penalty = 18.0 if _group_expression_issue(item) else 0.0
    return item.score - focus_penalty - expression_penalty


def _review_priority(item: PhotoResult, options: AnalysisOptions) -> float:
    """Prefer an open-eyed alternative without hard-rejecting ambiguous poses."""
    closed_eye_penalty = (
        12.0
        if item.frontal_face_count > 0 and _message(options, "closed") in item.reasons
        else 0.0
    )
    smile_priority = min(item.smiling_face_count, 2) * 24.0 if options.prefer_people else 0.0
    return item.score + smile_priority - closed_eye_penalty


def _diverse_selection(
    candidates: list[PhotoResult],
    limit: int,
    options: AnalysisOptions,
    priority: Callable[[PhotoResult], float] | None = None,
) -> list[PhotoResult]:
    """Take one strong frame per scene before filling with similar backups."""
    if limit <= 0:
        return []
    rank = priority or (lambda item: _ranking_score(item, options))
    ordered = sorted(candidates, key=rank, reverse=True)
    leads: dict[int, PhotoResult] = {}
    for item in ordered:
        leads.setdefault(item.group_id, item)
    selected = sorted(leads.values(), key=rank, reverse=True)[:limit]
    selected_paths = {item.path for item in selected}
    for item in ordered:
        if len(selected) >= limit:
            break
        if item.path not in selected_paths:
            selected.append(item)
            selected_paths.add(item.path)
    return selected


def _group_keep_quota(group: list[PhotoResult], options: AnalysisOptions) -> int:
    """Scale backups with burst length, respecting an optional user ceiling."""
    count = len(group)
    if count <= 0:
        return 0
    full_windows, remainder = divmod(count, 8)
    quota = full_windows * 2 + min(remainder, 2)
    if count >= 8 and max((item.frontal_face_count for item in group), default=0) >= 4:
        quota = max(quota, math.ceil(count / 3))
    keep_quota = max(options.picks_per_group, quota)
    if options.max_picks_per_group is not None:
        keep_quota = min(keep_quota, options.max_picks_per_group)
    return min(count, keep_quota)


def _sequence_priority_order(
    group: list[PhotoResult], priority: Callable[[PhotoResult], float]
) -> list[PhotoResult]:
    """Protect two strong frames from every consecutive eight-frame window."""
    windows = [group[start : start + 8] for start in range(0, len(group), 8)]
    ranked_windows = [sorted(window, key=priority, reverse=True) for window in windows]
    ordered: list[PhotoResult] = []
    selected_paths: set[Path] = set()
    for position in range(2):
        for window in ranked_windows:
            if position < len(window):
                ordered.append(window[position])
                selected_paths.add(window[position].path)
    ordered.extend(
        sorted(
            (item for item in group if item.path not in selected_paths),
            key=priority,
            reverse=True,
        )
    )
    return ordered


def _density_selection(
    candidates: list[PhotoResult],
    limit: int,
    groups: list[list[PhotoResult]],
    options: AnalysisOptions,
    priority: Callable[[PhotoResult], float],
) -> list[PhotoResult]:
    """Cover every scene, then add proportional backups, then fill globally."""
    if limit <= 0:
        return []
    candidate_paths = {item.path for item in candidates}
    candidate_groups: list[list[PhotoResult]] = []
    ranked_groups: list[list[PhotoResult]] = []
    source_group_lengths: list[int] = []
    for group in groups:
        group_candidates = [item for item in group if item.path in candidate_paths]
        if group_candidates:
            candidate_groups.append(group_candidates)
            ranked_groups.append(_sequence_priority_order(group_candidates, priority))
            source_group_lengths.append(len(group))

    # A photographer can set a separate cap for each detected duplicate
    # cluster.  It intentionally applies only when clustering is enabled;
    # independent photos remain free to satisfy the wider task target.
    group_limits = (
        {
            group_id: min(
                _group_keep_quota(group, options),
                options.max_duplicate_picks_per_group
                if options.max_duplicate_picks_per_group is not None
                else len(group),
            )
            for group_id, group in enumerate(groups, start=1)
        }
        if (
            options.max_picks_per_group is not None
            or options.max_duplicate_picks_per_group is not None
        )
        else {}
    )
    selected: list[PhotoResult] = []
    selected_paths: set[Path] = set()
    selected_per_group: dict[int, int] = {}

    def add(item: PhotoResult) -> None:
        group_limit = group_limits.get(item.group_id, len(candidates))
        if (
            len(selected) < limit
            and item.path not in selected_paths
            and selected_per_group.get(item.group_id, 0) < group_limit
        ):
            selected.append(item)
            selected_paths.add(item.path)
            selected_per_group[item.group_id] = selected_per_group.get(item.group_id, 0) + 1

    # Scene completeness precedes both singleton coverage and global score:
    # 3-5 consecutive frames keep at least one representative, and 6+ keep at
    # least two when the target count can accommodate all scene minimums.
    scene_minimums: list[PhotoResult] = []
    for source_length, ranked_group in zip(source_group_lengths, ranked_groups):
        minimum = 2 if source_length >= 6 else 1 if source_length >= 3 else 0
        scene_minimums.extend(ranked_group[:minimum])
    for item in sorted(scene_minimums, key=priority, reverse=True):
        add(item)

    # A natural smile is worth keeping even when another person looks away.
    # Protect the strongest smiling frame from each scene before ordinary
    # scene leads consume the remaining target slots.
    smile_leads = []
    for ranked_group in ranked_groups:
        smiling = [item for item in ranked_group if item.smiling_face_count > 0]
        if smiling:
            smile_leads.append(max(smiling, key=priority))
    for item in sorted(smile_leads, key=priority, reverse=True):
        add(item)

    leads = sorted((group[0] for group in ranked_groups), key=priority, reverse=True)
    for item in leads:
        add(item)

    for item in sorted(
        (item for item in candidates if item.scene_protected),
        key=priority,
        reverse=True,
    ):
        add(item)

    # A long continuous take can slowly change pose or framing while adjacent
    # hashes remain similar. Protect one frame from every eight-frame window.
    # When the final window contains exactly two sound frames, keep both; that
    # short tail is commonly a deliberate last pose rather than duplication.
    window_coverage: list[PhotoResult] = []
    for group in candidate_groups:
        windows = [group[start : start + 8] for start in range(0, len(group), 8)]
        for window in windows:
            window_coverage.append(max(window, key=priority))
        if windows and len(windows[-1]) == 2:
            window_coverage.extend(windows[-1])
    for item in sorted(
        {item.path: item for item in window_coverage}.values(),
        key=priority,
        reverse=True,
    ):
        add(item)

    maximum_quota = max(
        (_group_keep_quota(group, options) for group in ranked_groups),
        default=0,
    )
    for position in range(1, maximum_quota):
        round_candidates = [
            group[position]
            for group in ranked_groups
            if position < min(len(group), _group_keep_quota(group, options))
        ]
        for item in sorted(round_candidates, key=priority, reverse=True):
            add(item)

    for item in sorted(candidates, key=priority, reverse=True):
        add(item)

    # When explicitly requested, use additional frames from already capped
    # duplicate clusters to meet the task target.  The checkbox makes this
    # trade-off deliberate instead of silently adding near-duplicates.
    if options.fill_with_similar and len(selected) < limit:
        for item in sorted(candidates, key=priority, reverse=True):
            if len(selected) >= limit:
                break
            if item.path not in selected_paths:
                selected.append(item)
                selected_paths.add(item.path)
    return selected


def _three_star_score(item: PhotoResult, options: AnalysisOptions) -> float:
    """Within a scene, prefer a clear smile and complete group expressions."""
    return (
        _ranking_score(item, options)
        + item.smiling_face_count * 24.0
        + item.open_eye_face_count * 4.0
    )


def _assign_tiered_ratings(
    groups: list[list[PhotoResult]], options: AnalysisOptions
) -> None:
    """Assign cumulative 1★, 2★, and 3★ tiers from one target count."""
    eligible: list[PhotoResult] = []
    for group_id, group in enumerate(groups, 1):
        _mark_relative_exposure_outliers(group, options)
        _protect_coherent_soft_scene(group, options)
        _protect_scene_minimum(group, options)
        for item in group:
            item.group_id = group_id
            item.suggested_rating = 0
            if not _has_hard_fault(item, options):
                eligible.append(item)

    if not options.ai_selection_enabled:
        # Still perform local quality checks and grouping, but leave every
        # rating for the photographer to decide manually.
        for item in eligible:
            item.reasons.append("待人工复核" if options.language == "zh" else "Awaiting manual review")
        return

    ranked = sorted(eligible, key=lambda item: _ranking_score(item, options), reverse=True)
    target_floor = len(ranked) if options.target_keep_count is None else max(1, options.target_keep_count)
    target = target_floor
    if options.target_keep_max_count is not None:
        # The second number is a preferred quantity rather than a hard cap.
        # Scene coverage may legitimately go beyond it; the photographer's
        # first number remains the guaranteed minimum whenever enough sound
        # photos exist to meet it.
        coverage_floor = sum(
            1 for group in groups if any(item in eligible for item in group)
        )
        target = max(
            target_floor,
            max(1, options.target_keep_max_count),
            coverage_floor,
        )
    one_star_or_more = _density_selection(
        ranked,
        min(target, len(ranked)),
        groups,
        options,
        priority=lambda item: _review_priority(item, options),
    )

    # Per-burst and duplicate caps make the review wall manageable, but they
    # must not silently reduce an explicit minimum target.  If there are
    # enough eligible photos, relax those caps only until that floor is met.
    required_floor = min(target_floor, len(ranked))
    if len(one_star_or_more) < required_floor:
        selected_paths = {item.path for item in one_star_or_more}
        for item in ranked:
            if len(one_star_or_more) >= required_floor:
                break
            if item.path not in selected_paths:
                one_star_or_more.append(item)
                selected_paths.add(item.path)

    selected_by_group: dict[int, list[PhotoResult]] = {}
    for item in one_star_or_more:
        selected_by_group.setdefault(item.group_id, []).append(item)

    three_target = min((target + 9) // 10, len(one_star_or_more))
    best_per_group: dict[int, PhotoResult] = {}
    three_candidates = [
        item
        for item in one_star_or_more
        if not _people_focus_issue(item, options)
        and not _group_expression_issue(item)
    ]
    for item in sorted(
        three_candidates,
        key=lambda candidate: _three_star_score(candidate, options),
        reverse=True,
    ):
        best_per_group.setdefault(item.group_id, item)
    three_star = sorted(
        best_per_group.values(),
        key=lambda item: _three_star_score(item, options),
        reverse=True,
    )[:three_target]

    burst_leads = [
        max(items, key=lambda item: _three_star_score(item, options))
        for group_id, items in selected_by_group.items()
        if len(groups[group_id - 1]) >= 2
    ]
    mandatory_two = list({item.path: item for item in [*three_star, *burst_leads]}.values())
    two_target = min(
        max((target + 2) // 3, len(mandatory_two)),
        len(one_star_or_more),
    )
    two_star_or_more = mandatory_two
    two_paths = {item.path for item in two_star_or_more}
    for item in sorted(
        one_star_or_more,
        key=lambda candidate: _ranking_score(candidate, options),
        reverse=True,
    ):
        if len(two_star_or_more) >= two_target:
            break
        if item.path not in two_paths:
            two_star_or_more.append(item)
            two_paths.add(item.path)

    one_paths = {item.path for item in one_star_or_more}
    two_paths = {item.path for item in two_star_or_more}
    three_paths = {item.path for item in three_star}
    singleton_group_ids = {
        group_id for group_id, group in enumerate(groups, start=1) if len(group) == 1
    }
    single_scene_reason = _message(options, "single_scene")
    for item in eligible:
        if item.path in three_paths:
            item.suggested_rating = 3
            item.reasons.append(_message(options, "best"))
        elif item.path in two_paths:
            item.suggested_rating = 2
            item.reasons.append(_message(options, "backup"))
        elif item.path in one_paths:
            item.suggested_rating = 1
            item.reasons.append(_message(options, "review"))
        else:
            item.reasons.append(_message(options, "target"))
        if item.suggested_rating > 0 and item.group_id in singleton_group_ids:
            if single_scene_reason not in item.reasons:
                item.reasons.append(single_scene_reason)


def analyze_folder(
    folder: Path,
    options: AnalysisOptions = AnalysisOptions(),
    progress_callback: Callable[[int, int, Path], None] | None = None,
    pause_event: threading.Event | None = None,
    cancel_event: threading.Event | None = None,
) -> list[PhotoResult]:
    return analyze_paths(list(folder.iterdir()), options, progress_callback, pause_event, cancel_event)


def analyze_paths(
    paths: list[Path],
    options: AnalysisOptions = AnalysisOptions(),
    progress_callback: Callable[[int, int, Path], None] | None = None,
    pause_event: threading.Event | None = None,
    cancel_event: threading.Event | None = None,
) -> list[PhotoResult]:
    """Analyze exactly the dropped photos, automatically pairing RAW+JPG."""
    if options.picks_per_group < 1:
        raise ValueError("picks_per_group must be at least 1")
    if (
        options.max_picks_per_group is not None
        and options.max_picks_per_group < options.picks_per_group
    ):
        raise ValueError("max_picks_per_group must be at least picks_per_group")
    if (
        options.max_duplicate_picks_per_group is not None
        and options.max_duplicate_picks_per_group < 1
    ):
        raise ValueError("max_duplicate_picks_per_group must be at least 1")
    if (
        options.target_keep_max_count is not None
        and options.target_keep_count is not None
        and options.target_keep_max_count < options.target_keep_count
    ):
        raise ValueError("target_keep_max_count must be at least target_keep_count")
    entries = _representatives(_paths_for_selected_files(paths))
    cache = _load_analysis_cache()
    ordered_results: list[PhotoResult | None] = [None] * len(entries)
    total = len(entries)

    def wait_if_paused() -> None:
        while pause_event is not None and pause_event.is_set():
            if cancel_event is not None and cancel_event.is_set():
                raise AnalysisCancelled()
            time.sleep(0.15)
        if cancel_event is not None and cancel_event.is_set():
            raise AnalysisCancelled()

    if total:
        worker_count = min(_analysis_worker_count(), total)
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="event-cull")
        futures: dict[int, Future[tuple[PhotoResult, str, dict | None]]] = {}
        next_submit = 0

        def submit(index: int) -> None:
            photo, pairs = entries[index]
            futures[index] = executor.submit(_analyze_cached, photo, pairs, options, cache)

        try:
            wait_if_paused()
            while next_submit < worker_count:
                submit(next_submit)
                next_submit += 1
            for index, (photo, pairs) in enumerate(entries):
                wait_if_paused()
                try:
                    result, cache_key, new_record = futures.pop(index).result()
                    ordered_results[index] = result
                    if new_record is not None:
                        cache[cache_key] = new_record
                except Exception as exc:
                    ordered_results[index] = PhotoResult(
                        photo,
                        pairs,
                        reasons=[_message(options, "failed", error=str(exc))],
                    )
                if progress_callback is not None:
                    progress_callback(index + 1, total, photo)
                wait_if_paused()
                if next_submit < total:
                    submit(next_submit)
                    next_submit += 1
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    results = [result for result in ordered_results if result is not None]
    try:
        _write_analysis_cache(cache)
    except OSError:
        pass
    groups: list[list[PhotoResult]] = []
    for result in results:
        if not groups or not _same_group(groups[-1][-1], result, options):
            groups.append([])
        groups[-1].append(result)
    _assign_tiered_ratings(groups, options)
    return results
