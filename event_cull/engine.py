"""Deterministic first-pass culling for a folder of event images."""

from __future__ import annotations

import os
import csv
import json
import math
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from collections import deque
import copy
from functools import lru_cache
from pathlib import Path
from typing import Callable
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageOps
from event_cull.rating_policy import automatic_rating
from event_cull.wedding_rules import (
    CONTENT_TYPES, RULE_VERSION, add_evidence, apply_evidence_gates, eye_neutral,
)
from event_cull.wedding_content import (
    ContentModelCancelled, LocalWeddingContentModel, BundledWeddingContentModel, validate_observation,
)
from event_cull.bundled_inference import InferenceBudgetExpired

try:
    import rawpy
except ImportError:  # pragma: no cover - setup validation covers this
    rawpy = None

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - packaging validation covers this
    register_heif_opener = None

RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2"}
HEIC_EXTENSIONS = {".heic", ".heif"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", *HEIC_EXTENSIONS, *RAW_EXTENSIONS}
ATMOSPHERE_PORTRAIT_SCENE = "男生氛围写真"
ANALYSIS_CACHE_VERSION = 15
MAX_ANALYSIS_WORKERS = 8
_DETECTORS = threading.local()
_EMOTION_MODEL = Path("event_cull/models/emotion-ferplus-8.onnx")
_MUST_KEEP_RESOURCES = {
    "婚礼": Path("event_cull/data/wedding_must_keep_shots.csv"),
    "通用": Path("event_cull/data/general_must_keep_shots.csv"),
    "人像写真": Path("event_cull/data/general_must_keep_shots.csv"),
    "活动纪实": Path("event_cull/data/general_must_keep_shots.csv"),
    "儿童与家庭": Path("event_cull/data/general_must_keep_shots.csv"),
    "毕业照": Path("event_cull/data/other_must_keep_shots.csv"),
    "证件照": Path("event_cull/data/other_must_keep_shots.csv"),
    "美食": Path("event_cull/data/other_must_keep_shots.csv"),
}
MESSAGES = {
    "zh": {
        "blur": "明显虚焦或抖动",
        "over": "严重过曝",
        "under": "严重欠曝",
        "relative_over": "相邻连拍中突然过曝约 {ev} EV",
        "relative_under": "相邻连拍中突然欠曝约 {ev} EV",
        "flash_miss": "相邻连拍中人物肤色突然变暗约 {ev} EV，疑似闪光灯漏闪",
        "closed": "未检测到睁眼，可能是低头或侧脸，请复核",
        "eye_skip": "人脸过小或角度不明，闭眼不参与评分",
        "eye_intimate": "亲密互动镜头：闭眼不扣分，优先保留情绪",
        "eye_group_tolerated": "多人合影：少量闭眼不扣分",
        "eye_group_closed": "多人合影多数闭眼，建议复核",
        "must_keep": "必选镜头：{stage} · {name}",
        "id_face": "证件照：未检测到合格的单人正面脸",
        "id_eye": "证件照：双眼需清晰睁开并正视镜头",
        "people": "多人正面表情优先：{faces} 张清晰正面脸，{smiles} 张疑似笑脸",
        "people_focus": "人物清晰度低于前景，疑似对焦到前方物体",
        "expression": "多人表情不完整，不作为 3★ 首选",
        "soft_scene": "连续场景整体偏柔，可能是头纱或前景遮挡，保留代表画面复核",
        "scene_coverage": "连续场景保底：不依赖人脸，保留代表画面复核",
        "single_scene": "独立场景代表：无相似连拍，保留供复核",
        "best": "精修候选：本组优选",
        "backup": "人工确认",
        "review": "原片保留",
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
        "eye_skip": "Face is too small or the angle is uncertain; eye state does not affect scoring",
        "eye_intimate": "Intimate interaction: closed eyes are not penalized; preserve the emotion",
        "eye_group_tolerated": "Group portrait: a few closed eyes are not penalized",
        "eye_group_closed": "Most eyes appear closed in this group portrait; review recommended",
        "must_keep": "Must-keep shot: {stage} · {name}",
        "id_face": "ID portrait: one clear frontal face is required",
        "id_eye": "ID portrait: both eyes must be open and looking at the camera",
        "people": "People priority: {faces} clear frontal faces, {smiles} possible smiles",
        "people_focus": "People appear softer than the foreground; focus may be in front",
        "expression": "Group expressions are incomplete; not a first-choice 3-star frame",
        "soft_scene": "The continuous scene is uniformly soft, possibly through a veil or foreground layer; keep a representative frame for review",
        "scene_coverage": "Continuous-scene safeguard: keep a representative frame regardless of face detection",
        "single_scene": "Independent scene representative: no similar burst, retained for review",
        "best": "Retouch candidate: best in group",
        "backup": "Manual confirmation",
        "review": "Retained original",
        "target": "Outside the target selection count",
        "failed": "Could not analyze: {error}",
    },
}


@dataclass(frozen=True)
class MustKeepShot:
    """A photographer-labelled scene moment that must survive selection."""

    identifier: str
    stage: str
    name: str
    priority: str
    special_rule: str


@dataclass(frozen=True)
class EyePolicy:
    """Confidence-aware eye handling, separate from technical image quality."""

    mode: str
    penalty: float = 0.0
    reason_key: str | None = None


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
    scene_type: str = "通用"
    language: str = "zh"
    wedding_content_type: str = "自动（保守）"
    content_model_enabled: bool = False
    content_model_name: str = "qwen3-vl:4b"
    # Keep the historical API opt-in full mode; the desktop explicitly chooses
    # bounded review. Disabling the model still uses the unchanged fast path.
    content_review_mode: str = "full"
    # Supplied only for an applicable personal profile. Share the analysis
    # preview, so preference scoring never needs a second full RAW pass.
    calibration_feature_pipeline: str = ""
    # Experimental until independent retention and speed gates pass. Enabling
    # this never enables the language model or changes a user's active profile.
    human_analysis_enabled: bool = False
    human_face_recovery_enabled: bool = False


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
    face_coverage: float = 0.0
    closest_face_distance: float | None = None
    eye_detection_mode: str = "standard"
    eye_penalty: float = 0.0
    median_luminance: float | None = None
    subject_luminance: float | None = None
    relative_exposure_ev: float | None = None
    calibration_confidence: float | None = None
    scene_protected: bool = False
    coverage_protected: bool = False
    must_keep_id: str | None = None
    # Derived on every ranking pass, never reused from the feature cache.
    featured_blocked: bool = False
    max_auto_rating: int = 3
    rule_version: str = ""
    rule_ids: list[str] = field(default_factory=list)
    content_type: str = "auto"
    content_evidence: dict = field(default_factory=dict)
    visual_features: list[float] = field(default_factory=list)
    visual_feature_pipeline: str = ""
    person_evidence: dict = field(default_factory=dict)

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
    requested: dict[Path, set[str]] = {}
    expanded: set[Path] = set()
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        expanded.add(path)
        requested.setdefault(path.parent, set()).add(path.stem.casefold())
    # One directory scan per source, not one glob per frame (quadratic for
    # 5,000 imports). Literal stems also avoid glob metacharacter mispairing.
    for folder, stems in requested.items():
        with os.scandir(folder) as entries:
            for entry in entries:
                sibling = Path(entry.path)
                if (sibling.stem.casefold() in stems and sibling.suffix.lower() in IMAGE_EXTENSIONS
                        and entry.is_file()):
                    expanded.add(sibling)
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
            "无法生成 RAW 缩略图：当前设备的解码组件暂不支持此相机文件"
        )
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
        if image is None:
            raise RuntimeError("JPEG could not be decoded")
        return _resize_preview(image, maximum_dimension)
    if path.suffix.lower() in HEIC_EXTENSIONS:
        if register_heif_opener is None:
            raise RuntimeError("HEIC support requires the bundled pillow-heif component")
        # Pillow does not discover third-party codecs by itself. Registering
        # the bundled decoder here makes HEIC work in analysis, grid previews,
        # and close review without converting or altering the original file.
        register_heif_opener()
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


_CONTEXTUAL_EYE_SCENES = {
    "通用", "婚礼", "人像写真", ATMOSPHERE_PORTRAIT_SCENE,
    "活动纪实", "儿童与家庭", "毕业照"
}


def _has_scene_rules(options: AnalysisOptions) -> bool:
    return options.scene_type in _MUST_KEEP_RESOURCES


def _uses_contextual_eye_policy(options: AnalysisOptions) -> bool:
    return options.scene_type in _CONTEXTUAL_EYE_SCENES


def _scene_blur_floor(options: AnalysisOptions) -> float:
    """Respect the subject-specific sharpness tolerance of the selected scene."""
    if options.scene_type == "美食":
        # Macro food shots and steam naturally soften non-subject areas.
        return options.blur_floor * 0.72
    if options.scene_type == "证件照":
        return options.blur_floor * 1.18
    return options.blur_floor


def _resource_path(resource: Path) -> Path:
    """Locate a packaged resource both from source and a frozen app bundle."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / resource


@lru_cache(maxsize=None)
def _must_keep_shots(scene_type: str) -> tuple[MustKeepShot, ...]:
    """Load only the whitelist for the active scene from app-bundled data."""
    resource = _MUST_KEEP_RESOURCES.get(scene_type)
    if resource is None:
        return ()
    try:
        with _resource_path(resource).open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            rows = csv.reader(stream)
            next(rows, None)
            # The supplied keywords deliberately include ordinary commas.  The
            # stable fields are the first three and final two columns; retaining
            # that shape keeps the whitelist usable without silently assigning
            # a keyword fragment as a priority.
            return tuple(
                MustKeepShot(
                    identifier=row[0].strip(),
                    stage=row[1].strip(),
                    name=row[2].strip(),
                    priority=row[-2].strip().upper(),
                    special_rule=row[-1].strip(),
                )
                for row in rows
                if len(row) >= 5
                and row[0].strip()
                and row[2].strip()
                and (resource.name == "wedding_must_keep_shots.csv" or row[1].strip() == scene_type)
            )
    except (OSError, UnicodeDecodeError, csv.Error, KeyError):
        # The core culler must remain usable when a custom build accidentally
        # omits this optional policy resource.
        return ()


def _must_keep_match(path: Path, options: AnalysisOptions) -> MustKeepShot | None:
    """Match deliberate folder/file labels, never guess semantic content from a name.

    Camera counters do not reveal whether a frame is a dish, a kiss, or a
    diploma handover.
    A rule therefore activates only when the imported path already carries the
    photographer's stage/shot label (for example ``C9 仪式之吻``). Unlabelled
    folders still receive the active scene's confidence-aware base policy.
    """
    label = path.as_posix().casefold()
    for rule in _must_keep_shots(options.scene_type):
        if rule.name.casefold() in label:
            return rule
        identifier = re.escape(rule.identifier.casefold())
        if re.search(rf"(?<![a-z0-9]){identifier}(?![a-z0-9])", label):
            return rule
    return None


def _wedding_must_keep_shots() -> tuple[MustKeepShot, ...]:
    """Compatibility seam for historical integrations and focused tests."""
    return _must_keep_shots("婚礼")


def _eye_policy(
    options: AnalysisOptions,
    face_count: int,
    face_coverage: float,
    open_eye_count: int,
    closest_face_distance: float | None,
) -> EyePolicy:
    """Apply eye detection only where the measurement is trustworthy.

    Tiny/distant faces, profile-like evidence, intimate interactions, and group
    shots are not judged as if they were a single frontal portrait. It never
    converts an eye finding into a technical hard reject except in the explicit
    certificate-photo mode, where open eyes are a stated validity requirement.
    """
    if not options.detect_closed_eyes:
        return EyePolicy("disabled")

    if options.scene_type == "婚礼" and CONTENT_TYPES.get(options.wedding_content_type) in {
        "documentary", "portrait", "dynamic", "details",
    }:
        return EyePolicy("wedding_content_eye_neutral")

    if options.scene_type == "美食":
        return EyePolicy("food_no_eye_check")

    if options.scene_type == "证件照":
        if face_count != 1 or face_coverage < 0.07 or face_coverage > 0.65:
            return EyePolicy("id_face_invalid", penalty=60.0, reason_key="id_face")
        if open_eye_count != 1:
            return EyePolicy("id_eye_invalid", penalty=60.0, reason_key="id_eye")
        return EyePolicy("id_valid")

    if options.scene_type == "儿童与家庭":
        # Children naturally blink, laugh with closed eyes, and look away;
        # expression is useful, but eye state must not suppress the moment.
        return EyePolicy("family_eye_tolerant")

    if options.scene_type == ATMOSPHERE_PORTRAIT_SCENE:
        # Side profiles, looking away, lowered heads, and back views are
        # legitimate finished frames in an atmospheric portrait sequence.
        return EyePolicy("atmosphere_eye_neutral")

    if not _uses_contextual_eye_policy(options):
        if face_count == 1 and open_eye_count == 0:
            return EyePolicy("standard", penalty=12.0, reason_key="closed")
        if face_count >= 2 and 0 < open_eye_count < face_count:
            return EyePolicy("standard", reason_key="expression")
        return EyePolicy("standard")

    mode_prefix = "scene"
    if face_count == 0:
        return EyePolicy(f"{mode_prefix}_no_face")
    if face_coverage < 0.05:
        return EyePolicy(f"{mode_prefix}_small_face", reason_key="eye_skip")
    if closest_face_distance is not None and closest_face_distance < 1.5 and (
        options.scene_type != "婚礼" or face_count <= 2
    ):
        return EyePolicy(f"{mode_prefix}_intimate", reason_key="eye_intimate")
    if face_count >= 3:
        if open_eye_count == 0:
            return EyePolicy(f"{mode_prefix}_group_all_closed", penalty=8.0, reason_key="eye_group_closed")
        if open_eye_count < face_count:
            if options.scene_type == "婚礼":
                return EyePolicy(f"{mode_prefix}_group", reason_key="expression")
            return EyePolicy(f"{mode_prefix}_group", reason_key="eye_group_tolerated")
        return EyePolicy(f"{mode_prefix}_group")
    if open_eye_count == 0:
        # One or two large, frontal faces with no eye evidence are the one
        # contextual case where the detector is strong enough to matter.
        return EyePolicy(f"{mode_prefix}_portrait", penalty=24.0, reason_key="closed")
    return EyePolicy(f"{mode_prefix}_portrait")


def _face_metrics(
    gray: np.ndarray,
    options: AnalysisOptions,
    global_sharpness: float | None = None,
) -> tuple[
    list[str], float | None, int, int, int, float | None, float, float | None, EyePolicy
]:
    """Return face evidence plus a confidence-aware eye policy."""
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
    image_area = max(1, gray.shape[0] * gray.shape[1])
    face_coverage = sum(width * height for _x, _y, width, height in distinct) / image_area
    closest_face_distance: float | None = None
    clear_rectangles = [rect for rect, _sharpness in clear_faces]
    for index, (left_x, left_y, left_width, left_height) in enumerate(clear_rectangles):
        left_center = (left_x + left_width / 2, left_y + left_height / 2)
        for right_x, right_y, right_width, right_height in clear_rectangles[index + 1 :]:
            right_center = (right_x + right_width / 2, right_y + right_height / 2)
            centre_distance = math.hypot(
                left_center[0] - right_center[0], left_center[1] - right_center[1]
            )
            face_width = max(1.0, (left_width + right_width) / 2)
            ratio = centre_distance / face_width
            closest_face_distance = (
                ratio if closest_face_distance is None else min(closest_face_distance, ratio)
            )
    smiling_face_count = 0
    open_eye_face_count = 0
    smile_candidates = sorted(
        clear_faces,
        key=lambda item: item[0][2] * item[0][3],
        reverse=True,
    )
    # Wedding group comparisons need a matching numerator and denominator:
    # examining only eight faces makes a ten-person group look partly closed.
    if options.scene_type != "婚礼":
        smile_candidates = smile_candidates[:8]
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
    if _uses_contextual_eye_policy(options) and distinct and len(clear_faces) != len(distinct):
        # A partial/profile-like or soft detection cannot reliably distinguish
        # a blink from an obscured eye. Let quality and manual review decide.
        eye_policy = EyePolicy("scene_uncertain", reason_key="eye_skip")
    else:
        eye_policy = _eye_policy(
            options,
            len(clear_faces),
            face_coverage,
            open_eye_face_count,
            closest_face_distance,
        )
    reasons: list[str] = []
    # No-face scenes are expected for rings, flowers, venues, and wide shots;
    # avoid filling every card with an unnecessary explanation.  A detected
    # but too-small face remains visible so the photographer can audit it.
    if eye_policy.reason_key and not (
        eye_policy.reason_key == "eye_skip" and len(clear_faces) == 0
    ):
        reasons.append(_message(options, eye_policy.reason_key))
    return (
        reasons,
        subject_sharpness,
        len(clear_faces),
        smiling_face_count,
        open_eye_face_count,
        subject_luminance,
        face_coverage,
        closest_face_distance,
        eye_policy,
        distinct,
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
    from event_cull.platform_paths import cache_dir

    return cache_dir() / "analysis-v1.json"


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


def _human_model_signature(*, recover_faces=False):
    from src.core.image_features.human_analysis import model_signature
    return model_signature(recover_faces=recover_faces)


def _cache_signature(source: Path, options: AnalysisOptions) -> dict:
    stat = source.stat()
    signature = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "blur_floor": options.blur_floor,
        "clip_fraction": options.severe_clip_fraction,
        "detect_closed_eyes": options.detect_closed_eyes,
        "detect_blur": options.detect_blur,
        "detect_exposure": options.detect_exposure,
        "scene_type": options.scene_type,
        "language": options.language,
        "wedding_rule_version": RULE_VERSION if options.scene_type == "婚礼" else "",
        "wedding_content_type": options.wedding_content_type,
        "content_model_enabled": options.content_model_enabled,
        "content_model_name": options.content_model_name,
        "calibration_feature_pipeline": options.calibration_feature_pipeline,
        "human_analysis_enabled": options.human_analysis_enabled,
        "human_model_signature": (_human_model_signature(recover_faces=True) if options.human_face_recovery_enabled
                                  else _human_model_signature()) if options.human_analysis_enabled else "",
    }
    if options.human_face_recovery_enabled:
        from src.core.image_features.face_analysis import FACE_REGION_PIPELINE
        signature["human_face_region_pipeline"] = FACE_REGION_PIPELINE
    return signature


def _result_from_cache(photo: Path, pairs: list[Path], record: dict, signature: dict) -> PhotoResult | None:
    if record.get("signature") != signature:
        return None
    try:
        visual = record.get("visual_features") or []
        visual_pipeline = str(record.get("visual_feature_pipeline") or "")
        wanted_pipeline = signature.get("calibration_feature_pipeline", "")
        # The feature builder owns descriptor dimensions, including opt-in
        # provider-isolated versions. Same-length-but-wrong-version is rejected
        # separately by the signature and wanted-pipeline checks below.
        from event_cull.calibration import FEATURE_DIMENSIONS as dimensions
        if (not isinstance(visual, list) or any(not isinstance(v, (int,float)) for v in visual)
                or (visual and not np.isfinite(visual).all())
                or (wanted_pipeline and (visual_pipeline != wanted_pipeline or not visual))
                or (visual and visual_pipeline in dimensions and len(visual) != dimensions[visual_pipeline])):
            return None
        evidence = record.get("content_evidence", {})
        if not isinstance(evidence, dict):
            return None
        person = record.get("person_evidence", {})
        if signature.get("human_analysis_enabled"):
            from src.core.image_features.human_analysis import validate_observation as validate_person
            validate_person(person,require_details=True)
            if (signature.get("human_face_region_pipeline")
                    and person.get("region_pipeline") != signature["human_face_region_pipeline"]):
                return None
        if evidence.get("status") == "ok":
            validate_observation(evidence, {f"W{i:03}" for i in range(1, 241)})
            if (evidence.get("content_type") not in CONTENT_TYPES.values()
                    or type(evidence.get("section")) is not int or not 5 <= evidence["section"] <= 22):
                return None
        if not isinstance(record["reasons"], list) or any(not isinstance(reason, str) for reason in record["reasons"]):
            return None
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
            face_coverage=float(record.get("face_coverage", 0.0)),
            closest_face_distance=(
                float(record["closest_face_distance"])
                if record.get("closest_face_distance") is not None
                else None
            ),
            eye_detection_mode=str(record.get("eye_detection_mode") or "standard"),
            eye_penalty=float(record.get("eye_penalty", 0.0)),
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
            must_keep_id=(
                str(record["must_keep_id"])
                if record.get("must_keep_id")
                else None
            ),
            content_evidence=dict(evidence),
            visual_features=visual,
            visual_feature_pipeline=visual_pipeline,
            person_evidence=dict(person),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _cache_record(result: PhotoResult, signature: dict) -> dict:
    return {
        "signature": signature,
        "score": result.score,
        # Ranking adds reasons later. Never let it contaminate base evidence.
        "reasons": list(result.reasons),
        "sharpness": result.sharpness,
        "highlight_clip": result.highlight_clip,
        "shadow_clip": result.shadow_clip,
        "perceptual_hash": result.perceptual_hash,
        "subject_sharpness": result.subject_sharpness,
        "frontal_face_count": result.frontal_face_count,
        "smiling_face_count": result.smiling_face_count,
        "open_eye_face_count": result.open_eye_face_count,
        "face_coverage": result.face_coverage,
        "closest_face_distance": result.closest_face_distance,
        "eye_detection_mode": result.eye_detection_mode,
        "eye_penalty": result.eye_penalty,
        "median_luminance": result.median_luminance,
        "subject_luminance": result.subject_luminance,
        "must_keep_id": result.must_keep_id,
        "content_evidence": dict(result.content_evidence),
        "visual_features": list(result.visual_features),
        "visual_feature_pipeline": result.visual_feature_pipeline,
        "person_evidence": dict(result.person_evidence),
    }


def _apply_must_keep_rule(item: PhotoResult, options: AnalysisOptions) -> None:
    """Attach a whitelist rule when an imported path carries its shot label."""
    if not _has_scene_rules(options):
        return
    rule = _must_keep_match(item.path, options)
    if rule is None:
        return
    item.must_keep_id = rule.identifier
    if _must_keep_rule_exempts_eye_penalty(rule):
        item.eye_detection_mode = "scene_whitelist_eye_exempt"
        item.eye_penalty = 0.0
        eye_reasons = {
            _message(options, "closed"),
            _message(options, "expression"),
            _message(options, "eye_skip"),
            _message(options, "eye_intimate"),
            _message(options, "eye_group_tolerated"),
            _message(options, "eye_group_closed"),
        }
        item.reasons = [reason for reason in item.reasons if reason not in eye_reasons]
    reason = _message(options, "must_keep", stage=rule.stage, name=rule.name)
    if reason not in item.reasons:
        item.reasons.append(reason)


def _must_keep_priority(item: PhotoResult, options: AnalysisOptions) -> int:
    rule = next(
        (candidate for candidate in _must_keep_shots(options.scene_type) if candidate.identifier == item.must_keep_id),
        None,
    )
    return {"S": 3, "A": 2, "B": 1}.get(rule.priority if rule else "", 0)


def _must_keep_rule_exempts_eye_penalty(rule: MustKeepShot) -> bool:
    """Return whether the supplied rule explicitly makes eye state unreliable."""
    return any(
        phrase in rule.special_rule
        for phrase in ("闭眼豁免", "不扣分", "闭眼正常", "跳过闭眼", "不跑闭眼", "容忍")
    )


def _analyze(path: Path, pairs: list[Path], options: AnalysisOptions,
             content_model: LocalWeddingContentModel | None = None, *, human_model=None) -> PhotoResult:
    image = _load_preview(_analysis_source(path, pairs))
    if max(image.shape[:2]) > 1600:
        scale = 1600 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    highlights = float(np.mean(gray >= 250))
    shadows = float(np.mean(gray <= 5))
    metrics = _face_metrics(gray, options, sharpness)
    (
        face_reasons,
        subject_sharpness,
        frontal_faces,
        smiling_faces,
        open_eye_faces,
        subject_luminance,
    ) = metrics[:6]
    # Focused performance tests and third-party integrations historically
    # stubbed the six original values.  Keep that seam working while the
    # built-in detector supplies richer confidence evidence.
    if len(metrics) >= 9:
        face_coverage, closest_face_distance, eye_policy = metrics[6:9]
    else:
        face_coverage = 1.0 if frontal_faces else 0.0
        closest_face_distance = None
        eye_policy = _eye_policy(
            options,
            frontal_faces,
            face_coverage,
            open_eye_faces,
            closest_face_distance,
        )
    median_luminance = float(np.median(gray))
    effective_sharpness = _effective_sharpness(sharpness, subject_sharpness)
    reasons: list[str] = []
    if options.detect_blur and effective_sharpness < _scene_blur_floor(options):
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
    neutral_type = (options.scene_type == "婚礼" and CONTENT_TYPES.get(options.wedding_content_type) in {
        "documentary", "portrait", "dynamic", "details"})
    if options.prefer_people and options.scene_type != ATMOSPHERE_PORTRAIT_SCENE and not neutral_type:
        quality += _people_priority_bonus(frontal_faces, smiling_faces)
    result = PhotoResult(
        path,
        pairs,
        score=quality,
        reasons=reasons,
        sharpness=sharpness,
        highlight_clip=highlights,
        shadow_clip=shadows,
        perceptual_hash=_dhash(gray),
        subject_sharpness=subject_sharpness,
        frontal_face_count=frontal_faces,
        smiling_face_count=smiling_faces,
        open_eye_face_count=open_eye_faces,
        face_coverage=face_coverage,
        closest_face_distance=closest_face_distance,
        eye_detection_mode=eye_policy.mode,
        eye_penalty=eye_policy.penalty,
        median_luminance=median_luminance,
        subject_luminance=subject_luminance,
    )
    if human_model is not None:
        try:
            result.person_evidence = human_model.inspect(
                cv2.cvtColor(image,cv2.COLOR_BGR2RGB), face_regions=metrics[9] if len(metrics)>9 else ())
        except AnalysisCancelled:
            raise
        except Exception as exc:
            # A model failure is not a photo defect, and must be retried rather
            # than permanently cached as absent people or closed eyes.
            result.person_evidence = {"status":"error","error":str(exc)[:160]}
    if options.calibration_feature_pipeline:
        # Calibration owns the descriptor/model, analysis owns decoding. A
        # broken optional encoder must not make a readable photo a failed row.
        from event_cull.calibration import CalibrationError, _feature_matrix
        try:
            result.visual_features = _feature_matrix([image], options.calibration_feature_pipeline)[0].tolist()
            result.visual_feature_pipeline = options.calibration_feature_pipeline
        except CalibrationError:
            pass
    if _subject_motion_fault(result, options):
        blur_reason = _message(options, "blur")
        if blur_reason not in result.reasons:
            result.reasons.append(blur_reason)
    if severe_underexposure and _lit_wedding_subject(result, options):
        result.reasons = [reason for reason in result.reasons if reason != _message(options, "under")]
        result.reasons.append("暗背景但人物亮度与清晰度通过：不按整体阴影比例淘汰"
                              if options.language == "zh" else "Dark background with a measurable usable face")
    _apply_must_keep_rule(result, options)
    if content_model is not None:
        if _has_hard_fault(result, options):
            result.content_evidence = {"status": "skipped", "error": "技术检查未通过，未运行内容模型"}
        else:
            # Share the already decoded preview; RAW/HEIC is never decoded twice.
            result.content_evidence = content_model.inspect(
                image, CONTENT_TYPES.get(options.wedding_content_type, "auto"))
            if not neutral_type and eye_neutral(result, options):
                result.score -= _people_priority_bonus(frontal_faces, smiling_faces) if options.prefer_people else 0
    return result


def _analyze_cached(
    photo: Path,
    pairs: list[Path],
    options: AnalysisOptions,
    cache: dict[str, dict],
    content_model: LocalWeddingContentModel | None = None,
    *, human_model=None,
) -> tuple[PhotoResult, str, dict | None]:
    source = _analysis_source(photo, pairs)
    key = str(source.resolve())
    signature = _cache_signature(source, options)
    if content_model is not None:
        signature["content_fingerprint"] = content_model.fingerprint
    cached = _result_from_cache(photo, pairs, cache.get(key, {}), signature)
    if cached is not None:
        return cached, key, None
    if human_model is not None:
        result = _analyze(photo,pairs,options,content_model,human_model=human_model)
    else:
        result = (_analyze(photo, pairs, options) if content_model is None
                  else _analyze(photo, pairs, options, content_model))
    # Transient failures must be retried after the service/model becomes ready.
    cacheable = (result.content_evidence.get("status") not in {"error", "unavailable"}
                 and result.person_evidence.get("status") != "error")
    return result, key, _cache_record(result, signature) if cacheable else None


def _same_group(previous: PhotoResult, current: PhotoResult, options: AnalysisOptions) -> bool:
    if not options.detect_duplicates:
        return False
    if previous.path.parent.resolve() != current.path.parent.resolve():
        return False
    if options.scene_type == "婚礼":
        # Different observed subjects/categories are a safe split, never proof
        # that the same people have been recognised across different groups.
        a, b = previous.content_evidence, current.content_evidence
        if a.get("status") == b.get("status") == "ok" and (
            a["content_type"] != b["content_type"] or a["section"] != b["section"]
        ):
            return False
    try:
        gap = current.path.stat().st_mtime - previous.path.stat().st_mtime
    except OSError:
        # An external move during analysis must not crash all completed rows.
        return False
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


def _background_only_motion_evidence(item: PhotoResult, options: AnalysisOptions) -> bool:
    """An uncertain small subject passes absolute focus but loses to background.

    Shared by hard rejection and shortlist eligibility, so rescuing an original
    cannot accidentally promote the same ambiguous evidence to three stars.
    """
    return bool(
        options.scene_type == "婚礼"
        and options.detect_blur
        and item.eye_detection_mode in {"scene_small_face", "scene_uncertain"}
        and item.face_coverage < .05
        and item.frontal_face_count >= 2
        and item.open_eye_face_count == 0
        and item.sharpness is not None
        and item.subject_sharpness is not None
        and item.subject_sharpness >= _scene_blur_floor(options)
        and item.subject_sharpness < options.blur_floor * 8
        and item.subject_sharpness < item.sharpness * 0.45
    )


def _subject_motion_fault(item: PhotoResult, options: AnalysisOptions) -> bool:
    """Reject a soft multi-person subject despite sharp background texture.

    Whole-frame Laplacian variance can look excellent when foliage, lettering,
    or stonework is sharp behind motion-blurred people. Require multiple face
    detections and no reliable eye evidence before treating this as a hard
    fault, keeping single distant faces and environmental images conservative.
    """
    # Small/uncertain Haar crops plus a sharper leafy background do not prove
    # motion blur when the measured subject already passes the absolute floor.
    # The same shared predicate blocks shortlist promotion below.
    if _background_only_motion_evidence(item, options):
        return False
    return bool(
        options.detect_blur
        and item.frontal_face_count >= 2
        and item.open_eye_face_count == 0
        and item.sharpness is not None
        and item.subject_sharpness is not None
        and item.subject_sharpness < options.blur_floor * 8
        and item.subject_sharpness < item.sharpness * 0.45
    )


def _group_expression_issue(item: PhotoResult, options: AnalysisOptions | None = None) -> bool:
    if options is not None and eye_neutral(item, options):
        return False
    if options is not None and options.scene_type == ATMOSPHERE_PORTRAIT_SCENE:
        return False
    # Treat a confirmed mixed result as risky.  When the eye cascade finds no
    # eyes at all, keep it ambiguous so low-head and atmospheric frames survive.
    # A natural smile is a positive moment even when someone else looks away.
    if item.eye_detection_mode.startswith("scene_"):
        return False
    return (
        item.frontal_face_count >= 2
        and item.smiling_face_count == 0
        and 0 < item.open_eye_face_count < item.frontal_face_count
    )


def _focus_fault(item: PhotoResult, options: AnalysisOptions) -> bool:
    return options.detect_blur and item.sharpness is not None and _effective_sharpness(
        item.sharpness, item.subject_sharpness
    ) < _scene_blur_floor(options)


def _id_photo_fault(item: PhotoResult, options: AnalysisOptions) -> bool:
    """Reject the measurable hard requirements of a certificate-photo frame."""
    if options.scene_type != "证件照":
        return False
    return (
        item.frontal_face_count != 1
        or item.face_coverage < 0.07
        or item.face_coverage > 0.65
        or item.open_eye_face_count != 1
    )


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
    return (
        focus_fault
        or _subject_motion_fault(item, options)
        or _has_exposure_fault(item, options)
        or _id_photo_fault(item, options)
    )


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
        and not _lit_wedding_subject(item, options)
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


def _lit_wedding_subject(item: PhotoResult, options: AnalysisOptions) -> bool:
    """Dark background is not underexposure when a measured face is usable.

    No face/unknown subject, a dark subject, severe clipping or relative flash
    failure still follows the original hard gate. This does not identify people
    on stage or silently exempt all documentary images.
    """
    return (options.scene_type == "婚礼" and item.frontal_face_count > 0
            and item.subject_luminance is not None and 55 <= item.subject_luminance <= 215
            and item.subject_sharpness is not None and item.subject_sharpness >= _scene_blur_floor(options))


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
    """Rank on image quality; wedding eye penalties require reliable evidence."""
    focus_penalty = 35.0 if _people_focus_issue(item, options) else 0.0
    expression_penalty = 18.0 if _group_expression_issue(item, options) else 0.0
    eye_penalty = item.eye_penalty if _has_scene_rules(options) else 0.0
    content_penalty = (3 - item.max_auto_rating) * 30.0 if options.scene_type == "婚礼" else 0.0
    return item.score - focus_penalty - expression_penalty - eye_penalty - content_penalty


def _review_priority(item: PhotoResult, options: AnalysisOptions) -> float:
    """Prefer an open-eyed alternative without hard-rejecting ambiguous poses."""
    if item.calibration_confidence is not None:
        # This confidence describes ORIGINAL delivery, never retouch quality.
        # Keep technical/shortlist gates separate from a learned retention rank.
        return item.calibration_confidence * 100 + _ranking_score(item, options) * .01
    closed_eye_penalty = item.eye_penalty if _has_scene_rules(options) and item.eye_penalty else (
        12.0
        if item.frontal_face_count > 0 and _message(options, "closed") in item.reasons
        else 0.0
    )
    if eye_neutral(item, options):
        closed_eye_penalty = 0.0
    smile_priority = (
        min(item.smiling_face_count, 2) * 24.0
        if options.prefer_people and options.scene_type != ATMOSPHERE_PORTRAIT_SCENE
        else 0.0
    )
    if eye_neutral(item, options):
        smile_priority = 0.0
    content_penalty = (3 - item.max_auto_rating) * 30.0 if options.scene_type == "婚礼" else 0.0
    return item.score + smile_priority - closed_eye_penalty - content_penalty


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

    if any(item.calibration_confidence is not None for item in candidates):
        # Hash fragments are not true story beats. Do not force one slot per
        # fragment ahead of learned original-delivery preferences. Explicit
        # labelled must-keep moments are still protected by the caller.
        for item in sorted(candidates, key=priority, reverse=True):
            add(item)
        if options.fill_with_similar and len(selected) < limit:
            for item in sorted(candidates, key=priority, reverse=True):
                if len(selected) >= limit:
                    break
                if item.path not in selected_paths:
                    selected.append(item)
                    selected_paths.add(item.path)
        return selected

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
    if options.scene_type != ATMOSPHERE_PORTRAIT_SCENE:
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
    if options.scene_type == ATMOSPHERE_PORTRAIT_SCENE or eye_neutral(item, options):
        return _ranking_score(item, options)
    return (
        _ranking_score(item, options)
        + item.smiling_face_count * 24.0
        + item.open_eye_face_count * 4.0
    )


def _mark_wedding_group_shortlist(group: list[PhotoResult], options: AnalysisOptions) -> None:
    """Conservative shortlist gate, not a claim of gaze or confirmed blinking.

    Only compare technically sound frames within the existing burst. Unknown
    eye measurements stay available for review, never become hard rejects.
    Couples, documentary details and atmospheric portraits retain their policy.
    """
    if options.scene_type != "婚礼" or not options.prefer_people:
        return
    documentary_ids = {rule.identifier for rule in _must_keep_shots("婚礼")
                       if any(word in rule.special_rule for word in
                              ("情绪", "动态", "背影", "侧影", "不跑个体", "看稿", "化妆"))}
    portraits = [item for item in group if item.frontal_face_count >= 3
                 and item.must_keep_id not in documentary_ids
                 and not eye_neutral(item, options)
                 and not _has_hard_fault(item, options)
                 and not _people_focus_issue(item, options)]
    if not portraits:
        return
    complete = []
    for item in portraits:
        if options.detect_closed_eyes and (
            item.open_eye_face_count < item.frontal_face_count
            or item.eye_detection_mode in {"scene_uncertain", "scene_small_face"}
        ):
            item.featured_blocked = True
            item.reasons.append("合影睁眼证据不完整：保留复核，不自动精选" if options.language == "zh"
                                else "Group eye evidence incomplete: review, not automatic shortlist")
        else:
            complete.append(item)
    if not complete:
        return
    # Head count first prevents a turned-away face from improving the smile
    # ratio. This is burst-relative evidence, not facial identity tracking.
    def expression_key(item: PhotoResult) -> tuple[int, float]:
        return (item.frontal_face_count,
                item.smiling_face_count / max(1, item.frontal_face_count))
    best = max(expression_key(item) for item in complete)
    for item in group:
        if (item.frontal_face_count >= 2 and item.must_keep_id not in documentary_ids
                and not eye_neutral(item, options)
                and expression_key(item) < best):
            item.featured_blocked = True
            item.max_auto_rating = min(item.max_auto_rating, 2)
            item.reasons.append("同组合影有正面人数或笑脸比例更完整的照片，当前仅保留原片" if options.language == "zh"
                                else "A group alternative has fuller frontal-face or smile evidence")


def _assign_tiered_ratings(
    groups: list[list[PhotoResult]], options: AnalysisOptions
) -> None:
    """Choose retained originals, then promote a subset to retouch candidates.

    Automatic output is 0/1/3 only. Two stars are reserved for manual review.
    Quality checks and duplicate selection happen before either star is given.
    """
    eligible: list[PhotoResult] = []
    for group_id, group in enumerate(groups, 1):
        _mark_relative_exposure_outliers(group, options)
        _protect_coherent_soft_scene(group, options)
        _protect_scene_minimum(group, options)
        for item in group:
            item.group_id = group_id
            item.suggested_rating = 0
            item.featured_blocked = False
            _apply_must_keep_rule(item, options)
            technical_fault = _has_hard_fault(item, options)
            apply_evidence_gates(item, options, technical_fault=technical_fault)
            if _background_only_motion_evidence(item, options):
                # Block only the retouch shortlist: do not add a ranking penalty
                # that would undo the original-photo rescue at the delivery cap.
                item.featured_blocked = True
                add_evidence(item, "W008", "远景小脸对焦与表情证据不可靠，仅保留原片，不自动精选")
            if options.scene_type == "婚礼" and _people_focus_issue(item, options):
                item.max_auto_rating = min(item.max_auto_rating, 1)
                item.featured_blocked = True
                add_evidence(item, "W011", "主体疑似失焦，不用背景清晰度升级精选")
            if not _has_hard_fault(item, options):
                eligible.append(item)
        _mark_wedding_group_shortlist(group, options)
        for item in group:
            # A shared eligibility gate also constrains preference calibration.
            item.featured_blocked |= (_has_hard_fault(item, options)
                                     or _people_focus_issue(item, options)
                                     or _group_expression_issue(item, options))

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
        # A scene detector can split one real setting into many small groups
        # when the pose or camera angle changes. Never turn those groups into
        # a hidden minimum that overrides the photographer's requested cap.
        # Explicit must-keep moments are handled separately below.
        target = max(target_floor, options.target_keep_max_count)
    one_star_or_more = _density_selection(
        ranked,
        min(target, len(ranked)),
        groups,
        options,
        priority=lambda item: _review_priority(item, options),
    )
    # Correct only the representative inside an already selected burst; don't
    # make group portraits globally displace ceremony, detail or emotion shots.
    if options.scene_type == "婚礼" and options.prefer_people:
        indices_by_group: dict[int, list[int]] = {}
        for index, item in enumerate(one_star_or_more):
            indices_by_group.setdefault(item.group_id, []).append(index)
        for group in groups:
            alternatives = [item for item in group if item.frontal_face_count >= 3
                            and not item.featured_blocked]
            if not alternatives:
                continue
            best = max(alternatives, key=lambda item: _three_star_score(item, options))
            selected_indices = indices_by_group.get(best.group_id, [])
            if selected_indices and not any(one_star_or_more[index] is best for index in selected_indices):
                index = min(selected_indices, key=lambda index: _review_priority(one_star_or_more[index], options))
                if one_star_or_more[index].featured_blocked:
                    one_star_or_more[index] = best

    # Per-burst and duplicate caps make the review wall manageable, but they
    # must not silently reduce an explicit minimum target.  If there are
    # enough eligible photos, relax those caps only until that floor is met.
    required_floor = min(target_floor, len(ranked))
    if len(one_star_or_more) < required_floor and (
        options.scene_type != "婚礼" or options.fill_with_similar
    ):
        selected_paths = {item.path for item in one_star_or_more}
        for item in ranked:
            if len(one_star_or_more) >= required_floor:
                break
            if item.path not in selected_paths:
                one_star_or_more.append(item)
                selected_paths.add(item.path)

    if _has_scene_rules(options):
        # An explicitly labelled must-keep moment may lift the real total over
        # the target. This is intentional: target counts guide delivery size,
        # while the whitelist guards irreplaceable scene beats. Technical
        # hard faults were removed from ``eligible`` above and remain excluded.
        selected_paths = {item.path for item in one_star_or_more}
        retained_moments: set[tuple[str, int]] = {
            (item.must_keep_id, item.group_id) for item in one_star_or_more if item.must_keep_id
        }
        for item in sorted(
            (item for item in eligible if item.must_keep_id),
            key=lambda candidate: (
                _must_keep_priority(candidate, options),
                _review_priority(candidate, options),
            ),
            reverse=True,
        ):
            if options.scene_type == "婚礼":
                key = (item.must_keep_id, item.group_id)
                if key in retained_moments:
                    continue
                retained_moments.add(key)
            if item.path not in selected_paths:
                one_star_or_more.append(item)
                selected_paths.add(item.path)

    # At most one candidate per similar group and one fifth of the actual
    # retained pool (rounded up). A larger requested target cannot inflate
    # this shortlist, and missing quality evidence never forces a promotion.
    three_target = (len(one_star_or_more) + 4) // 5
    best_per_group: dict[int, PhotoResult] = {}
    three_candidates = [
        item
        for item in one_star_or_more
        if not _people_focus_issue(item, options)
        and not _group_expression_issue(item, options)
        and not item.featured_blocked
        and item.max_auto_rating >= 3
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

    one_paths = {item.path for item in one_star_or_more}
    three_paths = {item.path for item in three_star}
    singleton_group_ids = {
        group_id for group_id, group in enumerate(groups, start=1) if len(group) == 1
    }
    single_scene_reason = _message(options, "single_scene")
    for item in eligible:
        item.suggested_rating = automatic_rating(
            retained=item.path in one_paths,
            featured=item.path in three_paths,
            ceiling=item.max_auto_rating,
        )
        if item.path not in one_paths:
            item.reasons.append(_message(options, "target"))
        if item.suggested_rating:
            item.reasons.append(_message(options, {1: "review", 2: "backup", 3: "best"}[item.suggested_rating]))
            if options.scene_type == "婚礼":
                add_evidence(item, "W046", "同组去重后保留代表；目标张数不强制补重复照片")
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


def _result_groups(results: list[PhotoResult], options: AnalysisOptions) -> list[list[PhotoResult]]:
    groups: list[list[PhotoResult]] = []
    for result in results:
        same = bool(groups) and _same_group(groups[-1][-1], result, options)
        if same and options.scene_type == "婚礼":
            same = _same_group(groups[-1][0], result, options)
        if not same:
            groups.append([])
        groups[-1].append(result)
    return groups


def _review_budget_seconds(total: int) -> float:
    """User's speed target, not a guarantee for slow storage/RAW decoders."""
    return max(30.0, min(600.0, total * .12))


def _review_candidates(results: list[PhotoResult], options: AnalysisOptions) -> list[PhotoResult]:
    # Provisional ranking must not contaminate cached base evidence or add
    # selection reasons twice. Copies contain metrics only, never image pixels.
    groups = _result_groups(copy.deepcopy(results), replace(options, content_model_enabled=False))
    _assign_tiered_ratings(groups, replace(options, content_model_enabled=False))
    originals = {item.path: item for item in results}
    leaders, alternates = [], []
    for group in groups:
        eligible = sorted((item for item in group if not _has_hard_fault(item, options)),
                          key=lambda item: (item.suggested_rating == 3, item.suggested_rating > 0,
                                            _three_star_score(item, options)), reverse=True)
        if not eligible or not any(item.suggested_rating for item in eligible):
            continue
        leaders.append(originals[eligible[0].path])
        if len(eligible) > 1:
            alternates.append(originals[eligible[1].path])

    def spread(items):
        # Spend a small budget across the whole timeline rather than just the
        # first preparation frames. All leaders precede second choices.
        intervals = deque([(0, len(items))])
        while intervals:
            left, right = intervals.popleft()
            if left >= right:
                continue
            middle = (left + right) // 2
            yield items[middle]
            intervals.extend(((left, middle), (middle + 1, right)))

    return [*spread(leaders), *spread(alternates)]


def _limited_content_review(results, options, cache, started, checkpoint,
                            cancel_event, pause_event, phase_callback):
    """Reuse the decoder, metrics and model; no second full-folder analysis.

    Only selected review candidates need a fresh full-detail preview. Holding
    5,000 decoded images just to avoid these few reads would exhaust memory.
    Successful content observations have their own cache entry; they never
    overwrite the fast feature cache or become false evidence for other frames.
    """
    candidates = _review_candidates(results, options)
    deadline = started + _review_budget_seconds(len(results)) - 5.0
    for item in results:
        item.content_evidence = {"status": "not_reviewed", "error": "仅基础检测；限时复核未覆盖此照片"}
    if not candidates or time.monotonic() >= deadline:
        return
    model = BundledWeddingContentModel(options.content_model_name, cancel_event, pause_event)
    if model.runtime is not None:
        model.runtime.deadline = deadline
    reviewed = 0
    try:
        for item in candidates:
            checkpoint()
            if time.monotonic() >= deadline:
                break
            if phase_callback is not None:
                phase_callback(reviewed, len(candidates), item.path, max(0, deadline - time.monotonic()))
            try:
                source = _analysis_source(item.path, item.pair_paths)
                key = str(source.resolve()) + "::content-review"
                signature = _cache_signature(source, options)
                signature["content_fingerprint"] = model.fingerprint
                cached = _result_from_cache(item.path, item.pair_paths, cache.get(key, {}), signature)
                if cached is not None and cached.content_evidence.get("status") == "ok":
                    item.content_evidence = dict(cached.content_evidence)
                else:
                    # Same shared read-only RAW/HEIC path as the first pass.
                    image = _load_preview(source)
                    if time.monotonic() >= deadline:
                        break
                    item.content_evidence = model.inspect(image, CONTENT_TYPES.get(options.wedding_content_type, "auto"))
                    del image
                    if item.content_evidence.get("status") == "ok":
                        cache[key] = _cache_record(item, signature)
                if item.content_evidence.get("status") == "ok":
                    reviewed += 1
                    neutral_type = CONTENT_TYPES.get(options.wedding_content_type) in {
                        "documentary", "portrait", "dynamic", "details"}
                    if options.prefer_people and not neutral_type and eye_neutral(item, options):
                        item.score -= _people_priority_bonus(item.frontal_face_count, item.smiling_face_count)
                elif not model.ready:
                    break  # Missing/error component must not delay every frame.
            except InferenceBudgetExpired:
                item.content_evidence = {"status": "not_reviewed", "error": "限时复核预算结束；仅基础检测"}
                break
            except ContentModelCancelled as exc:
                raise AnalysisCancelled() from exc
            except Exception as exc:
                item.content_evidence = {"status": "error", "error": str(exc)[:160]}
    finally:
        model.close()


def analyze_paths(
    paths: list[Path],
    options: AnalysisOptions = AnalysisOptions(),
    progress_callback: Callable[[int, int, Path], None] | None = None,
    pause_event: threading.Event | None = None,
    cancel_event: threading.Event | None = None,
    *,
    phase_callback: Callable[[int, int, Path, float], None] | None = None,
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
    if options.content_review_mode not in {"limited", "full"}:
        raise ValueError("unsupported content review mode")
    if type(options.human_face_recovery_enabled) is not bool or (
            options.human_face_recovery_enabled and not options.human_analysis_enabled):
        raise ValueError("face recovery requires enabled human evidence")
    started = time.monotonic()
    limited = (options.content_model_enabled and options.scene_type == "婚礼"
               and options.content_review_mode == "limited")
    base_options = replace(options, content_model_enabled=False) if limited else options
    entries = _representatives(_paths_for_selected_files(paths))
    cache = _load_analysis_cache()
    ordered_results: list[PhotoResult | None] = [None] * len(entries)
    total = len(entries)
    content_model = (BundledWeddingContentModel(options.content_model_name, cancel_event, pause_event)
                     if total and options.scene_type == "婚礼" and options.content_model_enabled and not limited else None)
    human_model = None

    def wait_if_paused() -> None:
        while pause_event is not None and pause_event.is_set():
            if cancel_event is not None and cancel_event.is_set():
                raise AnalysisCancelled()
            time.sleep(0.15)
        if cancel_event is not None and cancel_event.is_set():
            raise AnalysisCancelled()
    if total and options.human_analysis_enabled:
        from src.core.image_features.human_process import HumanProcessService
        human_model = HumanProcessService(checkpoint=wait_if_paused,
                                          recover_faces=options.human_face_recovery_enabled)

    if total:
        worker_count = min(_analysis_worker_count(), total)
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="event-cull")
        futures: dict[int, Future[tuple[PhotoResult, str, dict | None]]] = {}
        next_submit = 0

        def submit(index: int) -> None:
            photo, pairs = entries[index]
            args = (photo, pairs, base_options, cache)
            if human_model is not None:
                futures[index] = executor.submit(_analyze_cached,*args,content_model,human_model=human_model)
            else:
                futures[index] = executor.submit(_analyze_cached, *args) if content_model is None else (
                    executor.submit(_analyze_cached, *args, content_model))

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
                except ContentModelCancelled as exc:
                    raise AnalysisCancelled() from exc
                except AnalysisCancelled:
                    raise
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
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            finally:
                if content_model is not None:
                    content_model.close()
                if human_model is not None:
                    human_model.close()

    results = [result for result in ordered_results if result is not None]
    wait_if_paused()
    if limited:
        _limited_content_review(results, options, cache, started, wait_if_paused,
                                cancel_event, pause_event, phase_callback)
    elif not options.content_model_enabled:
        for item in results:
            item.content_evidence = {"status": "base_only", "error": "快速筛选：未进行内容模型复核"}
    wait_if_paused()
    try:
        _write_analysis_cache(cache)
    except OSError:
        pass
    groups = _result_groups(results, options)
    _assign_tiered_ratings(groups, options)
    return results
