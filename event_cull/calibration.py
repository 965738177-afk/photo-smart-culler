"""Local, user-specific visual calibration with explicit learning objectives.

The profile stores only a small classifier and its normalization statistics.
It never stores source pixels and never writes photo metadata.
"""

from __future__ import annotations

import csv
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sqlite3
import zlib
import copy
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import cv2
import numpy as np

from event_cull.engine import IMAGE_EXTENSIONS, AnalysisOptions, PhotoResult, _load_preview
from event_cull.source_scan import scan_source_images
from event_cull.rating_policy import AUTOMATIC_RATINGS, automatic_rating
from event_cull.platform_paths import application_data_dir
from event_cull.deep_features import (
    PIPELINE_ID as DEEP_PIPELINE_ID,
    LEGACY_PIPELINE_ID as LEGACY_DEEP_PIPELINE_ID,
    CPU_PIPELINE_ID as CPU_DEEP_PIPELINE_ID,
    DeepFeatureUnavailable,
    encode_images,
    preferred_backend,
)


PROFILE_VERSION = 4
PROFILE_FILENAME = "personal-calibration-v1.json"
CORPUS_FILENAME = "personal-calibration-history-v1.sqlite3"
MIN_VALIDATION_AUC = 0.60
MIN_RETENTION_AGREEMENT = 0.75
_RATING_PATTERN = re.compile(rb"xmp:Rating=[\"'](-?\d+)[\"']")
HANDCRAFTED_PIPELINE_ID = "handcrafted-visual-v1"
FEATURE_DIMENSIONS = {HANDCRAFTED_PIPELINE_ID: 256, LEGACY_DEEP_PIPELINE_ID: 328,
                      DEEP_PIPELINE_ID: 832, CPU_DEEP_PIPELINE_ID: 832}
FEATURE_BATCH_SIZE = 16


class CalibrationError(RuntimeError):
    """Raised when supplied Lightroom results cannot form a safe profile."""


@dataclass(frozen=True)
class CalibrationProfile:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    featured_rating: int
    training_examples: int
    positive_examples: int
    validation_auc: float
    feature_pipeline: str = HANDCRAFTED_PIPELINE_ID
    objective: str = "featured"  # v1/v2 profiles retain their original meaning.
    scene_type: str = ""
    validation_precision: float | None = None
    validation_recall: float | None = None
    validation_f1: float | None = None
    validation_examples: int = 0
    validation_scope: str = "burst_holdout"
    task_count: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "version": PROFILE_VERSION,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "weights": list(self.weights),
            "bias": self.bias,
            "featured_rating": self.featured_rating,
            "training_examples": self.training_examples,
            "positive_examples": self.positive_examples,
            "validation_auc": self.validation_auc,
            "feature_pipeline": self.feature_pipeline,
            "objective": self.objective,
            "scene_type": self.scene_type,
            "validation_precision": self.validation_precision,
            "validation_recall": self.validation_recall,
            "validation_f1": self.validation_f1,
            "validation_examples": self.validation_examples,
            "validation_scope": self.validation_scope,
            "task_count": self.task_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CalibrationProfile":
        if not isinstance(value, dict) or value.get("version") not in {1, 2, 3, PROFILE_VERSION}:
            raise CalibrationError("选片偏好校准文件版本不兼容。")
        try:
            mean = tuple(float(item) for item in value["mean"])
            scale = tuple(float(item) for item in value["scale"])
            weights = tuple(float(item) for item in value["weights"])
            profile = cls(
                mean=mean,
                scale=scale,
                weights=weights,
                bias=float(value["bias"]),
                featured_rating=int(value["featured_rating"]),
                training_examples=int(value["training_examples"]),
                positive_examples=int(value["positive_examples"]),
                validation_auc=float(value["validation_auc"]),
                feature_pipeline=str(value.get("feature_pipeline") or HANDCRAFTED_PIPELINE_ID),
                objective=str(value.get("objective") or "featured"),
                scene_type=str(value.get("scene_type") or ""),
                validation_precision=value.get("validation_precision"),
                validation_recall=value.get("validation_recall"),
                validation_f1=value.get("validation_f1"),
                validation_examples=int(value.get("validation_examples") or 0),
                validation_scope=str(value.get("validation_scope") or "burst_holdout"),
                task_count=int(value.get("task_count", 1)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError("选片偏好校准文件不完整。") from exc
        if not mean or len(mean) != len(scale) or len(mean) != len(weights):
            raise CalibrationError("选片偏好校准特征维度无效。")
        numbers = [*mean, *scale, *weights, profile.bias, profile.validation_auc]
        if (not np.isfinite(numbers).all() or any(item <= 0 for item in scale)
                or profile.objective not in {"featured", "retention"}
                or not 0 <= profile.validation_auc <= 1 or profile.task_count < 1):
            raise CalibrationError("选片偏好校准参数无效。")
        for metric in (profile.validation_precision, profile.validation_recall, profile.validation_f1):
            if metric is not None and (not isinstance(metric, (int, float)) or not np.isfinite(metric) or not 0 <= metric <= 1):
                raise CalibrationError("选片偏好校准验证指标无效。")
        return profile

    def probability(self, image: np.ndarray) -> float:
        values = _feature_matrix([image], self.feature_pipeline)[0]
        return self.probability_for_feature(values)

    def probability_for_feature(self, values: np.ndarray) -> float:
        if len(values) != len(self.weights):
            raise CalibrationError("选片偏好校准特征与当前版本不匹配。请重新导入一次校准结果。")
        if not np.isfinite(values).all():
            raise CalibrationError("选片偏好校准特征含无效数值。")
        normalized = (values - np.asarray(self.mean)) / np.asarray(self.scale)
        return _sigmoid(float(normalized @ np.asarray(self.weights) + self.bias))


def default_profile_path() -> Path:
    return application_data_dir() / "calibration" / PROFILE_FILENAME


def default_corpus_path() -> Path:
    return application_data_dir() / "calibration" / CORPUS_FILENAME


@dataclass(frozen=True)
class CumulativeUpdate:
    """The newly active profile, or a pending batch that did not pass validation."""

    scene_type: str
    profile: CalibrationProfile | None
    task_count: int
    example_count: int
    positive_count: int
    replaced_task: bool
    pending_reason: str = ""


def load_profile(path: Path | None = None) -> CalibrationProfile | None:
    source = path or default_profile_path()
    if not source.is_file():
        return None
    try:
        return CalibrationProfile.from_dict(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, CalibrationError):
        return None


def save_profile(profile: CalibrationProfile, path: Path | None = None) -> Path:
    destination = path or default_profile_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    # Validate before replacing a working local preference file.
    payload = profile.to_dict()
    CalibrationProfile.from_dict(payload)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_xmp_rating(path: Path) -> int | None:
    """Read only the embedded XMP rating from an exported JPEG."""
    try:
        with path.open("rb") as stream:
            payload = stream.read()
    except OSError:
        return None
    match = _RATING_PATTERN.search(payload)
    return int(match.group(1)) if match else None


def exported_selection_files(folder: Path, *, recursive: bool = True) -> tuple[Path, ...]:
    """Collect supported feedback photos once; sidecars are never examples.

    Direct scans let an explicitly selected RAW directory remain independent
    of its rendered-export subdirectory. Recursive remains the existing default.
    """
    try:
        paths = folder.rglob("*") if recursive else folder.iterdir()
        return tuple(path for path in paths
                     if not path.name.startswith(".") and path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    except OSError as exc:
        raise CalibrationError("无法读取人工选片文件夹。") from exc


def exported_selection_stems(folder: Path, *, recursive: bool = True) -> set[str]:
    """Return the names of frames a photographer deliberately kept.

    The feedback signal is the photographer exporting (or collecting) a frame,
    not an XMP star embedded in that file.  Lightroom can omit XMP metadata,
    and photographers may use flags or collections instead of a star value.
    Matching by filename works for both rendered JPGs and selected originals.
    """
    return {path.stem.casefold() for path in exported_selection_files(folder, recursive=recursive)}


def _log_rows(log_path: Path) -> dict[str, tuple[str, int]]:
    """Map unique original names to their historical burst group and rating."""
    try:
        if log_path.suffix.lower() == ".json":
            payload = json.loads(log_path.read_text(encoding="utf-8"))
            rows = payload.get("photos", []) if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                rows = []
        else:
            with log_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError) as exc:
        raise CalibrationError("无法读取筛选日志（CSV 或自动保存的 JSON）。") from exc
    records: dict[str, tuple[str, int] | None] = {}
    for row in rows:
        name = Path(str(row.get("file") or "")).stem.casefold()
        if not name:
            continue
        try:
            rating = int(str(row.get("rating") or "0"))
        except ValueError:
            rating = 0
        record = (str(row.get("group") or name), rating)
        if name in records:
            records[name] = None
        else:
            records[name] = record
    return {name: record for name, record in records.items() if record is not None}


def _source_previews(
    folders: Iterable[Path], allowed_stems: set[str], *,
    excluded_folders: Iterable[Path] = (),
) -> dict[str, Path]:
    """Find one readable representative per old log record, preferring JPEG."""
    choices: dict[str, Path] = {}
    ambiguous: set[str] = set()
    for path in scan_source_images(folders, excluded_folders=excluded_folders):
        stem = path.stem.casefold()
        if stem not in allowed_stems or stem in ambiguous:
            continue
        previous = choices.get(stem)
        if previous is None:
            choices[stem] = path
        elif previous.parent == path.parent:
            # Prefer the JPEG already paired with a RAW, as it is also
            # what Lightroom exports and what the app previews first.
            if previous.suffix.lower() not in {".jpg", ".jpeg"} and path.suffix.lower() in {".jpg", ".jpeg"}:
                choices[stem] = path
        else:
            ambiguous.add(stem)
            choices.pop(stem, None)
    return choices


def _visual_descriptor(image: np.ndarray) -> np.ndarray:
    """Compact layout/colour/detail signature; no external model or network."""
    if image is None or image.size == 0:
        raise CalibrationError("有照片无法读取预览。")
    resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    tiles: list[float] = []
    for row in range(4):
        for column in range(4):
            tile = hsv[row * 32 : (row + 1) * 32, column * 32 : (column + 1) * 32]
            tiles.extend(tile.mean(axis=(0, 1)).tolist())
            tiles.extend(tile.std(axis=(0, 1)).tolist())
    thumbnail = cv2.resize(gray, (12, 12), interpolation=cv2.INTER_AREA).reshape(-1)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    edge_tiles = [
        float(magnitude[row * 32 : (row + 1) * 32, column * 32 : (column + 1) * 32].mean())
        for row in range(4)
        for column in range(4)
    ]
    return np.asarray([*tiles, *thumbnail.tolist(), *edge_tiles], dtype=np.float32)


def _preferred_feature_pipeline() -> str:
    """Use the bundled deep encoder when it is safely available locally."""
    return DEEP_PIPELINE_ID if preferred_backend() != "unavailable" else HANDCRAFTED_PIPELINE_ID


def _feature_matrix(images: list[np.ndarray], pipeline: str) -> np.ndarray:
    """Build stable local features, adding batched MPS embeddings when present."""
    if not images:
        return np.empty((0, 0), dtype=np.float32)
    handcrafted = np.vstack([_visual_descriptor(image) for image in images]).astype(np.float32)
    if pipeline == HANDCRAFTED_PIPELINE_ID:
        return handcrafted
    if pipeline not in {DEEP_PIPELINE_ID, LEGACY_DEEP_PIPELINE_ID, CPU_DEEP_PIPELINE_ID}:
        raise CalibrationError("选片偏好校准特征版本不兼容。请重新导入一次校准结果。")
    try:
        if pipeline == CPU_DEEP_PIPELINE_ID:
            deep_values, _backend = encode_images(images, backend="cpu")
            if _backend != "cpu":
                raise DeepFeatureUnavailable("CPU-only feature contract returned another backend.")
        else:
            deep_values, _backend = encode_images(images)
    except DeepFeatureUnavailable as exc:
        raise CalibrationError("深度评分组件不可用；请重新更新选片偏好校准。") from exc
    if len(deep_values) != len(handcrafted):
        raise CalibrationError("深度评分没有返回完整的画面特征。")
    if pipeline == LEGACY_DEEP_PIPELINE_ID:
        deep_values = deep_values.reshape(len(deep_values), 72, 8).mean(axis=2)
    return np.hstack((handcrafted, deep_values)).astype(np.float32)


def _feature_rows(
    paths: list[Path], pipeline: str, *,
    progress_callback: Callable[[int, int, Path], None] | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Share bounded image batches between training and applying preferences.

    Retain small feature vectors, never the entire batch of full-size previews.
    Check cancellation outside the decode exception handler so it propagates.
    """
    def decode(path):
        try:
            return _load_preview(path)
        except Exception:
            return None

    # The same shared decoder, bounded to one feature batch in memory. Serial
    # RAW reads made an otherwise compact encoder take ~15 minutes on 5,000
    # photos. Do not create an unbounded future/image list to gain concurrency.
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="calibration-preview") as executor:
        for start in range(0, len(paths), FEATURE_BATCH_SIZE):
            if checkpoint:
                checkpoint()
            batch = paths[start:start + FEATURE_BATCH_SIZE]
            futures = [executor.submit(decode, path) for path in batch]
            images, indexes = [], []
            try:
                for offset, future in enumerate(futures):
                    if checkpoint:
                        checkpoint()
                    image = future.result()
                    if image is not None and image.size:
                        images.append(image)
                        indexes.append(start + offset)
                    image = None
                    if progress_callback:
                        progress_callback(start + offset + 1, len(paths), batch[offset])
            finally:
                for future in futures:
                    future.cancel()
                futures.clear()
                future = None
            if checkpoint:
                checkpoint()
            if images:
                features = _feature_matrix(images, pipeline)
                if len(features) != len(indexes):
                    raise CalibrationError("选片偏好校准未返回完整特征。")
                images.clear()
                if checkpoint:
                    checkpoint()
                yield from zip(indexes, features)
                indexes.clear()


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def _fit_classifier(features: np.ndarray, labels: np.ndarray, *, checkpoint=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-4)
    values = (features - mean) / scale
    weights = np.zeros(values.shape[1], dtype=np.float64)
    positive = max(1, int(labels.sum()))
    negative = max(1, len(labels) - positive)
    sample_weights = np.where(labels > 0, len(labels) / (2 * positive), len(labels) / (2 * negative))
    # Correlated thumbnail/color channels can give hundreds of redundant
    # dimensions. A fixed 0.16 step then oscillates instead of fitting logistic
    # loss. Backtracking checks the actual regularized loss, without adding a
    # platform-specific training dependency or silently dropping features.
    def loss(w, b):
        logits = values @ w + b
        return float(np.mean((np.logaddexp(0, logits) - labels * logits) * sample_weights)
                     + .006 * (w @ w))
    bias = 0.0
    step = .16
    current_loss = loss(weights, bias)
    for iteration in range(360):
        if checkpoint and iteration % 20 == 0:
            checkpoint()
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(values @ weights + bias, -35.0, 35.0)))
        error = (probabilities - labels) * sample_weights
        gradient = (values.T @ error) / len(labels) + .012 * weights
        bias_gradient = float(error.mean())
        trial_step = min(.16, step * 1.2)
        for _ in range(32):
            trial_weights = weights - trial_step * gradient
            trial_bias = bias - trial_step * bias_gradient
            trial_loss = loss(trial_weights, trial_bias)
            if trial_loss <= current_loss + 1e-12:
                weights, bias, current_loss, step = trial_weights, trial_bias, trial_loss, trial_step
                break
            trial_step *= .5
        else:
            break
    return mean, scale, weights, bias


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Ties must get their average rank, otherwise a constant predictor appears
    # better/worse solely because the filenames were enumerated in a new order.
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    for start, end in zip(starts, np.r_[starts[1:], len(scores)]):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _holdout_mask(groups: list[str]) -> np.ndarray:
    return np.asarray([zlib.crc32(group.encode("utf-8")) % 5 == 0 for group in groups], dtype=bool)


def _has_class_balance(mask: np.ndarray, labels: np.ndarray) -> bool:
    held_out_positive = int(labels[mask].sum())
    training_positive = int(labels[~mask].sum())
    return (
        int(mask.sum()) >= 20
        and held_out_positive >= 4
        and int(mask.sum()) - held_out_positive >= 4
        and training_positive >= 8
        and int((~mask).sum()) - training_positive >= 8
    )


def _balanced_group_holdout(groups: list[str], labels: np.ndarray) -> np.ndarray:
    hashed = _holdout_mask(groups)
    if _has_class_balance(hashed, labels):
        return hashed
    for salt in range(1, 32):
        candidate = np.asarray(
            [zlib.crc32(f"{salt}:{group}".encode("utf-8")) % 5 == 0 for group in groups],
            dtype=bool,
        )
        if _has_class_balance(candidate, labels):
            return candidate
    # The group ids in older CSVs are often sequential. Choose a whole-group
    # fold rather than falling back to frame-level sampling, which would leak
    # near-identical burst images between training and validation.
    ordered = sorted(set(groups))
    for offset in range(5):
        held_groups = set(ordered[offset::5])
        candidate = np.asarray([group in held_groups for group in groups], dtype=bool)
        if _has_class_balance(candidate, labels):
            return candidate
    raise CalibrationError("人工留片样本分布过于集中，无法进行按连拍组隔离的校验。")


def train_profile(
    samples: list[tuple[np.ndarray, int, str]],
    featured_rating: int,
    *,
    feature_pipeline: str = HANDCRAFTED_PIPELINE_ID,
    objective: str = "featured",
    scene_type: str = "",
    validation_scope: str = "burst_holdout",
    task_count: int = 1,
    checkpoint: Callable[[], None] | None = None,
) -> CalibrationProfile:
    if len(samples) < 80:
        raise CalibrationError("可用于校准的原图不足 80 张。")
    labels = np.asarray([int(rating == featured_rating) for _feature, rating, _group in samples], dtype=np.int8)
    positive_count = int(labels.sum())
    if positive_count < 12 or positive_count == len(labels):
        raise CalibrationError("人工留片样本至少需要 12 张，且必须有未保留照片用于对比。")
    features = np.vstack([feature for feature, _rating, _group in samples]).astype(np.float64)
    if not np.isfinite(features).all():
        raise CalibrationError("画面特征含无效数值，保留旧学习结果。")
    groups = [group for _feature, _rating, group in samples]
    held_out = _balanced_group_holdout(groups, labels)
    training = ~held_out
    if validation_scope in {"continuous_100_frame_blocks", "cumulative_group_holdout"}:
        ordered = list(dict.fromkeys(groups))
        held_groups = {group for group, held in zip(groups, held_out, strict=True) if held}
        if validation_scope == "continuous_100_frame_blocks":
            held_blocks = {index for index, group in enumerate(ordered) if group in held_groups}
            forbidden = held_blocks | {index + offset for index in held_blocks for offset in (-1, 1)}
            positions = {group: index for index, group in enumerate(ordered)}
            training = np.asarray([positions[group] not in forbidden for group in groups])
        else:
            # Only neighbouring blocks from the SAME task are embargoed. A
            # different task's block number must never suppress its samples.
            held_blocks = set()
            for group in held_groups:
                match = re.fullmatch(r"block:([0-9a-f]{64}):(\d+)", group)
                if match:
                    held_blocks.update(f"block:{match[1]}:{int(match[2]) + offset}"
                                       for offset in (-1, 0, 1))
            training = np.asarray([group not in held_groups and group not in held_blocks for group in groups])
        if (labels[training].sum() < 8 or training.sum()-labels[training].sum() < 8):
            raise CalibrationError("隔离相邻连续片段后，训练样本不足；不降低验证标准启用学习。")
    mean, scale, weights, bias = _fit_classifier(features[training], labels[training], checkpoint=checkpoint)
    validation_values = (features[held_out] - mean) / scale
    validation_scores = validation_values @ weights + bias
    validation = _roc_auc(labels[held_out], validation_scores)
    if validation < MIN_VALIDATION_AUC:
        raise CalibrationError(
            f"这批资料的分组验证区分度只有 {validation:.0%}，不足以安全启用选片偏好校准。"
        )
    # Match the retention fraction from TRAINING labels; never tune a threshold
    # against the held-out filenames. AUC is not selection accuracy.
    take = max(1, min(int(held_out.sum()), round(held_out.sum() * labels[training].mean())))
    chosen = np.argsort(-validation_scores, kind="stable")[:take]
    hit = int(labels[held_out][chosen].sum())
    precision = hit / take
    recall = hit / max(1, int(labels[held_out].sum()))
    if objective == "retention" and (precision < MIN_RETENTION_AGREEMENT or recall < MIN_RETENTION_AGREEMENT):
        raise CalibrationError(
            f"原片学习留出验证：一致率 {precision:.1%}、找回率 {recall:.1%}，"
            f"未达到各 {MIN_RETENTION_AGREEMENT:.0%} 的启用门槛；未替换当前偏好，保留基础筛选。"
        )
    mean, scale, weights, bias = _fit_classifier(features, labels, checkpoint=checkpoint)
    return CalibrationProfile(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in weights),
        bias=float(bias),
        featured_rating=featured_rating,
        training_examples=len(samples),
        positive_examples=positive_count,
        validation_auc=validation,
        feature_pipeline=feature_pipeline,
        objective=objective, scene_type=scene_type,
        validation_precision=precision, validation_recall=recall,
        validation_f1=2 * precision * recall / max(precision + recall, 1e-12),
        validation_examples=int(held_out.sum()), validation_scope=validation_scope,
        task_count=task_count,
    )


def blocked_validation_groups(groups: list[str], minimum_frames: int = 100) -> list[str]:
    """Keep whole bursts in coarse continuous blocks for honest validation.

    Hash groups can fragment one scene. These larger blocks reduce adjacent
    frame leakage; they are not a claim of semantic scene/identity recognition.
    Used ONLY for splitting labels, never as an input to the classifier.
    """
    blocks, block, count, previous = [], 0, 0, None
    for group in groups:
        if count >= minimum_frames and group != previous:
            block += 1
            count = 0
        blocks.append(f"continuous-block-{block}")
        count += 1
        previous = group
    return blocks


def _feedback_sources(
    log_path: Path, exported_folder: Path, original_folders: list[Path],
) -> tuple[str, list[tuple[str, str, Path, int]], list[str], bool]:
    """Match complete human delivery against originals without reading pixels."""
    rows = _log_rows(log_path)
    selected_stems = exported_selection_stems(exported_folder)
    if not selected_stems:
        raise CalibrationError("人工选片文件夹中没有读取到 JPG 或 RAW 文件名。")
    if not set(rows).intersection(selected_stems):
        raise CalibrationError("人工选片文件名与筛选日志没有匹配；请确认导出时没有重命名照片。")
    sources = _source_previews(original_folders, set(rows), excluded_folders=[exported_folder])
    if selected_stems - set(sources):
        raise CalibrationError("人工结果存在无法唯一对应的原片，不能安全学习；请检查重命名或同名文件。")
    source_rows = [
        (stem, rows[stem][0], path, int(stem in selected_stems))
        for stem, path in sorted(sources.items())
    ]
    blocks = blocked_validation_groups([row[1] for row in source_rows])
    blocked = len(set(blocks)) >= 5
    if not blocked:
        blocks = [row[1] for row in source_rows]
    scene_type = "通用"
    if log_path.suffix.lower() == ".json":
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
            scene_type = str((payload.get("settings") or {}).get("scene_type") or scene_type)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError) as exc:
            raise CalibrationError("无法读取筛选日志的照片题材。") from exc
    return scene_type, source_rows, blocks, blocked


def build_profile(
    log_path: Path,
    exported_folder: Path,
    original_folders: list[Path],
    *,
    progress_callback: Callable[[int, int, Path], None] | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> CalibrationProfile:
    """Build a profile from an app log, kept filenames, and originals."""
    # Final original deliveries teach retention, not retouch candidacy. There
    # is no evidence here that every exported original is an ideal final hero.
    featured_rating = 1
    scene_type, source_rows, blocks, blocked = _feedback_sources(
        log_path, exported_folder, original_folders,
    )
    pipeline = _preferred_feature_pipeline()

    def samples_for(selected_pipeline):
        return [
            (feature, source_rows[index][3], blocks[index])
            for index, feature in _feature_rows(
                [row[2] for row in source_rows], selected_pipeline,
                progress_callback=progress_callback, checkpoint=checkpoint,
            )
        ]

    try:
        samples = samples_for(pipeline)
    except CalibrationError:
        # A damaged optional deep runtime must never block the proven local
        # pipeline. The saved profile records the fallback explicitly.
        pipeline = HANDCRAFTED_PIPELINE_ID
        samples = samples_for(pipeline)
    if checkpoint:
        checkpoint()
    profile = train_profile(samples, featured_rating, feature_pipeline=pipeline,
                            objective="retention", scene_type=scene_type,
                            validation_scope="continuous_100_frame_blocks" if blocked else "burst_holdout",
                            checkpoint=checkpoint)
    if checkpoint:
        checkpoint()
    return profile


def _open_corpus(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA busy_timeout=30000")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS feedback_tasks (
            task_key TEXT PRIMARY KEY, scene_type TEXT NOT NULL,
            feature_pipeline TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feedback_examples (
            scene_type TEXT NOT NULL, source_key TEXT NOT NULL,
            task_key TEXT NOT NULL, signature TEXT NOT NULL,
            group_key TEXT NOT NULL, label INTEGER NOT NULL CHECK(label IN (0, 1)),
            feature BLOB NOT NULL, PRIMARY KEY(scene_type, source_key)
        );
        CREATE INDEX IF NOT EXISTS feedback_by_task ON feedback_examples(task_key);
        CREATE TABLE IF NOT EXISTS active_profiles (
            scene_type TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL
        );
    """)
    return connection


def load_cumulative_profiles(path: Path | None = None) -> dict[str, CalibrationProfile]:
    """Read only validated profiles; a damaged history must not prevent launch."""
    source = path or default_corpus_path()
    if not source.is_file():
        return {}
    try:
        with closing(sqlite3.connect(source, timeout=3)) as connection:
            profiles = {}
            for scene_type, payload in connection.execute("SELECT scene_type, payload FROM active_profiles"):
                try:
                    profile = CalibrationProfile.from_dict(json.loads(payload))
                    if profile.scene_type == scene_type and profile.objective == "retention":
                        profiles[scene_type] = profile
                except (json.JSONDecodeError, CalibrationError):
                    continue
            return profiles
    except sqlite3.Error:
        return {}


def load_cumulative_history(path: Path | None = None) -> dict[str, tuple[int, int, int]]:
    """Return task/frame/retained counts, including feedback awaiting validation."""
    source = path or default_corpus_path()
    if not source.is_file():
        return {}
    try:
        with closing(sqlite3.connect(source, timeout=3)) as connection:
            return {
                scene: (int(tasks), int(examples), int(positives or 0))
                for scene, tasks, examples, positives in connection.execute(
                    "SELECT scene_type, COUNT(DISTINCT task_key), COUNT(*), SUM(label) "
                    "FROM feedback_examples GROUP BY scene_type"
                )
            }
    except sqlite3.Error:
        return {}


def _task_key(log_path: Path, scene_type: str, paths: list[Path]) -> str:
    """A task ID survives re-import; CSV falls back to its exact source set."""
    task_id = ""
    if log_path.suffix.lower() == ".json":
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            task = payload.get("task")
            task_id = str((task.get("id") if isinstance(task, dict) else None)
                          or payload.get("id") or "")
    identity = (f"id:{task_id}" if task_id else
                "sources:" + "\n".join(sorted(str(path.resolve()) for path in paths)))
    return hashlib.sha256(f"{scene_type}\n{identity}".encode()).hexdigest()


def _stored_feature(blob: bytes, pipeline: str) -> np.ndarray:
    vector = np.frombuffer(blob, dtype="<f4")
    if len(vector) != FEATURE_DIMENSIONS[pipeline] or not np.isfinite(vector).all():
        raise CalibrationError("本机累计学习记录有损坏特征；未修改现有偏好。")
    return vector.copy()


def update_cumulative_profile(
    log_path: Path,
    exported_folder: Path,
    original_folders: list[Path],
    *,
    path: Path | None = None,
    progress_callback: Callable[[int, int, Path], None] | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> CumulativeUpdate:
    """Accumulate exact feedback examples, then atomically activate only if safe.

    Re-importing a task replaces its examples. Overlapping source paths across
    tasks are upserted, so a frame is never counted twice in one scene. Invalid
    candidates remain in history for later learning but cannot replace the
    previously active, validated profile. Original pixels/paths are not stored.
    """
    scene_type, source_rows, blocks, blocked = _feedback_sources(
        log_path, exported_folder, original_folders,
    )
    if not source_rows:
        raise CalibrationError("没有找到可与筛选日志对应的原片。")
    task_key = _task_key(log_path, scene_type, [row[2] for row in source_rows])
    source_keys = [hashlib.sha256(str(row[2].resolve()).encode("utf-8")).hexdigest()
                   for row in source_rows]
    signatures = []
    for _, _, source, _ in source_rows:
        try:
            stat = source.stat()
        except OSError as exc:
            raise CalibrationError(f"无法读取原片状态：{source.name}") from exc
        signatures.append(f"{stat.st_size}:{stat.st_mtime_ns}")
    destination = path or default_corpus_path()
    try:
        with closing(_open_corpus(destination)) as connection:
            known = connection.execute(
                "SELECT feature_pipeline FROM feedback_tasks WHERE scene_type=? LIMIT 1",
                (scene_type,),
            ).fetchone()
            pipeline = known[0] if known else _preferred_feature_pipeline()
            if pipeline not in FEATURE_DIMENSIONS:
                raise CalibrationError("旧累计特征版本不兼容，未修改当前学习记录。")
            # Re-imported unchanged frames reuse their exact saved feature.
            cached = {
                key: (signature, blob)
                for key, signature, blob in connection.execute(
                    "SELECT source_key, signature, feature FROM feedback_examples WHERE scene_type=?",
                    (scene_type,),
                )
            }
            features: dict[int, np.ndarray] = {}
            decode_indexes = []
            for index, (key, signature) in enumerate(zip(source_keys, signatures, strict=True)):
                previous = cached.get(key)
                if previous is not None and previous[0] == signature:
                    features[index] = _stored_feature(previous[1], pipeline)
                    if progress_callback:
                        progress_callback(index + 1, len(source_rows), source_rows[index][2])
                else:
                    decode_indexes.append(index)
            def decode_missing(selected_pipeline: str) -> None:
                for local_index, feature in _feature_rows(
                    [source_rows[index][2] for index in decode_indexes], selected_pipeline,
                    progress_callback=(None if progress_callback is None else
                                       lambda current, total, source: progress_callback(
                                           current, len(source_rows), source)),
                    checkpoint=checkpoint,
                ):
                    features[decode_indexes[local_index]] = feature

            try:
                decode_missing(pipeline)
            except CalibrationError:
                if known or pipeline == HANDCRAFTED_PIPELINE_ID:
                    raise
                # The optional deep runtime may report ready but fail at the
                # first actual batch. With no existing scene vectors, retry
                # using the shared handcrafted encoder instead.
                pipeline = HANDCRAFTED_PIPELINE_ID
                features.clear()
                decode_indexes = list(range(len(source_rows)))
                decode_missing(pipeline)
            if len(features) != len(source_rows):
                raise CalibrationError("有原片无法生成预览特征；本次学习未保存。")
            if checkpoint:
                checkpoint()
            now = datetime.now(timezone.utc).isoformat()
            replaced_task = connection.execute(
                "SELECT 1 FROM feedback_tasks WHERE task_key=?", (task_key,),
            ).fetchone() is not None
            # One transaction covers task replacement, deduplication and model
            # activation; cancellation or I/O failure rolls it all back.
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR REPLACE INTO feedback_tasks VALUES (?, ?, ?, ?)",
                (task_key, scene_type, pipeline, now),
            )
            connection.execute("DELETE FROM feedback_examples WHERE task_key=?", (task_key,))
            for index, (_stem, group, _source, label) in enumerate(source_rows):
                group_key = (f"block:{task_key}:{blocks[index].rsplit('-', 1)[-1]}"
                             if blocked else f"burst:{task_key}:{group}")
                vector = np.asarray(features[index], dtype="<f4")
                connection.execute(
                    "INSERT OR REPLACE INTO feedback_examples VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (scene_type, source_keys[index], task_key, signatures[index],
                     group_key, label, vector.tobytes()),
                )
            stored = connection.execute(
                "SELECT feature, label, group_key, task_key FROM feedback_examples "
                "WHERE scene_type=? ORDER BY task_key, group_key, source_key",
                (scene_type,),
            ).fetchall()
            task_count = len({row[3] for row in stored})
            examples = [(_stored_feature(blob, pipeline), label, group_key)
                        for blob, label, group_key, _task in stored]
            positive_count = sum(label for _feature, label, _group in examples)
            try:
                profile = train_profile(
                    examples, 1, feature_pipeline=pipeline, objective="retention",
                    scene_type=scene_type, validation_scope="cumulative_group_holdout",
                    task_count=task_count, checkpoint=checkpoint,
                )
                pending_reason = ""
            except CalibrationError as exc:
                profile = None
                pending_reason = str(exc)
            if checkpoint:
                checkpoint()
            if profile is not None:
                connection.execute(
                    "INSERT OR REPLACE INTO active_profiles VALUES (?, ?, ?)",
                    (scene_type, json.dumps(profile.to_dict(), ensure_ascii=False), now),
                )
            connection.commit()
            return CumulativeUpdate(
                scene_type=scene_type, profile=profile,
                task_count=task_count, example_count=len(examples),
                positive_count=positive_count, replaced_task=replaced_task,
                pending_reason=pending_reason,
            )
    except sqlite3.Error as exc:
        raise CalibrationError("无法保存本机累计学习记录；原片未修改，旧偏好未替换。") from exc


def apply_profile(
    results: list[PhotoResult],
    profile: CalibrationProfile,
    *,
    progress_callback: Callable[[int, int, Path], None] | None = None,
    checkpoint: Callable[[], None] | None = None,
    protected_paths: frozenset[Path] = frozenset(),
    options: AnalysisOptions | None = None,
) -> int:
    """Apply only the saved objective; old retouch profiles stay retouch-only."""
    if getattr(profile, "objective", "featured") == "retention":
        return _apply_retention_profile(results, profile, options=options,
                                        progress_callback=progress_callback, checkpoint=checkpoint,
                                        protected_paths=protected_paths)
    existing_featured = sum(item.suggested_rating == 3 and item.path not in protected_paths for item in results)
    # Existing/manual two-star photos must never be promoted or demoted by
    # this automatic pass, even if a caller did not supply protected_paths.
    candidates = [item for item in results if item.suggested_rating in AUTOMATIC_RATINGS - {0}
                  and item.path not in protected_paths and not item.featured_blocked
                  and item.max_auto_rating >= 3]
    if not candidates or existing_featured == 0:
        return 0
    paths = [next((path for path in item.pair_paths if path.suffix.lower() in {".jpg", ".jpeg"}), item.path)
             for item in candidates]
    probabilities: dict[Path, float] = {}
    for index, feature in _feature_rows(paths, profile.feature_pipeline,
                                       progress_callback=progress_callback, checkpoint=checkpoint):
        item = candidates[index]
        probability = profile.probability_for_feature(feature)
        # Legacy retouch profiles must not leak a retention score into the
        # shared original selection strategy on a later ranking pass.
        probabilities[item.path] = probability
    if not probabilities:
        raise CalibrationError("当前照片无法生成校准特征，已保留原有推荐。")
    by_group: dict[int, PhotoResult] = {}
    for item in candidates:
        if item.path not in probabilities:
            continue
        previous = by_group.get(item.group_id)
        if previous is None or (
            probabilities[item.path], item.score
        ) > (probabilities[previous.path], previous.score):
            by_group[item.group_id] = item
    ranked = sorted(
        by_group.values(),
        key=lambda item: (probabilities[item.path], item.score),
        reverse=True,
    )[:existing_featured]
    featured_paths = {item.path for item in ranked}
    if checkpoint:
        checkpoint()
    for item in results:
        if item.path in protected_paths:
            continue
        if item.suggested_rating == 3 and item.path not in featured_paths:
            item.suggested_rating = automatic_rating(retained=True, featured=False)
            item.reasons = [reason for reason in item.reasons if reason not in {
                "本组最佳", "Best in group", "精修候选：本组优选", "Retouch candidate: best in group",
            }]
            item.reasons.append("偏好校准后保留原片，未列为精修候选")
        elif item.path in featured_paths:
            item.suggested_rating = 3
            item.reasons = [reason for reason in item.reasons if reason not in {
                "本组最佳", "Best in group", "精修候选：本组优选", "Retouch candidate: best in group", "原片保留",
            }]
            confidence = probabilities[item.path]
            item.reasons.append(f"选片偏好校准 3★ 置信度 {confidence:.0%}")
    return len(featured_paths)


def _apply_retention_profile(results, profile, *, options, progress_callback, checkpoint, protected_paths):
    """Transactionally change original delivery, preserving manual decisions.

    Originals are scored even when the base rules did not select them. Technical
    hard gates still run, and a learned original score NEVER upgrades retouch
    evidence. Use the engine's groups and rating policy, not a second culler.
    """
    from event_cull import engine
    options = options or AnalysisOptions(scene_type=profile.scene_type or "通用")
    if not options.ai_selection_enabled or (profile.scene_type and profile.scene_type != options.scene_type):
        return 0
    protected = protected_paths | frozenset(item.path for item in results if item.suggested_rating == 2)
    working = copy.deepcopy([item for item in results if item.path not in protected])
    if not working:
        return 0
    missing, indexes = [], []
    for index, item in enumerate(working):
        if checkpoint:
            checkpoint()
        if item.visual_feature_pipeline == profile.feature_pipeline and item.visual_features:
            item.calibration_confidence = profile.probability_for_feature(np.asarray(item.visual_features))
        else:
            missing.append(next((p for p in item.pair_paths if p.suffix.lower() in {".jpg", ".jpeg"}), item.path))
            indexes.append(index)
    for index, feature in _feature_rows(missing, profile.feature_pipeline,
                                       progress_callback=progress_callback, checkpoint=checkpoint):
        working[indexes[index]].calibration_confidence = profile.probability_for_feature(feature)
    if any(item.calibration_confidence is None for item in working):
        raise CalibrationError("部分原片无法生成学习特征，已保留完整基础结果。")
    manual_kept = sum(item.suggested_rating > 0 for item in results if item.path in protected)
    original_count = sum(item.suggested_rating > 0 for item in results)
    floor = max(0, (options.target_keep_count if options.target_keep_count is not None else original_count) - manual_kept)
    ceiling = max(floor, (options.target_keep_max_count if options.target_keep_max_count is not None else original_count) - manual_kept)
    # Learn delivery density, not a demand to fill every slot of the maximum.
    # Only TRAINING positive counts contribute; the user's minimum/maximum
    # remain authoritative. A 20% original delivery should not automatically
    # become 24% just because 1,200 is an allowed ceiling for 5,000 frames.
    learned_count = round(len(results) * profile.positive_examples / max(1, profile.training_examples)) - manual_kept
    desired = max(floor, min(ceiling, learned_count))
    # Engine uses a minimum of one when passed zero. A fully manual pool needs
    # no automatic slots at all; don't add a hidden extra original.
    if desired == 0:
        for item in working:
            item.suggested_rating = 0
    else:
        selection_reasons = {engine._message(options, key) for key in ("target", "review", "backup", "best", "single_scene")}
        for item in working:
            item.reasons = [reason for reason in item.reasons if reason not in selection_reasons]
        by_group = {}
        for item in working:
            by_group.setdefault(item.group_id, []).append(item)
        engine._assign_tiered_ratings(list(by_group.values()), replace(options, target_keep_count=floor,
                                                                      target_keep_max_count=desired))
    if checkpoint:
        checkpoint()
    updated = {item.path: item for item in working}
    for item in results:
        if item.path in updated:
            replacement = updated[item.path]
            item.suggested_rating = replacement.suggested_rating
            item.calibration_confidence = replacement.calibration_confidence
            item.reasons = replacement.reasons
            item.featured_blocked = replacement.featured_blocked
            item.max_auto_rating = replacement.max_auto_rating
            item.rule_ids = replacement.rule_ids
            item.rule_version = replacement.rule_version
            item.content_type = replacement.content_type
            item.relative_exposure_ev = replacement.relative_exposure_ev
            item.scene_protected = replacement.scene_protected
            item.coverage_protected = replacement.coverage_protected
    return sum(item.suggested_rating > 0 for item in working)
