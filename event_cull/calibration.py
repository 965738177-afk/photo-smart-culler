"""Local, user-specific visual calibration for featured culling picks.

The profile stores only a small classifier and its normalization statistics.
It never stores source pixels and never writes photo metadata.
"""

from __future__ import annotations

import csv
import json
import os
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from event_cull.engine import IMAGE_EXTENSIONS, PhotoResult, _load_preview


PROFILE_VERSION = 1
PROFILE_FILENAME = "personal-calibration-v1.json"
MIN_VALIDATION_AUC = 0.60
_RATING_PATTERN = re.compile(rb"xmp:Rating=[\"'](-?\d+)[\"']")


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
        }

    @classmethod
    def from_dict(cls, value: object) -> "CalibrationProfile":
        if not isinstance(value, dict) or value.get("version") != PROFILE_VERSION:
            raise CalibrationError("个人校准文件版本不兼容。")
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
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError("个人校准文件不完整。") from exc
        if not mean or len(mean) != len(scale) or len(mean) != len(weights):
            raise CalibrationError("个人校准特征维度无效。")
        return profile

    def probability(self, image: np.ndarray) -> float:
        values = _visual_descriptor(image)
        if len(values) != len(self.weights):
            raise CalibrationError("个人校准特征与当前版本不匹配。请重新导入一次校准结果。")
        normalized = (values - np.asarray(self.mean)) / np.asarray(self.scale)
        return _sigmoid(float(normalized @ np.asarray(self.weights) + self.bias))


def default_profile_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "EventCull" / "calibration" / PROFILE_FILENAME


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
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(profile.to_dict(), ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, destination)
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


def exported_ratings(folder: Path) -> dict[str, int]:
    """Return unambiguous stem-to-rating labels from Lightroom JPEG exports."""
    candidates: dict[str, list[int]] = {}
    try:
        files = folder.rglob("*")
    except OSError as exc:
        raise CalibrationError("无法读取 Lightroom 导出文件夹。") from exc
    for path in files:
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        rating = read_xmp_rating(path)
        if rating is not None:
            candidates.setdefault(path.stem.casefold(), []).append(rating)
    return {
        stem: values[0]
        for stem, values in candidates.items()
        if len(set(values)) == 1
    }


def _log_rows(log_path: Path) -> dict[str, tuple[str, int]]:
    """Map unique original names to their historical burst group and rating."""
    try:
        with log_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise CalibrationError("无法读取旧版筛选日志 CSV。") from exc
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


def _source_previews(folders: Iterable[Path], allowed_stems: set[str]) -> dict[str, Path]:
    """Find one readable representative per old log record, preferring JPEG."""
    choices: dict[str, Path] = {}
    ambiguous: set[str] = set()
    for folder in folders:
        try:
            paths = folder.rglob("*")
            for path in paths:
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
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
        except OSError:
            continue
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


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def _fit_classifier(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-4)
    values = (features - mean) / scale
    weights = np.zeros(values.shape[1], dtype=np.float64)
    positive = max(1, int(labels.sum()))
    negative = max(1, len(labels) - positive)
    sample_weights = np.where(labels > 0, len(labels) / (2 * positive), len(labels) / (2 * negative))
    bias = 0.0
    for _ in range(360):
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(values @ weights + bias, -35.0, 35.0)))
        error = (probabilities - labels) * sample_weights
        weights -= 0.16 * ((values.T @ error) / len(labels) + 0.012 * weights)
        bias -= 0.16 * float(error.mean())
    return mean, scale, weights, bias


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
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
    raise CalibrationError("最高星级样本分布过于集中，无法进行按连拍组隔离的校验。")


def train_profile(
    samples: list[tuple[np.ndarray, int, str]],
    featured_rating: int,
) -> CalibrationProfile:
    if len(samples) < 80:
        raise CalibrationError("可用于校准的原图不足 80 张。")
    labels = np.asarray([int(rating == featured_rating) for _feature, rating, _group in samples], dtype=np.int8)
    positive_count = int(labels.sum())
    if positive_count < 12 or positive_count == len(labels):
        raise CalibrationError("最高星级样本不足，至少需要 12 张。")
    features = np.vstack([feature for feature, _rating, _group in samples]).astype(np.float64)
    groups = [group for _feature, _rating, group in samples]
    held_out = _balanced_group_holdout(groups, labels)
    mean, scale, weights, bias = _fit_classifier(features[~held_out], labels[~held_out])
    validation_values = (features[held_out] - mean) / scale
    validation = _roc_auc(labels[held_out], validation_values @ weights + bias)
    if validation < MIN_VALIDATION_AUC:
        raise CalibrationError(
            f"这批资料的分组验证区分度只有 {validation:.0%}，不足以安全启用个人校准。"
        )
    mean, scale, weights, bias = _fit_classifier(features, labels)
    return CalibrationProfile(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in weights),
        bias=float(bias),
        featured_rating=featured_rating,
        training_examples=len(samples),
        positive_examples=positive_count,
        validation_auc=validation,
    )


def build_profile(
    log_path: Path,
    exported_folder: Path,
    original_folders: list[Path],
    *,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> CalibrationProfile:
    """Build a profile from a prior app CSV, Lightroom exports, and originals."""
    rows = _log_rows(log_path)
    ratings = exported_ratings(exported_folder)
    if not ratings:
        raise CalibrationError("没有从 Lightroom 导出 JPG 读到嵌入星级。")
    featured_rating = max(ratings.values())
    sources = _source_previews(original_folders, set(rows))
    samples: list[tuple[np.ndarray, int, str]] = []
    items = sorted(sources.items())
    for index, (stem, path) in enumerate(items, start=1):
        group, _old_rating = rows[stem]
        try:
            feature = _visual_descriptor(_load_preview(path))
        except Exception:
            continue
        samples.append((feature, ratings.get(stem, 0), group))
        if progress_callback:
            progress_callback(index, len(items), path)
    return train_profile(samples, featured_rating)


def apply_profile(
    results: list[PhotoResult],
    profile: CalibrationProfile,
    *,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> int:
    """Re-rank only the featured tier; selection coverage remains untouched."""
    existing_featured = sum(item.suggested_rating == 3 for item in results)
    candidates = [item for item in results if item.suggested_rating > 0]
    if not candidates or existing_featured == 0:
        return 0
    probabilities: dict[Path, float] = {}
    for index, item in enumerate(candidates, start=1):
        visual = next((path for path in item.pair_paths if path.suffix.lower() in {".jpg", ".jpeg"}), item.path)
        try:
            probability = profile.probability(_load_preview(visual))
        except Exception:
            continue
        item.calibration_confidence = probability
        probabilities[item.path] = probability
        if progress_callback:
            progress_callback(index, len(candidates), item.path)
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
    for item in results:
        if item.suggested_rating == 3 and item.path not in featured_paths:
            item.suggested_rating = 2
            item.reasons = [reason for reason in item.reasons if reason not in {"本组最佳", "Best in group"}]
            item.reasons.append("个人校准后保留为优质备用" if profile.featured_rating else "Personal calibration kept this as a backup")
        elif item.path in featured_paths:
            item.suggested_rating = 3
            item.reasons = [reason for reason in item.reasons if reason not in {"本组最佳", "Best in group"}]
            confidence = probabilities[item.path]
            item.reasons.append(f"个人校准 3★ 置信度 {confidence:.0%}")
    return len(featured_paths)
