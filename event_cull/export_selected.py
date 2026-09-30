"""Copy selected original photos into a new, collision-free delivery folder."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from event_cull.engine import PhotoResult
from event_cull.rating_policy import RATING_POLICY_VERSION, RETAINED_RATINGS


EXPORT_MARKER = ".event-cull-export.json"


class ExportCancelled(Exception):
    """The user stopped a copy before its new folders were completed."""


@dataclass(frozen=True)
class ExportSelection:
    photo_count: int
    review_count: int  # Legacy field name: count of retained one-star originals.
    files_by_folder: dict[Path, tuple[Path, ...]]

    @property
    def file_count(self) -> int:
        return sum(len(files) for files in self.files_by_folder.values())


@dataclass(frozen=True)
class ExportResult:
    folders: tuple[Path, ...]
    photo_count: int
    file_count: int


def collect_selected_files(
    results: Sequence[PhotoResult], included_tiers: frozenset[int] = RETAINED_RATINGS
) -> ExportSelection:
    """Export only the photographer's chosen nonzero rating tiers."""
    if not included_tiers or not included_tiers <= RETAINED_RATINGS:
        raise ValueError("请至少选择一种有效星级")
    grouped: dict[Path, dict[Path, None]] = {}
    photo_count = review_count = 0
    for item in results:
        if item.suggested_rating not in included_tiers:
            continue
        photo_count += 1
        review_count += item.suggested_rating == 1
        for photo in item.pair_paths or [item.path]:
            if not photo.is_file():
                raise FileNotFoundError(f"找不到待导出的原片：{photo}")
            source_folder = photo.parent.resolve()
            files = grouped.setdefault(source_folder, {})
            files[photo.resolve()] = None
            sidecar = photo.with_suffix(".xmp")
            if sidecar.is_file():
                files[sidecar.resolve()] = None
    return ExportSelection(
        photo_count,
        review_count,
        {
            folder: tuple(sorted(files, key=lambda path: path.name.casefold()))
            for folder, files in grouped.items()
        },
    )


def _reserve_export_folder(parent: Path) -> Path:
    """Never merge into or overwrite a prior 秒选 export."""
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    for number in range(1, 10000):
        name = "秒选" if number == 1 else f"秒选-{number}"
        candidate = parent / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"无法在 {parent} 创建新的秒选文件夹")


def _source_labels(folders: tuple[Path, ...]) -> dict[Path, str]:
    names = [folder.name.casefold() for folder in folders]
    return {
        folder: (
            folder.name
            if names.count(folder.name.casefold()) == 1
            else f"{folder.name}-{hashlib.sha256(str(folder).encode()).hexdigest()[:8]}"
        )
        for folder in folders
    }


def export_selected_files(
    selection: ExportSelection,
    *,
    custom_parent: Path | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> ExportResult:
    """Copy selected originals, paired files and sidecars; never alter sources.

    Each output folder is newly reserved. On cancellation or failure, only
    folders created by this call are removed, so incomplete exports cannot be
    mistaken for a completed delivery.
    """
    if not selection.photo_count or not selection.file_count:
        raise ValueError("没有可导出的已选图片")
    if custom_parent is not None:
        custom_parent = custom_parent.resolve()
        if not custom_parent.is_dir():
            raise NotADirectoryError(custom_parent)

    source_folders = tuple(sorted(selection.files_by_folder, key=lambda path: str(path).casefold()))
    destinations: list[Path] = []
    completed = 0
    try:
        if custom_parent is None:
            roots = {}
            for folder in source_folders:
                roots[folder] = _reserve_export_folder(folder)
                destinations.append(roots[folder])
            subfolders = {folder: roots[folder] for folder in source_folders}
        else:
            root = _reserve_export_folder(custom_parent)
            destinations.append(root)
            if len(source_folders) == 1:
                subfolders = {source_folders[0]: root}
            else:
                labels = _source_labels(source_folders)
                subfolders = {folder: root / labels[folder] for folder in source_folders}
                for folder in subfolders.values():
                    folder.mkdir()

        for root in destinations:
            (root / EXPORT_MARKER).write_text(
                json.dumps({"version": 1, "complete": False}, ensure_ascii=False),
                encoding="utf-8",
            )
        for source_folder in source_folders:
            output_folder = subfolders[source_folder]
            for source in selection.files_by_folder[source_folder]:
                if cancel_event is not None and cancel_event.is_set():
                    raise ExportCancelled()
                target = output_folder / source.name
                # Exclusive creation protects against filename collisions on
                # case-insensitive destination volumes as well as source races.
                with source.open("rb") as input_file, target.open("xb") as output_file:
                    while chunk := input_file.read(1024 * 1024):
                        if cancel_event is not None and cancel_event.is_set():
                            raise ExportCancelled()
                        output_file.write(chunk)
                shutil.copystat(source, target)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, selection.file_count, source)

        for root in destinations:
            (root / EXPORT_MARKER).write_text(
                json.dumps({
                    "version": 1,
                    "rating_policy": RATING_POLICY_VERSION,
                    "complete": True,
                    "selected_photos": selection.photo_count,
                    "copied_files": selection.file_count,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        return ExportResult(tuple(destinations), selection.photo_count, selection.file_count)
    except Exception:
        for root in destinations:
            shutil.rmtree(root, ignore_errors=True)
        raise
