"""Shared original-photo scanning for imports and local preference training."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from event_cull.engine import IMAGE_EXTENSIONS
from event_cull.export_selected import EXPORT_MARKER


_CURATED_FOLDER_NAMES = {"选", "已选", "selected"}


def _is_curated_folder(name: str) -> bool:
    """Recognize delivery folders, without excluding ordinary RAW folders."""
    return (name.casefold() in _CURATED_FOLDER_NAMES
            or name.endswith(("原图选好", "原图导出")))


def scan_source_images(
    folders: Iterable[Path], *, excluded_folders: Iterable[Path] = (),recursive: bool = True
) -> list[Path]:
    """Find originals without re-importing app exports or nested selections.

    A selected folder can still be imported directly; it is only skipped when
    it is nested below a broader original-source folder.
    """
    excluded = {folder.resolve() for folder in excluded_folders}
    files: list[Path] = []
    visited: set[Path] = set()
    for folder in folders:
        root = folder.resolve()
        if root in excluded:
            continue
        try:
            for parent, directories, filenames in os.walk(root):
                # History may list both a root and its children. Scan each
                # physical directory once so calibration cannot double-count.
                directory = Path(parent).resolve()
                if directory in visited:
                    directories[:] = []
                    continue
                visited.add(directory)
                directories[:] = [
                    name
                    for name in directories
                    if not name.startswith(".")
                    and (Path(parent) / name).resolve() not in excluded
                    and not _is_curated_folder(name)
                    and not (Path(parent) / name / EXPORT_MARKER).is_file()
                ]
                if not recursive:
                    directories[:] = []
                files.extend(
                    Path(parent, name)
                    for name in sorted(filenames, key=str.casefold)
                    if not name.startswith(".") and Path(name).suffix.lower() in IMAGE_EXTENSIONS
                )
        except OSError:
            continue
    return sorted(files, key=lambda path: (str(path.parent).casefold(), path.name.casefold()))
