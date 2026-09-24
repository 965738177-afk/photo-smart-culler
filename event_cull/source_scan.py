"""Shared original-photo scanning for imports and local preference training."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from event_cull.engine import IMAGE_EXTENSIONS
from event_cull.export_selected import EXPORT_MARKER


_CURATED_FOLDER_NAMES = {"选", "已选", "selected"}


def scan_source_images(
    folders: Iterable[Path], *, excluded_folders: Iterable[Path] = ()
) -> list[Path]:
    """Find originals without re-importing app exports or nested selections.

    A selected folder can still be imported directly; it is only skipped when
    it is nested below a broader original-source folder.
    """
    excluded = {folder.resolve() for folder in excluded_folders}
    files: list[Path] = []
    for folder in folders:
        root = folder.resolve()
        if root in excluded:
            continue
        try:
            for parent, directories, filenames in os.walk(root):
                directories[:] = [
                    name
                    for name in directories
                    if not name.startswith(".")
                    and (Path(parent) / name).resolve() not in excluded
                    and name.casefold() not in _CURATED_FOLDER_NAMES
                    and not (Path(parent) / name / EXPORT_MARKER).is_file()
                ]
                files.extend(
                    Path(parent, name)
                    for name in sorted(filenames, key=str.casefold)
                    if Path(name).suffix.lower() in IMAGE_EXTENSIONS
                )
        except OSError:
            continue
    return sorted(files, key=lambda path: (str(path.parent).casefold(), path.name.casefold()))
