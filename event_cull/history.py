"""Small, local-only index for completed and in-progress culling tasks."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

from event_cull.platform_paths import application_data_dir


HISTORY_VERSION = 1
MAX_HISTORY_RECORDS = 200


def application_support_path() -> Path:
    return application_data_dir()


def history_path() -> Path:
    return application_support_path() / "task-history.json"


def task_name(paths: Iterable[Path], created_at: datetime) -> str:
    """Create a readable task title from its timestamp and common source folder."""
    # The caller already validates imports.  Do not call ``is_file`` here:
    # keeping this function path-only also makes a task name stable if a
    # source volume is briefly unavailable while the history is rendered.
    source_paths = list(paths)
    folders = [str(path.parent.resolve()) for path in source_paths]
    if not folders:
        label = "未命名任务"
    else:
        try:
            label = Path(os.path.commonpath(folders)).name or "照片任务"
        except ValueError:
            label = "多文件夹"
    return f"{created_at:%Y-%m-%d %H:%M} · {label} · {len(source_paths):,} 张"


def load_history(path: Path | None = None) -> list[dict[str, object]]:
    source = path or history_path()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = payload.get("tasks", []) if isinstance(payload, dict) else []
        return [record for record in records if isinstance(record, dict)]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []


def upsert_task(record: dict[str, object], path: Path | None = None) -> Path:
    """Atomically save one task summary while retaining recent local history."""
    destination = path or history_path()
    task_id = str(record.get("id") or "")
    records = [item for item in load_history(destination) if item.get("id") != task_id]
    records.insert(0, record)
    payload = {"version": HISTORY_VERSION, "tasks": records[:MAX_HISTORY_RECORDS]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
