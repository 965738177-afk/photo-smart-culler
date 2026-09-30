"""One owner for user-writable paths in both desktop packages."""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def application_data_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "EventCull"
    return Path.home() / "Library" / "Application Support" / "EventCull"


def cache_dir() -> Path:
    if sys.platform == "win32":
        return application_data_dir() / "Cache"
    return Path.home() / "Library" / "Caches" / "照片原图快速筛选器"


def downloads_dir() -> Path:
    return Path.home() / "Downloads"


def device_label() -> str:
    return "Windows 电脑" if sys.platform == "win32" else "Mac"


def hidden_subprocess_options() -> dict[str, int]:
    """Console helpers must not flash a window in the frozen Windows GUI."""
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)} if sys.platform == "win32" else {}
