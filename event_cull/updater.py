"""Safe, platform-specific manifest-driven desktop update discovery.

The client only downloads a release after the user accepts an in-app prompt.
Every downloadable release must advertise a SHA-256 digest; a mismatched or
missing digest leaves the downloaded file quarantined and never opens it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from event_cull.platform_paths import downloads_dir


UPDATE_MANIFEST_URL = os.environ.get(
    "MIAOXUAN_UPDATE_MANIFEST_URL",
    "https://cull.lhxx.online/api/v1/app-updates/windows"
    if sys.platform == "win32" else
    "https://cull.lhxx.online/api/v1/app-updates/macos",
)
_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    download_url: str
    sha256: str
    notes: str = ""


def version_key(value: str) -> tuple[int, int, int]:
    match = _VERSION.fullmatch(value.strip())
    if match is None:
        raise ValueError("Invalid semantic version")
    return tuple(int(part) for part in match.groups())


def _update_from_manifest(payload: object) -> UpdateInfo | None:
    if not isinstance(payload, dict):
        return None
    version = payload.get("latest_version")
    url = payload.get("download_url")
    digest = payload.get("sha256")
    notes = payload.get("notes", "")
    if not all(isinstance(value, str) for value in (version, url, digest, notes)):
        return None
    if urllib.parse.urlparse(url).scheme != "https":
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        return None
    version_key(version)
    return UpdateInfo(version=version.lstrip("v"), download_url=url, sha256=digest.lower(), notes=notes)


def check_for_update(current_version: str, endpoint: str = UPDATE_MANIFEST_URL) -> UpdateInfo | None:
    """Return a verified newer manifest, or None for all offline/error states."""
    try:
        request = urllib.request.Request(
            endpoint,
            headers={"Accept": "application/json", "User-Agent": f"MiaoXuan/{current_version}"},
        )
        with urllib.request.urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
        update = _update_from_manifest(payload)
        if update is None or version_key(update.version) <= version_key(current_version):
            return None
        return update
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
        return None


def download_update(update: UpdateInfo, downloads: Path | None = None) -> Path:
    """Download the opted-in release and verify its digest before returning it."""
    target_directory = downloads or downloads_dir()
    target_directory.mkdir(parents=True, exist_ok=True)
    name = Path(urllib.parse.urlparse(update.download_url).path).name or f"照片智能筛选器-{update.version}.zip"
    destination = target_directory / name
    with tempfile.NamedTemporaryFile(prefix=f".{name}.", suffix=".download", dir=target_directory, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            request = urllib.request.Request(update.download_url, headers={"User-Agent": "MiaoXuan-updater"})
            with urllib.request.urlopen(request, timeout=30) as response:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if digest != update.sha256:
                raise ValueError("更新包校验失败，未打开下载文件。")
            os.replace(temporary, destination)
            return destination
        finally:
            temporary.unlink(missing_ok=True)
