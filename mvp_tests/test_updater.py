from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from event_cull import updater


def test_version_key_accepts_semver_and_rejects_other_labels() -> None:
    assert updater.version_key("v2.0.1") == (2, 0, 1)
    with pytest.raises(ValueError):
        updater.version_key("latest")


def test_manifest_requires_https_url_and_checksum() -> None:
    good = {
        "latest_version": "2.0.1",
        "download_url": "https://example.com/app.zip",
        "sha256": "a" * 64,
        "notes": "Fixes",
    }
    assert updater._update_from_manifest(good) is not None
    good["download_url"] = "http://example.com/app.zip"
    assert updater._update_from_manifest(good) is None


def test_download_update_verifies_sha256(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    body = b"signed release bytes"

    class Response:
        def read(self, _size: int = -1) -> bytes:
            nonlocal body
            chunk, body = body, b""
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(updater.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    update = updater.UpdateInfo("2.0.1", "https://example.com/app.zip", hashlib.sha256(b"signed release bytes").hexdigest())
    result = updater.download_update(update, tmp_path)
    assert result.read_bytes() == b"signed release bytes"
