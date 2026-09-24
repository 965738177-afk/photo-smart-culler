"""Windows-specific branches are exercised even on the macOS developer host."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from event_cull import platform_paths, xmp
from event_cull.app import EventCullApp
from event_cull.licensing import LicenseError, LicenseService


def test_windows_application_paths_use_local_app_data(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert platform_paths.application_data_dir() == tmp_path / "EventCull"
    assert platform_paths.cache_dir() == tmp_path / "EventCull" / "Cache"
    assert platform_paths.device_label() == "Windows 电脑"


def test_windows_shortcuts_match_menu_actions(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    bindings = {}
    app = object.__new__(EventCullApp)
    app.bind_all = lambda sequence, callback: bindings.setdefault(sequence, callback)
    EventCullApp._bind_menu_shortcuts(app)
    assert set(bindings) == {
        "<Control-Key-1>", "<Control-Key-2>", "<Control-Key-3>",
        "<Control-comma>", "<Control-Return>", "<Control-Key-p>",
        "<Control-period>", "<Control-Shift-Key-W>",
    }


def test_windows_editor_guard_detects_lightroom(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='"Lightroom.exe","1324","Console","1","456,000 K"\n')

    monkeypatch.setattr("event_cull.app.subprocess.run", fake_run)
    assert EventCullApp._metadata_editors_running() == ("Lightroom Classic",)
    assert calls == [["tasklist", "/fo", "csv", "/nh"]]


def test_windows_exiftool_uses_utf8_argument_file(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    root = tmp_path / "exiftool-windows"
    root.mkdir()
    (root / "exiftool.exe").touch()
    (root / "exiftool_files").mkdir()
    monkeypatch.setattr(xmp, "_bundled_exiftool_root", lambda: root)
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0, stdout="3\n", stderr="")

    monkeypatch.setattr(xmp.subprocess, "run", fake_run)
    assert xmp._run_bundled_exiftool("-XMP-xmp:Rating", "C:\\照片\\测试.HEIC") == "3"
    assert captured["command"] == [str(root / "exiftool.exe"), "-charset", "filename=UTF8", "-@", "-"]
    assert captured["input"] == "-XMP-xmp:Rating\nC:\\照片\\测试.HEIC\n"
    assert captured["encoding"] == "utf-8"


def test_windows_exiftool_missing_companion_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(xmp, "_bundled_exiftool_root", lambda: tmp_path)
    (tmp_path / "exiftool.exe").touch()
    with pytest.raises(RuntimeError, match="元数据组件不可用"):
        xmp._run_bundled_exiftool("-ver")


def test_windows_keyring_must_be_native(monkeypatch):
    import keyring

    monkeypatch.setattr("event_cull.licensing.platform.system", lambda: "Windows")
    native = type("WinVaultKeyring", (), {"__module__": "keyring.backends.Windows"})()
    monkeypatch.setattr(keyring, "get_keyring", lambda: native)
    assert LicenseService._keyring()[0] is keyring
    foreign = type("PlaintextKeyring", (), {"__module__": "keyrings.alt.file"})()
    monkeypatch.setattr(keyring, "get_keyring", lambda: foreign)
    with pytest.raises(LicenseError, match="安全凭据"):
        LicenseService._keyring()


@pytest.mark.skipif(sys.platform != "win32", reason="Requires native Windows ExifTool")
def test_windows_embedded_ratings_round_trip_with_chinese_path(tmp_path):
    from PIL import Image
    from pillow_heif import register_heif_opener

    register_heif_opener()
    folder = tmp_path / "中文照片"
    folder.mkdir()
    image = Image.new("RGB", (32, 32), (80, 130, 170))
    for extension in ("jpg", "heic"):
        photo = folder / f"测试照片.{extension}"
        image.save(photo)
        original = photo.read_bytes()
        assert xmp.write_rating(photo, 3) == "embedded-updated"
        assert xmp._read_embedded_rating(photo) == 3
        assert xmp.restore_managed_rating(photo) == "restored"
        assert photo.read_bytes() == original
