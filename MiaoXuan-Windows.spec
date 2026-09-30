# -*- mode: python ; coding: utf-8 -*-
"""Authoritative PyInstaller recipe for the Tk-based MiaoXuan Windows app."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files
from event_cull.bundled_inference import build_asset_tuples


datas = [
    ("event_cull/models/emotion-ferplus-8.onnx", "event_cull/models"),
    ("event_cull/models/mobilenet_v3_small_visual.onnx", "event_cull/models"),
    ("event_cull/data/wedding_must_keep_shots.csv", "event_cull/data"),
    ("docs/wedding-content-selection-policy-v2-240.md", "docs"),
    ("event_cull/data/general_must_keep_shots.csv", "event_cull/data"),
    ("event_cull/data/other_must_keep_shots.csv", "event_cull/data"),
    ("assets/branding/miaoxuan-logo.png", "assets/branding"),
    ("assets/models/face_landmarker.task", "models"),
    ("assets/models/pose_landmarker_lite.task", "models"),
]
exiftool_root = Path("event_cull/vendor/exiftool-windows")
if not (exiftool_root / "exiftool.exe").is_file() or not (exiftool_root / "exiftool_files").is_dir():
    raise SystemExit("Run scripts/prepare_windows_exiftool.py before packaging")
datas.extend((str(path), str(path.parent)) for path in exiftool_root.rglob("*") if path.is_file())

offline_datas, offline_binaries = build_asset_tuples()
datas += offline_datas
binaries = offline_binaries
hiddenimports = ["keyring.backends.Windows"]
datas += collect_data_files("cv2")
for package in ("tkinterdnd2", "cryptography", "keyring", "qrcode", "pillow_heif", "mediapipe"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    ["event_cull/app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torchvision", "onnx", "onnxscript", "AppKit", "Foundation", "Security", "objc"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="秒选",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon="build/miaoxuan.ico",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="秒选-Windows")
