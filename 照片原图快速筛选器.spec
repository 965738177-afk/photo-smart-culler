# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all
from pathlib import Path
import os
import re

version_source = Path('event_cull/build_info.py').read_text(encoding='utf-8')
version_match = re.search(r'APP_VERSION\s*=\s*["\']([^"\']+)', version_source)
if version_match is None:
    raise SystemExit('Could not read APP_VERSION from event_cull/build_info.py')
APP_VERSION = version_match.group(1)

datas = [
    ('event_cull/models/emotion-ferplus-8.onnx', 'event_cull/models'),
    ('event_cull/data/wedding_must_keep_shots.csv', 'event_cull/data'),
    ('docs/wedding-content-selection-policy-v2-240.md', 'docs'),
    ('event_cull/data/general_must_keep_shots.csv', 'event_cull/data'),
    ('event_cull/data/other_must_keep_shots.csv', 'event_cull/data'),
    ('assets/branding/miaoxuan-logo.png', 'assets/branding'),
    ('assets/models/face_landmarker.task', 'models'),
    ('assets/models/pose_landmarker_lite.task', 'models'),
]
if os.environ.get('MIAOXUAN_INTERNAL_BUILD') == '1':
    datas.append(('event_cull/internal_build_marker.txt', 'event_cull'))
for exiftool_file in Path('event_cull/vendor/exiftool').rglob('*'):
    if exiftool_file.is_file():
        datas.append((str(exiftool_file), str(exiftool_file.parent)))
# The optional 3.3 GB Qwen runtime has no demonstrated selection-quality gain.
# Keep it out of the standard fast release; the base image and metadata models
# listed above remain bundled.
binaries = []
hiddenimports = []
datas += collect_data_files('cv2')
tmp_ret = collect_all('tkinterdnd2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Personal calibration uses an inference-only ONNX model. Avoid collecting the
# complete PyTorch training runtime, which previously added hundreds of MB.
for package in ('cryptography', 'keyring', 'qrcode', 'pillow_heif', 'mediapipe'):
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Keyring discovers the native macOS backend through package metadata, so make
# the dynamic import explicit for a frozen application.
hiddenimports += ['keyring.backends.macOS', 'Foundation', 'Security', 'objc']

deep_model = Path('event_cull/models/mobilenet_v3_small_visual.onnx')
if not deep_model.is_file():
    raise SystemExit(
        'Missing MobileNet ONNX model. Run scripts/export_calibration_model.py before packaging.'
    )
datas += [(str(deep_model), 'event_cull/models')]

a = Analysis(
    ['event_cull/app.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'onnx', 'onnxscript'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='照片智能筛选器',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='照片智能筛选器',
)
app = BUNDLE(
    coll,
    name='照片智能筛选器.app',
    icon=None,
    bundle_identifier='online.lhxx.miaoxuan',
    info_plist={
        'CFBundleDisplayName': '秒选 — 照片智能筛选器',
        'CFBundleName': '秒选',
        'CFBundleShortVersionString': APP_VERSION,
        'CFBundleVersion': APP_VERSION,
        'CFBundleDevelopmentRegion': 'zh_CN',
        'CFBundleLocalizations': ['zh_CN', 'en'],
    },
)
