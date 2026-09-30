"""Build-time only: fetch pinned, verified inference assets for offline users.

Downloads come from the runtime's official GitHub release and Qwen's official
ModelScope repository. End users do not run this script or install another app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "event_cull/vendor/content"
RUNTIME_VERSION = "b11242"
RUNTIMES = {
    "darwin-arm64": ("llama-b11242-bin-macos-arm64.tar.gz", "76142db728d264a56da04592d69e4f3bdf798049183ca7c9b0511fbb561458e8"),
    "win32-amd64": ("llama-b11242-bin-win-cpu-x64.zip", "851fb7a4070a14789e9b4f617d0f1a58369e295b6466e518441d5067327a5320"),
}
MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
MODEL_REVISION = "625828cbd3489786522366e19e22b6f12278e41a"
MODEL_FILES = {
    "Qwen3VL-4B-Instruct-Q4_K_M.gguf": (2497281664, "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a"),
    "mmproj-Qwen3VL-4B-Instruct-F16.gguf": (836180256, "256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected: str, size: int | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (size is None or target.stat().st_size == size) and sha256(target) == expected:
        print(f"Verified existing {target.name}", flush=True)
        return
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.is_file() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                with partial.open("ab" if offset else "wb") as stream:
                    last_report = time.monotonic()
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_report > 15:
                            print(f"{target.name}: {offset / 1e9:.2f} GB", flush=True)
                            last_report = time.monotonic()
            if size is not None and partial.stat().st_size != size:
                raise ValueError("Downloaded file size mismatch")
            if sha256(partial) != expected:
                partial.unlink()
                raise ValueError("Downloaded file checksum mismatch")
            os.replace(partial, target)
            print(f"Verified {target.name}", flush=True)
            return
        except Exception as exc:
            print(f"Download retry {attempt + 1}: {type(exc).__name__}", flush=True)
            if attempt == 3:
                raise


def prepare(platform_key: str) -> None:
    name, checksum = RUNTIMES[platform_key]
    archive = ROOT / "build/content-downloads" / name
    download(f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_VERSION}/{name}", archive, checksum)
    runtime = ASSETS / "runtime" / platform_key
    runtime.mkdir(parents=True, exist_ok=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as packed:
            for member in packed.infolist():
                resolved = (runtime / member.filename).resolve()
                if not resolved.is_relative_to(runtime.resolve()):
                    raise ValueError("Unsafe runtime archive entry")
            packed.extractall(runtime)
    else:
        with tarfile.open(archive) as packed:
            packed.extractall(runtime, filter="data")
    executable_name = "llama-server.exe" if platform_key.startswith("win") else "llama-server"
    executable = next(runtime.rglob(executable_name))
    if platform_key.startswith("darwin"):
        executable.chmod(executable.stat().st_mode | 0o111)
    for filename, (size, checksum) in MODEL_FILES.items():
        url = f"https://modelscope.cn/api/v1/models/{MODEL_REPO}/repo?Revision={MODEL_REVISION}&FilePath={filename}"
        download(url, ASSETS / "models" / filename, checksum, size)
    notices = ASSETS / "licenses"
    notices.mkdir(parents=True, exist_ok=True)
    for url, filename in (
        (f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{RUNTIME_VERSION}/LICENSE", "llama.cpp-MIT.txt"),
        ("https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/LICENSE", "Qwen3-VL-Apache-2.0.txt"),
    ):
        with urllib.request.urlopen(url, timeout=30) as response:
            (notices / filename).write_bytes(response.read())
    manifest_path = ASSETS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest.update(version=1, model_id="Qwen3-VL-4B-Instruct-Q4_K_M", runtime_version=RUNTIME_VERSION,
                    model_repository=MODEL_REPO, model_revision=MODEL_REVISION,
                    weights={filename: {"size": size, "sha256": digest} for filename, (size, digest) in MODEL_FILES.items()})
    manifest.setdefault("executables", {})[platform_key] = str(executable.relative_to(ASSETS)).replace("\\", "/")
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"Offline model assets ready: {platform_key}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=tuple(RUNTIMES))
    prepare(parser.parse_args().platform)
