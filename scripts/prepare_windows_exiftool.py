"""Install the pinned official ExifTool Windows binary for packaging."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

VERSION = "13.59"
SHA256 = "44b512b25af500724ba579d0a53c8fc5851628b692dd5e5d94ae4a15c2cba9ec"
URL = f"https://sourceforge.net/projects/exiftool/files/exiftool-{VERSION}_64.zip/download"
DESTINATION = Path("event_cull/vendor/exiftool-windows")


def main() -> None:
    archive = Path("build") / f"exiftool-{VERSION}_64.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        urllib.request.urlretrieve(URL, archive)
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != SHA256:
        raise RuntimeError(f"ExifTool download checksum mismatch: {actual}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        prefix = f"exiftool-{VERSION}_64/"
        for member in source.infolist():
            relative = Path(member.filename.removeprefix(prefix))
            if not member.filename.startswith(prefix) or relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe ExifTool archive entry: {member.filename}")
            if member.is_dir() or not relative.parts:
                continue
            if relative.name == "exiftool(-k).exe":
                relative = Path("exiftool.exe")
            if relative.parts[0] not in {"exiftool_files", "exiftool.exe", "README.txt"}:
                continue
            target = DESTINATION / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    if not (DESTINATION / "exiftool.exe").is_file() or not (DESTINATION / "exiftool_files").is_dir():
        raise RuntimeError("Incomplete Windows ExifTool distribution")
    print(f"Prepared ExifTool {VERSION} in {DESTINATION}")


if __name__ == "__main__":
    main()
