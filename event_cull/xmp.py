"""Lightroom-compatible rating writes without ever touching RAW pixels.

RAW files use adjacent XMP sidecars, which is Lightroom's normal workflow.
JPEG, TIFF, DNG and Apple HEIC/HEIF ratings are stored in embedded XMP,
as Lightroom expects for these formats. Before a metadata-only update, the
complete original file is copied into the existing
``.event-cull-backups`` safety folder.
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

XMP = "adobe:ns:meta/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
X = "adobe:ns:meta/"

ET.register_namespace("x", X)
ET.register_namespace("rdf", RDF)
ET.register_namespace("xmp", "http://ns.adobe.com/xap/1.0/")
EVENT_CULL = "urn:event-cull:metadata:1.0/"
ET.register_namespace("eventcull", EVENT_CULL)

RATING_KEY = "{http://ns.adobe.com/xap/1.0/}Rating"
MANAGED_KEY = f"{{{EVENT_CULL}}}ManagedRating"
MANAGED_VALUE_KEY = f"{{{EVENT_CULL}}}ManagedRatingValue"
CREATED_SIDECAR_KEY = f"{{{EVENT_CULL}}}CreatedSidecar"
HEIC_EXTENSIONS = {".heic", ".heif"}
HEIC_BACKUP_MANIFEST = ".event-cull-heic-metadata.json"
EMBEDDED_EXTENSIONS = HEIC_EXTENSIONS | {".jpg", ".jpeg", ".tif", ".tiff", ".dng"}
EMBEDDED_BACKUP_MANIFEST = ".event-cull-embedded-metadata.json"
PAIRED_RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2"}


def _write_lightroom_xmp(tree: ET.ElementTree, sidecar: Path) -> None:
    """Write a complete Adobe-compatible XMP packet, not bare XML.

    Lightroom Classic is strict with some bare sidecar serializations. The
    xpacket wrapper, rdf:about and a plain standard namespace layout mirror
    the sidecars it writes itself.
    """
    root = tree.getroot()
    root.set(f"{{{X}}}xmptk", "Event Cull")
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=False).decode("utf-8")
    packet = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + payload
        + '\n<?xpacket end="w"?>\n'
    )
    temporary = sidecar.with_name(f".{sidecar.name}.event-cull-{uuid4().hex}.tmp")
    previous_mode = sidecar.stat().st_mode if sidecar.exists() else None
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(packet)
            stream.flush()
            os.fsync(stream.fileno())
        # Validate the complete packet before it can replace a good sidecar.
        ET.parse(temporary)
        if previous_mode is not None:
            temporary.chmod(previous_mode)
        os.replace(temporary, sidecar)
    finally:
        temporary.unlink(missing_ok=True)


def backup_existing_sidecars(photos: list[Path]) -> list[Path]:
    """Copy existing XMP files into timestamped per-folder backups."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_dirs: dict[Path, Path] = {}
    copied: list[Path] = []
    for photo in sorted(set(photos)):
        sidecar = sidecar_path(photo)
        if not sidecar.exists():
            continue
        backup_dir = backup_dirs.setdefault(
            sidecar.parent,
            sidecar.parent / ".event-cull-backups" / stamp,
        )
        backup_dir.mkdir(parents=True, exist_ok=True)
        destination = backup_dir / sidecar.name
        shutil.copy2(sidecar, destination)
        copied.append(destination)
    return copied


def sidecar_path(photo: Path) -> Path:
    """Return Lightroom's standard adjacent sidecar name: IMG_0001.xmp."""
    return photo.with_suffix(".xmp")


def _is_heic(photo: Path) -> bool:
    return photo.suffix.lower() in HEIC_EXTENSIONS


def _is_embedded(photo: Path) -> bool:
    return photo.suffix.lower() in EMBEDDED_EXTENSIONS


def _bundled_exiftool_root() -> Path:
    """Locate the separately executed ExifTool distribution.

    It is deliberately run through the system Perl interpreter instead of
    being imported into our Python process.  That keeps the GPL-licensed tool
    isolated as an accompanying command-line utility and lets it perform its
    well-tested ISO-BMFF metadata update without re-encoding HEIC pixels.
    """
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return bundle_root / "event_cull" / "vendor" / "exiftool"


def _run_bundled_exiftool(*arguments: str) -> str:
    root = _bundled_exiftool_root()
    script = root / "exiftool"
    library = root / "lib"
    perl = Path("/usr/bin/perl")
    if not script.is_file() or not library.is_dir() or not perl.is_file():
        raise RuntimeError("HEIC 元数据组件不可用；请重新安装秒选后再试")
    try:
        completed = subprocess.run(
            [str(perl), "-I", str(library), str(script), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:  # pragma: no cover - platform launch failure
        raise RuntimeError(f"无法启动 HEIC 元数据组件：{exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"HEIC 星级写入失败：{detail or 'ExifTool 未能完成更新'}")
    return completed.stdout.strip()


def _read_embedded_rating(photo: Path) -> int:
    value = _run_bundled_exiftool("-s3", "-XMP-xmp:Rating", str(photo)).strip()
    if not value:
        return 0
    try:
        rating = int(float(value.splitlines()[0].strip()))
    except ValueError as exc:
        raise RuntimeError(f"照片中的现有星级无法识别：{photo.name}") from exc
    if not 0 <= rating <= 5:
        raise RuntimeError(f"照片中的现有星级超出范围：{photo.name}")
    return rating


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.event-cull-{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _backup_embedded_before_write(photo: Path, previous_rating: int, rating: int) -> Path:
    """Create a recoverable full-file backup before an embedded XMP change."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = photo.parent / ".event-cull-backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / photo.name
    shutil.copy2(photo, backup)
    _write_json_atomically(
        backup_dir / EMBEDDED_BACKUP_MANIFEST,
        {
            "version": 1,
            "photo_name": photo.name,
            "previous_rating": previous_rating,
            "written_rating": rating,
            "original_size": photo.stat().st_size,
        },
    )
    return backup


def _latest_embedded_backup_record(photo: Path) -> tuple[Path, dict[str, object]] | None:
    root = photo.parent / ".event-cull-backups"
    if not root.is_dir():
        return None
    candidates = sorted(
        (*root.glob(f"*/{HEIC_BACKUP_MANIFEST}"), *root.glob(f"*/{EMBEDDED_BACKUP_MANIFEST}")),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for manifest in candidates:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("photo_name") != photo.name:
            continue
        backup = manifest.parent / photo.name
        if backup.is_file():
            return backup, payload
    return None


def _retire_obsolete_managed_sidecar(photo: Path) -> None:
    """Remove only our obsolete rating; preserve other sidecar metadata."""
    if photo.suffix.lower() in {".jpg", ".jpeg"} and any(
        sibling.stem.casefold() == photo.stem.casefold()
        and sibling.suffix.lower() in PAIRED_RAW_EXTENSIONS
        for sibling in photo.parent.iterdir()
    ):
        # The adjacent XMP may belong to a same-stem RAW, even when the RAW
        # was not among the photos selected for this particular write.
        return
    sidecar = sidecar_path(photo)
    if not sidecar.is_file():
        return
    try:
        tree = ET.parse(sidecar)
    except ET.ParseError:
        return
    desc = _description(tree)
    if desc.get(MANAGED_KEY) != "true":
        return
    recorded = desc.get(MANAGED_VALUE_KEY)
    if recorded is not None and recorded != desc.get(RATING_KEY):
        return
    backup_existing_sidecars([photo])
    own_keys = {RATING_KEY, MANAGED_KEY, MANAGED_VALUE_KEY, CREATED_SIDECAR_KEY}
    other_attributes = set(desc.attrib) - own_keys - {f"{{{RDF}}}about"}
    if not other_attributes and not list(desc):
        sidecar.unlink()
        return
    for key in own_keys:
        desc.attrib.pop(key, None)
    _write_lightroom_xmp(tree, sidecar)


def legacy_managed_embedded_photos(photos: list[Path]) -> list[Path]:
    """Find old managed sidecars for formats whose rating belongs in the file."""
    matches: list[Path] = []
    # A JPG paired with a camera RAW shares the RAW sidecar name.  That .xmp
    # belongs to the RAW and must never be migrated into, or removed for, JPG.
    paired_raw_stems = {
        (photo.parent.resolve(), photo.stem.casefold())
        for photo in photos
        if photo.suffix.lower() in PAIRED_RAW_EXTENSIONS
    }
    for photo in sorted(set(photos)):
        if not _is_embedded(photo):
            continue
        if photo.suffix.lower() in {".jpg", ".jpeg"} and (
            photo.parent.resolve(), photo.stem.casefold()
        ) in paired_raw_stems:
            continue
        sidecar = sidecar_path(photo)
        if not sidecar.is_file():
            continue
        try:
            tree = ET.parse(sidecar)
            description = _description(tree)
            rating = int(description.get(RATING_KEY, "0"))
        except (ET.ParseError, ValueError):
            continue
        if description.get(MANAGED_KEY) == "true" and 1 <= rating <= 5:
            matches.append(photo)
    return matches


def migrate_legacy_embedded_ratings(photos: list[Path]) -> dict[Path, str]:
    """Embed ratings already written by old versions without re-analysing.

    The migration is intentionally opt-in from the UI.  It only trusts the
    app's own marked sidecars, never a user-created sidecar, and it preserves
    any existing embedded HEIC rating in preference to the legacy value.
    """
    candidates = legacy_managed_embedded_photos(photos)
    backup_existing_sidecars(candidates)
    outcomes: dict[Path, str] = {}
    for photo in candidates:
        tree = ET.parse(sidecar_path(photo))
        rating = int(_description(tree).get(RATING_KEY, "0"))
        current = _read_embedded_rating(photo)
        if current:
            outcomes[photo] = "skipped-existing-rating"
            continue
        _backup_embedded_before_write(photo, current, rating)
        _run_bundled_exiftool(
            "-overwrite_original",
            f"-XMP-xmp:Rating={rating}",
            str(photo),
        )
        _retire_obsolete_managed_sidecar(photo)
        outcomes[photo] = "embedded-updated"
    return outcomes


def _write_embedded_rating(photo: Path, rating: int, mode: str) -> str:
    current = _read_embedded_rating(photo)
    if current and mode == "safe":
        return "skipped-existing-rating"
    sidecar = sidecar_path(photo)
    if mode == "safe" and sidecar.is_file():
        try:
            sidecar_desc = _description(ET.parse(sidecar))
        except ET.ParseError:
            return "skipped-existing-rating"
        if sidecar_desc.get(RATING_KEY, "0") not in {"", "0"}:
            return "skipped-existing-rating"
    record = _latest_embedded_backup_record(photo)
    if mode == "managed":
        if record is None or current != record[1].get("written_rating"):
            return "skipped-existing-rating"
    if rating == 0 and not current:
        return "skipped-unrated-new"

    _backup_embedded_before_write(photo, current, rating)
    _run_bundled_exiftool(
        "-overwrite_original",
        f"-XMP-xmp:Rating={rating}",
        str(photo),
    )
    _retire_obsolete_managed_sidecar(photo)
    return "embedded-updated"


def _new_tree() -> tuple[ET.ElementTree, ET.Element]:
    root = ET.Element(f"{{{X}}}xmpmeta")
    rdf = ET.SubElement(root, f"{{{RDF}}}RDF")
    # rdf:about is included for maximum compatibility with Lightroom's
    # sidecar importer, even though it is empty for an adjacent XMP file.
    desc = ET.SubElement(rdf, f"{{{RDF}}}Description", {f"{{{RDF}}}about": ""})
    return ET.ElementTree(root), desc


def _description(tree: ET.ElementTree) -> ET.Element:
    description = tree.find(f".//{{{RDF}}}Description")
    if description is None:
        rdf = tree.find(f".//{{{RDF}}}RDF")
        if rdf is None:
            rdf = ET.SubElement(tree.getroot(), f"{{{RDF}}}RDF")
        description = ET.SubElement(rdf, f"{{{RDF}}}Description")
    return description


def write_rating(photo: Path, rating: int, *, mode: str = "safe") -> str:
    """Write an Adobe rating, returning an action label.

    ``safe`` preserves every existing nonzero rating. ``managed`` may replace
    only a rating previously written by this app. ``force`` may replace any
    rating after the UI has obtained a second explicit confirmation.
    """
    if not 0 <= rating <= 5:
        raise ValueError("rating must be between 0 and 5")
    if mode not in {"safe", "managed", "force"}:
        raise ValueError("mode must be safe, managed, or force")
    if not photo.is_file():
        raise FileNotFoundError(photo)
    if _is_embedded(photo):
        return _write_embedded_rating(photo, rating, mode)

    sidecar = sidecar_path(photo)
    existed = sidecar.exists()
    if existed:
        try:
            tree = ET.parse(sidecar)
        except ET.ParseError as exc:
            raise ValueError(f"Existing XMP cannot be parsed: {sidecar.name}") from exc
    else:
        tree, _ = _new_tree()

    desc = _description(tree)
    current = desc.get(RATING_KEY)
    current_nonzero = current is not None and current.strip() not in {"", "0"}
    is_managed = desc.get(MANAGED_KEY) == "true"
    is_unchanged_managed = is_managed and current == desc.get(MANAGED_VALUE_KEY)
    if current_nonzero and mode == "safe":
        # Safe mode is deliberately byte-preserving for every existing
        # non-zero rating, including when it already matches our suggestion.
        # Re-serializing unchanged sidecars in bulk can discard packet details
        # owned by Lightroom/Camera Raw and makes a metadata refresh needlessly
        # risky.
        return "skipped-existing-rating"
    if mode == "managed" and not is_unchanged_managed:
        return "skipped-existing-rating"
    # Do not create hundreds of empty sidecars merely to represent an unrated
    # frame. Existing sidecars can still be cleared to Rating=0 in managed or
    # force mode.
    if rating == 0 and not existed:
        return "skipped-unrated-new"

    desc.set(RATING_KEY, str(rating))
    # This marker makes a later rollback safe: it never touches Lightroom
    # ratings that predated this app, nor a rating changed after our write.
    desc.set(MANAGED_KEY, "true")
    desc.set(MANAGED_VALUE_KEY, str(rating))
    desc.set(CREATED_SIDECAR_KEY, "true" if not existed else "false")
    _write_lightroom_xmp(tree, sidecar)
    return "updated" if existed else "created"


def write_ratings(
    ratings: dict[Path, int], *, mode: str = "safe"
) -> dict[Path, str]:
    backup_existing_sidecars(list(ratings))
    return {
        photo: write_rating(photo, rating, mode=mode)
        for photo, rating in ratings.items()
    }


def restore_managed_rating(photo: Path) -> str:
    """Undo only a rating previously written by Event Cull.

    An XMP sidecar created by the app is removed completely. For an existing
    sidecar, only this app's rating metadata is removed. A later Lightroom
    change is deliberately preserved instead of being overwritten.
    """
    if _is_embedded(photo):
        record = _latest_embedded_backup_record(photo)
        if record is None:
            return "not-managed"
        backup, metadata = record
        current = _read_embedded_rating(photo)
        if current != metadata.get("written_rating"):
            return "skipped-changed-after-write"
        temporary = photo.with_name(f".{photo.name}.event-cull-{uuid4().hex}.tmp")
        try:
            shutil.copy2(backup, temporary)
            os.replace(temporary, photo)
        finally:
            temporary.unlink(missing_ok=True)
        return "restored"

    sidecar = sidecar_path(photo)
    if not sidecar.exists():
        return "not-managed"
    try:
        tree = ET.parse(sidecar)
    except ET.ParseError as exc:
        raise ValueError(f"Existing XMP cannot be parsed: {sidecar.name}") from exc
    desc = _description(tree)
    if desc.get(MANAGED_KEY) != "true":
        return "not-managed"
    if desc.get(RATING_KEY) != desc.get(MANAGED_VALUE_KEY):
        return "skipped-changed-after-write"
    if desc.get(CREATED_SIDECAR_KEY) == "true":
        sidecar.unlink()
        return "removed-sidecar"
    for key in (RATING_KEY, MANAGED_KEY, MANAGED_VALUE_KEY, CREATED_SIDECAR_KEY):
        desc.attrib.pop(key, None)
    _write_lightroom_xmp(tree, sidecar)
    return "restored"


def restore_managed_ratings(photos: list[Path]) -> dict[Path, str]:
    backup_existing_sidecars(photos)
    return {photo: restore_managed_rating(photo) for photo in photos}
