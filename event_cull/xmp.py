"""Standard XMP sidecar updates. This module never writes image pixels or RAW files."""

from __future__ import annotations

import os
import shutil
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
    """Merge only XMP:Rating into a sidecar, returning an action label.

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
