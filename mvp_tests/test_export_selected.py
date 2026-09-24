import json
import threading
from pathlib import Path

import pytest

from event_cull.app import EventCullApp
from event_cull.engine import PhotoResult
from event_cull.export_selected import (
    EXPORT_MARKER,
    ExportCancelled,
    collect_selected_files,
    export_selected_files,
)


def test_every_rated_photo_is_selected_with_raw_jpeg_pair_and_xmp(tmp_path: Path):
    raw = tmp_path / "keep.ARW"
    jpeg = tmp_path / "keep.JPG"
    sidecar = tmp_path / "keep.xmp"
    rejected = tmp_path / "skip.JPG"
    for path in (raw, jpeg, sidecar, rejected):
        path.write_bytes(path.name.encode())
    results = [
        PhotoResult(path=raw, pair_paths=[raw, jpeg], suggested_rating=1),
        PhotoResult(path=rejected, pair_paths=[rejected], suggested_rating=0),
    ]

    selection = collect_selected_files(results)

    assert selection.photo_count == 1
    assert selection.review_count == 1
    assert selection.file_count == 3
    assert set(selection.files_by_folder[tmp_path.resolve()]) == {
        raw.resolve(), jpeg.resolve(), sidecar.resolve()
    }


def test_export_can_select_one_or_multiple_rating_tiers(tmp_path: Path):
    photos = [tmp_path / f"{rating}.JPG" for rating in (1, 2, 3)]
    for photo in photos:
        photo.write_bytes(photo.name.encode())
    results = [
        PhotoResult(path=photo, suggested_rating=rating)
        for rating, photo in enumerate(photos, start=1)
    ]

    only_best = collect_selected_files(results, frozenset({3}))
    assert only_best.photo_count == 1
    assert only_best.files_by_folder[tmp_path.resolve()] == (photos[2].resolve(),)

    backups_and_best = collect_selected_files(results, frozenset({2, 3}))
    assert backups_and_best.photo_count == 2
    assert backups_and_best.review_count == 0
    assert set(backups_and_best.files_by_folder[tmp_path.resolve()]) == {
        photos[1].resolve(), photos[2].resolve()
    }

    with pytest.raises(ValueError):
        collect_selected_files(results, frozenset())


def test_adjacent_export_copies_and_does_not_reimport_delivery(tmp_path: Path):
    photo = tmp_path / "one.HEIC"
    photo.write_bytes(b"original-pixels-and-metadata")
    selection = collect_selected_files([
        PhotoResult(path=photo, suggested_rating=3),
    ])

    exported = export_selected_files(selection)

    output = tmp_path / "秒选"
    assert exported.folders == (output,)
    assert (output / photo.name).read_bytes() == photo.read_bytes()
    assert json.loads((output / EXPORT_MARKER).read_text())["complete"] is True
    assert EventCullApp._files_in_folders([tmp_path]) == [photo]

    again = export_selected_files(selection)
    assert again.folders == (tmp_path / "秒选-2",)
    assert (output / photo.name).read_bytes() == b"original-pixels-and-metadata"


def test_nested_curated_folder_is_not_reimported_but_can_be_opened_directly(tmp_path: Path):
    original = tmp_path / "IMG_0001.HEIC"
    original.write_bytes(b"original")
    curated = tmp_path / "选"
    curated.mkdir()
    copy = curated / original.name
    copy.write_bytes(b"curated-copy")

    assert EventCullApp._files_in_folders([tmp_path]) == [original]
    assert EventCullApp._files_in_folders([curated]) == [copy]


def test_custom_export_separates_same_names_from_multiple_sources(tmp_path: Path):
    first = tmp_path / "camera-a"
    second = tmp_path / "camera-b"
    target = tmp_path / "delivery"
    first.mkdir()
    second.mkdir()
    target.mkdir()
    a = first / "IMG_0001.JPG"
    b = second / "IMG_0001.JPG"
    a.write_bytes(b"first")
    b.write_bytes(b"second")
    selection = collect_selected_files([
        PhotoResult(path=a, suggested_rating=1),
        PhotoResult(path=b, suggested_rating=2),
    ])

    exported = export_selected_files(selection, custom_parent=target)

    assert exported.photo_count == 2
    assert (target / "秒选" / "camera-a" / a.name).read_bytes() == b"first"
    assert (target / "秒选" / "camera-b" / b.name).read_bytes() == b"second"
    assert not (first / "秒选").exists()
    assert not (second / "秒选").exists()


def test_cancel_cleans_only_new_export_folder(tmp_path: Path):
    photos = [tmp_path / f"{index}.JPG" for index in range(2)]
    for photo in photos:
        photo.write_bytes(photo.name.encode())
    selection = collect_selected_files([
        PhotoResult(path=photo, suggested_rating=3) for photo in photos
    ])
    cancelled = threading.Event()

    with pytest.raises(ExportCancelled):
        export_selected_files(
            selection,
            cancel_event=cancelled,
            progress_callback=lambda *_args: cancelled.set(),
        )

    assert not (tmp_path / "秒选").exists()
    assert all(photo.is_file() for photo in photos)


def test_missing_source_aborts_without_leaving_partial_delivery(tmp_path: Path):
    photo = tmp_path / "gone.JPG"
    photo.write_bytes(b"before")
    selection = collect_selected_files([PhotoResult(path=photo, suggested_rating=2)])
    photo.unlink()

    with pytest.raises(FileNotFoundError):
        export_selected_files(selection)

    assert not (tmp_path / "秒选").exists()


def test_same_named_source_folders_get_distinct_custom_subfolders(tmp_path: Path):
    folders = (tmp_path / "a" / "photos", tmp_path / "b" / "photos")
    for folder in folders:
        folder.mkdir(parents=True)
        (folder / "image.JPG").write_bytes(str(folder).encode())
    custom = tmp_path / "custom"
    custom.mkdir()
    selection = collect_selected_files([
        PhotoResult(path=folder / "image.JPG", suggested_rating=3)
        for folder in folders
    ])

    export_selected_files(selection, custom_parent=custom)

    subfolders = list((custom / "秒选").glob("photos-*/image.JPG"))
    assert len(subfolders) == 2
    assert {photo.read_bytes() for photo in subfolders} == {str(folder).encode() for folder in folders}
