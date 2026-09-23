from pathlib import Path

import event_cull.xmp as xmp
from event_cull.xmp import restore_managed_rating, sidecar_path, write_rating, write_ratings


def test_writes_rating_to_sidecar_without_changing_original(tmp_path: Path):
    photo = tmp_path / "IMG_001.ARW"
    photo.write_bytes(b"original-raw-bytes")
    assert write_rating(photo, 3) == "created"
    assert photo.read_bytes() == b"original-raw-bytes"
    assert 'xmp:Rating="3"' in sidecar_path(photo).read_text()


def test_preserves_existing_nonzero_rating_without_override(tmp_path: Path):
    photo = tmp_path / "IMG_001.ARW"
    photo.write_bytes(b"raw")
    sidecar_path(photo).write_text('<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xmp="http://ns.adobe.com/xap/1.0/"><rdf:RDF><rdf:Description xmp:Rating="5" /></rdf:RDF></x:xmpmeta>')
    assert write_rating(photo, 3) == "skipped-existing-rating"
    assert 'xmp:Rating="5"' in sidecar_path(photo).read_text()


def test_safe_mode_leaves_same_existing_rating_byte_for_byte_unchanged(tmp_path: Path):
    photo = tmp_path / "IMG_001.ARW"
    photo.write_bytes(b"raw")
    original = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/"><rdf:RDF>'
        '<rdf:Description xmp:Rating="3" /></rdf:RDF></x:xmpmeta>'
    )
    sidecar_path(photo).write_text(original)

    assert write_rating(photo, 3) == "skipped-existing-rating"
    assert sidecar_path(photo).read_text() == original


def test_restores_only_rating_written_by_app(tmp_path: Path):
    photo = tmp_path / "IMG_001.ARW"
    photo.write_bytes(b"raw")
    assert write_rating(photo, 3) == "created"
    assert restore_managed_rating(photo) == "removed-sidecar"
    assert not sidecar_path(photo).exists()


def test_restore_preserves_rating_changed_after_app_write(tmp_path: Path):
    photo = tmp_path / "IMG_001.ARW"
    photo.write_bytes(b"raw")
    assert write_rating(photo, 3) == "created"
    content = sidecar_path(photo).read_text().replace('xmp:Rating="3"', 'xmp:Rating="5"')
    sidecar_path(photo).write_text(content)
    assert restore_managed_rating(photo) == "skipped-changed-after-write"
    assert 'xmp:Rating="5"' in sidecar_path(photo).read_text()


def test_batch_write_backs_up_existing_xmp_and_leaves_no_temp_file(tmp_path: Path):
    photo = tmp_path / "IMG_001.ARW"
    photo.write_bytes(b"raw")
    sidecar_path(photo).write_text(
        '<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/"><rdf:RDF>'
        '<rdf:Description xmp:Rating="0" /></rdf:RDF></x:xmpmeta>'
    )
    outcomes = write_ratings({photo: 3})
    assert outcomes[photo] == "updated"
    backups = list((tmp_path / ".event-cull-backups").glob("*/*.xmp"))
    assert len(backups) == 1
    assert 'xmp:Rating="0"' in backups[0].read_text()
    assert not list(tmp_path.glob("*.tmp"))


def test_managed_mode_overwrites_only_rating_previously_written_by_app(tmp_path: Path):
    managed = tmp_path / "MANAGED.ARW"
    external = tmp_path / "EXTERNAL.ARW"
    managed.write_bytes(b"raw")
    external.write_bytes(b"raw")
    assert write_rating(managed, 3) == "created"
    sidecar_path(external).write_text(
        '<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/"><rdf:RDF>'
        '<rdf:Description xmp:Rating="5" /></rdf:RDF></x:xmpmeta>'
    )

    assert write_rating(managed, 2, mode="managed") == "updated"
    assert write_rating(external, 2, mode="managed") == "skipped-existing-rating"
    assert 'xmp:Rating="2"' in sidecar_path(managed).read_text()
    assert 'xmp:Rating="5"' in sidecar_path(external).read_text()


def test_managed_mode_preserves_rating_changed_after_app_write(tmp_path: Path):
    photo = tmp_path / "IMG_001.ARW"
    photo.write_bytes(b"raw")
    assert write_rating(photo, 3) == "created"
    changed = sidecar_path(photo).read_text().replace('xmp:Rating="3"', 'xmp:Rating="5"')
    sidecar_path(photo).write_text(changed)

    assert write_rating(photo, 2, mode="managed") == "skipped-existing-rating"
    assert 'xmp:Rating="5"' in sidecar_path(photo).read_text()


def test_force_mode_overwrites_external_rating_and_can_clear_it(tmp_path: Path):
    photo = tmp_path / "IMG_001.ARW"
    photo.write_bytes(b"raw")
    sidecar_path(photo).write_text(
        '<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/"><rdf:RDF>'
        '<rdf:Description xmp:Rating="5" /></rdf:RDF></x:xmpmeta>'
    )

    assert write_rating(photo, 0, mode="force") == "updated"
    assert 'xmp:Rating="0"' in sidecar_path(photo).read_text()


def test_zero_rating_does_not_create_new_sidecar(tmp_path: Path):
    photo = tmp_path / "IMG_001.ARW"
    photo.write_bytes(b"raw")
    assert write_rating(photo, 0, mode="force") == "skipped-unrated-new"
    assert not sidecar_path(photo).exists()


def test_heic_rating_is_embedded_with_full_original_backup(tmp_path: Path, monkeypatch):
    photo = tmp_path / "IMG_001.HEIC"
    original = b"minimal-heic-container-for-command-test"
    photo.write_bytes(original)
    state = {"rating": 0}
    commands: list[tuple[str, ...]] = []

    def fake_exiftool(*arguments: str) -> str:
        commands.append(arguments)
        if arguments[:2] == ("-s3", "-XMP-xmp:Rating"):
            return str(state["rating"]) if state["rating"] else ""
        rating_argument = next(argument for argument in arguments if argument.startswith("-XMP-xmp:Rating="))
        state["rating"] = int(rating_argument.rsplit("=", 1)[1])
        return "1 image files updated"

    monkeypatch.setattr(xmp, "_run_bundled_exiftool", fake_exiftool)

    assert write_rating(photo, 3) == "embedded-updated"
    assert sidecar_path(photo).exists() is False
    backups = list((tmp_path / ".event-cull-backups").glob("*/IMG_001.HEIC"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert state["rating"] == 3
    assert any("-overwrite_original" in command for command in commands)


def test_managed_heic_rating_requires_the_app_last_write(tmp_path: Path, monkeypatch):
    photo = tmp_path / "IMG_001.HEIC"
    photo.write_bytes(b"heic")
    state = {"rating": 0}

    def fake_exiftool(*arguments: str) -> str:
        if arguments[:2] == ("-s3", "-XMP-xmp:Rating"):
            return str(state["rating"]) if state["rating"] else ""
        rating_argument = next(argument for argument in arguments if argument.startswith("-XMP-xmp:Rating="))
        state["rating"] = int(rating_argument.rsplit("=", 1)[1])
        return "1 image files updated"

    monkeypatch.setattr(xmp, "_run_bundled_exiftool", fake_exiftool)

    assert write_rating(photo, 3) == "embedded-updated"
    assert write_rating(photo, 2, mode="managed") == "embedded-updated"
    state["rating"] = 5  # A later Lightroom edit must remain protected.
    assert write_rating(photo, 1, mode="managed") == "skipped-existing-rating"


def test_migrates_old_app_heic_sidecar_into_embedded_rating(tmp_path: Path, monkeypatch):
    photo = tmp_path / "IMG_001.HEIC"
    photo.write_bytes(b"heic")
    sidecar_path(photo).write_text(
        '<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        'xmlns:eventcull="urn:event-cull:metadata:1.0/"><rdf:RDF><rdf:Description '
        'xmp:Rating="3" eventcull:ManagedRating="true" /></rdf:RDF></x:xmpmeta>'
    )
    state = {"rating": 0}

    def fake_exiftool(*arguments: str) -> str:
        if arguments[:2] == ("-s3", "-XMP-xmp:Rating"):
            return str(state["rating"]) if state["rating"] else ""
        rating_argument = next(argument for argument in arguments if argument.startswith("-XMP-xmp:Rating="))
        state["rating"] = int(rating_argument.rsplit("=", 1)[1])
        return "1 image files updated"

    monkeypatch.setattr(xmp, "_run_bundled_exiftool", fake_exiftool)

    assert xmp.legacy_managed_embedded_photos([photo]) == [photo]
    assert xmp.migrate_legacy_embedded_ratings([photo]) == {photo: "embedded-updated"}
    assert state["rating"] == 3
    assert not sidecar_path(photo).exists()


def test_jpeg_dng_and_heic_all_use_embedded_ratings_and_full_backups(tmp_path: Path, monkeypatch):
    ratings = {}

    def fake_exiftool(*arguments: str) -> str:
        path = Path(arguments[-1])
        if arguments[:2] == ("-s3", "-XMP-xmp:Rating"):
            return str(ratings[path]) if path in ratings else ""
        rating_argument = next(arg for arg in arguments if arg.startswith("-XMP-xmp:Rating="))
        ratings[path] = int(rating_argument.rsplit("=", 1)[1])
        return "1 image files updated"

    monkeypatch.setattr(xmp, "_run_bundled_exiftool", fake_exiftool)
    for suffix in (".JPG", ".DNG", ".HEIC"):
        photo = tmp_path / f"image{suffix}"
        original = f"original-{suffix}".encode()
        photo.write_bytes(original)

        assert write_rating(photo, 3) == "embedded-updated"
        assert ratings[photo] == 3
        assert not sidecar_path(photo).exists()
        backups = list((tmp_path / ".event-cull-backups").glob(f"*/{photo.name}"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == original


def test_old_managed_jpeg_sidecar_migrates_but_paired_raw_sidecar_is_preserved(tmp_path: Path):
    jpeg = tmp_path / "single.JPG"
    jpeg.write_bytes(b"standalone-jpeg")
    raw = tmp_path / "pair.ARW"
    raw.write_bytes(b"camera-raw")
    paired_jpeg = tmp_path / "pair.JPG"
    paired_jpeg.write_bytes(b"paired-jpeg")
    assert write_rating(raw, 3) == "created"

    # Simulate a pre-fix app's JPG sidecar. The paired RAW sidecar belongs to
    # the RAW even though a JPG with the same stem also exists.
    sidecar_path(jpeg).write_text(sidecar_path(raw).read_text())
    assert xmp.legacy_managed_embedded_photos([jpeg, raw, paired_jpeg]) == [jpeg]
    assert sidecar_path(raw).exists()


def test_direct_jpeg_write_does_not_remove_same_stem_raw_sidecar(tmp_path: Path, monkeypatch):
    raw = tmp_path / "pair.ARW"
    jpeg = tmp_path / "pair.JPG"
    raw.write_bytes(b"camera-raw")
    jpeg.write_bytes(b"paired-jpeg")
    assert write_rating(raw, 3) == "created"
    original_sidecar = sidecar_path(raw).read_bytes()

    def fake_exiftool(*arguments: str) -> str:
        if arguments[:2] == ("-s3", "-XMP-xmp:Rating"):
            return ""
        return "1 image files updated"

    monkeypatch.setattr(xmp, "_run_bundled_exiftool", fake_exiftool)
    assert write_rating(jpeg, 3, mode="force") == "embedded-updated"
    assert sidecar_path(raw).read_bytes() == original_sidecar
