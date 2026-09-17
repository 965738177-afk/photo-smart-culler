from pathlib import Path

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
