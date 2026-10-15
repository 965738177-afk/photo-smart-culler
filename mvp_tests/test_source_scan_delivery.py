from pathlib import Path

import pytest

from event_cull.source_scan import scan_source_images


@pytest.mark.parametrize("name", ["机位1原图选好", "1010原图导出", "机位2卡21010原图导出"])
def test_delivery_is_excluded_when_nested_but_can_be_imported_explicitly(tmp_path, name):
    original = tmp_path / "DSC00001.ARW"
    original.touch()
    delivery = tmp_path / name
    delivery.mkdir()
    exported = delivery / "DSC00001.jpg"
    exported.touch()
    assert scan_source_images([tmp_path]) == [original]
    assert scan_source_images([delivery]) == [exported]


def test_raw_folder_remains_a_source(tmp_path):
    folder = tmp_path / "原图"
    folder.mkdir()
    raw = folder / "DSC00001.ARW"
    raw.touch()
    assert scan_source_images([tmp_path]) == [raw]


def test_overlapping_roots_scan_directory_only_once(tmp_path, monkeypatch):
    from event_cull import source_scan
    folder = tmp_path / "card"
    folder.mkdir()
    raw = folder / "DSC00001.ARW"
    raw.touch()
    real_walk = source_scan.os.walk
    visited = []
    def counting_walk(root):
        for parent, directories, files in real_walk(root):
            visited.append(Path(parent))
            yield parent, directories, files
    monkeypatch.setattr(source_scan.os, "walk", counting_walk)
    assert scan_source_images([tmp_path, folder, tmp_path]) == [raw]
    # Repeated roots stop at the first yielded directory rather than traversing.
    assert len(visited) == 4
