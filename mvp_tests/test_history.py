from datetime import datetime
from pathlib import Path

from event_cull.history import load_history, task_name, upsert_task


def test_task_name_uses_timestamp_and_common_source_folder(tmp_path: Path):
    source = tmp_path / "婚礼原片"
    source.mkdir()
    photos = [source / "IMG_0001.ARW", source / "IMG_0002.JPG"]

    name = task_name(photos, datetime(2026, 9, 22, 14, 35))

    assert name == "2026-09-22 14:35 · 婚礼原片 · 2 张"


def test_upsert_task_replaces_matching_id_and_keeps_latest_first(tmp_path: Path):
    destination = tmp_path / "history.json"
    upsert_task({"id": "first", "name": "第一次", "status": "已导入"}, destination)
    upsert_task({"id": "second", "name": "第二次", "status": "已完成"}, destination)
    upsert_task({"id": "first", "name": "第一次", "status": "已完成"}, destination)

    records = load_history(destination)

    assert [record["id"] for record in records] == ["first", "second"]
    assert records[0]["status"] == "已完成"
