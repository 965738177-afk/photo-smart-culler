from pathlib import Path

from event_cull.app import EventCullApp, TEXT
from event_cull.engine import PhotoResult


def test_default_target_is_one_third_rounded_up_to_tens():
    assert EventCullApp._default_target_count(255) == 90
    assert EventCullApp._default_target_count(3000) == 1000
    assert EventCullApp._default_target_count(301) == 110
    assert EventCullApp._default_target_count(1) == 10
    assert EventCullApp._default_target_count(0) == 0


def test_result_filter_can_show_all_or_one_exact_rating():
    assert EventCullApp._matches_rating_filter(3, None)
    assert EventCullApp._matches_rating_filter(3, 3)
    assert not EventCullApp._matches_rating_filter(2, 3)
    assert EventCullApp._matches_rating_filter(0, 0)


def test_output_star_mapping_keeps_engine_tiers_but_supports_four_and_five_stars():
    assert [EventCullApp._mapped_rating(tier, 2, 3) for tier in range(4)] == [0, 1, 2, 3]
    assert [EventCullApp._mapped_rating(tier, 4, 5) for tier in range(4)] == [0, 1, 4, 5]


def test_repeat_import_appends_without_duplicate_files(tmp_path: Path):
    first = tmp_path / "A.JPG"
    second = tmp_path / "B.ARW"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    merged, added = EventCullApp._merge_unique_paths([first], [first, second, second])

    assert merged == [first, second]
    assert added == 1


def test_multiple_folders_are_combined_but_non_photos_are_ignored(tmp_path: Path):
    first_folder = tmp_path / "one"
    second_folder = tmp_path / "two"
    first_folder.mkdir()
    second_folder.mkdir()
    first_photo = first_folder / "A.JPG"
    second_photo = second_folder / "B.ARW"
    ignored = second_folder / "notes.txt"
    first_photo.write_bytes(b"a")
    second_photo.write_bytes(b"b")
    ignored.write_text("ignore")

    files = EventCullApp._files_in_folders([first_folder, second_folder])

    assert files == [first_photo, second_photo]


def test_remaining_time_waits_for_samples_then_estimates_from_real_progress():
    assert EventCullApp._remaining_seconds(1.0, 1, 100) is None
    assert EventCullApp._remaining_seconds(30.0, 3, 12) == 90
    assert EventCullApp._remaining_seconds(30.0, 12, 12) == 0


def test_log_record_keeps_rating_quality_and_review_reason():
    item = PhotoResult(
        Path("/photos/IMG_0001.ARW"),
        group_id=7,
        score=91.234,
        suggested_rating=3,
        reasons=["本组优选"],
        sharpness=120.456,
        subject_sharpness=98.765,
        frontal_face_count=2,
        smiling_face_count=1,
        calibration_confidence=.81234,
    )

    record = EventCullApp._log_record(item, 1)

    assert record["folder"] == "/photos"
    assert record["rating"] == 3
    assert record["score"] == 91.23
    assert record["subject_sharpness"] == 98.77
    assert record["calibration_confidence"] == 0.8123
    assert record["reasons"] == "本组优选"


def test_personal_calibration_copy_requires_all_lightroom_metadata():
    assert TEXT["zh"]["calibration_export_item"] == "Lightroom 导出带星级的 JPG（勾选所有元数据）"
    assert TEXT["zh"]["calibration_update"] == "更新个人校准"
    assert TEXT["zh"]["title"] == "照片智能筛选器"
