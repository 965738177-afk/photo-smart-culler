from pathlib import Path

from event_cull.app import EventCullApp, SCENE_TYPES, TEXT
from event_cull.engine import PhotoResult


def test_default_target_is_one_third_rounded_up_to_tens():
    assert EventCullApp._default_target_count(255) == 90
    assert EventCullApp._default_target_count(3000) == 1000
    assert EventCullApp._default_target_count(301) == 110
    assert EventCullApp._default_target_count(1) == 10
    assert EventCullApp._default_target_count(0) == 0


def test_scene_picker_includes_every_supported_scene_rule_set():
    assert SCENE_TYPES == ("通用", "婚礼", "人像写真", "活动纪实", "儿童与家庭", "毕业照", "证件照", "美食")


def test_main_window_uses_available_display_space_without_exceeding_release_size():
    assert EventCullApp._main_window_size_for_screen(1728, 1117) == (1680, 1033)
    assert EventCullApp._main_window_size_for_screen(2560, 1600) == (1700, 1050)


def test_mousewheel_normalization_supports_trackpads_and_notched_wheels():
    assert EventCullApp._wheel_units(1) == -1
    assert EventCullApp._wheel_units(-1) == 1
    assert EventCullApp._wheel_units(240) == -2
    assert EventCullApp._wheel_units(0) == 0


def test_preview_zoom_stays_within_safe_rendering_limits():
    assert EventCullApp._clamp_preview_zoom(0.01) == 0.10
    assert EventCullApp._clamp_preview_zoom(1.25) == 1.25
    assert EventCullApp._clamp_preview_zoom(9) == 4.0
    assert EventCullApp._preview_fit_scale((1600, 1000), (800, 800)) == 0.5


def test_view_controls_wrap_before_a_narrow_toolbar_can_clip_them():
    assert EventCullApp._grid_toolbar_is_compact(959)
    assert not EventCullApp._grid_toolbar_is_compact(960)


def test_virtual_grid_row_height_reserves_one_consistent_review_track():
    assert EventCullApp._grid_row_height(144, analyzed=False) == 214
    assert EventCullApp._grid_row_height(144, analyzed=True) == 360


def test_result_filter_can_show_all_or_one_exact_rating():
    assert EventCullApp._matches_rating_filter(3, None)
    assert EventCullApp._matches_rating_filter(3, 3)
    assert not EventCullApp._matches_rating_filter(2, 3)
    assert EventCullApp._matches_rating_filter(0, 0)


def test_chinese_filter_names_make_the_one_star_review_tier_explicit():
    assert TEXT["zh"]["filter_one"] == "仅 1★ 初选复核"


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


def test_folder_import_includes_photos_in_nested_camera_folders(tmp_path: Path):
    nested_folder = tmp_path / "card" / "DCIM" / "100SONY"
    nested_folder.mkdir(parents=True)
    photo = nested_folder / "DSC00001.ARW"
    photo.write_bytes(b"fixture")

    assert EventCullApp._files_in_folders([tmp_path]) == [photo]


def test_source_counts_prefer_raw_for_a_raw_jpg_pair(tmp_path: Path):
    paired_raw = tmp_path / "DSC0001.ARW"
    paired_jpg = tmp_path / "DSC0001.JPG"
    standalone_jpg = tmp_path / "DSC0002.JPG"
    other = tmp_path / "DSC0003.PNG"
    for path in (paired_raw, paired_jpg, standalone_jpg, other):
        path.write_bytes(b"fixture")

    counts = EventCullApp._source_photo_counts(
        [paired_raw, paired_jpg, standalone_jpg, other]
    )

    assert counts == {"jpg": 1, "raw": 1, "other": 1, "total": 3}


def test_source_file_counts_show_every_imported_jpg_and_raw(tmp_path: Path):
    paired_raw = tmp_path / "DSC0001.ARW"
    paired_jpg = tmp_path / "DSC0001.JPG"
    standalone_jpg = tmp_path / "DSC0002.JPG"
    other = tmp_path / "DSC0003.PNG"
    for path in (paired_raw, paired_jpg, standalone_jpg, other):
        path.write_bytes(b"fixture")

    counts = EventCullApp._source_file_counts(
        [paired_raw, paired_jpg, standalone_jpg, other]
    )

    assert counts == {"jpg": 2, "raw": 1, "other": 1, "total": 4}


def test_source_summary_names_other_supported_formats():
    app = object.__new__(EventCullApp)
    app.t = TEXT["zh"]

    text = app._source_count_text({"jpg": 2, "raw": 1, "other": 1, "total": 4})

    assert text == "JPG 2 张 · RAW 1 张 · 其他 1 张 · 合计 4 张照片"


def test_task_header_says_what_was_added_and_the_current_total():
    app = object.__new__(EventCullApp)
    app.t = TEXT["zh"]

    text = app._task_import_summary_text(
        {"jpg": 2, "raw": 3, "other": 0, "total": 5},
        {"jpg": 12, "raw": 20, "other": 0, "total": 32},
    )

    assert text == "本次新增 5 张 · JPG 2 张 · RAW 3 张 · 当前合计 32 张照片"


def test_task_header_uses_its_right_status_slot_for_analysis_progress():
    assert EventCullApp._analysis_header_status_text(
        218, 752, "预计剩余时间：2分14秒", language="zh"
    ) == "AI 分析 218 / 752 · 预计剩余时间：2分14秒"
    assert EventCullApp._analysis_header_status_text(
        12, 24, "", language="en"
    ) == "AI analysis 12 / 24"


def test_custom_duplicate_limit_accepts_direct_numeric_entry():
    class Value:
        def __init__(self, value: str):
            self.value = value

        def get(self) -> str:
            return self.value

    app = object.__new__(EventCullApp)
    app.keep_per_duplicate = Value("12")

    assert app._duplicate_keep_limit() == 12


def test_reset_picker_settings_defaults_duplicate_limit_to_two():
    class Value:
        def __init__(self, value):
            self.value = value

        def set(self, value):
            self.value = value

    app = object.__new__(EventCullApp)
    app.scene_type = Value("婚礼")
    app.ai_selection_enabled = Value(False)
    app.priority_more_faces = Value(False)
    app.fill_with_similar = Value(True)
    app.keep_existing_labels = Value(False)
    app.detect_closed_eyes = Value(False)
    app.detect_blur = Value(False)
    app.detect_exposure = Value(False)
    app.detect_duplicates = Value(False)
    app.blur_sensitivity = Value("强")
    app.exposure_sensitivity = Value("弱")
    app.duplicate_similarity = Value("宽松")
    app.keep_per_duplicate = Value("1")
    app._refresh_picker_confirmation_ui = lambda: None

    EventCullApp._reset_picker_settings(app)

    assert app.keep_per_duplicate.value == "2"
    assert app.ai_selection_enabled.value is True
    assert app.duplicate_similarity.value == "标准"


def test_picker_passes_the_selected_scene_type_to_the_culling_engine():
    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    app = object.__new__(EventCullApp)
    app.scene_type = Value("婚礼")
    app.blur_sensitivity = Value("标准")
    app.exposure_sensitivity = Value("标准")
    app.duplicate_similarity = Value("标准")
    app.detect_closed_eyes = Value(True)
    app.detect_blur = Value(True)
    app.detect_exposure = Value(True)
    app.detect_duplicates = Value(True)
    app.ai_selection_enabled = Value(True)
    app.priority_more_faces = Value(True)
    app.fill_with_similar = Value(False)
    app.language = "zh"

    options = app._analysis_options_from_picker(
        picks_min=2,
        picks_max=3,
        target_min=200,
        target_max=300,
        duplicate_keep_limit=2,
    )

    assert options.scene_type == "婚礼"


def test_reanalysis_restores_only_manual_ratings_when_requested():
    path = Path("/photos/IMG_0001.ARW")
    result = PhotoResult(path, suggested_rating=3)

    EventCullApp._restore_manual_ratings([result], {path: 1}, "zh")

    assert result.suggested_rating == 1
    assert "保留手动评级" in result.reasons


def test_picker_summary_makes_active_detection_rules_explicit():
    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    app = object.__new__(EventCullApp)
    app.picks = Value("2")
    app.picks_max = Value("3")
    app.target = Value("200")
    app.target_max = Value("300")
    app.detect_closed_eyes = Value(True)
    app.detect_blur = Value(True)
    app.detect_exposure = Value(True)
    app.detect_duplicates = Value(True)

    assert app._picker_rule_summary_text() == "目标 至少 200，建议 300 张 · 每组优选 2–3 张 · 闭眼、模糊、曝光、重复检测"


def test_start_analysis_opens_rule_confirmation_before_any_analysis_work():
    app = object.__new__(EventCullApp)
    app.imported_files = [Path("/photos/IMG_0001.ARW")]
    app.picker_settings_confirmed = False
    app.show_picker_settings = lambda: setattr(app, "settings_opened", True)

    EventCullApp.start_analysis(app)

    assert app.settings_opened is True


def test_appending_photos_updates_task_count_without_changing_task_id(tmp_path: Path):
    class Title:
        value = ""

        def set(self, value: str) -> None:
            self.value = value

    source = tmp_path / "婚礼原片"
    source.mkdir()
    first = source / "IMG_0001.JPG"
    second = source / "IMG_0002.JPG"
    app = object.__new__(EventCullApp)
    app.current_task_id = None
    app.current_task_name = ""
    app.current_task_created_at = None
    app.task_title = Title()

    app._ensure_task_identity([first])
    task_id = app.current_task_id
    app._ensure_task_identity([first, second])

    assert app.current_task_id == task_id
    assert app.current_task_name.endswith("婚礼原片 · 2 张")
    assert app.task_title.value == app.current_task_name


def test_default_target_range_leaves_a_twenty_percent_scene_coverage_allowance():
    assert EventCullApp._default_target_max_count(1000) == 1200
    assert EventCullApp._default_target_max_count(90) == 110


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


def test_selection_preference_copy_uses_final_filenames_without_metadata():
    assert TEXT["zh"]["calibration_export_item"] == "人工最终选出的 JPG / 原片文件夹（只读取文件名）"
    assert TEXT["zh"]["calibration"] == "选片偏好校准"
    assert TEXT["zh"]["calibration_update"] == "更新选片偏好校准"
    assert TEXT["zh"]["title"] == "照片智能筛选器"
