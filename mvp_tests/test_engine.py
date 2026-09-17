from pathlib import Path

import threading

import pytest

from event_cull.engine import AnalysisCancelled, AnalysisOptions, PhotoResult, analyze_folder


def test_target_count_keeps_one_lead_per_group_before_filling_global_slots(monkeypatch, tmp_path: Path):
    paths = []
    for number in range(4):
        path = tmp_path / f"IMG_{number}.jpg"
        path.write_bytes(b"fixture")
        paths.append(path)
    values = {
        paths[0]: PhotoResult(paths[0], score=90, perceptual_hash=1),
        paths[1]: PhotoResult(paths[1], score=80, perceptual_hash=1),
        paths[2]: PhotoResult(paths[2], score=70, perceptual_hash=10),
        paths[3]: PhotoResult(paths[3], score=60, perceptual_hash=10),
    }
    import event_cull.engine as engine
    monkeypatch.setattr(engine, "_representatives", lambda folder: [(path, [path]) for path in paths])
    monkeypatch.setattr(engine, "_analyze", lambda path, pairs, options: values[path])
    monkeypatch.setattr(engine, "_same_group", lambda previous, current, options: previous.perceptual_hash == current.perceptual_hash)
    results = analyze_folder(tmp_path, AnalysisOptions(target_keep_count=2))
    assert [item.suggested_rating for item in results] == [3, 0, 2, 0]


def test_short_burst_keeps_best_and_one_backup(monkeypatch, tmp_path: Path):
    paths = []
    for number in range(3):
        path = tmp_path / f"IMG_{number}.jpg"
        path.write_bytes(b"fixture")
        paths.append(path)
    values = {
        paths[0]: PhotoResult(paths[0], score=90, perceptual_hash=1),
        paths[1]: PhotoResult(paths[1], score=80, perceptual_hash=1),
        paths[2]: PhotoResult(paths[2], score=70, perceptual_hash=1),
    }
    import event_cull.engine as engine
    monkeypatch.setattr(engine, "_representatives", lambda folder: [(path, [path]) for path in paths])
    monkeypatch.setattr(engine, "_analyze", lambda path, pairs, options: values[path])
    monkeypatch.setattr(engine, "_same_group", lambda previous, current, options: True)

    results = analyze_folder(tmp_path, AnalysisOptions(picks_per_group=2, target_keep_count=100))

    assert [item.suggested_rating for item in results] == [3, 2, 2]


def test_same_filename_in_two_folders_is_not_collapsed(tmp_path: Path):
    import event_cull.engine as engine

    first_folder = tmp_path / "card-a"
    second_folder = tmp_path / "card-b"
    first_folder.mkdir()
    second_folder.mkdir()
    first_raw = first_folder / "DSC00001.ARW"
    first_jpg = first_folder / "DSC00001.JPG"
    second_raw = second_folder / "DSC00001.ARW"
    second_jpg = second_folder / "DSC00001.JPG"
    for path in (first_raw, first_jpg, second_raw, second_jpg):
        path.write_bytes(b"fixture")

    representatives = engine._representatives(
        [first_raw, first_jpg, second_raw, second_jpg]
    )

    assert len(representatives) == 2
    assert {primary.parent for primary, _pairs in representatives} == {
        first_folder,
        second_folder,
    }
    assert all(len(pairs) == 2 for _primary, pairs in representatives)


def test_grouping_never_crosses_source_folder(tmp_path: Path):
    import event_cull.engine as engine

    first_folder = tmp_path / "card-a"
    second_folder = tmp_path / "card-b"
    first_folder.mkdir()
    second_folder.mkdir()
    first = first_folder / "DSC00001.JPG"
    second = second_folder / "DSC00002.JPG"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    second.touch()

    assert not engine._same_group(
        PhotoResult(first, perceptual_hash=1),
        PhotoResult(second, perceptual_hash=1),
        AnalysisOptions(),
    )


def test_dynamic_density_keeps_two_of_eight_and_four_of_fifteen():
    import event_cull.engine as engine

    short_group = [
        PhotoResult(Path(f"S_{index}.jpg"), score=100 - index, perceptual_hash=1)
        for index in range(8)
    ]
    long_group = [
        PhotoResult(Path(f"L_{index}.jpg"), score=100 - index, perceptual_hash=2)
        for index in range(15)
    ]

    assert engine._group_keep_quota(short_group, AnalysisOptions()) == 2
    assert engine._group_keep_quota(long_group, AnalysisOptions()) == 4


def test_large_group_photo_burst_keeps_five_of_fifteen():
    import event_cull.engine as engine

    group = [
        PhotoResult(
            Path(f"G_{index}.jpg"),
            score=100 - index,
            perceptual_hash=1,
            frontal_face_count=8,
        )
        for index in range(15)
    ]

    assert engine._group_keep_quota(group, AnalysisOptions()) == 5


def test_long_burst_protects_two_frames_from_each_eight_frame_window():
    import event_cull.engine as engine

    group = [
        PhotoResult(Path(f"IMG_{index:02d}.jpg"), score=200 - index, perceptual_hash=1)
        for index in range(26)
    ]
    # The final two frames have lower global scores than the beginning of the
    # burst, but form their own consecutive remainder and must both survive.
    engine._assign_tiered_ratings(
        [group],
        AnalysisOptions(target_keep_count=8, picks_per_group=2),
    )

    assert group[-2].suggested_rating > 0
    assert group[-1].suggested_rating > 0
    assert sum(item.suggested_rating > 0 for item in group) == 8


def test_two_frame_tail_pose_survives_competing_scene_scores():
    import event_cull.engine as engine

    burst = [
        PhotoResult(Path(f"BURST_{index:02d}.jpg"), score=200 - index, perceptual_hash=1)
        for index in range(8)
    ] + [
        PhotoResult(Path("TAIL_A.jpg"), score=40, perceptual_hash=1),
        PhotoResult(Path("TAIL_B.jpg"), score=45, perceptual_hash=1),
    ]
    other_scene = [PhotoResult(Path("OTHER.jpg"), score=300, perceptual_hash=2)]

    engine._assign_tiered_ratings(
        [burst, other_scene],
        AnalysisOptions(target_keep_count=4, picks_per_group=2),
    )

    assert burst[-2].suggested_rating > 0
    assert burst[-1].suggested_rating > 0
    assert other_scene[0].suggested_rating > 0


def test_long_uniformly_soft_face_scene_keeps_three_representatives_for_review():
    import event_cull.engine as engine

    group = []
    for index in range(12):
        sharpness = 42 + index % 4
        group.append(
            PhotoResult(
                Path(f"VEIL_{index:02d}.jpg"),
                score=50 + index,
                reasons=["明显虚焦或抖动"],
                sharpness=sharpness,
                subject_sharpness=48 if index < 8 else None,
                highlight_clip=0,
                shadow_clip=0,
                perceptual_hash=1,
            )
        )

    engine._assign_tiered_ratings(
        [group],
        AnalysisOptions(target_keep_count=3, picks_per_group=2),
    )

    selected = [item for item in group if item.suggested_rating > 0]
    assert len(selected) == 3
    assert all(item.scene_protected for item in selected)
    assert all("明显虚焦或抖动" not in item.reasons for item in selected)
    assert all(any("整体偏柔" in reason for reason in item.reasons) for item in selected)


def test_four_frame_scene_without_faces_keeps_one_representative():
    import event_cull.engine as engine

    scene = [
        PhotoResult(Path(f"SCENE_{index}.jpg"), score=40 - index, sharpness=100, perceptual_hash=1)
        for index in range(4)
    ]
    competing = [
        PhotoResult(Path(f"HIGH_{index}.jpg"), score=200 - index, sharpness=100, perceptual_hash=2 + index)
        for index in range(3)
    ]

    engine._assign_tiered_ratings(
        [scene, *[[item] for item in competing]],
        AnalysisOptions(target_keep_count=4),
    )

    assert sum(item.suggested_rating > 0 for item in scene) == 1


def test_three_frame_scene_without_faces_keeps_one_representative():
    import event_cull.engine as engine

    scene = [
        PhotoResult(Path(f"SCENE_{index}.jpg"), score=40 - index, sharpness=100, perceptual_hash=1)
        for index in range(3)
    ]
    competing = [
        PhotoResult(Path(f"HIGH_{index}.jpg"), score=200 - index, sharpness=100, perceptual_hash=2 + index)
        for index in range(3)
    ]

    engine._assign_tiered_ratings(
        [scene, *[[item] for item in competing]],
        AnalysisOptions(target_keep_count=4),
    )

    assert sum(item.suggested_rating > 0 for item in scene) == 1


def test_scene_minimum_prefers_normal_face_over_suspected_closed_eye():
    import event_cull.engine as engine

    options = AnalysisOptions(target_keep_count=1)
    suspected_closed = PhotoResult(
        Path("closed.jpg"),
        score=100,
        sharpness=100,
        perceptual_hash=1,
        frontal_face_count=1,
        open_eye_face_count=0,
        reasons=[engine._message(options, "closed")],
    )
    normal = PhotoResult(
        Path("normal.jpg"),
        score=95,
        sharpness=100,
        perceptual_hash=1,
        frontal_face_count=1,
        open_eye_face_count=1,
    )
    alternate = PhotoResult(
        Path("alternate.jpg"),
        score=80,
        sharpness=100,
        perceptual_hash=1,
    )

    engine._assign_tiered_ratings([[suspected_closed, normal, alternate]], options)

    assert normal.suggested_rating > 0
    assert suspected_closed.suggested_rating == 0


def test_six_frame_scene_without_faces_keeps_two_representatives():
    import event_cull.engine as engine

    scene = [
        PhotoResult(Path(f"SCENE_{index}.jpg"), score=40 - index, sharpness=100, perceptual_hash=1)
        for index in range(6)
    ]
    competing = [
        PhotoResult(Path(f"HIGH_{index}.jpg"), score=200 - index, sharpness=100, perceptual_hash=2 + index)
        for index in range(3)
    ]

    engine._assign_tiered_ratings(
        [scene, *[[item] for item in competing]],
        AnalysisOptions(target_keep_count=5),
    )

    assert sum(item.suggested_rating > 0 for item in scene) == 2


def test_moderately_soft_five_frame_scene_without_faces_gets_review_safeguard():
    import event_cull.engine as engine

    scene = [
        PhotoResult(
            Path(f"SOFT_{index}.jpg"),
            score=40 - index,
            reasons=["明显虚焦或抖动"],
            sharpness=35 + index,
            highlight_clip=0,
            shadow_clip=0,
            perceptual_hash=1,
        )
        for index in range(5)
    ]

    engine._assign_tiered_ratings(
        [scene],
        AnalysisOptions(target_keep_count=1),
    )

    selected = [item for item in scene if item.suggested_rating > 0]
    assert len(selected) == 1
    assert selected[0].coverage_protected
    assert any("连续场景保底" in reason for reason in selected[0].reasons)


def test_catastrophically_blurred_scene_is_not_forced_into_selection():
    import event_cull.engine as engine

    scene = [
        PhotoResult(
            Path(f"BLUR_{index}.jpg"),
            score=100,
            sharpness=10,
            highlight_clip=0,
            shadow_clip=0,
            perceptual_hash=1,
        )
        for index in range(5)
    ]

    engine._assign_tiered_ratings(
        [scene],
        AnalysisOptions(target_keep_count=5),
    )

    assert all(item.suggested_rating == 0 for item in scene)


def test_relative_one_stop_outlier_is_rejected_from_stable_burst():
    import event_cull.engine as engine

    normal_left = PhotoResult(
        Path("left.jpg"), score=90, sharpness=100, perceptual_hash=1,
        median_luminance=100,
    )
    bright = PhotoResult(
        Path("bright.jpg"), score=120, sharpness=100, perceptual_hash=1,
        median_luminance=145,
    )
    normal_right = PhotoResult(
        Path("right.jpg"), score=80, sharpness=100, perceptual_hash=1,
        median_luminance=102,
    )

    engine._assign_tiered_ratings(
        [[normal_left, bright, normal_right]],
        AnalysisOptions(target_keep_count=3),
    )

    assert bright.relative_exposure_ev is not None
    assert bright.relative_exposure_ev >= 1.0
    assert bright.suggested_rating == 0
    assert any("突然过曝" in reason for reason in bright.reasons)


def test_relative_subject_darkening_is_a_flash_misfire_and_is_not_selected():
    import event_cull.engine as engine

    normal_left = PhotoResult(
        Path("left.jpg"), score=80, sharpness=100, perceptual_hash=42,
        median_luminance=130, subject_luminance=160,
    )
    flash_miss = PhotoResult(
        Path("flash-miss.jpg"), score=98, sharpness=100, perceptual_hash=42,
        median_luminance=128, subject_luminance=100,
    )
    normal_right = PhotoResult(
        Path("right.jpg"), score=82, sharpness=100, perceptual_hash=42,
        median_luminance=129, subject_luminance=162,
    )

    engine._assign_tiered_ratings(
        [[normal_left, flash_miss, normal_right]],
        AnalysisOptions(target_keep_count=3),
    )

    assert flash_miss.relative_exposure_ev is not None
    assert flash_miss.relative_exposure_ev <= -1.0
    assert flash_miss.suggested_rating == 0
    assert any("疑似闪光灯漏闪" in reason for reason in flash_miss.reasons)


def test_best_smile_gets_three_stars_and_another_frame_survives():
    import event_cull.engine as engine

    plain = PhotoResult(
        Path("plain.jpg"), score=100, sharpness=100, perceptual_hash=1,
        frontal_face_count=1, open_eye_face_count=1,
    )
    smiling = PhotoResult(
        Path("smiling.jpg"), score=92, sharpness=100, perceptual_hash=1,
        frontal_face_count=1, smiling_face_count=1, open_eye_face_count=1,
    )
    backup = PhotoResult(
        Path("backup.jpg"), score=80, sharpness=100, perceptual_hash=1,
        frontal_face_count=1, open_eye_face_count=1,
    )

    engine._assign_tiered_ratings(
        [[plain, smiling, backup]],
        AnalysisOptions(target_keep_count=2),
    )

    assert smiling.suggested_rating == 3
    assert sum(item.suggested_rating > 0 for item in (plain, smiling, backup)) == 2


def test_mixed_group_expression_and_front_focus_cannot_be_three_star():
    import event_cull.engine as engine

    bad_expression = PhotoResult(
        Path("expression.jpg"), score=140, sharpness=100, subject_sharpness=100,
        perceptual_hash=1, frontal_face_count=3, open_eye_face_count=2,
    )
    front_focus = PhotoResult(
        Path("front-focus.jpg"), score=130, sharpness=180, subject_sharpness=30,
        perceptual_hash=1, frontal_face_count=1, open_eye_face_count=1,
    )
    safe = PhotoResult(
        Path("safe.jpg"), score=90, sharpness=100, subject_sharpness=100,
        perceptual_hash=1, frontal_face_count=3, open_eye_face_count=3,
    )

    engine._assign_tiered_ratings(
        [[bad_expression, front_focus, safe]],
        AnalysisOptions(target_keep_count=3),
    )

    assert safe.suggested_rating == 3
    assert bad_expression.suggested_rating != 3
    assert front_focus.suggested_rating != 3


def test_smile_survives_when_another_person_is_not_looking_at_camera():
    import event_cull.engine as engine

    smiling_moment = PhotoResult(
        Path("smile.jpg"), score=90, sharpness=100, subject_sharpness=100,
        perceptual_hash=1, frontal_face_count=2, smiling_face_count=1,
        open_eye_face_count=1,
    )
    neutral = PhotoResult(
        Path("neutral.jpg"), score=105, sharpness=100, subject_sharpness=100,
        perceptual_hash=1, frontal_face_count=2, smiling_face_count=0,
        open_eye_face_count=2,
    )
    alternate = PhotoResult(
        Path("alternate.jpg"), score=80, sharpness=100, perceptual_hash=1,
    )

    engine._assign_tiered_ratings(
        [[smiling_moment, neutral, alternate]],
        AnalysisOptions(target_keep_count=1),
    )

    assert smiling_moment.suggested_rating == 3
    assert not engine._group_expression_issue(smiling_moment)


def test_each_smiling_scene_gets_a_smile_before_ordinary_scene_leads():
    import event_cull.engine as engine

    smiling_scene = [
        PhotoResult(
            Path("smile.jpg"), score=50, sharpness=100, perceptual_hash=1,
            frontal_face_count=1, smiling_face_count=1,
        ),
        PhotoResult(Path("neutral.jpg"), score=60, sharpness=100, perceptual_hash=1),
    ]
    ordinary_scenes = [
        [PhotoResult(Path(f"ordinary-{index}.jpg"), score=200 - index, sharpness=100, perceptual_hash=10 + index)]
        for index in range(3)
    ]

    engine._assign_tiered_ratings(
        [smiling_scene, *ordinary_scenes],
        AnalysisOptions(target_keep_count=3),
    )

    assert smiling_scene[0].suggested_rating > 0


def test_default_similarity_accepts_small_burst_composition_changes(tmp_path: Path):
    left = tmp_path / "left.jpg"
    right = tmp_path / "right.jpg"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    left.touch()
    right.touch()
    import event_cull.engine as engine
    first = PhotoResult(left, perceptual_hash=0)
    second = PhotoResult(right, perceptual_hash=(1 << 16) - 1)

    assert engine._hamming(first.perceptual_hash, second.perceptual_hash) == 16
    assert engine._same_group(first, second, AnalysisOptions())


def test_missing_eye_hint_does_not_reject_a_unique_action(monkeypatch, tmp_path: Path):
    path = tmp_path / "IMG_0.jpg"
    path.write_bytes(b"fixture")
    import event_cull.engine as engine
    options = AnalysisOptions(target_keep_count=100)
    result = PhotoResult(
        path,
        score=90,
        reasons=[engine._message(options, "closed")],
        sharpness=100,
        perceptual_hash=1,
    )
    monkeypatch.setattr(engine, "_representatives", lambda folder: [(path, [path])])
    monkeypatch.setattr(engine, "_analyze", lambda path, pairs, options: result)

    analyzed = analyze_folder(tmp_path, options)

    assert analyzed[0].suggested_rating == 3


def test_people_priority_never_promotes_a_blurry_frame(monkeypatch, tmp_path: Path):
    blurry = tmp_path / "IMG_0.jpg"
    clear = tmp_path / "IMG_1.jpg"
    blurry.write_bytes(b"blurry")
    clear.write_bytes(b"clear")
    values = {
        blurry: PhotoResult(
            blurry,
            score=150,
            sharpness=10,
            perceptual_hash=1,
            frontal_face_count=6,
            smiling_face_count=6,
        ),
        clear: PhotoResult(clear, score=80, sharpness=100, perceptual_hash=1),
    }
    import event_cull.engine as engine
    monkeypatch.setattr(engine, "_representatives", lambda folder: [(path, [path]) for path in (blurry, clear)])
    monkeypatch.setattr(engine, "_analyze", lambda path, pairs, options: values[path])
    monkeypatch.setattr(engine, "_same_group", lambda previous, current, options: True)

    analyzed = analyze_folder(tmp_path, AnalysisOptions(target_keep_count=1))

    assert analyzed[0].suggested_rating == 0
    assert analyzed[1].suggested_rating == 3


def test_white_highlights_alone_are_not_severe_overexposure():
    import event_cull.engine as engine

    wedding_frame = PhotoResult(
        Path("wedding.jpg"),
        sharpness=100,
        highlight_clip=0.30,
        shadow_clip=0.0,
        perceptual_hash=1,
        median_luminance=128,
        subject_luminance=165,
    )

    assert not engine._has_hard_fault(wedding_frame, AnalysisOptions())


def test_whole_frame_or_subject_midtone_overexposure_is_a_hard_fault():
    import event_cull.engine as engine

    whole_frame_white = PhotoResult(
        Path("whole-white.jpg"),
        sharpness=100,
        highlight_clip=0.30,
        shadow_clip=0.0,
        perceptual_hash=1,
        median_luminance=220,
        subject_luminance=None,
    )
    blown_skin = PhotoResult(
        Path("blown-skin.jpg"),
        sharpness=100,
        highlight_clip=0.20,
        shadow_clip=0.0,
        perceptual_hash=2,
        median_luminance=150,
        subject_luminance=230,
    )

    assert engine._has_hard_fault(whole_frame_white, AnalysisOptions())
    assert engine._has_hard_fault(blown_skin, AnalysisOptions())


def test_single_capture_stays_in_review_instead_of_becoming_an_unrated_reject(monkeypatch, tmp_path: Path):
    path = tmp_path / "only.jpg"
    path.write_bytes(b"fixture")
    result = PhotoResult(path, score=80, perceptual_hash=1)
    import event_cull.engine as engine
    monkeypatch.setattr(engine, "_representatives", lambda folder: [(path, [path])])
    monkeypatch.setattr(engine, "_analyze", lambda path, pairs, options: result)
    results = analyze_folder(tmp_path)
    assert results[0].suggested_rating == 3
    assert "本组最佳" in results[0].reasons


def test_english_analysis_reasons_follow_requested_language(monkeypatch, tmp_path: Path):
    path = tmp_path / "only.jpg"
    path.write_bytes(b"fixture")
    result = PhotoResult(path, score=80, perceptual_hash=1)
    import event_cull.engine as engine
    monkeypatch.setattr(engine, "_representatives", lambda folder: [(path, [path])])
    monkeypatch.setattr(engine, "_analyze", lambda path, pairs, options: result)
    results = analyze_folder(tmp_path, AnalysisOptions(language="en"))
    assert "Best in group" in results[0].reasons


def test_target_count_creates_cumulative_rating_tiers():
    import event_cull.engine as engine

    groups = [
        [PhotoResult(Path(f"IMG_{index:04d}.jpg"), score=2000 - index, perceptual_hash=index + 1)]
        for index in range(1000)
    ]

    engine._assign_tiered_ratings(groups, AnalysisOptions(target_keep_count=300))

    counts = {
        rating: sum(item.suggested_rating == rating for group in groups for item in group)
        for rating in range(4)
    }
    assert counts == {0: 700, 1: 200, 2: 70, 3: 30}


def test_only_one_three_star_is_selected_from_each_similar_group():
    import event_cull.engine as engine

    groups = [
        [
            PhotoResult(Path(f"G{group_index}_{item_index}.jpg"), score=100 - item_index, perceptual_hash=group_index + 1)
            for item_index in range(5)
        ]
        for group_index in range(3)
    ]

    engine._assign_tiered_ratings(
        groups,
        AnalysisOptions(picks_per_group=2, target_keep_count=15),
    )

    assert all(sum(item.suggested_rating == 3 for item in group) <= 1 for group in groups)


def test_analysis_reports_each_processed_photo(monkeypatch, tmp_path: Path):
    paths = []
    for number in range(2):
        path = tmp_path / f"IMG_{number}.jpg"
        path.write_bytes(b"fixture")
        paths.append(path)
    values = {
        paths[0]: PhotoResult(paths[0], score=80, perceptual_hash=1),
        paths[1]: PhotoResult(paths[1], score=70, perceptual_hash=10),
    }
    import event_cull.engine as engine
    monkeypatch.setattr(engine, "_representatives", lambda folder: [(path, [path]) for path in paths])
    monkeypatch.setattr(engine, "_analyze", lambda path, pairs, options: values[path])
    callbacks = []
    analyze_folder(tmp_path, progress_callback=lambda current, total, path: callbacks.append((current, total, path.name)))
    assert callbacks == [(1, 2, "IMG_0.jpg"), (2, 2, "IMG_1.jpg")]


def test_raw_jpeg_pair_uses_jpeg_for_analysis_but_keeps_raw_as_rating_target(monkeypatch, tmp_path: Path):
    raw = tmp_path / "IMG_0001.ARW"
    jpeg = tmp_path / "IMG_0001.JPG"
    raw.write_bytes(b"raw")
    jpeg.write_bytes(b"jpeg")
    import event_cull.engine as engine
    seen = []
    monkeypatch.setattr(engine, "_load_preview", lambda path: seen.append(path) or __import__("numpy").zeros((100, 100, 3), dtype="uint8"))
    result = engine._analyze(raw, [raw, jpeg], AnalysisOptions())
    assert seen == [jpeg]
    assert result.path == raw


def test_cancel_stops_before_next_photo(monkeypatch, tmp_path: Path):
    path = tmp_path / "IMG_0.jpg"
    path.write_bytes(b"fixture")
    import event_cull.engine as engine
    monkeypatch.setattr(engine, "_representatives", lambda folder: [(path, [path])])
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(AnalysisCancelled):
        analyze_folder(tmp_path, cancel_event=cancelled)
