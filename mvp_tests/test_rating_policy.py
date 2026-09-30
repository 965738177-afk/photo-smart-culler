"""The 1/3 automatic delivery contract is shared across all rating boundaries."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from event_cull import calibration, engine
from event_cull.app import EventCullApp, SCENE_TYPES, TEXT
from event_cull.export_selected import collect_selected_files
from event_cull.rating_policy import AUTOMATIC_RATINGS, automatic_rating


@pytest.mark.parametrize("ceiling", [0, 1, 2, 3])
@pytest.mark.parametrize("retained,featured", [(False, False), (False, True), (True, False), (True, True)])
def test_automatic_policy_never_uses_manual_two_stars(ceiling, retained, featured):
    rating = automatic_rating(retained=retained, featured=featured, ceiling=ceiling)
    assert rating in AUTOMATIC_RATINGS
    assert (rating > 0) == (retained and ceiling > 0)
    assert (rating == 3) == (retained and featured and ceiling == 3)


@pytest.mark.parametrize("scene", SCENE_TYPES)
def test_every_scene_automatically_emits_only_zero_one_and_three(scene):
    groups = [[engine.PhotoResult(
        Path(f"{index}.jpg"), score=100-index, sharpness=150, subject_sharpness=150,
        perceptual_hash=index+1, frontal_face_count=1, open_eye_face_count=1,
        smiling_face_count=1, face_coverage=.2,
    )] for index in range(15)]
    engine._assign_tiered_ratings(groups, engine.AnalysisOptions(scene_type=scene, target_keep_count=10))
    ratings = [group[0].suggested_rating for group in groups]
    assert set(ratings) <= AUTOMATIC_RATINGS
    assert sum(rating > 0 for rating in ratings) == 10
    assert sum(rating == 3 for rating in ratings) == 2


def test_quality_ceiling_two_becomes_retained_one_not_manual_approval():
    item = engine.PhotoResult(Path("group.jpg"), score=100, sharpness=150,
                              subject_sharpness=150, perceptual_hash=1,
                              frontal_face_count=5, open_eye_face_count=5,
                              smiling_face_count=3, face_coverage=.2)
    engine._assign_tiered_ratings([[item]], engine.AnalysisOptions(scene_type="婚礼"))
    assert item.max_auto_rating == 2
    assert item.suggested_rating == 1


def test_three_star_promotion_never_expands_retained_pool(monkeypatch):
    groups = [[engine.PhotoResult(Path(f"{index}.jpg"), score=1000-index,
                                 perceptual_hash=index+1)] for index in range(1000)]
    retained = []
    original_selection = engine._density_selection

    def capture(*args, **kwargs):
        picked = original_selection(*args, **kwargs)
        retained.extend(item.path for item in picked)
        return picked

    monkeypatch.setattr(engine, "_density_selection", capture)
    engine._assign_tiered_ratings(groups, engine.AnalysisOptions(target_keep_count=300))
    items = [group[0] for group in groups]
    assert len(retained) == 300
    assert {item.path for item in items if item.suggested_rating > 0} == set(retained)
    assert {item.path for item in items if item.suggested_rating == 3} <= set(retained)
    assert sum(item.suggested_rating == 2 for item in items) == 0


def test_hard_faults_remain_unrated_even_when_target_exceeds_batch():
    failed = engine.PhotoResult(Path("undecodable.jpg"), score=999)
    blurred = engine.PhotoResult(Path("blurred.jpg"), score=999, perceptual_hash=1,
                                 sharpness=1, subject_sharpness=1)
    sound = engine.PhotoResult(Path("sound.jpg"), score=90, perceptual_hash=2,
                              sharpness=150, subject_sharpness=150)
    engine._assign_tiered_ratings([[failed], [blurred], [sound]],
                                 engine.AnalysisOptions(target_keep_count=300))
    assert failed.suggested_rating == blurred.suggested_rating == 0
    assert sound.suggested_rating == 3


def test_calibration_keeps_manual_two_without_decoding_it(monkeypatch):
    old = engine.PhotoResult(Path("old.jpg"), group_id=1, suggested_rating=3)
    replacement = engine.PhotoResult(Path("better.jpg"), group_id=1, suggested_rating=1)
    manual = engine.PhotoResult(Path("manual.jpg"), group_id=1, suggested_rating=2,
                               reasons=["photographer approved"])
    reject = engine.PhotoResult(Path("reject.jpg"), group_id=2, suggested_rating=0)
    decoded = []

    def features(paths, *_args, **_kwargs):
        decoded.extend(paths)
        for index in range(len(paths)):
            yield index, [index / 10]

    monkeypatch.setattr(calibration, "_feature_rows", features)
    profile = SimpleNamespace(feature_pipeline="fixture", probability_for_feature=lambda feature: feature[0])
    before = {item.path for item in (old, replacement, manual, reject) if item.suggested_rating > 0}
    calibration.apply_profile([old, replacement, manual, reject], profile)
    assert decoded == [old.path, replacement.path]
    assert (old.suggested_rating, replacement.suggested_rating) == (1, 3)
    assert manual.suggested_rating == 2 and manual.reasons == ["photographer approved"]
    assert reject.suggested_rating == 0
    assert {item.path for item in (old, replacement, manual, reject) if item.suggested_rating > 0} == before


def test_reanalysis_restores_manual_two_after_automatic_rating():
    path = Path("approved.jpg")
    item = engine.PhotoResult(path, score=100, sharpness=150, perceptual_hash=1)
    engine._assign_tiered_ratings([[item]], engine.AnalysisOptions())
    assert item.suggested_rating == 3
    EventCullApp._restore_manual_ratings([item], {path: 2}, "zh")
    assert item.suggested_rating == EventCullApp._mapped_rating(item.suggested_rating, 4, 5) == 2


def test_full_delivery_export_includes_retained_manual_and_retouch_without_duplicates(tmp_path):
    photos = [tmp_path / f"{rating}.JPG" for rating in range(4)]
    for photo in photos:
        photo.touch()
    items = [engine.PhotoResult(photo, suggested_rating=rating) for rating, photo in enumerate(photos)]
    full = collect_selected_files(items)
    retouch = collect_selected_files(items, frozenset({3}))
    assert full.photo_count == 3 and retouch.photo_count == 1
    assert set(full.files_by_folder[tmp_path.resolve()]) == set(photos[1:])
    assert set(retouch.files_by_folder[tmp_path.resolve()]) == {photos[3]}
    assert all(EventCullApp._matches_rating_filter(rating, -1) == (rating > 0) for rating in range(4))
    assert "1星以上" in TEXT["zh"]["filter_retained"]
