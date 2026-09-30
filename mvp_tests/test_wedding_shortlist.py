"""Wedding shortlist policy tests: detector evidence is not ground truth."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from event_cull import calibration, engine


def portrait(name, faces=5, eyes=5, smiles=5, score=90, **kwargs):
    return engine.PhotoResult(Path(name), score=score, sharpness=150,
                              subject_sharpness=150, perceptual_hash=1,
                              frontal_face_count=faces, open_eye_face_count=eyes,
                              smiling_face_count=smiles, face_coverage=.2,
                              eye_detection_mode="scene_group", **kwargs)


def rank(group, **kwargs):
    engine._assign_tiered_ratings([group], engine.AnalysisOptions(
        scene_type="婚礼", target_keep_count=len(group), **kwargs))


@pytest.mark.parametrize("eyes,smiles", [(4, 5), (0, 5), (5, 0), (5, 2)])
def test_complete_smiling_group_beats_higher_score(eyes, smiles):
    good = portrait("good.jpg")
    bad = portrait("bad.jpg", eyes=eyes, smiles=smiles, score=140)
    rank([bad, good])
    assert good.suggested_rating == 3
    assert bad.suggested_rating == 1
    assert bad.featured_blocked


def test_fewer_frontal_faces_cannot_win_by_smile_ratio():
    fewer = portrait("turned.jpg", faces=3, eyes=3, smiles=3, score=140)
    full = portrait("full.jpg", smiles=4)
    rank([fewer, full])
    # New evidence policy refuses a 3-star all-smiling claim when one smile
    # is unobserved. Retain the original without occupying manual two stars.
    assert full.suggested_rating == 1
    assert fewer.featured_blocked


def test_only_one_pick_still_selects_complete_expression():
    good = portrait("good.jpg")
    bad = portrait("bad.jpg", eyes=2, score=140)
    engine._assign_tiered_ratings([[bad, good]], engine.AnalysisOptions(
        scene_type="婚礼", target_keep_count=1, max_duplicate_picks_per_group=1))
    assert good.suggested_rating == 3
    assert bad.suggested_rating == 0


def test_no_complete_group_does_not_force_three_stars():
    group = [portrait("a.jpg", eyes=3), portrait("b.jpg", eyes=0)]
    rank(group)
    assert all(0 < item.suggested_rating < 3 for item in group)


def test_couple_kiss_and_details_remain_eligible():
    couple = portrait("kiss.jpg", faces=2, eyes=0, smiles=0)
    detail = portrait("rings.jpg", faces=0, eyes=0, smiles=0)
    rank([couple])
    assert couple.suggested_rating == 3
    rank([detail])
    assert detail.suggested_rating == 3


def test_detector_disabled_does_not_impose_open_eye_gate():
    item = portrait("a.jpg", eyes=0)
    rank([item], detect_closed_eyes=False)
    assert item.suggested_rating == 3


def test_nearest_pair_does_not_exempt_whole_group():
    options = engine.AnalysisOptions(scene_type="婚礼")
    assert engine._eye_policy(options, 5, .2, 3, 1.0).mode == "scene_group"
    assert engine._eye_policy(options, 2, .2, 0, 1.0).mode == "scene_intimate"


def test_calibration_cannot_promote_blocked_photo(monkeypatch):
    good = portrait("good.jpg", suggested_rating=3)
    bad = portrait("bad.jpg", suggested_rating=2, featured_blocked=True)
    seen = []
    def features(paths, *args, **kwargs):
        seen.extend(paths)
        yield 0, [1]
    monkeypatch.setattr(calibration, "_feature_rows", features)
    profile = SimpleNamespace(feature_pipeline="test", featured_rating=3,
                              probability_for_feature=lambda feature: .9)
    calibration.apply_profile([good, bad], profile)
    assert seen == [good.path]
    assert good.suggested_rating == 3
    assert bad.suggested_rating == 2


def test_labelled_documentary_emotion_is_not_posed_group():
    item = portrait("C4/交接.jpg", eyes=1, smiles=0)
    rank([item])
    assert item.must_keep_id == "C4"
    assert not item.featured_blocked


def test_all_ten_faces_are_examined_in_wedding_mode(monkeypatch):
    rectangles = [(10 + 55 * (index % 5), 10 + 60 * (index // 5), 40, 40)
                  for index in range(10)]
    face_detector = SimpleNamespace(detectMultiScale=lambda *a, **k: rectangles)
    feature_detector = SimpleNamespace(detectMultiScale=lambda *a, **k: [(1, 1, 12, 12)])
    monkeypatch.setattr(engine, "_DETECTORS", SimpleNamespace(
        faces=face_detector, eyes=feature_detector, smiles=feature_detector))
    gray = np.random.default_rng(1).integers(0, 256, (160, 300), dtype=np.uint8)
    metrics = engine._face_metrics(gray, engine.AnalysisOptions(scene_type="婚礼"))
    assert metrics[2:5] == (10, 10, 10)


def test_non_wedding_group_policy_unchanged():
    item = portrait("family.jpg", eyes=1, smiles=3)
    engine._assign_tiered_ratings([[item]], engine.AnalysisOptions(scene_type="儿童与家庭"))
    assert not item.featured_blocked
