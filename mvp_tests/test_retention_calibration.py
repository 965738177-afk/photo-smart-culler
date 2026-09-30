"""Shared original-delivery preference contract, not filename memorization."""
from dataclasses import replace
from pathlib import Path
import threading

import numpy as np
import pytest

from event_cull import calibration, deep_features, engine
from scripts.evaluate_retention_calibration import purged_folds


def profile():
    return calibration.CalibrationProfile((0.,), (1.,), (1.,), 0., 1, 100, 20, .7,
        feature_pipeline="fixture", objective="retention", scene_type="婚礼")


def photo(name, value, rating=0, group=1):
    return engine.PhotoResult(Path(name), group_id=group, suggested_rating=rating, score=90,
        sharpness=200, perceptual_hash=123, shadow_clip=0., highlight_clip=0.,
        median_luminance=100, visual_features=[value], visual_feature_pipeline="fixture")


def test_original_preferences_can_recover_a_previously_unselected_frame_without_decode(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("must reuse features from the shared analysis preview")
    monkeypatch.setattr(calibration, "_load_preview", fail)
    old, new = photo("old.jpg", -2, 3, 1), photo("new.jpg", 2, 0, 2)
    opts = engine.AnalysisOptions(scene_type="婚礼", target_keep_count=1, target_keep_max_count=1)
    assert calibration.apply_profile([old, new], profile(), options=opts) == 1
    assert old.suggested_rating == 0 and new.suggested_rating > 0


def test_manual_zero_and_two_are_preserved_and_count_towards_the_delivery_ceiling():
    zero = photo("manual-zero.jpg", 99, 0)
    two = photo("manual-two.jpg", -99, 2)
    low, high = photo("low.jpg", -2, 3, 2), photo("high.jpg", 2, 0, 3)
    opts = engine.AnalysisOptions(scene_type="婚礼", target_keep_count=2, target_keep_max_count=2)
    calibration.apply_profile([zero, two, low, high], profile(), options=opts,
                              protected_paths=frozenset({zero.path}))
    assert [p.suggested_rating for p in [zero, two, low]] == [0, 2, 0]
    assert high.suggested_rating in {1, 3}
    assert sum(p.suggested_rating > 0 for p in [zero, two, low, high]) == 2


def test_personal_preference_never_rescues_a_true_technical_failure():
    good, bad = photo("good.jpg", 0, 1), photo("broken.jpg", 99, 0)
    bad.perceptual_hash = None
    opts = engine.AnalysisOptions(scene_type="婚礼", target_keep_count=1, target_keep_max_count=1)
    calibration.apply_profile([good, bad], profile(), options=opts)
    assert good.suggested_rating > 0 and bad.suggested_rating == 0


def test_retention_preferences_do_not_cross_scene_or_override_manual_only_mode(monkeypatch):
    monkeypatch.setattr(calibration, "_feature_rows", lambda *a, **k: pytest.fail("not an applicable objective"))
    item = photo("one.jpg", 1, 1)
    for options in (engine.AnalysisOptions(scene_type="男生氛围写真"),
                    engine.AnalysisOptions(scene_type="婚礼", ai_selection_enabled=False)):
        assert calibration.apply_profile([item], profile(), options=options) == 0
        assert item.suggested_rating == 1


def test_retention_cancel_is_transactional():
    items = [photo("a.jpg", -2, 3), photo("b.jpg", 2, 0)]
    count = 0
    def stop():
        nonlocal count
        count += 1
        if count == 3:
            raise engine.AnalysisCancelled()
    with pytest.raises(engine.AnalysisCancelled):
        calibration.apply_profile(items, profile(), checkpoint=stop)
    assert [p.suggested_rating for p in items] == [3, 0]
    assert all(p.calibration_confidence is None for p in items)


def test_zero_automatic_budget_does_not_add_a_hidden_extra_photo():
    manual, other = photo("manual.jpg", -1, 2), photo("other.jpg", 1, 3, 2)
    calibration.apply_profile([manual, other], profile(), options=engine.AnalysisOptions(
        scene_type="婚礼", target_keep_count=1, target_keep_max_count=1))
    assert manual.suggested_rating == 2 and other.suggested_rating == 0


def test_constant_auc_is_half_regardless_of_row_order():
    labels = np.asarray([0, 0, 1, 1])
    assert calibration._roc_auc(labels, np.ones(4)) == .5
    assert calibration._roc_auc(labels[::-1], np.ones(4)) == .5


@pytest.mark.parametrize("field,value", [("scale", [0.]), ("weights", [float("nan")]),
                                         ("objective", "unknown"), ("validation_precision", 2)])
def test_invalid_saved_profiles_cannot_enable_bad_scoring(field, value):
    payload = profile().to_dict()
    payload[field] = value
    with pytest.raises(calibration.CalibrationError):
        calibration.CalibrationProfile.from_dict(payload)


def test_v2_profiles_do_not_change_the_original_delivery_objective():
    payload = profile().to_dict()
    payload["version"] = 2
    del payload["objective"]
    restored = calibration.CalibrationProfile.from_dict(payload)
    assert restored.objective == "featured"


def test_deep_feature_channels_are_not_arbitrarily_averaged():
    values = np.arange(576, dtype=np.float32).reshape(1, 576)
    assert np.array_equal(deep_features.DeepVisualEncoder._compress(values), values)


def test_old_deep_feature_profiles_still_have_their_old_dimensions(monkeypatch):
    monkeypatch.setattr(calibration, "encode_images", lambda images: (np.ones((len(images),576)), "cpu"))
    image = np.zeros((20,20,3), np.uint8)
    assert calibration._feature_matrix([image], deep_features.LEGACY_PIPELINE_ID).shape == (1,328)
    assert calibration._feature_matrix([image], deep_features.PIPELINE_ID).shape == (1,832)


def test_continuous_validation_blocks_never_split_one_burst_and_purge_neighbours():
    bursts = [str(i//7) for i in range(1000)]
    blocks = calibration.blocked_validation_groups(bursts)
    for burst in set(bursts):
        assert len({block for b,block in zip(bursts,blocks) if b == burst}) == 1
    seen = np.zeros(1000, dtype=int)
    for training, held_out in purged_folds(blocks):
        seen += held_out
        assert not (training & held_out).any()
        forbidden = {int(block.rsplit('-',1)[1]) + offset
                     for block in np.asarray(blocks)[held_out] for offset in (-1,0,1)}
        assert all(int(block.rsplit('-',1)[1]) not in forbidden for block in np.asarray(blocks)[training])
    assert (seen == 1).all()


@pytest.mark.parametrize("scene,subject,sharp,faces,expected", [
    ("婚礼", 100, 200, 1, False), ("婚礼", 30, 200, 1, True),
    ("婚礼", None, 200, 0, True), ("婚礼", 100, 10, 1, True),
    ("通用", 100, 200, 1, True),
])
def test_background_darkness_requires_a_usable_measured_wedding_subject(scene, subject, sharp, faces, expected):
    item = photo("stage.jpg", 1)
    item.subject_luminance, item.subject_sharpness = subject, sharp
    item.frontal_face_count = faces
    item.shadow_clip, item.median_luminance = .6, 20
    assert engine._has_exposure_fault(item, engine.AnalysisOptions(scene_type=scene)) is expected
    item.relative_exposure_ev = -1.2
    assert engine._has_exposure_fault(item, engine.AnalysisOptions(scene_type=scene))


def test_analysis_encoder_reuses_the_already_decoded_image(monkeypatch, tmp_path):
    path = tmp_path / "test.jpg"
    image = np.zeros((32,32,3),np.uint8)
    decoded, encoded = [], []
    monkeypatch.setattr(engine, "_load_preview", lambda p: decoded.append(p) or image)
    monkeypatch.setattr(engine, "_face_metrics", lambda *a: ([],None,0,0,0,None))
    monkeypatch.setattr(calibration, "_feature_matrix", lambda images, pipeline: encoded.append(images[0]) or np.asarray([[1.]]))
    result = engine._analyze(path, [path], engine.AnalysisOptions(calibration_feature_pipeline="fixture"))
    assert decoded == [path] and len(encoded) == 1 and encoded[0] is image
    assert result.visual_features == [1.] and result.visual_feature_pipeline == "fixture"


def test_shared_encoder_initializes_once_under_concurrent_calls(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    made = []
    class Fake:
        backend = "cpu"
        def __init__(self, backend):
            made.append(backend)
        def encode(self, images):
            return np.ones((len(images),576))
    monkeypatch.setattr(deep_features, "preferred_backend", lambda: "cpu")
    monkeypatch.setattr(deep_features, "DeepVisualEncoder", Fake)
    monkeypatch.setattr(deep_features, "_ENCODERS", {})
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda _: deep_features.encode_images([np.zeros((1,1,3))]), range(20)))
    assert made == ["cpu"]


def test_invalid_feature_cache_retries_instead_of_disabling_preferences_forever(tmp_path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"file signature fixture")
    options = engine.AnalysisOptions(calibration_feature_pipeline="handcrafted-visual-v1")
    signature = engine._cache_signature(path, options)
    item = photo(str(path), 1)
    item.visual_features = [0.] * 256
    item.visual_feature_pipeline = "handcrafted-visual-v1"
    record = engine._cache_record(item, signature)
    assert engine._result_from_cache(path, [path], record, signature) is not None
    record["visual_features"][0] = float("nan")
    assert engine._result_from_cache(path, [path], record, signature) is None
    record["visual_features"] = [0.]
    assert engine._result_from_cache(path, [path], record, signature) is None


def test_profile_fit_is_cancellable_without_waiting_for_all_iterations():
    calls = []
    def stop():
        calls.append(True)
        raise engine.AnalysisCancelled()
    with pytest.raises(engine.AnalysisCancelled):
        calibration._fit_classifier(np.ones((100,4)), np.tile([0,1],50), checkpoint=stop)
    assert len(calls) == 1


def test_correlated_high_dimension_features_do_not_make_training_oscillate():
    labels = np.tile([0,1],100)
    features = np.repeat(labels[:,None],832,axis=1).astype(float)
    mean, scale, weights, bias = calibration._fit_classifier(features,labels)
    logits = (features-mean)/scale @ weights + bias
    assert np.mean(np.logaddexp(0,logits)-labels*logits) < .1
    assert calibration._roc_auc(labels,logits) == 1.


def test_retention_activation_requires_precision_and_recall_not_just_auc(monkeypatch):
    samples = [(np.asarray([float(i % 5 == 0)]), 1 if i%5 == 0 else 0, str(i)) for i in range(100)]
    # Held-out set has a different delivery density. Perfect separation/AUC is
    # not enough when the training-based quantity still misses most originals.
    holdout = np.asarray([i < 50 if i%5 == 0 else i < 25 for i in range(100)])
    monkeypatch.setattr(calibration, "_balanced_group_holdout", lambda *args: holdout)
    with pytest.raises(calibration.CalibrationError, match="75%"):
        calibration.train_profile(samples, 1, objective="retention")
