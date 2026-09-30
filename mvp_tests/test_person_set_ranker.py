"""Per-person association/runtime tests, not wedding accuracy evidence."""
from pathlib import Path

import numpy as np
import pytest
import torch

from event_cull.engine import PhotoResult
from event_cull.person_set_ranker import (PersonSetRanker, person_sets, _make_network,
                                         FACE_INPUT_DIMENSION, MAX_FACES, body_sets,
                                         BODY_INPUT_DIMENSION)
from src.core.image_features import human_analysis as human


def face(smile=.2, blink=.2, *, size=.2, reliable=True):
    features = np.zeros(human.FACE_DIMENSION, np.float32)
    for key, value in (("mouthSmileLeft", smile), ("mouthSmileRight", smile),
                       ("eyeBlinkLeft", blink), ("eyeBlinkRight", blink)):
        features[human.BLENDSHAPES.index(key)] = value
    features[-1] = 5
    return {"features": features.tolist(), "bbox": [.1, .1, .1+size, .1+size],
            "reliable_eyes": reliable}


def item(faces):
    return PhotoResult(Path("ignored.ARW"), person_evidence=human.aggregate_observation(faces, []))


def test_shared_old_pooling_really_loses_same_person_expression_association():
    first = item([face(size=.5), face(.9, .9), face(.1, .1)])
    second = item([face(size=.5), face(.9, .1), face(.1, .9)])
    assert np.array_equal(first.person_evidence["features"], second.person_evidence["features"])
    sets, mask = person_sets([first, second])
    assert mask.all(axis=1).sum() == 0  # No dummy person gets observation status.
    assert not np.array_equal(sets[0], sets[1])


def test_padding_and_unreliable_eyes_are_distinct_unknown_states(monkeypatch):
    from event_cull import engine
    monkeypatch.setattr(engine, "_load_preview", lambda *a, **k: pytest.fail("no extra image decode"))
    sets, mask = person_sets([item([]), item([face(reliable=False)])])
    assert not mask[0].any() and not sets[0].any()
    assert mask[1, 0] and not mask[1, 1:].any() and sets[1, 0, -1] == 0


def test_person_cap_uses_geometry_not_names_or_labels():
    records = [face(smile=i/20, size=.02+i*.01) for i in range(16)]
    source = item(records)
    sets, mask = person_sets([source])
    assert mask.sum() == MAX_FACES and sets.shape == (1, MAX_FACES, FACE_INPUT_DIMENSION)
    assert sets[0, 0, human.BLENDSHAPES.index("mouthSmileLeft")] == pytest.approx(.75)
    source.path = Path("renamed.jpg")
    source.suggested_rating = 3
    assert np.array_equal(sets, person_sets([source])[0])
    assert len(source.person_evidence["faces"]) == 16


@pytest.mark.parametrize("change", ["pipeline", "nan", "bbox", "reliability", "null-set", "null-record"])
def test_incomplete_person_records_fail_instead_of_silent_subset(change):
    record = item([face()])
    if change == "pipeline":
        record.person_evidence["pipeline"] = "wrong"
    elif change == "nan":
        record.person_evidence["faces"][0]["features"][0] = float("nan")
    elif change == "bbox":
        record.person_evidence["faces"][0]["bbox"] = [1, 1, 0, 0]
    elif change == "reliability":
        record.person_evidence["faces"][0].pop("reliable_eyes")
    elif change == "null-set":
        record.person_evidence["faces"] = None
    else:
        record.person_evidence["faces"][0] = None
    with pytest.raises(ValueError):
        person_sets([record])


def test_network_is_person_order_invariant_and_ignores_padding():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = _make_network(3, FACE_INPUT_DIMENSION, width=8).eval()
        x = torch.randn((2, 3))
        faces = torch.randn((2, 5, FACE_INPUT_DIMENSION))
        mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        with torch.inference_mode():
            before = model(x, faces, mask)
            order = [4, 2, 0, 3, 1]
            after = model(x, faces[:, order], mask[:, order])
            faces[~mask] = 100000
            padded = model(x, faces, mask)
        assert torch.isfinite(before).all()
        assert torch.allclose(before, after, atol=1e-6) and torch.allclose(before, padded, atol=1e-6)


def sample_training():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(160, 3)).astype(np.float32)
    labels = (x[:, 0] > 0).astype(np.int8)
    faces = rng.normal(size=(160, 3, FACE_INPUT_DIMENSION)).astype(np.float32)
    mask = np.asarray([[True, True, False]]*160)
    return x, labels, faces, mask


def test_training_only_scaling_masks_prediction_and_restores_caller_cpu_state():
    x, labels, faces, mask = sample_training()
    owner = PersonSetRanker(epochs=2, width=8)
    threads, state = torch.get_num_threads(), torch.random.get_rng_state()
    owner.fit(x, labels, faces, mask, groups=np.arange(160)//4)
    assert threads == torch.get_num_threads() and torch.equal(state, torch.random.get_rng_state())
    mean = owner._normalization[0].copy()
    face_mean = owner._normalization[2].copy()
    before = owner.predict(x[:10], faces[:10], mask[:10])
    changed = faces[:10].copy()
    changed[~mask[:10]] = 10000
    after = owner.predict(x[:10], changed, mask[:10])
    owner.predict(x[:10]+10000, faces[:10]+10000, mask[:10])
    assert np.array_equal(mean, owner._normalization[0])
    assert np.array_equal(face_mean, owner._normalization[2])
    assert np.allclose(before, after) and owner.training_pairs <= 8*len(x)
    owner.close()
    with pytest.raises(ValueError, match="not been trained"):
        owner.predict(x[:10], faces[:10], mask[:10])


def test_cancelled_refit_keeps_last_model_and_delivery_density():
    calls, enabled = [0], [False]
    def checkpoint():
        calls[0] += 1
        if enabled[0] and calls[0] >= 2:
            raise InterruptedError("cancelled")
    x, labels, faces, mask = sample_training()
    owner = PersonSetRanker(checkpoint=checkpoint, epochs=2, width=8).fit(x, labels, faces, mask)
    model, density = owner._model, owner.positive_fraction
    scores = owner.predict(x, faces, mask)
    calls[0], enabled[0] = 0, True
    with pytest.raises(InterruptedError):
        owner.fit(x, np.asarray([1]*120+[0]*40), faces, mask)
    enabled[0] = False
    assert owner._model is model and owner.positive_fraction == density
    assert np.array_equal(scores, owner.predict(x, faces, mask))
    owner.close()


def test_inference_batches_remain_bounded_and_checkpointed():
    x, labels, faces, mask = sample_training()
    owner = PersonSetRanker(epochs=1, width=8).fit(x, labels, faces, mask)
    calls = []
    hook = owner._model.register_forward_pre_hook(lambda module, inputs: calls.append(len(inputs[0])))
    prediction = owner.predict(np.tile(x, (6, 1)), np.tile(faces, (6, 1, 1)), np.tile(mask, (6, 1)))
    hook.remove()
    assert len(prediction) == 960 and calls == [256, 256, 256, 192]
    owner.close()


def observed_body(offset=0):
    joints = np.zeros((33, 4), np.float32)
    joints[:20, :3] = offset
    joints[:20, 3] = .9
    return {"joints":joints.tolist(), "bbox":[.1,.1,.8,.8],
            "visible_fraction":20/33, "bounds_are_complete_body":False}


def test_individual_joints_are_kept_and_legacy_average_is_not_recovered():
    records = [observed_body(.2), observed_body(-.2)]
    source = item([])
    source.person_evidence = human.aggregate_observation(
        [], [p["joints"] for p in records], pose_records=records)
    bodies, mask = body_sets([source])
    assert mask.sum() == 2 and bodies.shape == (1, human.MAX_POSES, BODY_INPUT_DIMENSION)
    assert bodies[0, 0, 0] == pytest.approx(.2) and bodies[0, 1, 0] == pytest.approx(-.2)
    source.person_evidence = human.aggregate_observation([], [p["joints"] for p in records])
    with pytest.raises(ValueError, match="not recoverable"):
        body_sets([source])


def test_joint_network_is_set_order_invariant_and_unknown_padding_is_ignored():
    with torch.random.fork_rng(devices=[]):
        model = _make_network(3, FACE_INPUT_DIMENSION, width=8, body_dimension=BODY_INPUT_DIMENSION).eval()
        x = torch.randn(2, 3)
        faces, face_mask = torch.randn(2, 3, FACE_INPUT_DIMENSION), torch.zeros(2, 3, dtype=torch.bool)
        bodies = torch.randn(2, human.MAX_POSES, BODY_INPUT_DIMENSION)
        mask = torch.tensor([[1,1,0,0,0,0], [0,0,0,0,0,0]], dtype=torch.bool)
        with torch.inference_mode():
            before = model(x, faces, face_mask, bodies, mask)
            order = [4,1,3,0,2,5]
            after = model(x, faces, face_mask, bodies[:,order], mask[:,order])
            bodies[~mask] = 100000
            padded = model(x, faces, face_mask, bodies, mask)
        assert torch.isfinite(before).all()
        assert torch.allclose(before, after, atol=1e-6) and torch.allclose(before, padded, atol=1e-6)


def test_joint_fit_owns_training_normalization_and_requires_complete_sets():
    x, labels, faces, face_mask = sample_training()
    rng = np.random.default_rng(6)
    bodies = rng.normal(size=(160, human.MAX_POSES, BODY_INPUT_DIMENSION)).astype(np.float32)
    mask = np.tile([True,True,False,False,False,False], (160,1))
    owner = PersonSetRanker(with_bodies=True, epochs=2, width=8)
    with pytest.raises(ValueError, match="joint-set"):
        owner.fit(x, labels, faces, face_mask)
    owner.fit(x, labels, faces, face_mask, bodies=bodies, body_mask=mask)
    mean = owner._normalization[4].copy()
    before = owner.predict(x, faces, face_mask, bodies=bodies, body_mask=mask)
    bodies[~mask] = 10000
    after = owner.predict(x, faces, face_mask, bodies=bodies, body_mask=mask)
    assert np.array_equal(mean, owner._normalization[4]) and np.allclose(before, after)
    owner.close()


def test_corrupt_joint_record_is_validation_error_not_attribute_crash():
    source = item([])
    source.person_evidence.update(pose_count=1, poses=["corrupt record"])
    with pytest.raises(ValueError, match="joint record"):
        body_sets([source])


def test_out_of_frame_or_unseen_joints_are_unknown_even_with_high_model_confidence():
    from types import SimpleNamespace
    points = [SimpleNamespace(x=.1+i*.01, y=.2+i*.005, z=.1,
                              visibility=.9, presence=.9) for i in range(33)]
    points[0].x = -1
    points[1].visibility = .1
    pose = human.pose_record(points, 1000, 600)
    joints = np.asarray(pose["joints"])
    assert not joints[0].any() and not joints[1,:3].any()
    assert joints[1,3] == pytest.approx(.1) and not pose["bounds_are_complete_body"]
    evidence = human.aggregate_observation([], [pose["joints"]], pose_records=[pose])
    human.validate_observation(evidence, require_details=True)
