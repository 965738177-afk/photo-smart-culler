"""Cached experiment safety and grouping tests; no image pipeline is started."""
import numpy as np
import pytest

from scripts import evaluate_cached_preference as candidate
from scripts.evaluate_retention_calibration import purged_folds


def observation(path="/source/a.ARW", timestamp=1, phash="0000000000000000", faces=1):
    return {"path": path, "timestamp": timestamp, "phash": phash, "frontal_faces": faces}


@pytest.mark.parametrize("changes", [
    {"path": "/other/b.ARW"}, {"timestamp": 5}, {"timestamp": 0},
    {"timestamp": None}, {"phash": None}, {"phash": "ffffffffffffffff"},
    {"phash": "10000000000000000"}, {"frontal_faces": 2},
])
def test_conservative_group_veto(changes):
    a = observation()
    b = dict(observation(path="/source/b.ARW", timestamp=2), **changes)
    assert not candidate.compatible(a, b)


def test_all_pairs_prevent_similarity_chaining():
    # A-B and B-C pass, but A-C differs by 12 bits.
    rows = [observation(), observation("/source/b.ARW", 2, "000000000000003f"),
            observation("/source/c.ARW", 3, "0000000000000fff")]
    assert candidate.microbursts(rows) == [[0, 1], [2]]


def test_unknown_faces_are_not_identity_evidence():
    assert candidate.compatible(observation(faces=0), observation(timestamp=2, faces=3))


def test_context_uses_only_observed_quality_and_missing_stays_missing():
    values = np.array([[1., np.nan], [3., np.nan], [10., 5.]])
    result = candidate.context_features(values, [[0, 1], [2]])
    assert result.shape == (3, 7)
    np.testing.assert_array_equal(result[:, 3], [-1, 1, 0])
    assert np.isnan(result[:2, 4]).all()
    np.testing.assert_array_equal(result[:, 5], [2, 0, 0])


def test_count_policy_cannot_use_held_out_labels():
    selected = candidate.select_by_training_rate([1, 4, 2, 3], [True, False])
    assert selected.tolist() == [False, True, False, True]
    with pytest.raises(ValueError):
        candidate.select_by_training_rate([np.nan], [True])


def test_blocks_do_not_split_old_groups_or_new_bursts():
    rows = [{"path": f"/source/{i}.ARW", "group": i // 7} for i in range(1400)]
    bursts = [list(range(i, min(i+9, 1400))) for i in range(0, 1400, 9)]
    blocks = candidate.validation_blocks(rows, bursts)
    for group in bursts:
        assert len({blocks[i] for i in group}) == 1
    for i in range(1, len(rows)):
        if rows[i]["group"] == rows[i-1]["group"]:
            assert blocks[i] == blocks[i-1]
    for training, held in purged_folds(blocks):
        assert not (training & held).any()
        unique = list(dict.fromkeys(blocks))
        for block in {b for b,h in zip(blocks, held) if h}:
            index = unique.index(block)
            forbidden = set(unique[max(0,index-1):index+2])
            assert not any(t and b in forbidden for b,t in zip(blocks, training))


def test_invalid_cache_refuses_before_training(tmp_path):
    path = tmp_path / "a.ARW"
    path.write_bytes(b"fixture")
    inventory = {str(path): [path.stat().st_size, path.stat().st_mtime_ns]}
    reference = {"photos": [{"path": str(path), "human_retained": True, "ai_rating": 1}]}
    observations = {"results": [{"path": str(path)}], "failures": []}
    assert candidate.validate_inputs(reference, observations, inventory, {})
    with pytest.raises(ValueError, match="pure automatic"):
        candidate.validate_inputs(reference, observations, inventory, {"manual_ratings": {str(path): 2}})
    with pytest.raises(ValueError, match="repeated"):
        candidate.validate_inputs(reference, {"results": observations["results"]*2}, inventory, {})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        candidate.validate_inputs(reference, observations, inventory, {})


def test_delivery_snapshot_detects_additions_and_edits(tmp_path):
    path = tmp_path / "a.jpg"
    path.write_bytes(b"delivery")
    inventory = {str(path): [path.stat().st_size, path.stat().st_mtime_ns]}
    assert candidate.validate_delivery_inventory(inventory) == {tmp_path}
    extra = tmp_path / "renamed.JPG"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="file list"):
        candidate.validate_delivery_inventory(inventory)
    extra.unlink()
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="delivery changed"):
        candidate.validate_delivery_inventory(inventory)
