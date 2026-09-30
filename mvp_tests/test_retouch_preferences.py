"""Partial retouch references never make unlisted delivered frames negative."""
import numpy as np
import pytest
from scripts.evaluate_retouch_preferences import reference_states, select_retouch, validate_retention_evaluation


def test_unlisted_retained_originals_stay_unknown():
    assert reference_states([1,1,0,1,0], [1,0,0,0,0]).tolist() == [1,-1,0,-1,0]


def test_reference_must_be_subset_of_original_delivery():
    with pytest.raises(ValueError, match="retained originals"):
        reference_states([0,1], [1,0])


def test_featured_set_is_bounded_retention_subset_even_if_rejected_has_best_score():
    kept = np.asarray([False]+[True]*10)
    selected = select_retouch(np.arange(11)[::-1], kept)
    assert selected.sum() == 2 and not (selected & ~kept).any()
    assert np.flatnonzero(selected).tolist() == [1,2]


def test_empty_kept_set_never_produces_three_stars():
    assert not select_retouch([.9,.4], [False,False]).any()


@pytest.mark.parametrize("scores,retained", [([float("nan")],[1]), ([.5,.9],[1])])
def test_nonfinite_or_mismatched_predictions_fail(scores,retained):
    with pytest.raises(ValueError):
        select_retouch(scores,retained)


def test_frozen_retention_scope_rejects_renumbered_groups_or_other_sources():
    baseline = {"photos":[{"folder":"/original", "file":"x.ARW", "group":1}]}
    evaluation = {"scope":"same_wedding_purged_5_fold_continuous_100_frame_blocks",
                  "active_profile_changed":False, "photos":[{"file":"x.ARW", "rating":1}]}
    assert validate_retention_evaluation(baseline,baseline,evaluation).tolist() == [True]
    changed = {"photos":[dict(baseline["photos"][0], group=2)]}
    with pytest.raises(ValueError, match="blocked groups"):
        validate_retention_evaluation(baseline,changed,evaluation)
    changed["photos"][0].update(group=1, folder="/another_wedding")
    with pytest.raises(ValueError, match="identities"):
        validate_retention_evaluation(baseline,changed,evaluation)
