"""Fixed-method ownership and cross-case isolation, not selection quality."""
from types import SimpleNamespace

import numpy as np
import pytest

from event_cull.retention_ranker import (PairedRetentionRanker, RetentionRanker,
                                        combine_pairwise_scores)


def test_blend_matches_the_existing_fixed_equation():
    absolute, relative = np.array([0., .1, .5, .9, 1.]), np.array([.8, .3, .5, .7, .2])
    a, r = np.clip(absolute,1e-6,1-1e-6), np.clip(relative,1e-6,1-1e-6)
    expected = 1/(1+np.exp(-np.clip(np.log(a/(1-a))+.5*np.log(r/(1-r)), -30,30)))
    assert np.array_equal(combine_pairwise_scores(absolute,relative),expected)


@pytest.mark.parametrize("absolute,relative", [([np.nan],[.5]),([1.1],[.5]),([.5],[-.1]),
                                                ([.5,.5],[.5]),([[.5]],[[.5]])])
def test_bad_blend_inputs_are_refused(absolute,relative):
    with pytest.raises(ValueError,match="ranking scores"):
        combine_pairwise_scores(absolute,relative)


def test_cancelled_refit_keeps_prior_pair_and_closes_only_candidates():
    created = []
    fail = [False]
    class Absolute:
        def __init__(self, **kwargs):
            self.closed = False
            self.positive_fraction = .4
            created.append(self)
        def fit(self, x, labels):
            self.training = np.array(x).copy()
        def predict(self, x):
            return np.full(len(x),.6)
        def close(self):
            self.closed = True
    class Pair(Absolute):
        def fit_pairs(self, x, labels, groups):
            if fail[0]:
                raise InterruptedError("cancelled fit")
        def compare(self, x, groups):
            return np.full(len(x),.7)
    x, y, groups = np.ones((8,2)), np.tile([0,1],4), np.arange(8)//2
    owner = PairedRetentionRanker(absolute_factory=Absolute,pair_factory=Pair).fit(x,y,groups)
    before = owner.predict(x,groups)
    fail[0] = True
    with pytest.raises(InterruptedError):
        owner.fit(x+5,y,groups)
    assert not any(c.closed for c in created[:2]) and all(c.closed for c in created[2:])
    assert owner.positive_fraction == .4 and np.array_equal(owner.predict(x,groups),before)
    owner.close()
    owner.close()
    assert all(c.closed for c in created)
    with pytest.raises(ValueError,match="not been trained"):
        owner.predict(x,groups)


def test_fixed_case_fitting_uses_eligible_training_rows_only(monkeypatch):
    from scripts import evaluate_burst_sets as evaluation
    from event_cull import engine
    monkeypatch.setattr(engine,"_load_preview",lambda *a,**k:pytest.fail("No additional image decoding"))
    passed = []
    class Fake:
        def fit(self, x, labels, groups):
            passed.extend((x.copy(),labels.copy(),groups.copy()))
        def close(self):
            pass
    monkeypatch.setattr(evaluation,"PairedRetentionRanker",Fake)
    data = SimpleNamespace(features=np.arange(24).reshape(8,3),labels=np.tile([0,1],4),
                           groups=np.array(["first","first","second","second"]*2))
    eligible = np.arange(8)<4
    owner, report = evaluation.fit_fixed_pairwise(data,eligible)
    assert np.array_equal(passed[0],data.features[eligible])
    assert np.array_equal(passed[1],data.labels[eligible])
    assert np.array_equal(passed[2],data.groups[eligible])
    assert report["training_examples"]==4 and report["validation_examples"]==0
    assert report["test_labels_used_for_training"] is False
    owner.close()


def test_absolute_close_releases_model_without_refitting():
    owner = RetentionRanker()
    owner._model, owner.positive_fraction = object(), .4
    owner.close()
    assert owner._model is None and owner.positive_fraction is None
    with pytest.raises(ValueError,match="not been trained"):
        owner.predict(np.ones((1,2)))
