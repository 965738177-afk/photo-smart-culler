from pathlib import Path
import pytest

from event_cull.engine import PhotoResult, AnalysisOptions
from event_cull.candidate_story_policy import StoryObservation, plan_originals


OPTIONS = AnalysisOptions(detect_blur=False, detect_exposure=False)


def frame(name, kept=0, score=70):
    return PhotoResult(Path('/source') / name, score=score, perceptual_hash=1, suggested_rating=kept)


def evidence(action='drink', people='anonymous-set-1', origin='model'):
    return StoryObservation(status='ok', provider='fixture-not-real-detector', origin=origin,
                            segment='tea-scene', participants=people, action=action,
                            category='tea', subject_usable='pass', visibility='pass',
                            expression_appropriate='pass')


def test_unknown_evidence_is_baseline_not_a_fake_improvement():
    items = [frame('a', 1), frame('b')]
    result = plan_originals(items, {}, options=OPTIONS, budget=2)
    assert result.retained == {items[0].path} and result.measured_observations == 0
    assert not result.added and not result.removed
    assert [i.suggested_rating for i in items] == [1, 0]


def test_photographer_labels_do_not_become_model_predictions():
    item = frame('a')
    result = plan_originals([item], {item.path: evidence(origin='photographer')}, options=OPTIONS, budget=1)
    assert not result.retained and result.measured_observations == 0


def test_different_people_and_actions_get_separate_slots():
    items = [frame('a'), frame('b'), frame('c')]
    obs = {items[0].path: evidence(), items[1].path: evidence(people='set-2'),
           items[2].path: evidence(action='hand_over')}
    assert len(plan_originals(items, obs, options=OPTIONS, budget=3).added) == 3


def test_duplicate_frees_slot_but_does_not_remove_another_unique_story():
    a, b, c, d = frame('a', 1, 80), frame('b', 1, 60), frame('c'), frame('d', 1, 50)
    items = [a,b,c,d]
    obs = {a.path: evidence(), b.path: evidence(), c.path: evidence(action='gift'),
           d.path: evidence(action='kiss')}
    plan = plan_originals(items, obs, options=OPTIONS, budget=3)
    assert plan.retained == {a.path, c.path, d.path}
    assert [i.suggested_rating for i in items] == [1,1,0,1]


@pytest.mark.parametrize('issue', ['visibility', 'expression_appropriate', 'subject_usable'])
def test_failed_or_unknown_observation_routes_to_review_not_delete(issue):
    from dataclasses import replace
    item=frame('a',1)
    obs=replace(evidence(), **{issue:'fail'})
    plan=plan_originals([item], {item.path:obs}, options=OPTIONS, budget=1)
    assert item.path in plan.retained and item.path in plan.review


def test_manual_zero_and_two_are_protected():
    a,b=frame('a'),frame('b',2)
    plan=plan_originals([a,b], {a.path:evidence(),b.path:evidence()}, options=OPTIONS,
                       budget=2, protected_paths={a.path})
    assert plan.retained == {b.path}


def test_decode_failure_cannot_be_saved_by_story():
    item=frame('broken')
    item.perceptual_hash=None
    assert not plan_originals([item],{item.path:evidence()},options=OPTIONS,budget=1).retained


def test_cancellation_propagates_without_mutation():
    item=frame('a',1)
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        plan_originals([item],{},options=OPTIONS,budget=1,checkpoint=cancel)
    assert item.suggested_rating==1


def test_foreign_evidence_refused():
    with pytest.raises(ValueError,match='identity'):
        plan_originals([frame('a')],{Path('/foreign'):evidence()},options=OPTIONS,budget=1)
