from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from event_cull.duplicate_review import DuplicateReviewSession
from event_cull.engine import PhotoResult


def photo(name, *, group=1, score=70, rating=1):
    return PhotoResult(path=Path("/originals") / name, group_id=group, score=score, suggested_rating=rating)


def test_uncertain_queue_is_existing_group_scoped_and_excludes_manual_stars():
    a, b, c = photo("a", score=72), photo("b"), photo("c", score=10)
    other = photo("other", group=2)
    manual = {c.path: 3}
    session = DuplicateReviewSession([a, b, c, other], manual)
    assert session.pair == (a, b)
    assert session.total_groups == 1
    assert c.suggested_rating == 1


def test_groups_from_different_source_folders_are_never_merged():
    a, b = photo("a"), photo("b")
    b.path = Path("/other-card/b")
    assert not DuplicateReviewSession([a, b], {}).pair


def test_score_gap_is_only_a_queue_heuristic_and_all_groups_can_be_reviewed():
    a, b = photo("a", score=90), photo("b", score=20)
    assert not DuplicateReviewSession([a, b], {}).pair
    assert DuplicateReviewSession([a, b], {}, uncertain_only=False).pair == (a, b)


@pytest.mark.parametrize("action,expected", [("a", (2, 0)), ("b", (0, 2)),
                                            ("both", (2, 2)), ("neither", (0, 0))])
def test_decision_touches_only_the_visible_pair_and_undo_restores_exact_state(action, expected):
    a, b, c = photo("a", rating=3), photo("b", rating=0), photo("c", rating=1)
    manual = {}
    session = DuplicateReviewSession([a, b, c], manual)
    assert session.decide(action)
    assert (a.suggested_rating, b.suggested_rating) == expected
    assert c.suggested_rating == 1
    assert set(manual) == {a.path, b.path}
    assert session.undo()
    assert [p.suggested_rating for p in (a, b, c)] == [3, 0, 1]
    assert manual == {} and session.decisions == []
    assert session.pair == (a, b)


def test_winner_faces_next_candidate_and_undo_preserves_earlier_confirmation():
    a, b, c = photo("a"), photo("b"), photo("c")
    manual = {}
    session = DuplicateReviewSession([a, b, c], manual)
    session.decide("a")
    assert session.pair == (a, c)
    session.decide("b")
    assert not session.pair and manual[a.path] == 0 and manual[c.path] == 2
    session.undo()
    assert manual[a.path] == 2 and manual[b.path] == 0 and c.path not in manual


def test_skip_and_revisit_never_change_ratings():
    a, b = photo("a"), photo("b")
    manual = {}
    session = DuplicateReviewSession([a, b], manual)
    session.decide("skip")
    assert not session.pair and len(session.deferred) == 1 and manual == {}
    session.revisit_deferred()
    assert session.pair == (a, b)
    assert a.suggested_rating == b.suggested_rating == 1


def test_manual_edit_between_screen_openings_invalidates_undo_and_protects_photo():
    a, b, c = photo("a"), photo("b"), photo("c")
    manual = {}
    session = DuplicateReviewSession([a, b, c], manual)
    session.decide("a")
    a.suggested_rating = 3
    manual[a.path] = 3  # Independent edit in the main grid.
    session.reconcile_manual_edits()
    assert not session.pair and not session.can_undo
    assert not session.undo() and manual[a.path] == 3


def test_concurrent_ab_and_zoom_consumers_share_one_raw_decode(tmp_path, monkeypatch):
    from event_cull import app as module
    subject = object.__new__(module.EventCullApp)
    subject._preview_load_locks = [threading.Lock() for _ in range(64)]
    subject._preview_lock = threading.RLock()
    subject._zoom_preview_memory = OrderedDict()
    source = tmp_path / "a.ARW"
    source.touch()
    calls = []
    monkeypatch.setattr(module, "_load_preview", lambda *_a, **_k: calls.append(1) or np.zeros((8, 8, 3), np.uint8))
    with ThreadPoolExecutor(max_workers=2) as workers:
        images = list(workers.map(subject._zoom_preview_image, [source, source]))
    assert calls == [1]
    assert all(image.size == (8, 8) for image in images)
