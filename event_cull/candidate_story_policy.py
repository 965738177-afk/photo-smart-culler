"""Experimental evidence-only original-retention plan, not an active rule.

Consumes shared PhotoResults and injected model observations; never decodes,
loads models, writes stars, or reads photographer answer labels. Missing scene,
participant-set or action evidence must not be fabricated from filenames/hash
groups. This policy is intentionally not called by the production pipeline.
"""
from dataclasses import dataclass
from collections import defaultdict
import math
from pathlib import Path
from typing import Callable

from event_cull.engine import AnalysisOptions, PhotoResult, _has_hard_fault, _review_priority

VERSION = "candidate-story-coverage-20261012.1"
CATEGORIES = frozenset({"makeup", "detail", "pet", "tea", "gift", "interaction", "portrait", "other"})


@dataclass(frozen=True)
class StoryObservation:
    """Model-produced structured evidence; keys require an independently tested provider.

    A participant-set is an anonymous local observation, not a family role.
    Quality/visibility are ternary; looking away is not a visibility failure.
    """
    status: str = "unknown"
    provider: str = ""
    origin: str = "model"
    segment: str | None = None
    participants: str | None = None
    action: str | None = None
    category: str = "other"
    subject_usable: str = "unknown"
    expression_appropriate: str = "unknown"
    visibility: str = "unknown"

    def __post_init__(self):
        if self.status not in {"ok", "unknown", "error"} or self.origin not in {"model", "photographer"}:
            raise ValueError("Invalid evidence provenance")
        if self.category not in CATEGORIES or any(v not in {"pass", "fail", "unknown"} for v in
                (self.subject_usable, self.expression_appropriate, self.visibility)):
            raise ValueError("Invalid story observation")
        if any(v is not None and (not isinstance(v, str) or not v.strip()) for v in
               (self.segment, self.participants, self.action)):
            raise ValueError("Invalid scene/person/action key")

    @property
    def complete(self):
        return (self.status == "ok" and self.origin == "model" and bool(self.provider)
                and all((self.segment, self.participants, self.action)))


@dataclass(frozen=True)
class CandidatePlan:
    retained: frozenset[Path]
    review: frozenset[Path]
    added: frozenset[Path]
    removed: frozenset[Path]
    unresolved_coverage: tuple[tuple, ...]
    measured_observations: int


def plan_originals(results: list[PhotoResult], observations: dict[Path, StoryObservation], *,
                   options: AnalysisOptions, budget: int, protected_paths=frozenset(),
                   checkpoint: Callable[[], None] = lambda: None) -> CandidatePlan:
    """Reserve one usable representative for each observed scene/person/action.

    No forced duplicate filling. A story reservation never overrides a hard
    technical fault or a manual choice. Unknown/bad visible evidence routes to
    review; subjective expression failure alone never deletes a story frame.
    Existing retouch candidates are untouched because this returns a plan only.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("Invalid candidate budget")
    paths = [item.path for item in results]
    if len(paths) != len(set(paths)) or set(observations) - set(paths):
        raise ValueError("Duplicate or foreign evidence identity")
    locked = set(protected_paths) | {i.path for i in results if i.suggested_rating == 2}
    if locked - set(paths):
        raise ValueError("Unknown protected path")
    baseline = {i.path for i in results if i.suggested_rating > 0}
    eligible, review, coverage, measured = {}, set(), defaultdict(list), 0
    for item in results:
        checkpoint()
        if not math.isfinite(item.score):
            raise ValueError("Nonfinite quality score")
        obs = observations.get(item.path, StoryObservation())
        if not obs.complete:
            review.add(item.path)
            continue
        measured += 1
        if item.path in locked:
            continue
        if _has_hard_fault(item, options) or obs.subject_usable != "pass":
            review.add(item.path)
            continue
        if obs.visibility != "pass" or obs.expression_appropriate != "pass":
            review.add(item.path)
            continue
        key = (str(item.path.parent), obs.segment, obs.participants, obs.action, obs.category)
        coverage[key].append(item)
        eligible[item.path] = item
    # Unknown semantic evidence leaves the frozen baseline exactly unchanged.
    selected = set(baseline)
    reserved = {max((i for i in items if i.path in selected),
                    key=lambda i: _review_priority(i, options)).path
                for items in coverage.values() if any(i.path in selected for i in items)}
    unresolved = []
    for key, items in coverage.items():
        checkpoint()
        if any(i.path in selected for i in items):
            reserved.add(max((i for i in items if i.path in selected),
                             key=lambda i: _review_priority(i, options)).path)
            continue
        best = max(items, key=lambda i: _review_priority(i, options))
        if len(selected) >= budget:
            # Only an explicitly observed repeated story slot can make room.
            # Unknown baseline frames are not silently sacrificed as duplicates.
            expendable = [eligible[p] for p in selected if p in eligible and p not in locked and p not in reserved]
            if not expendable:
                unresolved.append(key)
                review.update(i.path for i in items)
                continue
            loser = min(expendable, key=lambda i: _review_priority(i, options))
            selected.remove(loser.path)
        selected.add(best.path)
        reserved.add(best.path)
    # Preserve manual rejections as well as manual keeps.
    selected.difference_update(locked - baseline)
    selected.update(locked & baseline)
    return CandidatePlan(frozenset(selected), frozenset(review), frozenset(selected-baseline),
                         frozenset(baseline-selected), tuple(unresolved), measured)
