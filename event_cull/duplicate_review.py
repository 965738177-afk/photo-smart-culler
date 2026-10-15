"""Human-only A/B decisions; no decoding, training, deletion or metadata writes."""
from collections import OrderedDict
from datetime import datetime, timezone
import math


class DuplicateReviewSession:
    """Tournament within existing groups, never inventing a confidence score.

    Close scores are a review heuristic, not evidence that frames are identical.
    Existing manual ratings are excluded. Each action affects only the visible
    pair; all unseen frames retain their previous ratings.
    """

    def __init__(self, results, manual_ratings, *, uncertain_only=True, score_gap=5.0):
        self.manual_ratings = manual_ratings
        self._known_manual = dict(manual_ratings)
        groups = OrderedDict()
        for item in results:
            if item.group_id is None or item.group_id < 1 or item.path in manual_ratings:
                continue
            groups.setdefault((item.path.parent, item.group_id), []).append(item)
        self.pending = []
        for group in groups.values():
            ranked = sorted(group, key=lambda p: p.score if math.isfinite(p.score) else -math.inf,
                            reverse=True)
            if len(ranked) < 2:
                continue
            close = (math.isfinite(ranked[0].score) and math.isfinite(ranked[1].score)
                     and abs(ranked[0].score - ranked[1].score) <= score_gap)
            if not uncertain_only or close:
                self.pending.append(ranked)
        self.total_groups = len(self.pending)
        self.deferred = []
        self.decisions = []
        self._undo = []

    @property
    def pair(self):
        return tuple(self.pending[0][:2]) if self.pending else ()

    @property
    def can_undo(self):
        return bool(self._undo)

    def reconcile_manual_edits(self):
        """Protect ratings edited outside this review screen between openings."""
        if self.manual_ratings == self._known_manual:
            return
        changed = {path for path in self.manual_ratings.keys() | self._known_manual.keys()
                   if self.manual_ratings.get(path) != self._known_manual.get(path)}
        for name in ("pending", "deferred"):
            groups = [[item for item in group if item.path not in changed] for group in getattr(self, name)]
            setattr(self, name, [group for group in groups if len(group) >= 2])
        self._undo.clear()  # Never undo past an independent photographer edit.
        self._known_manual = dict(self.manual_ratings)

    def decide(self, action):
        self.reconcile_manual_edits()
        if action not in {"a", "b", "both", "neither", "skip"}:
            raise ValueError("Unknown A/B action")
        if not self.pair:
            return False
        a, b = self.pair
        self._undo.append(([list(g) for g in self.pending], [list(g) for g in self.deferred],
                           [(p, p.suggested_rating, self.manual_ratings.get(p.path)) for p in (a, b)]))
        group = self.pending[0]
        if action == "skip":
            self.deferred.append(self.pending.pop(0))
        else:
            for item, keep in ((a, action in {"a", "both"}), (b, action in {"b", "both"})):
                # Tier 2 is explicitly reserved for photographer confirmation.
                item.suggested_rating = 2 if keep else 0
                self.manual_ratings[item.path] = item.suggested_rating
            remaining = ([a] if action == "a" else [b] if action == "b" else []) + group[2:]
            if len(remaining) >= 2:
                self.pending[0] = remaining
            else:
                self.pending.pop(0)
        self.decisions.append({"at": datetime.now(timezone.utc).isoformat(),
                               "a": str(a.path), "b": str(b.path), "action": action})
        self._known_manual = dict(self.manual_ratings)
        return True

    def undo(self):
        self.reconcile_manual_edits()
        if not self._undo:
            return False
        self.pending, self.deferred, previous = self._undo.pop()
        for item, rating, manual in previous:
            item.suggested_rating = rating
            if manual is None:
                self.manual_ratings.pop(item.path, None)
            else:
                self.manual_ratings[item.path] = manual
        self.decisions.pop()
        self._known_manual = dict(self.manual_ratings)
        return True

    def revisit_deferred(self):
        if not self.pending and self.deferred:
            self.pending, self.deferred = self.deferred, []
            self._undo.clear()  # Do not undo across a queue navigation boundary.
