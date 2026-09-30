"""Shared delivery-rating contract for selection, calibration and UI.

Two stars belong to the photographer. Automatic stages may only emit zero,
one or three; a promotion never creates an additional retained original.
"""

RATING_POLICY_VERSION = "originals-1-retouch-3-manual-2.v1"
AUTOMATIC_RATINGS = frozenset({0, 1, 3})
RETAINED_RATINGS = frozenset({1, 2, 3})
RATING_LABELS = {
    "zh": {0: "未入选", 1: "原片保留", 2: "人工确认", 3: "精修候选"},
    "en": {0: "Not selected", 1: "Retained original", 2: "Manual confirmation", 3: "Retouch candidate"},
}


def automatic_rating(*, retained: bool, featured: bool, ceiling: int = 3) -> int:
    """A former two-star quality ceiling means retain, not manual approval."""
    if not retained or ceiling < 1:
        return 0
    return 3 if featured and ceiling >= 3 else 1
