"""Compare frozen AI advice with a confirmed human delivery, without training.

Reuses the calibration filename convention. Does not decode images, invoke a
model, change stars or equate non-delivery with a technical defect.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from event_cull.calibration import exported_selection_files


def group_allocation_diagnostic(groups: dict) -> dict:
    """Separate missed group allocations from within-group exact-file gaps.

    This is an evaluation-only ceiling with the *current* heuristic groups and
    per-group AI counts frozen. It is not a prediction, nor proof that photos
    in one group are interchangeable story moments.
    """
    human_count = ai_count = exact = slot_ceiling = 0
    human_uncovered = ai_uncovered = 0
    for group in groups.values():
        ai = set(group["ai_target"])
        human = set(group["human_target"])
        ai_count += len(ai)
        human_count += len(human)
        exact += len(ai & human)
        slot_ceiling += min(len(ai), len(human))
        if not ai:
            human_uncovered += len(human)
        if not human:
            ai_uncovered += len(ai)
    return {
        "human_photos_in_unselected_groups": human_uncovered,
        "ai_photos_in_nonhuman_groups": ai_uncovered,
        "same_group_slot_max_exact": slot_ceiling,
        "same_group_slot_headroom": slot_ceiling - exact,
        "same_group_slot_max_recall": slot_ceiling / human_count if human_count else None,
        "note": ("仅冻结当前启发式分组及各组 AI 留片数量时的逐文件命中上界；"
                 "要超越此值，必须改变组间名额或分组。同组照片不一定等价，"
                 "该上界不是照片质量、场景完整性或可实现准确率。"),
    }


def compare_selection(payload: dict, selected_folder: Path, *, confirmed_complete: bool = False,
                      objective: str = "retention", recursive: bool = True) -> dict:
    """Evaluate one objective without treating incomplete positives as negatives.

    Retention means automatic 1/3 stars. Retouch means 3 stars only; a human
    retouch reference does not become an original-delivery or a 2-star label.
    """
    if objective not in {"retention", "retouch"}:
        raise ValueError("未知的选片对比目标")
    records = payload.get("photos")
    if not isinstance(records, list) or any(not isinstance(row, dict) or not row.get("file") for row in records):
        raise ValueError("筛选基线没有有效的照片列表")
    if not selected_folder.is_dir():
        raise ValueError("人工结果文件夹不存在")
    selected_files = exported_selection_files(selected_folder, recursive=recursive)
    selected = {path.stem.casefold() for path in selected_files}
    counts = Counter(Path(str(row["file"])).stem.casefold() for row in records)
    # RAW+JPG in one exported directory remains one capture; names occurring
    # in different exported directories cannot safely identify an original.
    destinations = defaultdict(set)
    for path in selected_files:
        destinations[path.stem.casefold()].add(str(path.parent.resolve()))
    ambiguous = {stem for stem, n in counts.items() if n > 1}
    ambiguous |= {stem for stem, folders in destinations.items() if len(folders) > 1}
    unknown_selected = selected - set(counts)
    manual = payload.get("manual_ratings") or {}
    manual = set(manual) if isinstance(manual, dict) else set()
    can_score = confirmed_complete and not unknown_selected
    rows, groups = [], defaultdict(lambda: {"ai_retained": [], "human_retained": [],
                                            "ai_featured": [], "ai_target": [], "human_target": []})
    truth, advice, known_advice = set(), set(), set()
    for row in records:
        file = str(row["file"])
        folder = str(row.get("folder") or "")
        identity = str(Path(folder) / file)
        stem = Path(file).stem.casefold()
        rating = row.get("rating")
        ai_known = (stem not in ambiguous and identity not in manual
                    and type(rating) is int and rating in {0, 1, 2, 3})
        # Two stars and edited rows cannot be called original automatic advice.
        ai_known = ai_known and rating != 2
        human = True if stem in selected and stem not in ambiguous else (False if can_score and stem not in ambiguous else None)
        ai = (rating == 3 if objective == "retouch" else rating > 0) if ai_known else None
        if ai:
            known_advice.add(identity)
        state = ("unmatched_or_unfinished" if ai is None or human is None else
                 "both_retained" if ai and human else "human_only" if human else
                 "ai_only_not_delivered" if ai else "both_not_retained")
        if objective == "retouch" and state != "unmatched_or_unfinished":
            state = {"both_retained": "both_featured", "human_only": "human_retouch_only",
                     "ai_only_not_delivered": "ai_featured_not_in_reference",
                     "both_not_retained": "both_not_featured"}[state]
        entry = {"path": identity, "group": row.get("group"), "ai_rating": rating if ai_known else None,
                 "human_retained": human if objective == "retention" else None,
                 "human_retouch_candidate": human if objective == "retouch" else None,
                 "human_target_positive": human, "difference": state, "ai_reason": row.get("reasons"),
                 "content_evidence": row.get("content_evidence")}
        rows.append(entry)
        if ai_known and human is not None:
            if ai: advice.add(identity)
            if human: truth.add(identity)
            group_key = str(row.get("group") or identity)
            if rating > 0: groups[group_key]["ai_retained"].append(identity)
            if human and objective == "retention": groups[group_key]["human_retained"].append(identity)
            if rating == 3: groups[group_key]["ai_featured"].append(identity)
            if ai: groups[group_key]["ai_target"].append(identity)
            if human: groups[group_key]["human_target"].append(identity)
    tp = len(truth & advice)
    precision = tp / len(advice) if advice else None
    recall = tp / len(truth) if truth else None
    metrics = None
    if can_score and objective == "retention":
        metrics = {"scope": "仅唯一匹配且未人工改星的原片", "both_retained": tp,
                   "human_only": len(truth-advice), "ai_only_not_delivered": len(advice-truth),
                   "retained_precision": precision, "retained_recall": recall,
                   "retained_f1": (2*tp/(len(truth)+len(advice)) if truth or advice else None),
                   "retouch_precision": None,
                   "retouch_note": "尚无独立人工精修标签，不能计算精修准确率"}
    elif can_score:
        metrics = {"scope": "仅唯一匹配且未人工改星的精修候选；不是原片留片准确率",
                   "both_featured": tp, "human_retouch_only": len(truth-advice),
                   "ai_featured_not_in_reference": len(advice-truth),
                   "retouch_precision": precision, "retouch_recall": recall,
                   "retouch_f1": (2*tp/(len(truth)+len(advice)) if truth or advice else None)}
    positive_stems = {Path(path).stem.casefold() for path in truth}
    return {"objective": objective, "recursive_reference_scan": recursive,
            "confirmed_complete": confirmed_complete, "original_count": len(records),
            "reference_paths": [str(path) for path in selected_files],
            "human_unique_stems": len(selected), "ambiguous_stems": sorted(ambiguous),
            "unmatched_human_stems": sorted(unknown_selected), "metrics": metrics,
            "positive_reference": {"comparable_positive_count": len(truth), "exact_hits": tp,
                                   "positive_coverage": recall,
                                   "ai_target_count_known": len(known_advice),
                                   "unscored_positive_stems": sorted(selected-positive_stems),
                                   "note": "只描述已提供正例；未确认完整时不计算额外推荐错误或精确率。"},
            "differences": dict(Counter(row["difference"] for row in rows)), "photos": rows,
            "groups": dict(groups),
            "caution": ("精修与原片交付分开；未进入精修名单不等于废片；同组替代不计同文件命中。"
                        if objective == "retouch" else
                        "AI 额外保留仅是交付分歧，不等于废片；同组替代需比较画面，未完成结果不训练。")}
