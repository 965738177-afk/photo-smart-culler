"""Known-answer selection-gate diagnostic, NEVER a model accuracy result.

Uses complete original-delivery references as hypothetical scores to identify
post-ranking losses. Reuses frozen measurements and shared rating assignment;
does not decode, infer, train, activate or write photo metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from event_cull import engine
from event_cull.selection_compare import compare_selection
from scripts.evaluate_retention_calibration import result_from_row, metrics


def diagnose_case(case, payload):
    if (case.get("source_preculled") or not case.get("retention_labels_complete")
            or "positive_control" in case.get("role", "")):
        raise ValueError("Preculled/partial references cannot diagnose full-selection gates")
    rows = payload.get("photos", [])
    source = Path(case["originals"]).resolve()
    if len(rows) != case["original_count"] or any(Path(row["folder"]).resolve() != source for row in rows):
        raise ValueError("Incomplete or different case source")
    compared = compare_selection(payload,Path(case["selected_originals"]),confirmed_complete=True)
    if (compared["metrics"] is None or compared["human_unique_stems"] != case["human_retained_count"]
            or any(row["ai_rating"] is None for row in compared["photos"])
            or compared["ambiguous_stems"] or compared["unmatched_human_stems"]):
        raise ValueError("Unique complete original-delivery reference required")
    labels = np.asarray([row["human_retained"] for row in compared["photos"]],bool)
    results = [result_from_row(row) for row in rows]
    options = engine.AnalysisOptions(**payload["settings"])
    original_groups = [result.group_id for result in results]
    groups = {}
    for result, label in zip(results,labels):
        result.calibration_confidence = float(label)  # Deliberate answer leakage: diagnostic only.
        groups.setdefault(result.group_id,[]).append(result)
    human_per_group = Counter(group for group,label in zip(original_groups,labels) if label)
    quota_excess = sum(max(0,human_per_group[key]-engine._group_keep_quota(group,options))
                       for key,group in groups.items())
    engine._assign_tiered_ratings(list(groups.values()),options)
    retained = np.asarray([result.suggested_rating>0 for result in results])
    hard = np.asarray([engine._has_hard_fault(result,options) for result in results])
    return {"case_id": case["case_id"], "answers_used_as_scores": True,
            "scope": "hypothetical_known_answer_scorer_through_shared_gates",
            "hypothetical_metrics": metrics(labels,retained),
            "configured_keep_min": options.target_keep_count,
            "configured_keep_max": options.target_keep_max_count,
            "hard_gate_human_retained_blocked": int((labels&hard).sum()),
            "eligible_human_retained_not_kept": int((labels&~hard&~retained).sum()),
            "manual_count_above_group_quota_before_gates": quota_excess,
            "groups": len(groups), "group_manual_count_distribution": dict(Counter(human_per_group.values())),
            "selection_accuracy": None, "passed_acceptance": False, "release_approved": False,
            "additional_image_decodes": 0, "model_training_performed": False,
            "active_profile_changed": False, "photo_metadata_changed": False,
            "is_strict_mathematical_upper_bound": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--case-id",required=True)
    args = parser.parse_args()
    manifest_raw = args.manifest.read_bytes()
    cases = json.loads(manifest_raw)["cases"]
    if len({case["case_id"] for case in cases}) != len(cases):
        parser.error("Unique case identities required")
    case = next((case for case in cases if case["case_id"] == args.case_id),None)
    if case is None:
        parser.error("Known case required")
    if args.output.exists() or any(args.output.resolve().is_relative_to(Path(case[key]).resolve())
                                  for key in ("originals","selected_originals")):
        parser.error("New report outside original/delivery directories required")
    raw = args.baseline.read_bytes()
    report = diagnose_case(case,json.loads(raw))
    report.update(baseline_sha256=hashlib.sha256(raw).hexdigest(),
                  manifest_sha256=hashlib.sha256(manifest_raw).hexdigest())
    args.output.mkdir(parents=True)
    with (args.output/"diagnostic.json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
