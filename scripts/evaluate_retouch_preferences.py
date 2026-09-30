"""Read-only positive/unknown retouch transfer, separate from original retention.

Use confirmed non-delivered originals as retouch negatives (3 stars must be
retained). Other delivered originals not in the partial retouch reference are
UNKNOWN, never retouch negatives. Every frame has purged out-of-fold scores.
This reports supplied-positive coverage, NOT retouch precision or accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from event_cull import calibration
from event_cull.person_set_ranker import PersonSetRanker, person_sets, body_sets, JOINT_SET_PIPELINE
from event_cull.retention_ranker import observed_features
from event_cull.selection_compare import compare_selection
from scripts.evaluate_person_retention import frozen_group_ids
from scripts.evaluate_retention_calibration import purged_folds, result_from_row


def reference_states(retained, retouch):
    """0=confirmed excluded, 1=supplied retouch positive, -1=unknown."""
    retained, retouch = np.asarray(retained, bool), np.asarray(retouch, bool)
    if retained.shape != retouch.shape or retained.ndim != 1 or np.any(retouch & ~retained):
        raise ValueError("Retouch positives must uniquely match retained originals")
    states = np.full(len(retained), -1, np.int8)
    states[~retained], states[retouch] = 0, 1
    return states


def select_retouch(scores, retained, *, fraction=.2):
    """Explicit experimental budget, not inferred from a partial reference."""
    scores, retained = np.asarray(scores), np.asarray(retained, bool)
    if (scores.shape != retained.shape or scores.ndim != 1 or not np.isfinite(scores).all()
            or not 0 <= fraction <= 1):
        raise ValueError("Invalid retouch score/budget")
    candidates = np.flatnonzero(retained)
    count = int(round(len(candidates)*fraction))
    # Stable ties do not enter the learned model as filename/position inputs.
    chosen = candidates[np.argsort(-scores[candidates], kind="stable")[:count]]
    featured = np.zeros(len(scores), bool)
    featured[chosen] = True
    return featured


def validate_retention_evaluation(payload, prior_payload, retention):
    """Retained advice must come from the same frozen identities/blocked groups."""
    identities = lambda data: [(row["folder"], row["file"], row["group"]) for row in data["photos"]]
    if (identities(payload) != identities(prior_payload)
            or retention.get("scope") != "same_wedding_purged_5_fold_continuous_100_frame_blocks"
            or retention.get("active_profile_changed") is not False
            or [row["file"] for row in retention["photos"]] != [row["file"] for row in payload["photos"]]
            or any(type(row["rating"]) is not int or row["rating"] not in {0,1,3} for row in retention["photos"])):
        raise ValueError("Matching immutable automatic retention identities and blocked groups required")
    return np.asarray([row["rating"] > 0 for row in retention["photos"]], bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "original_delivery", "retouch_reference", "features", "retention_evaluation", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    original = compare_selection(payload, args.original_delivery, confirmed_complete=True)
    reference = compare_selection(payload, args.retouch_reference, objective="retouch", recursive=False)
    for report in (original, reference):
        if report["ambiguous_stems"] or report["unmatched_human_stems"] or any(
                row["ai_rating"] is None for row in report["photos"]):
            parser.error("unique matched originals, pure frozen AI baseline and retouch reference required")
    sources = {Path(row["folder"]).resolve() for row in payload["photos"]}
    sources.update((args.original_delivery.resolve(), args.retouch_reference.resolve()))
    if args.output.exists() or any(args.output.resolve().is_relative_to(p) for p in sources):
        parser.error("new non-overwriting evaluation output outside photo directories required")
    retained = np.asarray([row["human_retained"] for row in original["photos"]], bool)
    positive = np.asarray([row["human_retouch_candidate"] is True for row in reference["photos"]], bool)
    states = reference_states(retained, positive)
    if positive.sum() < 12:
        parser.error("insufficient matched retouch positives")
    results = [result_from_row(row) for row in payload["photos"]]
    for result, row in zip(results, payload["photos"]):
        result.person_evidence = row["person_evidence"]
    with np.load(args.features, allow_pickle=False) as saved:
        from src.core.image_features.semantic_concepts import PIPELINE_ID
        if (list(saved["paths"]) != [str(r.path) for r in results] or str(saved["pipeline"]) != PIPELINE_ID
                or not np.isfinite(saved["features"]).all()):
            parser.error("matching frozen content features required; no additional image decode")
        visual = saved["features"].copy()
    retention = json.loads(args.retention_evaluation.read_text(encoding="utf-8"))
    prior_payload = json.loads(Path(retention["baseline"]).read_text(encoding="utf-8"))
    try:
        automatic_retained = validate_retention_evaluation(payload, prior_payload, retention)
    except ValueError as exc:
        parser.error(str(exc))
    started = time.monotonic()
    features = observed_features(results, visual)
    faces, face_mask = person_sets(results)
    bodies, body_mask = body_sets(results)
    groups = frozen_group_ids(results)
    blocks = calibration.blocked_validation_groups(groups)
    scores, featured = np.zeros(len(results)), np.zeros(len(results), bool)
    folds = []
    for number, (training, held) in enumerate(purged_folds(blocks)):
        known = training & (states >= 0)
        owner = PersonSetRanker(with_bodies=True, progress=lambda epoch,total: print(
            f"retouch fold {number}: epoch {epoch}/{total}", flush=True))
        try:
            owner.fit(features[known], states[known], faces[known], face_mask[known],
                      bodies=bodies[known], body_mask=body_mask[known], groups=groups[known])
            scores[held] = owner.predict(features[held], faces[held], face_mask[held],
                                         bodies=bodies[held], body_mask=body_mask[held])
        finally:
            owner.close()
        featured[held] = select_retouch(scores[held], automatic_retained[held])
        fold = {"fold":number, "training_positive":int((known & positive).sum()),
                "training_confirmed_negative":int((known & ~retained).sum()),
                "training_unknown_excluded":int((training & (states < 0)).sum()),
                "held_positive":int((held & positive).sum()),
                "positive_original_retained":int((held & positive & automatic_retained).sum()),
                "positive_featured":int((held & positive & featured).sum())}
        folds.append(fold)
        print(json.dumps(fold), flush=True)
    hits = int((featured & positive).sum())
    report = {"baseline":str(args.baseline), "retention_evaluation":str(args.retention_evaluation),
              "original_delivery":str(args.original_delivery), "retouch_reference":str(args.retouch_reference),
              "features":str(args.features), "pipeline":JOINT_SET_PIPELINE,
              "scope":"same_wedding_purged_5_fold_continuous_100_frame_blocks",
              "original_count":len(results), "provided_positive_count":int(positive.sum()),
              "unknown_retained_count":int((states < 0).sum()),
              "retouch_positive_list_confirmed_complete":False,
              "retouch_precision":None, "retouch_accuracy":None,
              "exact_positive_hits":hits, "provided_positive_coverage":hits/int(positive.sum()),
              "automatic_retained":int(automatic_retained.sum()),
              "automatic_retained_positive":int((positive & automatic_retained).sum()),
              "automatic_featured":int(featured.sum()), "experimental_featured_fraction":.2,
              "scores_are_calibrated_probabilities":False, "evaluation_seconds":time.monotonic()-started,
              "folds":folds, "exploratory_validation":True, "release_approved":False,
              "active_profile_changed":False, "additional_image_decodes":0,
              "photos":[{"file":r.path.name, "score":float(scores[i]), "provided_positive":bool(positive[i]),
                         "reference_state":int(states[i]),
                         "rating":3 if featured[i] else 1 if automatic_retained[i] else 0}
                        for i,r in enumerate(results)]}
    args.output.mkdir(parents=True)
    with (args.output/"evaluation.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "photos"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
