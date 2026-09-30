"""Read-only original-delivery evaluation with purged continuous-block folds.

No active profile, originals, XMP, task history or running app is modified.
Every evaluated frame is scored by a classifier that did not train on its
continuous block OR the neighbouring blocks. Filename/group/position are not
classifier inputs. This is same-wedding validation, not cross-wedding evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
import time

import numpy as np

from event_cull import calibration, engine
from event_cull.selection_compare import compare_selection


def result_from_row(row):
    item = engine.PhotoResult(Path(row["folder"]) / row["file"])
    mapping = {"group": "group_id", "rating": "suggested_rating", "frame_sharpness": "sharpness",
               "frontal_faces": "frontal_face_count", "smiling_faces": "smiling_face_count",
               "open_eye_faces": "open_eye_face_count", "scene_softness_protected": "scene_protected",
               "scene_coverage_protected": "coverage_protected"}
    for key in ("score", "perceptual_hash", "highlight_clip", "shadow_clip", "subject_sharpness",
                "subject_luminance", "face_coverage", "closest_face_distance", "eye_detection_mode",
                "eye_penalty", "must_keep_id", "median_luminance", "relative_exposure_ev"):
        if key in row:
            setattr(item, key, row[key])
    for key, attr in mapping.items():
        if key in row:
            setattr(item, attr, row[key])
    item.reasons = re.split(r"[;；]",str(row.get("reasons") or ""))
    item.content_evidence = {"status": "base_only"}
    return item


def purged_folds(blocks, folds=5):
    """Exclude neighbouring continuous blocks, not just the same hash group."""
    unique = list(dict.fromkeys(blocks))
    numbers = {block: index for index, block in enumerate(unique)}
    ids = np.asarray([numbers[block] for block in blocks])
    for fold in range(folds):
        held = {index for index in range(len(unique)) if index % folds == fold}
        forbidden = held | {index + offset for index in held for offset in (-1, 1)}
        yield np.asarray([index not in forbidden for index in ids]), np.asarray([index in held for index in ids])


def train_side_validation(blocks, eligible):
    """Inner whole-block split, excluding adjacent blocks and outer test rows."""
    eligible=np.asarray(eligible,bool)
    if eligible.ndim!=1 or len(eligible)!=len(blocks):
        raise ValueError("Invalid outer training mask")
    unique=list(dict.fromkeys(blocks))
    ids={value:index for index,value in enumerate(unique)}
    block_ids=np.asarray([ids[value] for value in blocks])
    available=list(dict.fromkeys(block_ids[eligible]))
    if len(available)<5:
        raise ValueError("Insufficient training-side blocks for nested validation")
    held=set(available[::5])
    forbidden=held|{index+offset for index in held for offset in (-1,1)}
    return eligible&~np.isin(block_ids,list(forbidden)),eligible&np.isin(block_ids,list(held))


def metrics(labels, kept):
    tp = int((labels & kept).sum())
    return {"human_retained": int(labels.sum()), "ai_retained": int(kept.sum()), "both_retained": tp,
            "precision": tp / max(1, int(kept.sum())), "recall": tp / max(1, int(labels.sum())),
            "f1": 2 * tp / max(1, int(kept.sum() + labels.sum()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("manual", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--features", type=Path, help="reuse this evaluation's matching immutable feature file")
    args = parser.parse_args()
    payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    comparison = compare_selection(payload, args.manual, confirmed_complete=True)
    if comparison["ambiguous_stems"] or comparison["unmatched_human_stems"] or comparison["metrics"] is None:
        parser.error("manual originals must be complete and uniquely matched")
    if any(row["ai_rating"] is None for row in comparison["photos"]):
        parser.error("baseline must be frozen pure AI advice without manual edits")
    sources = {Path(row["folder"]).resolve() for row in payload["photos"]} | {args.manual.resolve()}
    if any(args.output.resolve() == p or args.output.resolve().is_relative_to(p) for p in sources):
        parser.error("output must be outside original and manual photo directories")
    if args.output.exists():
        parser.error("output must be a new non-overwriting evaluation directory")
    args.output.mkdir(parents=True)
    results = [result_from_row(row) for row in payload["photos"]]
    labels = np.asarray([row["human_retained"] for row in comparison["photos"]], dtype=bool)
    paths = [item.path for item in results]
    pipeline = calibration._preferred_feature_pipeline()
    started, last = time.monotonic(), 0

    def progress(current, total, path):
        nonlocal last
        now = time.monotonic()
        if now - last > 20 or current == total:
            print(f"visual features {current}/{total}, {now-started:.1f}s", flush=True)
            last = now

    if args.features:
        with np.load(args.features, allow_pickle=False) as saved:
            if list(saved["paths"]) != [str(p) for p in paths] or str(saved["pipeline"]) != pipeline:
                parser.error("feature identity/pipeline differs from baseline")
            features = saved["features"].copy()
    else:
        rows = list(calibration._feature_rows(paths, pipeline, progress_callback=progress))
        if [index for index, _ in rows] != list(range(len(results))):
            parser.error("not all originals could be decoded; no silent subset evaluation")
        features = np.vstack([feature for _, feature in rows])
        np.savez_compressed(args.output / "features.npz", features=features,
                            paths=np.asarray([str(p) for p in paths]), pipeline=np.asarray(pipeline))
    if len(features) != len(results) or not np.isfinite(features).all():
        parser.error("incomplete/nonfinite feature matrix")
    feature_seconds = time.monotonic() - started
    for item, feature in zip(results, features):
        item.visual_features = feature.tolist()
        item.visual_feature_pipeline = pipeline
    blocks = calibration.blocked_validation_groups([str(item.group_id) for item in results])
    options = engine.AnalysisOptions(**payload["settings"])
    predictions = np.zeros(len(results), dtype=bool)
    scores = np.zeros(len(results))
    folds = []
    for number, (training, held_out) in enumerate(purged_folds(blocks)):
        if labels[training].sum() < 12 or labels[held_out].sum() < 4:
            parser.error("insufficient class coverage in purged folds")
        mean, scale, weights, bias = calibration._fit_classifier(features[training], labels[training].astype(np.int8))
        profile = calibration.CalibrationProfile(tuple(mean), tuple(scale), tuple(weights), bias, 1,
                    int(training.sum()), int(labels[training].sum()), .5, pipeline,
                    objective="retention", scene_type=options.scene_type)
        indexes = np.flatnonzero(held_out)
        items = [results[index] for index in indexes]
        from dataclasses import replace
        fold_options = replace(options, target_keep_count=round((options.target_keep_count or 1023)*held_out.mean()),
                              target_keep_max_count=round((options.target_keep_max_count or 1200)*held_out.mean()))
        calibration.apply_profile(items, profile, options=fold_options)
        scores[held_out] = [(features[i]-mean) / scale @ weights + bias for i in indexes]
        predictions[held_out] = [item.suggested_rating > 0 for item in items]
        result = metrics(labels[held_out], predictions[held_out])
        result.update(fold=number, training_examples=int(training.sum()), held_out_examples=int(held_out.sum()))
        folds.append(result)
        print(f"purged fold {number}: {json.dumps(result)}", flush=True)
    outcome = metrics(labels, predictions)
    outcome["auc"] = calibration._roc_auc(labels.astype(np.int8), scores)
    outcome["required_precision_and_recall"] = calibration.MIN_RETENTION_AGREEMENT
    outcome["passed_acceptance"] = (outcome["precision"] >= calibration.MIN_RETENTION_AGREEMENT
                                    and outcome["recall"] >= calibration.MIN_RETENTION_AGREEMENT)
    report = {"baseline": str(args.baseline), "manual": str(args.manual), "rule_version": engine.RULE_VERSION,
              "feature_pipeline": pipeline, "feature_dimensions": features.shape[1],
              "feature_seconds": feature_seconds, "total_seconds": time.monotonic()-started,
              "scope": "same_wedding_purged_5_fold_continuous_100_frame_blocks",
              "baseline_metrics": comparison["metrics"], "out_of_fold": outcome, "folds": folds,
              "rating_counts": dict(Counter(item.suggested_rating for item in results)),
              "retouch_accuracy": None, "active_profile_changed": False,
              "caution": "This is same-wedding held-out prediction, not a new-wedding guarantee. Non-delivery is not a technical defect.",
              "photos": [{"file": item.path.name, "fold": int(blocks[i].rsplit('-',1)[1]) % 5,
                          "human_retained": bool(labels[i]), "predicted_rating": item.suggested_rating}
                         for i, item in enumerate(results)]}
    with (args.output / "evaluation.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "photos"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
