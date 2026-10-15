"""Freeze and evaluate multiple complete deliveries; train only isolated candidates.

Usage: --case NAME LOG SELECTED (repeat) --output NEW_DIRECTORY [--learn]
No source pixels, metadata, running tasks or active preference profiles are written.
Continuous blocks and their neighbours are kept out of training. Filenames,
camera names and group IDs are never classifier inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np

from event_cull import calibration, engine
from event_cull.selection_compare import compare_selection, group_allocation_diagnostic
from scripts.evaluate_retention_calibration import metrics, purged_folds, result_from_row


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def evaluate_case(name, log, selected, output, learn, log_only=False):
    output.mkdir()
    raw = log.read_bytes()
    with (output / "baseline.json").open("xb") as stream:
        stream.write(raw)
    payload = json.loads(raw)
    report = compare_selection(payload, selected, confirmed_complete=True)
    report["group_metrics"] = group_allocation_diagnostic(report["groups"])
    report["captured_at"] = datetime.now(timezone.utc).isoformat()
    report["baseline_sha256"] = hashlib.sha256(raw).hexdigest()
    report["baseline_path"] = str(log)
    report["settings"] = payload["settings"]
    write_json(output / "comparison.json", report)
    summary = {"name": name, "originals": len(payload["photos"]),
               "human": report["human_unique_stems"], "metrics": report["metrics"],
               "ambiguous_stems": report["ambiguous_stems"],
               "unmatched": report["unmatched_human_stems"], "manual": payload.get("manual_ratings")}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not learn:
        return summary
    # Unknown originals are excluded explicitly, never guessed from a same-name
    # derivative. Preserve original comparison and disclose the smaller scope.
    comparable = {r["path"]: r for r in report["photos"]
                  if r["ai_rating"] is not None and r["human_retained"] is not None}
    rows = [r for r in payload["photos"] if str(Path(r["folder"]) / r["file"]) in comparable]
    if report["unmatched_human_stems"] or not rows:
        raise ValueError("Incomplete reference cannot teach negative labels")
    labels = np.array([comparable[str(Path(r["folder"]) / r["file"])]["human_retained"]
                       for r in rows], dtype=bool)
    items = [result_from_row(r) for r in rows]
    pipeline = calibration.HANDCRAFTED_PIPELINE_ID
    started = time.monotonic()
    missing = [str(i.path) for i in items if not i.path.is_file()]
    if missing and not log_only:
        write_json(output / "unavailable-originals.json", {"count": len(missing), "paths": missing})
        raise ValueError(f"{len(missing)} originals missing; restore full source or use --log-features-only")
    features = np.empty((len(rows), calibration.FEATURE_DIMENSIONS[pipeline]))
    seen = set()
    def progress(done, total, path):
        if done % 500 == 0 or done == total:
            print(f"{name}: features {done}/{total}, {time.monotonic()-started:.1f}s", flush=True)
    if not log_only:
        for index, value in calibration._feature_rows([i.path for i in items], pipeline,
                                                     progress_callback=progress):
            features[index] = value
            seen.add(index)
        if len(seen) != len(items):
            raise ValueError("Missing visual features; no partial result scored")
        np.savez_compressed(output / "features.npz", features=features, labels=labels,
                            paths=np.array([str(i.path) for i in items]), pipeline=np.array(pipeline))
    blocks = calibration.blocked_validation_groups([str(i.group_id) for i in items])
    numeric_keys = ("score", "frame_sharpness", "subject_sharpness", "subject_luminance",
                    "frontal_faces", "smiling_faces", "open_eye_faces", "face_coverage",
                    "highlight_clip", "shadow_clip", "median_luminance", "eye_penalty")
    numeric = np.array([[np.nan if r.get(k) is None else float(r[k]) for k in numeric_keys] for r in rows])
    if log_only:
        # Missingness remains an explicit observation; no image content is
        # inferred from a filename, a rating, or a human delivery label.
        features = np.column_stack((np.nan_to_num(numeric, nan=0), np.isnan(numeric)))
    settings = payload["settings"]
    options = engine.AnalysisOptions(scene_type=settings.get("scene_type", "通用"))
    experiments = {}
    for model_name in (("log_quality_logistic", "log_quality_trees") if log_only else
                       ("existing_visual_logistic", "visual_and_quality_trees")):
        predictions = np.zeros(len(rows), dtype=bool)
        folds = []
        for fold, (training, held) in enumerate(purged_folds(blocks)):
            if model_name.endswith("logistic"):
                mean, scale, weights, bias = calibration._fit_classifier(features[training], labels[training].astype(np.int8))
                scores = (features[held]-mean)/scale @ weights+bias
            else:
                # Fixed candidate from mature sklearn; no search using test labels.
                from sklearn.ensemble import HistGradientBoostingClassifier
                from threadpoolctl import threadpool_limits
                values = numeric if log_only else np.column_stack((features, numeric))
                model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                           min_samples_leaf=30, l2_regularization=2, early_stopping=False, random_state=42)
                with threadpool_limits(limits=2):
                    model.fit(values[training], labels[training])
                    scores = model.predict_proba(values[held])[:, 1]
            indexes = np.flatnonzero(held)
            # Density comes only from training labels. Do not use held-out count.
            count = round(len(indexes)*labels[training].mean())
            chosen = np.argsort(-scores, kind="stable")[:count]
            predictions[indexes[chosen]] = True
            outcome = metrics(labels[held], predictions[held])
            outcome.update(fold=fold, training=int(training.sum()), held=int(held.sum()))
            folds.append(outcome)
        outcome = metrics(labels, predictions)
        outcome["passed_75"] = outcome["precision"] > .75 and outcome["recall"] > .75
        experiments[model_name] = {"metrics": outcome, "folds": folds}
        print(f"{name}: {model_name} {json.dumps(outcome)}", flush=True)
    # This evaluates classifier rankings, not the production group/quota
    # selector. Even a passing isolated profile cannot establish app quality.
    candidate = {"activated": False, "reason": "Log-only candidates cannot become visual preference profiles"}
    if not log_only:
        for item, feature in zip(items, features):
            item.visual_features = feature.tolist()
            item.visual_feature_pipeline = pipeline
        try:
            profile = calibration.train_profile([(f, int(y), b) for f,y,b in zip(features,labels,blocks)],
                       1, feature_pipeline=pipeline, objective="retention", scene_type=options.scene_type,
                       validation_scope="continuous_100_frame_blocks")
        except calibration.CalibrationError as exc:
            candidate = {"activated": False, "reason": str(exc)}
        else:
            write_json(output / "candidate-profile.json", profile.to_dict())
            candidate = {"activated": False, "reason": "Passed one split; requires independent wedding validation"}
    summary["evaluation"] = {"scope": "same_wedding_purged_5fold_100frame_blocks",
                             "excluded_rows": len(payload["photos"])-len(rows),
                             "missing_originals": len(missing), "log_features_only": log_only,
                             "numeric_fields": numeric_keys, "experiments": experiments,
                             "profile": candidate, "elapsed_seconds": time.monotonic()-started,
                             "active_profile_changed": False}
    write_json(output / "evaluation.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=3, action="append", required=True, metavar=("NAME", "LOG", "SELECTED"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--log-features-only", action="store_true",
                        help="Evaluate saved numeric observations when complete originals are unavailable")
    args = parser.parse_args()
    # Refuse any output beneath source or reference folders, before writing.
    for _, log, selected in args.case:
        p = json.loads(Path(log).read_text())
        sources = [Path(f).resolve() for f in p.get("source_folders", [])] + [Path(selected).resolve()]
        if any(args.output.resolve() == f or args.output.resolve().is_relative_to(f) for f in sources):
            parser.error("Output must be separate from originals and manual deliveries")
    args.output.mkdir(exist_ok=False, parents=True)
    summaries = [evaluate_case(name, Path(log), Path(selected), args.output / str(index), args.learn, args.log_features_only)
                 for index, (name, log, selected) in enumerate(args.case, 1)]
    write_json(args.output / "summary.json", summaries)


if __name__ == "__main__":
    main()
