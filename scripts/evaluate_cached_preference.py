"""Isolated cached-feature ablation; never activates a production rule/model.

Uses an EXIF-verified reference and native Fast observations. No image decoding,
cloud service or third-party application startup. Counts depend on training
labels only. Same-wedding purged validation is NOT cross-wedding acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from event_cull.calibration import blocked_validation_groups
from event_cull.selection_compare import validate_delivery_inventory
from scripts.evaluate_retention_calibration import metrics, purged_folds


LOG_FIELDS = ("score", "frame_sharpness", "subject_sharpness", "subject_luminance",
              "frontal_faces", "smiling_faces", "open_eye_faces", "face_coverage",
              "highlight_clip", "shadow_clip", "median_luminance", "eye_penalty")
FAST_FIELDS = ("blur_score", "brightness_mean", "brightness_std", "overexposed_ratio",
               "underexposed_ratio", "entropy", "salient_sharpness", "blur_combined",
               "motion_anisotropy", "edge_width_pix", "focus_ratio", "horizon_tilt_deg",
               "composition")


def number(value):
    """Unknown observations remain missing, rather than becoming perfect scores."""
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (ValueError, TypeError):
        return np.nan


def compatible(a, b):
    """Conservative microburst hypothesis, NOT proof of identical people/story.

    All members must agree, preventing single-linkage chaining. Face counts only
    veto merges when both are observed; zero/unknown faces cannot prove identity.
    """
    if Path(a["path"]).parent != Path(b["path"]).parent:
        return False
    delta = number(b.get("timestamp")) - number(a.get("timestamp"))
    if not np.isfinite(delta) or not 0 <= delta <= 3:
        return False
    try:
        ah, bh = int(a["phash"], 16), int(b["phash"], 16)
        if not (0 <= ah < 2**64 and 0 <= bh < 2**64) or (ah ^ bh).bit_count() > 10:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    af, bf = a.get("frontal_faces", 0), b.get("frontal_faces", 0)
    return not (af and bf and af != bf)


def microbursts(observations):
    groups = []
    for index, row in enumerate(observations):
        if groups and all(compatible(observations[i], row) for i in groups[-1]):
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def context_features(values, groups):
    """Group length + relative quality, without reference labels or group IDs."""
    context = np.full((len(values), 1 + 2 * values.shape[1]), np.nan)
    for indexes in groups:
        subset = values[indexes]
        # Avoid warnings for entirely unobserved columns.
        med = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any()
                        else np.nan for col in subset.T])
        maximum = np.array([np.max(col[np.isfinite(col)]) if np.isfinite(col).any()
                            else np.nan for col in subset.T])
        context[indexes, 0] = np.log1p(len(indexes))
        context[indexes, 1:1 + values.shape[1]] = subset - med
        context[indexes, 1 + values.shape[1]:] = maximum - subset
    return np.column_stack((values, context))


def validation_blocks(rows, groups):
    """Never split either a cached old group or a proposed microburst."""
    ids = np.empty(len(rows), dtype=int)
    for group_id, indexes in enumerate(groups):
        ids[indexes] = group_id
    # Coarse segments end only at a boundary of BOTH groupings.
    segments, segment = [], 0
    for i, row in enumerate(rows):
        if i and ids[i] != ids[i-1] and (
                Path(row["path"]).parent != Path(rows[i-1]["path"]).parent
                or row.get("group") != rows[i-1].get("group")):
            segment += 1
        segments.append(str(segment))
    return blocked_validation_groups(segments)


def select_by_training_rate(scores, training_labels):
    """Held-out human count is deliberately absent from this API."""
    labels = np.asarray(training_labels, bool)
    scores = np.asarray(scores, float)
    if not len(labels) or not np.isfinite(scores).all():
        raise ValueError("Invalid training labels or scores")
    count = round(len(scores) * labels.mean())
    selected = np.zeros(len(scores), bool)
    selected[np.argsort(-scores, kind="stable")[:count]] = True
    return selected


def new_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def validate_inputs(reference, observations, inventory, baseline):
    rows = reference["photos"]
    paths = [r["path"] for r in rows if r["path"] in inventory]
    obs_paths = [r["path"] for r in observations["results"]]
    if (observations.get("failures") or len(obs_paths) != len(set(obs_paths))
            or len(paths) != len(set(paths)) or set(paths) != set(obs_paths)
            or set(paths) != set(inventory)):
        raise ValueError("Incomplete or repeated cache identities")
    if baseline.get("manual_ratings") or any(
            r.get("excluded_manual") or type(r.get("human_retained")) is not bool
            or r.get("ai_rating") not in (0, 1, 3) for r in rows if r["path"] in inventory):
        raise ValueError("Reference must be pure automatic advice and complete delivery labels")
    for path, stats in inventory.items():
        p = Path(path)
        if not p.is_file() or [p.stat().st_size, p.stat().st_mtime_ns] != list(stats):
            raise ValueError("Original changed or unavailable: " + path)
    return [r for r in rows if r["path"] in inventory]


def run(cache, baseline_path, delivery_inventory_path, output):
    inputs = {name: (cache / name).read_bytes() for name in
              ("reference-snapshot.json", "native-observations.json", "input-inventory.json")}
    baseline_raw = baseline_path.read_bytes()
    delivery_raw = delivery_inventory_path.read_bytes()
    delivery_inventory = json.loads(delivery_raw)
    manual_folders = validate_delivery_inventory(delivery_inventory)
    reference, observations, inventory = [json.loads(inputs[name]) for name in inputs]
    baseline = json.loads(baseline_raw)
    if hashlib.sha256(baseline_raw).hexdigest() != reference["baseline_sha256"]:
        raise ValueError("Frozen baseline hash does not match capture reference")
    rows = validate_inputs(reference, observations, inventory, baseline)
    roots = [Path(p).parent.resolve() for p in inventory] + [cache.resolve(), baseline_path.parent.resolve(),
             *(p.resolve() for p in manual_folders)]
    if any(output.resolve().is_relative_to(root) for root in roots):
        raise ValueError("Output must be separate from source and frozen evidence")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()}
    hashes["baseline.json"] = hashlib.sha256(baseline_raw).hexdigest()
    hashes["delivery_inventory.json"] = hashlib.sha256(delivery_raw).hexdigest()
    policy = {"scope": "same_wedding_purged_5fold_100frame_blocks",
              "activated": False, "full_production_pipeline": False,
              "hashes": hashes, "feature_fields": {"app": LOG_FIELDS, "fast": FAST_FIELDS},
              "group_policy": {"max_span_seconds": 3, "max_all_pair_phash_distance": 10},
              "count_policy": "training_label_fraction_only_or_fixed_probability_0.5",
              "model": {"max_iter": 120, "max_leaf_nodes": 15, "min_samples_leaf": 30,
                        "l2_regularization": 2, "early_stopping": False, "random_state": 42},
              "excluded_reference_paths": [r["path"] for r in reference["photos"] if r["path"] not in inventory]}
    new_json(output / "protocol.json", policy)
    old = {str(Path(r["folder"]) / r["file"]): r for r in baseline["photos"]}
    native = {r["path"]: r for r in observations["results"]}
    ordered = [dict(native[r["path"]], frontal_faces=old[r["path"]].get("frontal_faces", 0)) for r in rows]
    app_values = np.array([[number(old[r["path"]].get(k)) for k in LOG_FIELDS] for r in rows])
    fast_values = np.array([[number(native[r["path"]]["quality"].get(k)) for k in FAST_FIELDS] for r in rows])
    combined = np.column_stack((app_values, fast_values))
    groups = microbursts(ordered)
    blocks = validation_blocks(rows, groups)
    labels = np.array([r["human_retained"] for r in rows], bool)
    variants = {"app_features": app_values, "app_plus_fast": combined,
                "app_plus_fast_microburst_context": context_features(combined, groups)}
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    outcomes, photo_predictions = {}, {}
    for name in ("app_score_training_count", "native_score_training_count", *variants):
        predictions = np.zeros(len(rows), bool)
        confidence = np.zeros(len(rows), bool)
        folds = []
        for fold, (training, held) in enumerate(purged_folds(blocks)):
            if len(np.unique(labels[training])) != 2 or not held.any():
                raise ValueError("Insufficient class/block coverage")
            if name == "app_score_training_count":
                scores = app_values[held, 0]
            elif name == "native_score_training_count":
                scores = np.array([ordered[i]["composite_score"] for i in np.flatnonzero(held)])
            else:
                model = HistGradientBoostingClassifier(**policy["model"])
                with threadpool_limits(limits=2):
                    model.fit(variants[name][training], labels[training])
                    scores = model.predict_proba(variants[name][held])[:, 1]
                confidence[held] = scores >= .5
            predictions[held] = select_by_training_rate(scores, labels[training])
            folds.append(dict(metrics(labels[held], predictions[held]), fold=fold,
                              training=int(training.sum()), held=int(held.sum())))
        outcome = metrics(labels, predictions)
        outcome["passed_75"] = outcome["precision"] >= .75 and outcome["recall"] >= .75
        outcomes[name] = {"metrics": outcome, "folds": folds}
        photo_predictions[name] = predictions.tolist()
        if name in variants:
            outcomes[name + "_threshold_0.5"] = {"metrics": metrics(labels, confidence)}
            photo_predictions[name + "_threshold_0.5"] = confidence.tolist()
        print(name, json.dumps(outcome), flush=True)
    # Recheck originals AND evidence after the experiment; never report a moving corpus.
    validate_inputs(reference, observations, inventory, baseline)
    validate_delivery_inventory(delivery_inventory)
    if (baseline_path.read_bytes() != baseline_raw or delivery_inventory_path.read_bytes() != delivery_raw
            or any((cache / n).read_bytes() != raw for n, raw in inputs.items())):
        raise ValueError("Frozen evidence changed during evaluation")
    outcome = {"completed_at": datetime.now(timezone.utc).isoformat(), **policy,
               "originals": len(rows), "human": int(labels.sum()),
               "baseline_metrics": metrics(labels, np.array([r["ai_rating"] > 0 for r in rows])),
               "candidate_microburst_sizes": dict(Counter(map(len, groups))),
               "experiments": outcomes, "elapsed_seconds": time.monotonic()-started,
               "caution": "开发集消融实验，非新婚礼盲测；未交付不是技术废片；未启用模型、修改任务或照片。"}
    new_json(output / "predictions.json", {"paths": [r["path"] for r in rows], "predictions": photo_predictions})
    new_json(output / "summary.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--delivery-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.baseline, args.delivery_inventory, args.output)


if __name__ == "__main__":
    main()
