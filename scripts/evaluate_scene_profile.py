"""Offline, read-only evaluation of a photographer's selected folder.

Example: python scripts/evaluate_scene_profile.py LOG.json ORIGINALS SELECTED
The script never writes image files or a calibration profile.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from event_cull.calibration import (
    HANDCRAFTED_PIPELINE_ID,
    _feature_matrix,
    _fit_classifier,
    _roc_auc,
    exported_selection_stems,
)
from event_cull.deep_features import PIPELINE_ID, preferred_backend
from event_cull.engine import _load_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("originals", type=Path)
    parser.add_argument("selected", type=Path)
    parser.add_argument("--mode", choices=("handcrafted", "deep"), default="handcrafted")
    parser.add_argument("--reject", action="append", default=[])
    args = parser.parse_args()

    rows = json.loads(args.log.read_text(encoding="utf-8"))["photos"]
    selected = exported_selection_stems(args.selected)
    rejects = {Path(name).stem.casefold() for name in args.reject}
    pipeline = PIPELINE_ID if args.mode == "deep" else HANDCRAFTED_PIPELINE_ID
    if args.mode == "deep" and preferred_backend() == "unavailable":
        raise SystemExit("Deep visual encoder is unavailable")
    values: list[np.ndarray] = []
    usable: list[dict] = []
    batch: list[np.ndarray] = []
    started = time.monotonic()
    for row in rows:
        path = args.originals / row["file"]
        try:
            batch.append(_load_preview(path))
            usable.append(row)
        except Exception as exc:
            print(f"skip {path.name}: {exc}", flush=True)
        if len(batch) >= 16:
            values.extend(_feature_matrix(batch, pipeline))
            batch.clear()
        if len(usable) and len(usable) % 100 == 0:
            print(f"decoded {len(usable)} in {time.monotonic() - started:.1f}s", flush=True)
    if batch:
        values.extend(_feature_matrix(batch, pipeline))
    features = np.asarray(values, dtype=np.float64)
    labels = np.asarray([Path(row["file"]).stem.casefold() in selected for row in usable], dtype=np.int8)
    old_picks = np.asarray([int(row["rating"]) > 0 for row in usable], dtype=bool)
    target = int(old_picks.sum())
    blocks = np.asarray([(int(row["index"]) - 1) // 20 for row in usable], dtype=np.int32)
    crossfit = np.zeros(len(usable), dtype=np.float64)
    for fold in range(5):
        test = blocks % 5 == fold
        mean, scale, weights, bias = _fit_classifier(features[~test], labels[~test])
        crossfit[test] = ((features[test] - mean) / scale) @ weights + bias
    picked = np.zeros(len(usable), dtype=bool)
    picked[np.argsort(crossfit)[-target:]] = True
    rejected = np.asarray([Path(row["file"]).stem.casefold() in rejects for row in usable], dtype=bool)
    print(json.dumps({
        "mode": args.mode,
        "images": len(usable),
        "positive_examples": int(labels.sum()),
        "old_selected": target,
        "old_positive_recall": int((old_picks & (labels > 0)).sum()),
        "crossfit_auc_by_20_frame_blocks": round(_roc_auc(labels, crossfit), 4),
        "crossfit_top_selected_positive_recall": int((picked & (labels > 0)).sum()),
        "old_selected_explicit_rejects": int((old_picks & rejected).sum()),
        "crossfit_selected_explicit_rejects": int((picked & rejected).sum()),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
