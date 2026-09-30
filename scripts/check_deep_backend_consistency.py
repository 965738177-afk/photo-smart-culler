"""Read-only, bounded single/batch/provider conditioning check.

Use the existing preview and encoder owners. This is NOT a selection accuracy
test, and does not update models, feature caches, profiles or photo metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from event_cull import deep_features, engine


def difference(first, second):
    values = np.abs(first-second)
    return {"maximum": float(values.max()), "mean": float(values.mean())}


def check_backend(images, backend, *, factory=deep_features.DeepVisualEncoder):
    """Same pixels in every ordering; the actual backend is checked per call."""
    if not 2 <= len(images) <= 16:
        raise ValueError("A bounded multi-image control is required")
    owner = factory(backend)
    stages = []
    def encode(inputs, name):
        values = np.asarray(owner.encode(inputs), np.float32)
        if values.shape != (len(inputs), deep_features.FEATURE_DIMENSION) or not np.isfinite(values).all():
            raise ValueError("Incomplete/nonfinite visual conditioning control")
        stages.append({"stage": name, "actual_backend": owner.backend})
        return values
    try:
        singles = np.vstack([encode([im], f"single-{i}") for i, im in enumerate(images)])
        batch = encode(images, "batch")
        reverse = encode(images[::-1], "reverse-batch")[::-1]
        repeat = np.vstack([encode([im], f"repeat-{i}") for i, im in enumerate(images)])
        delta = {"single_vs_batch": difference(singles, batch),
                 "batch_vs_reverse": difference(batch, reverse),
                 "single_vs_repeat": difference(singles, repeat)}
        changed = len({item["actual_backend"] for item in stages}) > 1
        return {"requested_backend": backend, "stages": stages, "backend_switched": changed,
                "differences": delta, "tolerance": 1e-4,
                "consistent_on_control": not changed and all(d["maximum"] <= 1e-4 for d in delta.values())}
    finally:
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    raw = args.baseline.read_bytes()
    rows = json.loads(raw).get("photos", [])
    if not rows or not 2 <= args.limit <= min(16, len(rows)):
        parser.error("2–16 controls within a complete baseline required")
    sources = {Path(row["folder"]).resolve() for row in rows}
    if args.output.exists() or any(args.output.resolve().is_relative_to(p) for p in sources):
        parser.error("new output outside original directories required")
    chosen = [rows[i] for i in np.linspace(0, len(rows)-1, args.limit, dtype=int)]
    paths = [Path(row["folder"])/row["file"] for row in chosen]
    started = time.monotonic()
    images = [engine._load_preview(p) for p in paths]
    reports = [check_backend(images, "cpu")]
    if deep_features.preferred_backend() == "coreml":
        reports.append(check_backend(images, "coreml"))
    report = {"baseline": str(args.baseline), "baseline_sha256": hashlib.sha256(raw).hexdigest(),
              "paths": [str(p) for p in paths], "shared_preview_decodes": len(paths),
              "images_used_to_choose_controls": "uniform_without_manual_labels",
              "seconds": time.monotonic()-started, "backends": reports,
              "selection_accuracy": None, "profile_changed": False, "photo_metadata_changed": False}
    args.output.mkdir(parents=True)
    with (args.output/"diagnostic.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
