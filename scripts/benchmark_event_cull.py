"""Read-only full-pipeline benchmark; isolated feature cache and explicit scope.

Run from the project root with `python -m scripts.benchmark_event_cull FOLDER`.
Never writes originals, XMP, task history or the running application's cache.
Generated reports go only to an explicitly requested output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import numpy as np

from event_cull import engine
from event_cull.app import EventCullApp
from event_cull.build_info import APP_VERSION
from event_cull.source_scan import scan_source_images


def sampled_paths(paths, limit, *, sample_mode="first",keep_min=1023,keep_max=1200):
    """Deterministic unlabeled sampling and proportional retention budgets."""
    if (not paths or (limit is not None and limit < 1) or sample_mode not in {"first", "evenly"}
            or type(keep_min) is not int or type(keep_max) is not int or not 1 <= keep_min <= keep_max):
        raise ValueError("Valid nonempty benchmark inventory/sampling required")
    count = len(paths) if limit is None else min(limit, len(paths))
    indexes = (np.linspace(0, len(paths)-1, count, dtype=int) if sample_mode == "evenly"
               else np.arange(count))
    sampled = [paths[i] for i in indexes]
    fraction = count/len(paths)
    floor = min(count, max(1, round(keep_min*fraction)))
    ceiling = min(count, max(floor, round(keep_max*fraction)))
    return sampled, floor, ceiling


def sample_logical_photos(paths,limit,**kwargs):
    """Pair once before sampling, so N means captures, not half-pairs/files.

    Reuse the engine's grouping owner. The engine will discover JPEG buddies
    for the sampled RAW primaries; no extra image decoder/scanner is created.
    """
    return sampled_paths([primary for primary,_pairs in engine._representatives(paths)],limit,**kwargs)


def frozen_visual_features(results, pipeline):
    """Archive the descriptors already computed by the shared preview owner.

    A missing row must fail explicitly, not silently create a shorter training
    matrix. No additional image scan, decoder, detector or inference runs here.
    """
    if not results or any(item.visual_feature_pipeline != pipeline for item in results):
        raise ValueError("Incomplete shared visual feature pipeline")
    try:
        values = np.asarray([item.visual_features for item in results], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Incomplete shared visual feature dimensions") from exc
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Incomplete/nonfinite shared visual features")
    from event_cull.calibration import FEATURE_DIMENSIONS
    if pipeline in FEATURE_DIMENSIONS and values.shape[1] != FEATURE_DIMENSIONS[pipeline]:
        raise ValueError("Incomplete shared visual feature dimensions")
    return {"features": values, "paths": np.asarray([str(item.path) for item in results]),
            "pipeline": np.asarray(pipeline)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-mode", choices=("first", "evenly"), default="first")
    parser.add_argument("--mode", choices=("base", "limited", "full"), default="base")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--direct", action="store_true", help="match an already imported top-level-only batch")
    parser.add_argument("--human-evidence", action="store_true", help="experimental local face/body evidence; does not activate a profile")
    parser.add_argument("--face-recovery", action="store_true", help="experimental shared YuNet-guided face recovery; requires human evidence")
    parser.add_argument("--keep-min",type=int,default=1023,help="explicit whole-batch retention floor")
    parser.add_argument("--keep-max",type=int,default=1200,help="explicit whole-batch retention ceiling")
    parser.add_argument("--visual-features",choices=("none","deep","deep-cpu","handcrafted"),default="none",
                        help="collect shared-preview descriptors without activating personal preferences")
    args = parser.parse_args()
    if (not args.folder.is_dir() or (args.limit is not None and args.limit < 1)
            or not 1 <= args.keep_min <= args.keep_max):
        parser.error("valid folder and positive limit required")
    if args.face_recovery and not args.human_evidence:
        parser.error("face recovery requires --human-evidence")
    if args.output and (args.output.resolve() == args.folder.resolve()
                        or args.output.resolve().is_relative_to(args.folder.resolve())):
        parser.error("reports must be outside the original photo folder")
    if args.output and args.output.exists():
        parser.error("a new output directory is required; do not rerun before checking the existing process/report")
    started = time.monotonic()
    paths = scan_source_images([args.folder],recursive=not args.direct)
    if not paths:
        parser.error("source contains no supported photos")
    physical_files=len(paths)
    paths, floor, ceiling = sample_logical_photos(paths, args.limit, sample_mode=args.sample_mode,
                                                keep_min=args.keep_min,keep_max=args.keep_max)
    scan_seconds = time.monotonic() - started
    from event_cull.calibration import DEEP_PIPELINE_ID,CPU_DEEP_PIPELINE_ID,HANDCRAFTED_PIPELINE_ID
    visual_pipeline={"none":"","deep":DEEP_PIPELINE_ID,"deep-cpu":CPU_DEEP_PIPELINE_ID,
                     "handcrafted":HANDCRAFTED_PIPELINE_ID}[args.visual_features]
    options = engine.AnalysisOptions(scene_type="婚礼", picks_per_group=2, max_picks_per_group=3,
                                     target_keep_count=floor, target_keep_max_count=ceiling,
                                     content_model_enabled=args.mode != "base",
                                     content_review_mode="limited" if args.mode == "limited" else "full",
                                     human_analysis_enabled=args.human_evidence,
                                     human_face_recovery_enabled=args.face_recovery,
                                     calibration_feature_pipeline=visual_pipeline)
    last_report = 0

    def progress(current, total, path):
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= 20 or current == total:
            print(f"base {current}/{total}, elapsed {now-started:.2f}s", flush=True)
            last_report = now

    def review(current, total, path, seconds):
        print(f"content {current}/{total}, budget left {seconds:.2f}s", flush=True)

    with tempfile.TemporaryDirectory(prefix="miaoxuan-cull-benchmark-") as cache:
        with patch.object(engine, "_cache_path", lambda: Path(cache) / "features.json"):
            results = engine.analyze_paths(paths, options, progress, phase_callback=review)
            feature_archive = frozen_visual_features(results, visual_pipeline) if visual_pipeline else None
            cold_seconds = time.monotonic() - started
            warm_started = time.monotonic()
            warm = engine.analyze_paths(paths, options)
            warm_seconds = time.monotonic() - warm_started
    summary = {"app_version": APP_VERSION, "mode": args.mode, "source": str(args.folder),
               "logical_photos": len(results), "scan_seconds": round(scan_seconds, 3),
               "physical_source_files":physical_files,"sampled_logical_photos":len(paths),
               "whole_batch_keep_min":args.keep_min,"whole_batch_keep_max":args.keep_max,
               "sample_mode":args.sample_mode,
               "cold_seconds": round(cold_seconds, 3), "warm_seconds": round(warm_seconds, 3),
               "ratings": dict(Counter(item.suggested_rating for item in results)),
               "content_status": dict(Counter(item.content_evidence.get("status", "base_only") for item in results)),
               "failed": sum(any(reason.startswith("无法分析") for reason in item.reasons) for item in results),
               "person_status": dict(Counter(item.person_evidence.get("status","disabled") for item in results)),
               "visual_features_requested":args.visual_features,
               "visual_feature_status":dict(Counter(item.visual_feature_pipeline or "missing" for item in results)),
               "workers": engine._analysis_worker_count(), "opencv_threads": engine.cv2.getNumThreads(),
               "warm_ratings_identical": [item.suggested_rating for item in warm] == [item.suggested_rating for item in results],
               "options": asdict(options)}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if feature_archive is not None:
            with (args.output / "features.npz").open("xb") as stream:
                np.savez_compressed(stream, **feature_archive)
        # Refuse to overwrite an evaluation baseline on a second invocation.
        for name, payload in (("benchmark.json", summary), ("baseline.json", {
            "benchmark_baseline": True, "software_version": APP_VERSION,
            "source_folders": [str(args.folder)], "settings": asdict(options),
            "photos": [EventCullApp._log_record(item, i) for i, item in enumerate(results, 1)],
        })):
            with (args.output / name).open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
