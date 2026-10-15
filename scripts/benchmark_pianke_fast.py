"""Isolated full-corpus Pianke Fast test: no Flask, cloud or source writes.

Native modules are imported unchanged. Initial pass/reject is the native
prescreen, NOT the final A/B delivery. Equal-count rankings are explicitly
labelled proxy diagnostics, never represented as Pianke final selections.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional

import cv2
import numpy as np


def json_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def measure(paths, truth):
    paths = set(paths)
    exact = len(paths & truth)
    return {"retained": len(paths), "human_retained": len(truth), "exact": exact,
            "precision": exact / len(paths) if paths else None,
            "recall": exact / len(truth) if truth else None,
            "human_only": len(truth - paths), "extra_not_delivered": len(paths - truth),
            "f1": 2 * exact / (len(paths) + len(truth)) if paths or truth else None}


def native_ranker(repo):
    """Load only pure native scoring functions, avoiding Flask/key-file setup."""
    names = {"_quality_score", "_aesthetic_score", "_face_quality_score", "_subject_sharpness",
             "_composite_score", "_quality_flags"}
    tree = ast.parse((repo / "app.py").read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        raise ValueError("Native score function contract changed")
    namespace = {"Optional": Optional, "ImageInfo": object}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(repo / "app.py"), "exec"), namespace)
    return namespace["_composite_score"]


def prepare_worker(runtime, repo):
    sys.path[:0] = [runtime, repo]
    cv2.setNumThreads(1)


def process_photo(path):
    from pic_selecter.grouper import _process_one
    return _process_one(path, strength="standard", engine="fast", face_aware=False, companions=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    # Use the independently frozen, EXIF-verified identity set. Never test on
    # only delivered positives or accidentally import nested delivery JPGs.
    missing_rows = [row for row in reference["photos"] if not Path(row["path"]).is_file()]
    rows = [row for row in reference["photos"] if Path(row["path"]).is_file()]
    paths = [Path(row["path"]) for row in rows]
    if any(args.output.resolve().is_relative_to(p.parent.resolve()) for p in paths):
        parser.error("Outputs must be outside source folders")
    args.output.mkdir(parents=True, exist_ok=False)
    json_new(args.output / "excluded-missing-originals.json", missing_rows)
    initial_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    json_new(args.output / "input-inventory.json", initial_stats)
    json_new(args.output / "reference-snapshot.json", reference)
    sys.path[:0] = [str(args.runtime), str(args.repo)]
    from pic_selecter.grouper import ImageInfo  # noqa: F401; native worker result deserialization
    rank = native_ranker(args.repo)
    cv2.setNumThreads(1)
    commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    source_hashes = {str(p.relative_to(args.repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [args.repo / "app.py", *sorted((args.repo / "pic_selecter").glob("*.py"))]}
    json_new(args.output / "configuration.json", {
        "repository": str(args.repo), "commit": commit, "source_sha256": source_hashes,
        "mode": "fast", "strength": "standard", "workers": 4, "scheduler": "isolated-process-pool",
        "source_originals": len(paths), "cloud": False, "source_writes": False,
        "expert_dependencies": {name: bool(importlib.util.find_spec(name)) for name in
                                ("torch", "torchvision", "transformers", "insightface", "onnxruntime", "pyiqa", "timm")}})
    started = time.monotonic()
    results, failures, infos = [], [], []
    with (args.output / "progress.jsonl").open("x", encoding="utf-8") as progress:
        with ProcessPoolExecutor(max_workers=4, initializer=prepare_worker,
                                 initargs=(str(args.runtime), str(args.repo))) as workers:
            jobs = {workers.submit(process_photo, str(p)): p for p in paths}
            for done, future in enumerate(as_completed(jobs), 1):
                path = jobs[future]
                info, error = future.result()
                if info is None:
                    failures.append({"path": str(path), "error": error})
                else:
                    infos.append(info)
                    result = {"path": info.path, "quality": info.quality, "composite_score": rank(info),
                              "timestamp": info.timestamp, "exif_summary": info.exif_summary,
                              "phash": info.phash, "dhash": info.dhash, "whash": info.whash,
                              "ahash": info.ahash, "mtime": info.mtime, "size": info.size}
                    results.append(result)
                if done % 100 == 0 or done == len(paths):
                    status = {"done": done, "total": len(paths), "failed": len(failures),
                              "elapsed_seconds": round(time.monotonic() - started, 1)}
                    print(json.dumps(status), flush=True)
                    progress.write(json.dumps(status) + "\n")
                    progress.flush()
    results.sort(key=lambda r: r["path"])
    infos.sort(key=lambda r: r.path)
    json_new(args.output / "native-observations.json", {"results": results, "failures": failures})
    # Preserve all native grouping features without object-pickle serialization.
    descs = [i.orb_descs if i.orb_descs is not None else np.empty((0, 32), np.uint8) for i in infos]
    points = [i.orb_kps if i.orb_kps is not None else np.empty((0, 2), np.float32) for i in infos]
    np.savez_compressed(args.output / "native-grouping-features.npz",
                        descriptors=np.concatenate(descs), keypoints=np.concatenate(points),
                        offsets=np.r_[0, np.cumsum([len(d) for d in descs])],
                        hist=np.array([i.color_hist if i.color_hist is not None else np.zeros(144) for i in infos]),
                        hist_missing=np.array([i.color_hist is None for i in infos]))
    changed = [str(p) for p in paths if not p.is_file() or
               (p.stat().st_size, p.stat().st_mtime_ns) != initial_stats[str(p)]]
    if failures or changed:
        json_new(args.output / "incomplete.json", {"failures": failures, "changed_sources": changed})
        raise ValueError("Incomplete or changed inputs; no full-corpus accuracy reported")
    # Labels are read only after every image was scored by the third-party code.
    truth = {r["path"] for r in rows if r["human_retained"]}
    native_pass = {r["path"] for r in results if not r["quality"]["auto_reject"]}
    app_kept = {row["path"] for row in rows if row["ai_rating"] in (1, 3)}
    count = len(app_kept)  # Existing app count, NOT human answer count.
    ranked = sorted(results, key=lambda r: (-r["composite_score"], r["path"]))
    proxy = [r["path"] for r in ranked[:count]]
    json_new(args.output / "equal-count-proxy-selection.json", proxy)
    json_new(args.output / "native-prescreen-pass.json", sorted(native_pass))
    summary = {"completed_at": datetime.now(timezone.utc).isoformat(), "commit": commit,
               "mode": "fast-standard", "originals": len(paths), "human": len(truth),
               "excluded_missing_originals": len(missing_rows),
               "processed": len(results), "failed": len(failures),
               "native_prescreen": measure(native_pass, truth),
               "native_rejected_human_deliveries": sorted(truth - native_pass),
               "equal_count_native_score_proxy": measure(proxy, truth),
               "existing_app_same_available_corpus": measure(app_kept, truth),
               "elapsed_seconds": time.monotonic() - started,
               "pianke_final_delivery_tested": False, "expert_model_tested": False,
               "caution": "Prescreen survivors require A/B review. Top-K score ranking is a controlled proxy, not native final output. No human A/B was simulated."}
    json_new(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "native_rejected_human_deliveries"},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
