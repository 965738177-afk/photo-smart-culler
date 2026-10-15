"""Read-only identity audit for delivery renames and repeated camera counters.

Never infer which card a name belongs to: require capture metadata agreement.
No image decode, model inference, training or source metadata writes.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from event_cull.calibration import exported_selection_files
from event_cull.engine import RAW_EXTENSIONS


def base_stem(path):
    stem = path.stem.casefold()
    return stem.removesuffix("_副本")


def matching_candidates(delivery, candidates):
    """Time/model must exist and match; available subsecond/serial must agree."""
    matches = []
    for original in candidates:
        if not all(delivery.get(key) and original.get(key) and str(delivery[key]) == str(original[key])
                   for key in ("DateTimeOriginal", "Model")):
            continue
        if any(delivery.get(key) is not None and original.get(key) is not None
               and str(delivery[key]) != str(original[key])
               for key in ("SubSecTimeOriginal", "SerialNumber", "ImageUniqueID")):
            continue
        matches.append(original)
    return matches


def exif_records(paths, *, evidence_output=None):
    if not paths:
        return {}
    tool = Path(__file__).resolve().parents[1] / "event_cull/vendor/exiftool/exiftool"
    command = [str(tool), "-j", "-DateTimeOriginal", "-SubSecTimeOriginal", "-Model",
               "-SerialNumber", "-ImageUniqueID", "-@", "-"]
    process = subprocess.run(command, input="\n".join(str(p) for p in paths) + "\n",
                             capture_output=True, text=True, encoding="utf-8", timeout=180)
    records = json.loads(process.stdout)
    if evidence_output is not None:
        write_json(evidence_output, {"returncode": process.returncode, "stderr": process.stderr,
                                     "records": records})
    if process.returncode:
        raise ValueError("EXIF extraction failed: " + process.stderr[:1000])
    if len(records) != len(paths) or any("Error" in row for row in records):
        raise ValueError("EXIF extraction incomplete; no full result calculated")
    return {row["SourceFile"]: row for row in records}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("selected", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.baseline.read_bytes()
    payload = json.loads(raw)
    photos = payload["photos"]
    roots = [Path(p).resolve() for p in payload["source_folders"]] + [args.selected.resolve()]
    if any(args.output.resolve().is_relative_to(root) for root in roots):
        parser.error("Report output must be outside source and delivery folders")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "baseline.json").open("xb") as stream:
        stream.write(raw)
    originals = {str(Path(row["folder"]) / row["file"]): row for row in photos
                 if Path(row["file"]).suffix.lower() in RAW_EXTENSIONS}
    if len(originals) != sum(Path(row["file"]).suffix.lower() in RAW_EXTENSIONS for row in photos):
        raise ValueError("Repeated original paths in log")
    human = exported_selection_files(args.selected)
    input_paths = [Path(p) for p in originals] + list(human)
    input_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in input_paths}
    write_json(args.output / "input-inventory.json", input_stats)
    missing = [p for p in originals if not Path(p).is_file()]
    if missing:
        write_json(args.output / "missing-originals.json", missing)
        raise ValueError(f"{len(missing)} source RAW files unavailable for identity verification")
    print(f"Reading capture metadata only: {len(originals)} RAW, {len(human)} delivery files", flush=True)
    metadata = exif_records(input_paths, evidence_output=args.output / "capture-metadata.json")
    if exported_selection_files(args.selected) != human or any(
            not p.is_file() or (p.stat().st_size, p.stat().st_mtime_ns) != input_stats[str(p)] for p in input_paths):
        raise ValueError("Source or delivery changed during comparison; retry against a stable folder")
    by_stem = defaultdict(list)
    for path in originals:
        by_stem[base_stem(Path(path))].append(metadata[path])
    matching, unresolved, matched_paths = [], [], set()
    for delivery in human:
        candidates = by_stem.get(base_stem(delivery), [])
        matches = matching_candidates(metadata[str(delivery)], candidates)
        if len(matches) != 1:
            unresolved.append({"delivery": str(delivery), "candidate_count": len(matches),
                               "metadata": metadata[str(delivery)],
                               "name_candidates": [r["SourceFile"] for r in candidates]})
            continue
        source = matches[0]["SourceFile"]
        matched_paths.add(source)
        matching.append({"delivery": str(delivery), "original": source,
                         "renamed": delivery.stem.casefold() != Path(source).stem.casefold(),
                         "delivery_metadata": metadata[str(delivery)], "original_metadata": matches[0]})
    write_json(args.output / "capture-matching.json", {"matches": matching, "unresolved": unresolved})
    if unresolved:
        raise ValueError(f"{len(unresolved)} deliveries unresolved; no full hit rate reported")
    manual = payload.get("manual_ratings") or {}
    ai = {p for p, r in originals.items() if r["rating"] in (1, 3) and p not in manual}
    excluded = {p for p, r in originals.items() if p in manual or r["rating"] == 2}
    truth = matched_paths - excluded
    hit = ai & truth
    metrics = {"ai_retained": len(ai), "human_retained": len(truth), "both_retained": len(hit),
               "human_only": len(truth - ai), "ai_only_not_delivered": len(ai - truth),
               "precision": len(hit) / len(ai) if ai else None,
               "recall": len(hit) / len(truth) if truth else None,
               "f1": 2 * len(hit) / (len(ai) + len(truth)) if ai or truth else None}
    rows = [{"path": p, "group": r["group"], "ai_rating": r["rating"],
             "human_retained": p in truth, "excluded_manual": p in excluded,
             "difference": "excluded_manual" if p in excluded else
             "both_retained" if p in hit else "human_only" if p in truth else
             "ai_only_not_delivered" if p in ai else "both_not_retained",
             "ai_reason": r.get("reasons")} for p, r in originals.items()]
    sections = []
    for folder in dict.fromkeys(r["folder"] for r in originals.values()):
        paths = {p for p, r in originals.items() if r["folder"] == folder}
        a, h = ai & paths, truth & paths
        sections.append({"folder": folder, "original_count": len(paths), "ai_retained": len(a),
                         "human_retained": len(h), "both_retained": len(a & h),
                         "precision": len(a & h) / len(a) if a else None,
                         "recall": len(a & h) / len(h) if h else None})
    excluded_nonraw = [r for r in photos if Path(r["file"]).suffix.lower() not in RAW_EXTENSIONS]
    featured = {p for p, r in originals.items() if r["rating"] == 3 and p not in manual}
    report = {"scope": "Original RAW captures only; same basename requires EXIF identity agreement",
              "captured_at": datetime.now(timezone.utc).isoformat(),
              "task": payload["task"], "baseline_sha256": hashlib.sha256(raw).hexdigest(),
              "baseline_path": str(args.baseline), "settings": payload["settings"],
              "logged_rows": len(photos), "original_raw_count": len(originals),
              "delivery_file_count": len(human), "matched_delivery_files": len(matching),
              "unique_delivery_captures": len(matched_paths), "renamed_files": sum(r["renamed"] for r in matching),
              "excluded_manual_count": len(excluded), "excluded_nonraw_rows": excluded_nonraw,
              "excluded_nonraw_retained": sum(r["rating"] > 0 for r in excluded_nonraw),
              "metrics": metrics, "sections": sections,
              "featured_in_delivery": {"featured_count": len(featured), "in_delivery": len(featured & truth),
                                       "note": "Reference is original delivery, not retouch labels"},
              "photos": rows,
              "caution": "Not delivered does not mean technically unusable; no models trained or ratings written"}
    write_json(args.output / "comparison.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"photos", "excluded_nonraw_rows"}},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
