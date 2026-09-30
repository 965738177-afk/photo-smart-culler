"""Read-only, objective-specific comparison against human retouch candidates.

Uses the shared filename matcher and preview decoder. No models, training,
active profiles, XMP writes or original-file mutations are performed. Absence
from an unconfirmed reference is unknown, never a negative quality label.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from event_cull.calibration import exported_selection_files, exported_selection_stems
from event_cull.engine import _load_preview
from event_cull.selection_compare import compare_selection


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("selected", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--confirmed-complete", action="store_true")
    parser.add_argument("--direct", action="store_true", help="ignore rendered-export subdirectories")
    parser.add_argument("--original-delivery", type=Path)
    parser.add_argument("--reference-previews", type=Path, help="unique JPEG copies for a labelled visual index")
    parser.add_argument("--verify-identical-copies", action="store_true", help="hash copied reference files against originals")
    args = parser.parse_args()
    baseline_bytes = args.baseline.read_bytes()
    payload = json.loads(baseline_bytes)
    report = compare_selection(payload, args.selected, objective="retouch",
                               confirmed_complete=args.confirmed_complete, recursive=not args.direct)
    sources = {Path(row["folder"]).resolve() for row in payload["photos"]}
    sources.add(args.selected.resolve())
    if args.original_delivery:
        sources.add(args.original_delivery.resolve())
    if args.reference_previews:
        sources.add(args.reference_previews.resolve())
    destination = args.output.resolve()
    if any(destination == source or destination.is_relative_to(source) for source in sources):
        parser.error("report must be outside all photo directories")
    if destination.exists():
        parser.error("report directory must be new; previous reports are never overwritten")
    # Reuse the shared matcher's inventory rather than rescan the same folder.
    reference_files = tuple(Path(path) for path in report["reference_paths"])
    originals = {Path(row["path"]).stem.casefold(): Path(row["path"]) for row in report["photos"]}
    comparable = [row for row in report["photos"] if row["human_retouch_candidate"] is True]
    ratings = Counter(str(row["ai_rating"]) for row in comparable)
    all_featured_groups = {str(row["group"] or row["path"]) for row in report["photos"]
                           if row["ai_rating"] == 3}
    human_groups = {str(row["group"] or row["path"]) for row in comparable}
    covered_groups = human_groups & all_featured_groups
    retained = sum(row["ai_rating"] in {1, 3} for row in comparable)
    manifest = [{"path": str(path.resolve()), "size": path.stat().st_size,
                 "mtime_ns": path.stat().st_mtime_ns} for path in sorted(reference_files)]
    if args.verify_identical_copies:
        if report["ambiguous_stems"] or report["unmatched_human_stems"]:
            parser.error("copy verification requires unambiguous matched identities")
        def verify(entry):
            path = Path(entry["path"])
            original = originals[path.stem.casefold()]
            if original.suffix.casefold() != path.suffix.casefold():
                return {**entry, "copy_verification": "different_format_not_byte_comparable"}
            reference_hash = digest(path)
            source_hash = reference_hash if path.samefile(original) else digest(original)
            return {**entry, "sha256": reference_hash, "source_sha256": source_hash,
                    "copy_verification": "identical" if reference_hash == source_hash else "different_bytes"}
        with ThreadPoolExecutor(max_workers=4) as executor:
            manifest = list(executor.map(verify, manifest))
        if any(row["copy_verification"] == "different_bytes" for row in manifest):
            parser.error("reference originals differ in bytes; inspect identity before scoring")
    report.update(created_utc=datetime.now(timezone.utc).isoformat(), baseline=str(args.baseline.resolve()),
                  baseline_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
                  reference_folder=str(args.selected.resolve()), reference_manifest=manifest,
                  reference_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  software_version=payload.get("software_version"), ai_ratings=dict(Counter(
                      str(row.get("rating")) for row in payload["photos"])),
                  reference_ai_rating_counts=dict(ratings), reference_retained_count=retained,
                  reference_entirely_rejected_count=ratings.get("0", 0),
                  group_comparison={"reference_groups": len(human_groups),
                                    "groups_with_any_ai_three_star": len(covered_groups),
                                    "note": "启发式同组不保证动作/故事等价，不作为同文件命中。"},
                  active_profile_changed=False, training_performed=False, release_approved=False)
    if args.original_delivery:
        delivered = exported_selection_stems(args.original_delivery)
        reference_stems = {path.stem.casefold() for path in reference_files}
        report["original_delivery"] = {"folder": str(args.original_delivery.resolve()),
                                       "unique_count": len(delivered),
                                       "reference_in_delivery": len(reference_stems & delivered),
                                       "reference_not_in_delivery": sorted(reference_stems-delivered)}
    # This is a reference contact index, not an edited-JPEG vs RAW quality test.
    # Use supplied unique JPEG exports only. Never silently decode every RAW.
    visual_sheets, visual_errors = [], []
    images = {}
    if args.reference_previews:
        for path in exported_selection_files(args.reference_previews, recursive=False):
            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            stem = path.stem.casefold()
            if stem in images:
                parser.error("preview copies must have unique stems")
            images[stem] = path
    destination.mkdir(parents=True)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default(size=16)
    if images:
        for start in range(0, len(comparable), 20):
            rows = comparable[start:start+20]
            sheet = Image.new("RGB", (1280, 320*((len(rows)+3)//4)), "#17191b")
            draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(rows):
                identity = Path(row["path"])
                x, y = (index % 4)*320, (index // 4)*320
                preview = images.get(identity.stem.casefold())
                try:
                    if preview is None:
                        raise FileNotFoundError("No unique JPEG reference preview")
                    image = Image.fromarray(cv2.cvtColor(_load_preview(preview, maximum_dimension=640), cv2.COLOR_BGR2RGB))
                    image.thumbnail((308, 265))
                    sheet.paste(image, (x+(320-image.width)//2, y+(265-image.height)//2))
                except Exception as exc:
                    visual_errors.append({"path": str(identity), "error": str(exc)})
                    draw.text((x+8, y+100), "Preview unavailable", font=font, fill="white")
                draw.text((x+8, y+270), identity.name, font=font, fill="white")
                draw.text((x+8, y+294), f"AI {row['ai_rating']}star / G{row['group']}", font=font, fill="#cbb487")
            filename = f"reference-{start//20+1:02}.jpg"
            sheet.save(destination/filename, quality=92)
            visual_sheets.append(filename)
    report.update(visual_sheets=visual_sheets, visual_errors=visual_errors,
                  visual_note="人工精修候选 JPEG 索引；可能有后期，不用于和 RAW 比较技术画质。")
    with (destination/"comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    positive = report["positive_reference"]
    lines = ["# 独立三星精修候选对比", "", "只读对比；不训练、不改原片/XMP/当前应用。", "",
             f"- 人工唯一候选：{report['human_unique_stems']}；可比较：{positive['comparable_positive_count']}。",
             f"- 三星同文件命中：{positive['exact_hits']}；已提供正例覆盖率：{positive['positive_coverage']:.2%}。"
             if positive['positive_coverage'] is not None else "- 没有可比较正例，覆盖率不适用。",
             f"- 候选中旧 AI 星级：{dict(ratings)}；原图保留：{retained}，完全未保留：{ratings.get('0', 0)}。",
             f"- 人工候选覆盖 {len(human_groups)} 个启发式组；AI 三星覆盖其中 {len(covered_groups)} 组。不能把同组换帧当命中。",
             f"- 完整名单已确认：{args.confirmed_complete}。",
             f"- 基线 SHA-256：`{report['baseline_sha256']}`；不是当前 GUI 任务的新分析结果。", ""]
    if report["metrics"]:
        metrics = report["metrics"]
        p = metrics["retouch_precision"]
        precision = "不适用" if p is None else f"{p:.2%}"
        lines += [f"完整精修名单下的一致率：{precision}；这不是原片留片准确率。", ""]
    else:
        lines += ["名单尚未确认完整：未列出的照片未知，不计算精修精确率或额外推荐错误数；不把它们训练为负例。", ""]
    lines += [report["caution"], "", "## 全部人工正例", "",
              "| 原 RAW | 旧 AI 星级 | 连拍组 |", "| --- | ---: | ---: |"]
    for row in comparable:
        lines.append(f"| [{Path(row['path']).name}](<{row['path']}>) | {row['ai_rating']} | {row['group']} |")
    lines += ["", report["visual_note"], ""]
    for filename in visual_sheets:
        lines += [f"![]({filename})", ""]
    with (destination/"对比摘要.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"output": str(destination), "positive_reference": positive,
                      "reference_ai_rating_counts": dict(ratings), "metrics": report["metrics"],
                      "visual_errors": len(visual_errors)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
