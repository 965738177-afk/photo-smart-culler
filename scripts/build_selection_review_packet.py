"""Build a local, read-only review packet from a frozen whole-wedding evaluation.

The packet asks the photographer to distinguish missed moments, extra
deliveries and alternative burst frames. It never treats a non-delivery as a
technical defect, trains a model, or modifies source photos and ratings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from event_cull.calibration import exported_selection_files
from event_cull.engine import _load_preview
from event_cull.selection_compare import validate_delivery_inventory


def _evenly_spaced(items: list, limit: int) -> list:
    if limit <= 0:
        raise ValueError("每类抽样数量必须为正数")
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


def _rated_rows(baseline: dict, evaluation: dict, baseline_sha256: str) -> list[dict]:
    """Reject partial or misaligned evaluations before inspecting photos."""
    frozen_sha = evaluation.get("evaluated_baseline_sha256")
    if frozen_sha and frozen_sha != baseline_sha256:
        raise ValueError("评估与冻结基线哈希不一致")
    if evaluation.get("complete_evaluated_case") is not True:
        raise ValueError("只接受整场完整评估，不能从局部样本推断")
    original, scored = baseline.get("photos"), evaluation.get("photos")
    if not isinstance(original, list) or not isinstance(scored, list) or len(original) != len(scored):
        raise ValueError("评估照片数量与基线不一致")
    rows = []
    for photo, result in zip(original, scored):
        rating = result.get("ai_rating")
        score = result.get("score")
        if (photo.get("file") != result.get("file") or result.get("evaluated") is not True
                or type(rating) is not int or rating not in {0, 1, 3}
                or type(result.get("human_retained")) is not bool
                or photo.get("group") is None
                or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError("评估照片顺序、星级或人工标签不完整")
        rows.append({"path": str(Path(photo["folder"]) / photo["file"]),
                     "group": (str(photo["folder"]), photo.get("group")),
                     "index": len(rows), "ai_rating": rating,
                     "human_retained": result["human_retained"],
                     "score": score})
    return rows


def _review_candidates(rows: list[dict], limit: int) -> dict[str, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    candidates = {"same_group_different_frame": [], "human_group_missed": [],
                  "ai_group_not_delivered": []}
    for (folder, group_id), group_rows in groups.items():
        ai = [row for row in group_rows if row["ai_rating"] > 0]
        human = [row for row in group_rows if row["human_retained"]]
        if ai and human and not {r["path"] for r in ai} & {r["path"] for r in human}:
            category = "same_group_different_frame"
        elif human and not ai:
            category = "human_group_missed"
        elif ai and not human:
            category = "ai_group_not_delivered"
        else:
            continue
        rank = lambda row: float(row["score"] if row["score"] is not None else -1)
        candidates[category].append({
            "group": str(group_id), "folder": folder,
            "first_index": group_rows[0]["index"], "group_size": len(group_rows),
            "ai_path": max(ai, key=rank)["path"] if ai else None,
            "human_path": max(human, key=rank)["path"] if human else None,
            "ai_count": len(ai), "human_count": len(human),
        })
    return {name: _evenly_spaced(sorted(items, key=lambda row: row["first_index"]), limit)
            for name, items in candidates.items()}


def _font(size: int):
    for path in ("/System/Library/Fonts/Hiragino Sans GB.ttc",
                 "/System/Library/Fonts/STHeiti Medium.ttc",
                 "C:/Windows/Fonts/msyh.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _draw_sheet(samples: list[dict], output: Path, load_preview, errors: list[dict]) -> None:
    font = _font(18)
    sheet = Image.new("RGB", (1120, max(1, len(samples)) * 300), "#17191b")
    draw = ImageDraw.Draw(sheet)
    for row_number, item in enumerate(samples):
        top = row_number * 300
        draw.text((12, top + 4), f"组 {item['group']} · {item['group_size']} 张 · "
                  f"AI 留 {item['ai_count']} / 人工留 {item['human_count']}", font=font, fill="#e7e7e7")
        for column, key in enumerate(("ai_path", "human_path")):
            path = item[key]
            x = column * 560
            label = "AI 留片" if column == 0 else "人工交付"
            if path:
                try:
                    image = Image.fromarray(cv2.cvtColor(load_preview(Path(path)), cv2.COLOR_BGR2RGB))
                    image.thumbnail((530, 240))
                    sheet.paste(image, (x + (560 - image.width) // 2, top + 35 + (240 - image.height) // 2))
                except Exception as exc:
                    errors.append({"path": path, "error": str(exc)})
                    draw.text((x + 20, top + 150), "预览读取失败", font=font, fill="#e69b7b")
                draw.text((x + 12, top + 275), f"{label}: {Path(path).name}", font=font, fill="#ffffff")
            else:
                draw.text((x + 20, top + 150), "本组无留片", font=font, fill="#aaaaaa")
                draw.text((x + 12, top + 275), label, font=font, fill="#ffffff")
    sheet.save(output, quality=90)


def build_packet(baseline: Path, evaluation: Path, manual: Path, output: Path, *, per_category: int = 8,
                 load_preview=_load_preview) -> dict:
    if not 1 <= per_category <= 24:
        raise ValueError("每类抽样仅允许 1–24 组，避免生成过大的预览图")
    baseline_bytes = baseline.read_bytes()
    payload = json.loads(baseline_bytes)
    result = json.loads(evaluation.read_bytes())
    rows = _rated_rows(payload, result, hashlib.sha256(baseline_bytes).hexdigest())
    if not manual.is_dir():
        raise ValueError("人工交付目录不存在")
    delivered = {path.stem.casefold() for path in exported_selection_files(manual, recursive=True)}
    labelled = {Path(row["path"]).stem.casefold() for row in rows if row["human_retained"]}
    if delivered != labelled:
        raise ValueError("人工交付目录与冻结评估标签不一致")
    folders = {Path(row["path"]).parent.resolve() for row in rows} | {manual.resolve()}
    destination = output.resolve()
    if output.exists() or any(destination == folder or destination.is_relative_to(folder) for folder in folders):
        raise ValueError("输出必须是原片目录外一个不存在的新目录")
    samples = _review_candidates(rows, per_category)
    output.mkdir(parents=True)
    errors: list[dict] = []
    titles = {"same_group_different_frame": "同组选帧分歧",
              "human_group_missed": "人工保留但整组漏选",
              "ai_group_not_delivered": "AI 留片但整组未交付"}
    for category, items in samples.items():
        _draw_sheet(items, output / f"{category}.jpg", load_preview, errors)
    report = {"baseline": str(baseline), "evaluation": str(evaluation), "manual_folder": str(manual),
              "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
              "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
              "sampling": "每类按原始拍摄顺序均匀抽取；不是挑好看的例子，也不是质量结论。",
              "samples": samples, "visual_errors": errors, "training_performed": False,
              "source_modified": False}
    with (output / "review.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    lines = ["# 选片分歧人工复核", "", "仅本机预览；原片和星级未修改。与人工不同不等于 AI 照片一定是废片。",
             "", "请在每组后写：技术问题 / 闭眼与视线 / 表情 / 动作未完成 / 构图与遮挡 / 故事价值 / 基本等价 / 其他，以及一句具体理由。",
             ""]
    for category, items in samples.items():
        lines += [f"## {titles[category]}", "", f"![{titles[category]}]({category}.jpg)", ""]
        for item in items:
            lines += [f"- 组 {item['group']}：AI {Path(item['ai_path']).name if item['ai_path'] else '无'}；"
                      f"人工 {Path(item['human_path']).name if item['human_path'] else '无'}。原因：____", ""]
    if errors:
        lines += [f"有 {len(errors)} 张预览读取失败；不算已看图，详情见 review.json。"]
    (output / "请标注原因.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def build_capture_packet(baseline: Path, reference: Path, inventory: Path, manual: Path,
                         output: Path, *, per_category: int = 8, load_preview=_load_preview) -> dict:
    """Use verified capture paths, never basename labels across renamed counters.

    This is a sampled reason-gathering packet, not another accuracy evaluation.
    Missing source files are disclosed and not substituted by exported JPEGs.
    """
    if not 1 <= per_category <= 24:
        raise ValueError("每类抽样仅允许 1–24 组")
    baseline_bytes, reference_bytes, inventory_bytes = [p.read_bytes() for p in (baseline, reference, inventory)]
    payload, comparison, stats = [json.loads(raw) for raw in (baseline_bytes, reference_bytes, inventory_bytes)]
    digest = hashlib.sha256(baseline_bytes).hexdigest()
    if comparison.get("baseline_sha256") != digest or payload.get("manual_ratings"):
        raise ValueError("需要相同冻结基线及纯自动建议")
    folders = validate_delivery_inventory(stats)
    if folders != {manual.resolve()}:
        raise ValueError("人工交付目录与冻结身份核验不一致")
    originals = {str(Path(r["folder"]) / r["file"]): r for r in payload["photos"]}
    rows, missing = [], []
    for row in comparison["photos"]:
        path = row["path"]
        if type(row.get("human_retained")) is not bool or row.get("excluded_manual") or row.get("ai_rating") not in (0, 1, 3):
            raise ValueError("存在未知人工标签或人工改星，不能自动生成取舍样本")
        p = Path(path)
        if not p.is_file():
            missing.append(path)
            continue
        if path not in stats or [p.stat().st_size, p.stat().st_mtime_ns] != stats[path]:
            raise ValueError("原片已改变，请重新核验身份")
        original = originals[path]
        if not isinstance(original.get("score"), (int, float)) or not math.isfinite(original["score"]):
            raise ValueError("冻结基线评分无效")
        rows.append({"path": path, "group": (str(p.parent), row["group"]), "index": len(rows),
                     "ai_rating": row["ai_rating"], "human_retained": row["human_retained"],
                     "score": original["score"]})
    protected = {Path(p).parent.resolve() for p in stats} | {baseline.parent.resolve(), reference.parent.resolve()}
    if output.exists() or any(output.resolve().is_relative_to(p) for p in protected):
        raise ValueError("输出必须是原片、人工交付和冻结证据外的新目录")
    samples = _review_candidates(rows, per_category)
    output.mkdir(parents=True)
    errors, previews = [], {}

    def cached_preview(path):
        # Same file may illustrate multiple categories; shared loader runs once.
        if path not in previews:
            previews[path] = load_preview(path)
        return previews[path]

    titles = {"same_group_different_frame": "同组选帧分歧",
              "human_group_missed": "人工保留但当前组漏选",
              "ai_group_not_delivered": "AI 留片但当前组未交付"}
    lines = ["# 1011 选片分歧：请标注原因", "", "这是已有启发式分组的抽样，不代表真实故事分组。",
             "未交付不等于废片；请标注表情、动作、构图、人物组合、故事价值、基本等价或其他原因。",
             "不要只回答 A/B；每组补一句原因，更有利于识别下一步缺少的特征。", ""]
    for category, items in samples.items():
        _draw_sheet(items, output / f"{category}.jpg", cached_preview, errors)
        lines += [f"## {titles[category]}", "", f"![对照]({category}.jpg)", ""]
        for item in items:
            ai = Path(item["ai_path"]).name if item["ai_path"] else "无"
            human = Path(item["human_path"]).name if item["human_path"] else "无"
            lines += [f"- {Path(item['folder']).name} · 组 {item['group']}：AI {ai}；人工 {human}。原因：____", ""]
    validate_delivery_inventory(stats)
    for row in rows:
        p = Path(row["path"])
        if not p.is_file() or [p.stat().st_size, p.stat().st_mtime_ns] != stats[str(p)]:
            raise ValueError("原片生成期间改变，不发布完成报告")
    if any(p.read_bytes() != raw for p,raw in zip((baseline, reference, inventory),
                                                   (baseline_bytes, reference_bytes, inventory_bytes))):
        raise ValueError("冻结证据生成期间改变")
    report = {"identity_mode": "EXIF_verified_original_path", "baseline_sha256": digest,
              "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
              "samples": samples, "excluded_missing_originals": missing, "visual_errors": errors,
              "unique_decodes": len(previews), "training_performed": False, "source_modified": False,
              "sampling": "每类按原始拍摄顺序均匀抽取；不作为模型验收成绩。"}
    with (output / "review.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    with (output / "请标注原因.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("manual", type=Path, help="manual delivery; protected from report output")
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--capture-inventory", type=Path,
                        help="Use evaluation as EXIF-verified capture reference, with its frozen inventory")
    args = parser.parse_args()
    if args.capture_inventory:
        report = build_capture_packet(args.baseline, args.evaluation, args.capture_inventory,
                                      args.manual, args.output, per_category=args.per_category)
    else:
        report = build_packet(args.baseline, args.evaluation, args.manual, args.output,
                              per_category=args.per_category)
    print(json.dumps({"output": str(args.output), "counts": {k: len(v) for k, v in report["samples"].items()},
                      "visual_errors": len(report["visual_errors"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
