"""Generate a local, read-only discrepancy report for a frozen culling batch."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from event_cull.engine import _load_preview
from event_cull.selection_compare import compare_selection, group_allocation_diagnostic


def spread(items, limit):
    if len(items) <= limit:
        return items
    return [items[i*(len(items)-1)//(limit-1)] for i in range(limit)]


def percentage(value):
    return "不适用" if value is None else f"{value:.2%}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("selected", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--confirmed-complete", action="store_true")
    parser.add_argument("--no-previews", action="store_true", help="metadata-only comparison: do not decode sample photos")
    args = parser.parse_args()
    payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    report = compare_selection(payload, args.selected, confirmed_complete=args.confirmed_complete)
    source_folders = [Path(folder).resolve() for folder in payload.get("source_folders", [])]
    destination = args.output.resolve()
    if any(destination == folder or destination.is_relative_to(folder) for folder in [*source_folders, args.selected.resolve()]):
        parser.error("report output must not be inside original or selected photos")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("report folder must be empty; existing baselines/reports are never overwritten")
    report["baseline_path"] = str(args.baseline)
    report["baseline_sha256"] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    report["software_version"] = payload.get("software_version")
    report["settings"] = payload.get("settings")
    data = {str(Path(str(row.get("folder") or "")) / row["file"]): row for row in payload["photos"]}
    human = {row["path"] for row in report["photos"] if row["human_retained"]}
    groups = report["groups"]
    overlap_groups = sum(bool(g["ai_retained"]) and bool(g["human_retained"]) for g in groups.values())
    human_groups = sum(bool(g["human_retained"]) for g in groups.values())
    group_stats = {
        "all_heuristic_groups": len({row.get("group") for row in payload["photos"]}),
        "ai_retained_groups": sum(bool(g["ai_retained"]) for g in groups.values()),
        "human_retained_groups": human_groups, "both_retained_groups": overlap_groups,
        "human_group_coverage": overlap_groups/human_groups if human_groups else None,
        "both_groups_without_identical_frame": sum(bool(g["ai_retained"]) and bool(g["human_retained"])
            and not set(g["ai_retained"])&set(g["human_retained"]) for g in groups.values()),
        "ai_only_groups": sum(bool(g["ai_retained"]) and not g["human_retained"] for g in groups.values()),
        "human_only_groups": sum(bool(g["human_retained"]) and not g["ai_retained"] for g in groups.values()),
        "note": "分组来自当前启发式算法；同组不等于语义等价或故事完整。",
    }
    if report["metrics"] is not None:
        group_stats.update(group_allocation_diagnostic(groups))
    report["group_metrics"] = group_stats
    featured = [row for row in payload["photos"] if row["rating"] == 3]
    report["featured_in_original_delivery"] = sum(str(Path(row["folder"])/row["file"]) in human for row in featured)
    report["ai_ratings"] = dict(Counter(row["rating"] for row in payload["photos"]))
    pair_sets = defaultdict(list)
    for gid, group in sorted(groups.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else 0):
        ai, manual = group["ai_retained"], group["human_retained"]
        if ai and manual and not set(ai)&set(manual):
            pair_sets["same-group-alternative"].append((gid, ai[0], manual[0]))
        stars = [p for p in group["ai_featured"] if p not in human]
        if stars and manual:
            pair_sets["three-star-disagreement"].append((gid, stars[0], manual[0]))
    for path in sorted(human):
        row = data[path]
        if row["rating"] == 0 and any(word in row.get("reasons", "") for word in ("明显虚焦", "严重欠曝", "严重过曝")):
            group = groups.get(str(row["group"]))
            if group is None:
                continue  # Manual/ambiguous rows are not original AI evidence.
            ai = group["ai_retained"]
            pair_sets["technical-disagreement"].append((str(row["group"]), ai[0] if ai else path, path))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default(size=18)
    sheets, visual_errors = [], []
    for category, pairs in (() if args.no_previews else pair_sets.items()):
        pairs = spread(pairs, 8)
        sheet = Image.new("RGB", (1040, 360*len(pairs)), "#17191b")
        draw = ImageDraw.Draw(sheet)
        for i, (gid, ai_path, human_path) in enumerate(pairs):
            for j, path in enumerate((ai_path, human_path)):
                row = data[path]
                try:
                    image = Image.fromarray(cv2.cvtColor(_load_preview(Path(path)), cv2.COLOR_BGR2RGB))
                    image.thumbnail((500, 300))
                except Exception as exc:
                    # A missing/unreadable sample must not erase the metadata
                    # comparison or pretend the failed preview was inspected.
                    image = Image.new("RGB", (500, 300), "#323538")
                    visual_errors.append({"path": path, "error": str(exc)})
                    ImageDraw.Draw(image).text((20, 140), "Preview unavailable", font=font, fill="#ffffff")
                x, y = j*520 + (520-image.width)//2, i*360
                sheet.paste(image, (x, y))
                label = ("AI" if j == 0 else "HUMAN") + f"  G{gid}  {row['file']}  {row['rating']}star"
                draw.text((j*520+10, y+305), label, fill="#ffffff", font=font)
                draw.text((j*520+10, y+331), f"score {row['score']:.0f} / sharp {row.get('frame_sharpness')} / faces {row.get('frontal_faces')}", fill="#cbb487", font=font)
        name = category + ".jpg"
        sheet.save(args.output/name, quality=92)
        sheets.append({"file": name, "category": category, "pairs": pairs})
    report["visual_samples"] = sheets
    report["visual_errors"] = visual_errors
    report["previews_requested"] = not args.no_previews
    with (args.output/"comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    metrics = report["metrics"]
    summary = ["# 本批婚礼人工选片对比", "", "本报告未写星级、未修改照片、未训练模型。", "",
               f"- 原片：{len(data)}；人工交付：{len(human)} 个唯一文件名。",
               f"- 基线：已冻结的共享筛选引擎结果，软件 {report['software_version']}。",
               f"- 基线路径：`{args.baseline}`。SHA-256：`{report['baseline_sha256']}`。",
               f"- 目标保留：{payload['settings']['target_keep_count']}–{payload['settings']['target_keep_max_count']} 张。",
               f"- AI 星级分布：{report['ai_ratings']}。自动只有 0、1、3 星，2 星保留给人工。", ""]
    if metrics:
        summary += ["## 原片保留对比", "", "| 项目 | 数量 |", "| --- | ---: |",
                    f"| 双方保留相同文件 | {metrics['both_retained']} |",
                    f"| 人工保留、AI 未保留 | {metrics['human_only']} |",
                    f"| AI 保留、人工未交付 | {metrics['ai_only_not_delivered']} |",
                    f"| 双方均未保留 | {report['differences'].get('both_not_retained', 0)} |", "",
                    f"AI 保留中进入人工交付的比例：{percentage(metrics['retained_precision'])}；人工交付被 AI 找回的比例：{percentage(metrics['retained_recall'])}；F1：{percentage(metrics['retained_f1'])}。这是逐文件一致性，不等同于客观质量分数。", ""]
    else:
        summary += ["## 原片保留对比", "", "人工结果未确认完整或存在未匹配文件，暂不计算精确率和召回率。", ""]
    summary += ["## 连拍组对比", "", "| 项目 | 组数 |", "| --- | ---: |",
               f"| 全批启发式分组 | {group_stats['all_heuristic_groups']} |",
               f"| 人工保留覆盖 | {group_stats['human_retained_groups']} |",
               f"| AI 保留覆盖 | {group_stats['ai_retained_groups']} |",
               f"| 双方覆盖 | {group_stats['both_retained_groups']} |",
               f"| 双方覆盖、但没有同一帧 | {group_stats['both_groups_without_identical_frame']} |",
               f"| 仅人工覆盖 | {group_stats['human_only_groups']} |",
               f"| 仅 AI 覆盖 | {group_stats['ai_only_groups']} |", "",
               "同组不同帧需要判断：明显更好、基本等价、主观偏好、无法确认。启发式组可能拆错或合错，不能把组覆盖称为故事完整率。", "",
               f"AI 3 星共 {len(featured)} 张，其中 {report['featured_in_original_delivery']} 张进入人工原片交付；不是精修命中率，因为人工提供的是原片文件夹。", ""]
    if metrics:
        ceiling = group_stats["same_group_slot_max_exact"]
        headroom = group_stats["same_group_slot_headroom"]
        summary += [
            "## 场景名额与组内帧误差", "",
            f"- 人工保留但整组无 AI 留片：{group_stats['human_photos_in_unselected_groups']} 张；"
            f"AI 留片但整组无人工交付：{group_stats['ai_photos_in_nonhuman_groups']} 张。",
            f"- 冻结现有分组和每组 AI 留片数量，仅在组内换帧时，逐文件最多命中 {ceiling} 张"
            f"（{percentage(group_stats['same_group_slot_max_recall'])}）；相比当前还有 {headroom} 个组内名额差。",
            "这只是组合上界，不能视为照片等价或实际可达质量；超过它须调整组间名额或分组。", "",
        ]
    summary += ["## 分层抽样", "", "每类最多抽取 8 组，跨时间线取样，各类可能重复；不是人工看完所有照片。左侧是 AI 建议，右侧是人工保留的对应原片；均由原 RAW 预览生成，未用处理后 JPG 的色调比较。两侧的 star 数字都是软件建议星级，不是人工打星；右侧 0star 表示软件漏选了人工保留的这张。技术分歧且无 AI 组内代表时，两侧展示同一张，不能叫换片。", ""]
    for sheet in sheets:
        summary.extend((f"### {sheet['category']}", "", f"![]({sheet['file']})", ""))
    if args.no_previews:
        start = summary.index("## 分层抽样")
        summary[start:] = ["## 分层抽样", "", "本次仅比较冻结清单，没有读取或解码照片，也没有生成或查看画面对照。", ""]
    summary += ["## 完整清单与使用边界", "", "[逐组差异清单](逐组差异清单.md) · [全部逐图与逐组数据](comparison.json)", "",
                report["caution"], "未对图中每个人的视线、身份、动作或表情做自动事实标注。分歧照片需要画面对比；未知证据不训练成事实，原片交付标签不训练成精修标签。"]
    if visual_errors:
        summary.append(f"有 {len(visual_errors)} 次抽样预览读取失败，未计为已看图；错误记录在 comparison.json。")
    (args.output/"对比摘要.md").write_text("\n".join(summary), encoding="utf-8")
    group_lines = ["# 逐组差异清单", "", "点击文件名查看对应原 RAW。同组不代表动作等价或人物关系已识别，以下类别仅描述保留集合的差别。", "",
                   "| 组 | AI 1 星 | AI 3 星 | 人工保留 | 差异 |", "| --- | --- | --- | --- | --- |"]
    def links(paths):
        return "、".join(f"[{Path(path).name}](<{path}>)" for path in paths) or "—"
    for gid, group in sorted(groups.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else 0):
        ai, manual = set(group["ai_retained"]), set(group["human_retained"])
        if ai == manual:
            continue
        label = ("仅人工保留" if not ai else "仅 AI 保留" if not manual else
                 "同组不同帧" if not ai & manual else "部分同留，其余需对照")
        originals = [path for path in group["ai_retained"] if path not in group["ai_featured"]]
        group_lines.append(f"| {gid} | {links(originals)} | {links(group['ai_featured'])} | {links(group['human_retained'])} | {label} |")
    (args.output/"逐组差异清单.md").write_text("\n".join(group_lines), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "groups": group_stats, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
