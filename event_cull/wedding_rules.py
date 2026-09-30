"""Versioned wedding policy and evidence gates shared by ranking and the UI.

The catalogue is also supplied to the optional local vision model. A catalogue
entry is not a claim that the program can observe every condition in that entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import sys

RULE_VERSION = "wedding-v2-240.4"
CATALOG_RESOURCE = Path("docs/wedding-content-selection-policy-v2-240.md")
CONTENT_TYPES = {
    "自动（保守）": "auto",
    "正式合影": "formal_group",
    "纪实 / 仪式 / 互动": "documentary",
    "新人 / 艺术肖像": "portrait",
    "动态抓拍": "dynamic",
    "静物 / 场地": "details",
}
BASE_RULE_IDS = ("W001", "W002", "W005", "W006", "W008")
# These have executable state/evidence guards. The remaining rules are model
# context or human review criteria, not silently advertised as implemented.
PROGRAM_RULE_IDS = frozenset({
    *BASE_RULE_IDS, "W003", "W004", "W011", "W017", "W019", "W022", "W032",
    "W043", "W046", "W047", "W048", "W113", "W114", "W118",
    "W134", "W036", "W038", "W161", "W169", "W226", "W228", "W230", "W240",
})


@dataclass(frozen=True)
class WeddingRule:
    identifier: str
    section: int
    section_name: str
    capabilities: str
    action: str
    text: str


@lru_cache(maxsize=1)
def rules() -> tuple[WeddingRule, ...]:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    entries = []
    section_name = ""
    for line in (base / CATALOG_RESOURCE).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section_name = line[3:]
        match = re.fullmatch(r"(W\d{3}) 【([^｜]+)｜([^】]+)】(.+)", line)
        if match:
            identifier, capabilities, action, description = match.groups()
            entries.append(WeddingRule(identifier, (int(identifier[1:]) - 1) // 10 + 1,
                                       section_name, capabilities, action, description))
    if [entry.identifier for entry in entries] != [f"W{i:03}" for i in range(1, 241)]:
        raise ValueError("婚礼规则库缺失或编号不完整，请重新安装完整版本。")
    return tuple(entries)


def prompt_rules(sections: set[int]) -> str:
    """Route all 240 rules by content; never infer roles from appearance."""
    return "\n".join(f"{rule.identifier} {rule.text}" for rule in rules()
                     if rule.section in sections)


def add_evidence(item, identifier: str, text: str) -> None:
    if identifier not in item.rule_ids:
        item.rule_ids.append(identifier)
    reason = f"[{identifier}] {text}"
    if reason not in item.reasons:
        item.reasons.append(reason)


def content_type(item, options) -> str:
    specified = CONTENT_TYPES.get(options.wedding_content_type, "auto")
    if specified != "auto":
        return specified
    if item.content_evidence.get("status") == "ok":
        return item.content_evidence["content_type"]
    return "auto"


def eye_neutral(item, options) -> bool:
    if CONTENT_TYPES.get(options.wedding_content_type) == "formal_group":
        return False
    return options.scene_type == "婚礼" and (
        content_type(item, options) in {"documentary", "portrait", "dynamic", "details"}
        or item.eye_detection_mode == "scene_whitelist_eye_exempt")


def apply_evidence_gates(item, options, *, technical_fault: bool) -> None:
    """Limit automatic stars; never mutate files or override a manual rating.

    Haar face counts are observations, not identities or verified gaze. Unknown
    measurements cap a formal portrait rather than pretending a blink is known.
    Model failures never become content rejects; the deterministic path remains.
    """
    if options.scene_type != "婚礼":
        return
    item.rule_version = RULE_VERSION
    for identifier in BASE_RULE_IDS:
        if identifier not in item.rule_ids:
            item.rule_ids.append(identifier)
    item.max_auto_rating = 3
    context = content_type(item, options)
    item.content_type = context
    evidence = item.content_evidence
    if eye_neutral(item, options):
        item.eye_penalty = 0.0
        add_evidence(item, "W004",
                     "当前内容类型不要求看镜头或睁眼；清晰度门槛仍保留")
    if technical_fault or item.scene_protected or item.coverage_protected:
        item.max_auto_rating = 0 if technical_fault else 1
        add_evidence(item, "W011", "技术问题限制自动精选；场景保底不提高质量星级")
    formal = context == "formal_group" or (context == "auto" and item.frontal_face_count >= 3)
    if formal and not eye_neutral(item, options) and options.detect_closed_eyes:
        if (item.frontal_face_count == 0 or item.open_eye_face_count < item.frontal_face_count
                or item.eye_detection_mode in {"scene_uncertain", "scene_small_face"}):
            item.max_auto_rating = min(item.max_auto_rating, 1)
            add_evidence(item, "W022", "合影眼部证据不足，不能确认全员状态；只留代表复核")
        elif item.smiling_face_count < item.frontal_face_count:
            item.max_auto_rating = min(item.max_auto_rating, 2)
            add_evidence(item, "W043", "合影笑脸证据不完整，不自动推荐为全员表情最佳")
    if evidence.get("status") == "ok":
        checks = evidence["checks"]
        if context == "auto":
            item.max_auto_rating = min(item.max_auto_rating, 2)
            add_evidence(item, "W004", "内容类型无法确认，不自动升为精选")
        if checks["subject_usable"] == "fail":
            item.max_auto_rating = min(item.max_auto_rating, 1)
            add_evidence(item, "W011", "内容模型疑似主体不可用，需复核；不删除原片")
        if checks["action_complete"] == "fail":
            item.max_auto_rating = min(item.max_auto_rating, 1)
            add_evidence(item, "W118" if context == "portrait" else "W134", "内容模型疑似动作半成品，暂不精选")
        if checks["expression_appropriate"] == "fail":
            item.max_auto_rating = min(item.max_auto_rating, 1)
            add_evidence(item, "W032" if formal else "W036", "内容模型疑似表情不合用途，暂不精选")
        required = ["subject_usable", "expression_appropriate", "action_complete"]
        if formal:
            required += ["group_complete", "gaze_coherent"]
        if any(checks[field] == "unknown" for field in required):
            item.max_auto_rating = min(item.max_auto_rating, 2)
            add_evidence(item, "W008", "内容证据仍有未知项，仅保留原片而不强推 3 星")
        if formal and any(checks[field] == "fail" for field in ("group_complete", "gaze_coherent")):
            item.max_auto_rating = min(item.max_auto_rating, 1)
            add_evidence(item, "W048", "内容模型疑似合影不完整或视线不协调，需人工确认")
        for rule_id in evidence["rule_ids"]:
            if rule_id not in item.rule_ids:
                item.rule_ids.append(rule_id)
        observations = "；".join(evidence["observations"])
        if observations:
            add_evidence(item, "W008", f"内容模型观察（可能有误）：{observations}")
    elif options.content_model_enabled:
        item.max_auto_rating = min(item.max_auto_rating, 1)
        explanation = str(evidence.get("error") or "没有可用内容证据")[:160]
        add_evidence(item, "W008", f"内置内容分析未完成：{explanation}；仅按基础检测保留原片，不推荐精修候选")
    item.featured_blocked |= item.max_auto_rating < 3


def selection_summary(results, requested_min: int | None) -> str:
    selected = sum(item.suggested_rating > 0 for item in results)
    featured = sum(item.suggested_rating == 3 for item in results)
    confirmed = sum(item.suggested_rating == 2 for item in results)
    model_complete = sum(item.content_evidence.get("status") == "ok" for item in results)
    shortfall = max(0, (requested_min or 0) - selected)
    parts = [f"婚礼规则 {RULE_VERSION} · 保留原片 {selected} · 精修候选 {featured} · 人工确认 {confirmed}"]
    if shortfall:
        parts.append(f"目标差 {shortfall} 张（未用重复图补足）")
    if any(item.content_evidence.get("status") not in {None, "base_only"} for item in results):
        parts.append(f"内容分析完成 {model_complete}/{len(results)}")
    elif any(item.content_evidence.get("status") == "base_only" for item in results):
        parts.append("快速筛选 · 未启用内容模型复核")
    return "；".join(parts)
