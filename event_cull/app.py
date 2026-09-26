"""Small local review UI for the culling MVP; run with ``python -m event_cull.app``."""

from __future__ import annotations

import csv
from collections import OrderedDict
import hashlib
from io import BytesIO
import json
import locale
import math
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageOps, ImageTk
from tkinterdnd2 import COPY, DND_FILES, REFUSE_DROP, TkinterDnD

try:  # Prefer native SF Symbols where available.
    from AppKit import NSImage
except ImportError:
    NSImage = None

from event_cull.calibration import (
    CalibrationError,
    CalibrationProfile,
    apply_profile,
    build_profile,
    load_profile,
    save_profile,
)
from event_cull.build_info import APP_VERSION
from event_cull.deep_features import status_label as deep_status_label
from event_cull.history import load_history, task_name as history_task_name, upsert_task
from event_cull.licensing import LicenseController, license_label
from event_cull.license_ui import LicenseDialog
from event_cull.updater import UpdateInfo, check_for_update, download_update
from event_cull.engine import (
    ATMOSPHERE_PORTRAIT_SCENE,
    HEIC_EXTENSIONS,
    RAW_EXTENSIONS,
    AnalysisCancelled,
    AnalysisOptions,
    PhotoResult,
    _load_preview,
    analyze_folder,
    analyze_paths,
)
from event_cull.export_selected import (
    ExportCancelled,
    collect_selected_files,
    export_selected_files,
)
from event_cull.source_scan import scan_source_images
from event_cull.platform_paths import application_data_dir
from event_cull.platform_process import running_executable_names
from event_cull.xmp import (
    EMBEDDED_EXTENSIONS,
    legacy_managed_embedded_photos,
    migrate_legacy_embedded_ratings,
    restore_managed_ratings,
    write_ratings,
)


SUPPORTED_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2",
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif",
}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
DISPLAY_PREVIEW_MAX_DIMENSION = 320
MEMORY_PREVIEW_CACHE_LIMIT = 64
ZOOM_PREVIEW_MAX_DIMENSION = 1600
ZOOM_PREVIEW_MEMORY_CACHE_LIMIT = 3
GRID_THUMBNAIL_CACHE_LIMIT = 120
GRID_CARD_GUTTER = 3
PRODUCT_TITLE_ZH = "秒选 — 照片智能筛选器"
BRAND_LOGO_RESOURCE = Path("assets/branding/miaoxuan-logo.png")
SCENE_TYPES = ("通用", "婚礼", "人像写真", ATMOSPHERE_PORTRAIT_SCENE, "活动纪实", "儿童与家庭", "毕业照", "证件照", "美食")

def app_resource_path(relative_path: Path) -> Path:
    """Resolve a bundled asset in both source and PyInstaller builds."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return bundle_root / relative_path


TEXT = {
    "zh": {"title": "照片智能筛选器", "safe": "本地分析 · 原图只读", "choose": "选择原片文件夹", "folder": "原片来源", "per_group": "每组优选", "target": "目标 3★ 数量（约）", "analyze": "分析这一组", "pause": "暂停", "resume": "继续", "stop": "中止", "paused": "已暂停；可继续或中止。", "stopping": "正在中止；会在当前图片处理完成后停止。", "cancelled": "分析已中止；没有写入 XMP，原图未改动。", "results": "筛选结果", "legend": "3★ 推荐保留　2★ 可选　1★ 复核　— 不标星", "preview": "预览", "empty_preview": "拖入文件夹或照片\n在这里查看细节与清晰度", "decision": "本张决策", "xmp_hint": "星级只在确认写入时生成 XMP", "empty_detail": "从左侧选择一张照片\n查看评分与原因", "manual": "手动评级", "clear": "清除", "write": "确认写入 XMP 星级", "repair": "修复 LR 同步", "repair_confirm": "重写本次评分为 Lightroom 兼容 XMP？", "repair_confirm_detail": "将重写本次有星级的 XMP 封包，仅更新评级字段，不会移动、删除或改写 ARW/JPG。\n用于修复旧版本侧车无法被 Lightroom 读取的问题。", "repaired": "已重写 {count} 个 Lightroom 兼容 XMP。请回到 Lightroom 执行“元数据 → 从文件读取元数据”。", "restore": "还原本次星级", "restore_confirm": "还原本次写入的星级？", "restore_confirm_detail": "只撤回本 App 写入且之后未被 Lightroom 改动的星级。\n不会删除或修改你原有的 Lightroom 星级、照片或其他 XMP 信息。", "restore_nothing": "没有可还原的本次星级。", "restored": "已还原 {restored} 张本次写入的星级；跳过 {skipped} 张后续被修改或非本 App 写入的照片。请在 Lightroom 执行“元数据 → 从文件读取元数据”（不是“同步元数据”）。", "safe_note": "不会移动、删除或改写原图", "pick_folder": "选择一个或多个原片文件夹", "pick_needed": "需要照片来源", "pick_error": "请先选择或拖入一个或多个原片文件夹/照片文件。", "bad_settings": "参数不正确", "bad_settings_detail": "优选张数和目标数量都必须是正整数。", "analyzing": "正在本地分析；RAW 只读取预览，不修改原图…", "scanning": "正在扫描文件并准备第一张照片预览…", "progress": "正在检测 {current}/{total} · {name}", "analysis_failed": "分析失败", "best": "本组优选", "done": "分析完成：{count} 个逻辑原片，推荐保留 {keep} 张 3★；请复核后再写入 XMP。", "raw_preview": "RAW 预览请在 Lightroom 中核对", "score": "评分 {score:.0f}  清晰度 {sharpness:.0f}", "not_written": "已修改建议星级；尚未写入磁盘。", "nothing": "没有可写入内容", "nothing_detail": "目前没有带星级的照片。", "confirm": "确认写入 XMP", "confirm_detail": "将为 {count} 张照片创建或合并相邻 .xmp 文件。\n不会移动、删除或改写任何原图。\n已有非零星级会跳过。", "written": "XMP 写入完成：写入 {updated}，保留现有星级 {skipped}。请在 Lightroom 执行“元数据 → 从文件读取元数据”（不是“同步元数据”）。", "drop": "已导入：{folder}", "drop_error": "请拖入一个或多个原片文件夹/照片文件。"},
    "en": {"title": "Event Cull", "safe": "LOCAL · ORIGINALS READ-ONLY", "choose": "Choose source folders", "folder": "PHOTO SOURCES", "per_group": "PICKS PER GROUP", "target": "TARGET 3★ COUNT", "analyze": "Start analysis", "pause": "Pause", "resume": "Resume", "stop": "Stop", "paused": "Paused. Resume or stop when ready.", "stopping": "Stopping after the current photo finishes.", "cancelled": "Analysis stopped. No XMP was written and originals are unchanged.", "results": "CULL RESULTS", "legend": "3★ Keep  ·  2★ Optional  ·  1★ Review  ·  — Unrated", "preview": "PREVIEW", "empty_preview": "Drop folders or photo files here\nto review detail and sharpness", "decision": "THIS FRAME", "xmp_hint": "Ratings are written to XMP only when confirmed", "empty_detail": "Select a frame on the left\nto review its score and reasons", "manual": "MANUAL RATING", "clear": "Clear", "write": "Write ratings to XMP", "safe_note": "Never moves, deletes, or rewrites originals", "pick_folder": "Choose one or more source folders", "pick_needed": "Choose photo sources", "pick_error": "Choose or drop one or more source folders or photo files first.", "bad_settings": "Invalid settings", "bad_settings_detail": "Picks per group and target count must be positive integers.", "analyzing": "Analyzing locally; RAW previews are read-only…", "scanning": "Scanning files and reading the first RAW preview…", "progress": "Analyzing {current}/{total} · {name}", "analysis_failed": "Analysis failed", "best": "Group pick", "done": "Analysis complete: {count} logical photos, {keep} recommended 3★ picks. Review before writing XMP.", "raw_preview": "Review RAW preview in Lightroom", "score": "Score {score:.0f}  Sharpness {sharpness:.0f}", "not_written": "Rating updated locally; not written to disk yet.", "nothing": "Nothing to write", "nothing_detail": "No frames have a star rating yet.", "confirm": "Write XMP ratings", "confirm_detail": "This creates or merges adjacent .xmp files for {count} photos.\nIt will not move, delete, or rewrite any original.\nExisting non-zero ratings are skipped.", "written": "XMP complete: wrote {updated}, preserved existing ratings {skipped}. In Lightroom, choose Read Metadata from File.", "drop": "Imported: {folder}", "drop_error": "Drop one or more source folders or photo files."},
}

TEXT["zh"].update({
    "safe_note": "分析阶段只读；确认后才写入星级元数据",
    "export_selected": "导出已选图片",
    "export_cancel": "取消导出",
    "confirm_detail": "将为 {count} 张照片写入星级：相机 RAW 使用旁车 XMP；JPG、DNG、TIFF、HEIC 使用文件内 XMP。文件内写入前会完整备份原文件，已有非零星级默认跳过。",
    "restore_confirm_detail": "只撤回本 App 写入且之后未被 Lightroom 改动的星级。文件内写入的照片会从备份还原；请确认写入后没有继续编辑这些原文件。",
    "empty_preview": "请在任意位置拖入文件或文件夹",
    "raw_preview": "无法生成 RAW 缩略图；原片未被修改",
    "new_task": "新建任务",
    "history": "任务历史",
    "history_title": "任务历史",
    "history_empty": "还没有已保存的任务。导入照片后会自动建立记录。",
    "history_open_log": "在 Finder 中显示日志",
    "history_columns": ("任务", "照片", "已选", "3★", "状态"),
    "menu_file": "文件",
    "menu_edit": "编辑",
    "menu_window": "窗口",
    "menu_help": "帮助",
    "menu_about": "关于照片智能筛选器",
    "menu_quit": "退出照片智能筛选器",
    "menu_import_files": "导入照片文件…",
    "menu_import_folders": "导入原片文件夹…",
    "menu_new_task": "新建任务",
    "menu_history": "查看任务历史…",
    "menu_copy": "复制",
    "menu_minimize": "最小化",
    "menu_help_content": "使用说明",
    "menu_license": "授权与订阅…",
    "menu_workflow": "操作",
    "menu_culling": "照片筛选",
    "menu_export": "写入与导出",
    "menu_calibration": "选片偏好校准",
    "menu_picker_settings": "挑图设置…",
    "menu_start_analysis": "开始分析",
    "menu_pause_resume": "暂停或继续分析",
    "menu_stop_analysis": "中止分析",
    "menu_write_ratings": "确认写入 XMP 星级",
    "about_content": "照片智能筛选器\n本地 RAW 初筛、人工校准与只读预览\n不会移动、删除照片；确认写入星级后，部分格式会更新文件内元数据。",
    "choose_files": "选择文件",
    "pick_files": "选择一组照片文件",
    "new_task_confirm": "开始新任务？",
    "new_task_confirm_detail": "将清空当前筛选列表、手动评级和进度。\n不会删除已写入的 XMP、原图或 Lightroom 目录。",
    "analyze": "开始分析",
    "legend": "1★ 复核　2★ 可选　3★ 推荐保留　— 不标星",
    "stats_empty": "分析完成后显示星级统计",
    "stats": "1★ {one}　2★ {two}　3★ {three}　已选合计 {selected}　未入选 {zero}　共 {total} 张",
    "target": "目标选图数量",
    "legend": "1★ 初选复核　2★ 优质备用　3★ 精选推荐　— 未入选",
    "filter_label": "显示",
    "filter_all": "全部照片",
    "filter_three": "仅 3★",
    "filter_two": "仅 2★",
    "filter_one": "仅 1★ 初选复核",
    "filter_zero": "仅未入选",
    "write_mode": "写入方式",
    "per_group": "相似图保留",
    "range_separator": "～",
    "bad_settings_detail": "相似图保留范围和目标选图数量范围都必须是正整数，且左侧不能大于右侧。",
    "count_unit": "张",
    "drop_busy_title": "正在分析",
    "drop_busy_detail": "请先中止当前分析，等待停止后再拖入新照片。",
    "mode_safe": "安全写入｜跳过已有星级",
    "mode_managed": "更新本软件上次写入的星级",
    "mode_force": "强制覆盖全部星级",
    "score_subject": "评分 {score:.0f}  主体清晰度 {sharpness:.0f}",
    "score_frame": "评分 {score:.0f}  整图清晰度 {sharpness:.0f}",
    "pick_folder": "选择一个或多个原片文件夹",
    "added": "✓ 已导入 · JPG {jpg} 张 · RAW {raw} 张 · 合计 {total} 张照片",
    "added_other": "✓ 已导入 · JPG {jpg} 张 · RAW {raw} 张 · 其他 {other} 张 · 合计 {total} 张照片",
    "already_added": "所选照片已在当前任务中，没有重复添加。",
    "selected_summary": "JPG {jpg} 张 · RAW {raw} 张 · 合计 {total} 张照片",
    "selected_summary_other": "JPG {jpg} 张 · RAW {raw} 张 · 其他 {other} 张 · 合计 {total} 张照片",
    "task_import_summary": "本次新增 {added_total} 张 · JPG {added_jpg} 张 · RAW {added_raw} 张 · 当前合计 {total} 张照片",
    "task_import_summary_other": "本次新增 {added_total} 张 · JPG {added_jpg} 张 · RAW {added_raw} 张 · 其他 {added_other} 张 · 当前合计 {total} 张照片",
    "import_ready": "已导入 {total} 张照片\n点击“开始分析”查看完整筛选结果",
    "raw_import_ready": "已导入 RAW {raw} 张 · JPG {jpg} 张\n已准备 RAW 缩略图；点击“开始分析”开始筛选",
    "append_status": "已添加照片；再次分析时会重新计算当前任务的全部照片。",
    "eta_estimating": "预计剩余时间：正在估算…",
    "eta_remaining": "预计剩余时间：{minutes}分{seconds}秒",
    "eta_paused": "预计剩余时间：已暂停",
    "eta_complete": "已完成 ✅",
    "done": "分析完成：共 {count} 张，已选 {selected} 张，其中 3★ 精选 {keep} 张；请复核后再写入 XMP。",
    "export_log": "导出筛选日志",
    "export_log_title": "导出本次筛选日志",
    "export_log_done": "筛选日志已导出：{path}",
    "export_log_failed": "日志导出失败：{error}",
    "calibration": "选片偏好校准",
    "calibration_active": "选片偏好校准：已启用",
    "calibration_idle": "选片偏好校准",
    "calibration_update": "更新选片偏好校准",
    "calibration_intro": "根据人工最终选片，在本机微调后续推荐",
    "calibration_privacy": "本地偏好学习；不上传、不修改原片或 Lightroom 星级",
    "sync_summary": "写入 XMP 后，请在 Lightroom 中读取元数据即可同步。",
    "sync_show": "查看 Lightroom 同步说明",
    "sync_hide": "收起 Lightroom 同步说明",
    "calibration_prepare": "当前任务只需选择人工最终选出的照片文件夹",
    "calibration_log_item": "本次筛选日志：软件已自动保存",
    "calibration_export_item": "人工最终选出的 JPG / 原片文件夹（只读取文件名）",
    "calibration_source_item": "本次原片来源：软件自动关联",
    "calibration_log": "选择筛选日志（CSV 或自动保存的 JSON）",
    "calibration_export": "选择人工最终选出的 JPG / 原片文件夹",
    "calibration_sources": "选择校准原片文件夹（可多选）",
    "calibration_running": "正在本地学习你的最终选片偏好；只读取文件名与画面特征，不修改原图…",
    "calibration_progress": "正在提取本地画面特征 {current}/{total} · {name}",
    "calibration_apply": "正在按你的历史选片偏好复核 3★ {current}/{total} · {name}",
    "calibration_done": "选片偏好校准已启用：学习 {samples} 张原图，其中 {positive} 张为最高星级；分组验证区分度 {auc:.0%}。",
    "calibration_failed": "选片偏好校准未启用：{error}",
    "calibration_apply_skipped": "选片偏好校准未应用到本次结果：{error}；已保留基础筛选结果。",
})
TEXT["en"].update({
    "safe_note": "Read-only analysis; ratings are written only after confirmation",
    "export_selected": "Export selected photos",
    "export_cancel": "Cancel export",
    "confirm_detail": "Write ratings for {count} photos: camera RAW uses XMP sidecars; JPEG, DNG, TIFF and HEIC use embedded XMP. Full originals are backed up before embedded writes. Existing nonzero ratings are skipped by default.",
    "restore_confirm_detail": "Undo only app-written ratings that Lightroom has not changed. Embedded files are restored from backup; make sure the originals have not been edited since the write.",
    "empty_preview": "Drop files or folders anywhere",
    "raw_preview": "A RAW thumbnail could not be generated; the original is unchanged",
    "new_task": "New task",
    "history": "Task history",
    "history_title": "Task history",
    "history_empty": "No saved tasks yet. A local record is created when photos are imported.",
    "history_open_log": "Reveal log in Finder",
    "history_columns": ("Task", "Photos", "Selected", "3★", "Status"),
    "menu_file": "File",
    "menu_edit": "Edit",
    "menu_window": "Window",
    "menu_help": "Help",
    "menu_about": "About Photo Smart Culler",
    "menu_quit": "Quit Photo Smart Culler",
    "menu_import_files": "Import photo files…",
    "menu_import_folders": "Import source folders…",
    "menu_new_task": "New task",
    "menu_history": "Show task history…",
    "menu_copy": "Copy",
    "menu_minimize": "Minimize",
    "menu_help_content": "User guide",
    "menu_license": "Authorization & subscription…",
    "menu_workflow": "Actions",
    "menu_culling": "Photo culling",
    "menu_export": "Write & export",
    "menu_calibration": "Selection preference tuning",
    "menu_picker_settings": "Culling settings…",
    "menu_start_analysis": "Start analysis",
    "menu_pause_resume": "Pause or resume analysis",
    "menu_stop_analysis": "Stop analysis",
    "menu_write_ratings": "Confirm writing XMP ratings",
    "about_content": "Photo Smart Culler\nLocal RAW culling, selection preferences, and read-only previews.\nNever moves or deletes photos; confirmed ratings update embedded metadata in some formats.",
    "choose_files": "Choose files",
    "pick_files": "Choose photo files",
    "new_task_confirm": "Start a new task?",
    "new_task_confirm_detail": "This clears the current results, manual ratings, and progress.\nWritten XMP, originals, and the Lightroom catalog are not deleted.",
    "legend": "1★ Review  ·  2★ Optional  ·  3★ Keep  ·  — Unrated",
    "stats_empty": "Rating totals appear after analysis",
    "stats": "1★ {one}   2★ {two}   3★ {three}   Selected {selected}   Not selected {zero}   Total {total}",
    "target": "TARGET SELECTION COUNT",
    "legend": "1★ Review  ·  2★ Strong backup  ·  3★ Best  ·  — Not selected",
    "filter_label": "SHOW",
    "filter_all": "All photos",
    "filter_three": "3★ only",
    "filter_two": "2★ only",
    "filter_one": "1★ review only",
    "filter_zero": "Not selected",
    "write_mode": "WRITE MODE",
    "per_group": "SIMILAR PHOTOS KEPT",
    "range_separator": "–",
    "bad_settings_detail": "Both ranges must contain positive integers, and the left value cannot exceed the right value.",
    "count_unit": "photos",
    "drop_busy_title": "Analysis in progress",
    "drop_busy_detail": "Stop the current analysis and wait for it to finish before dropping new photos.",
    "mode_safe": "Safe · preserve existing ratings",
    "mode_managed": "Update ratings written by this app",
    "mode_force": "Force overwrite all ratings",
    "score_subject": "Score {score:.0f}  Subject sharpness {sharpness:.0f}",
    "score_frame": "Score {score:.0f}  Frame sharpness {sharpness:.0f}",
    "pick_folder": "Choose one or more source folders",
    "added": "✓ Imported · JPG {jpg} · RAW {raw} · {total} photos total",
    "added_other": "✓ Imported · JPG {jpg} · RAW {raw} · Other {other} · {total} photos total",
    "already_added": "Those photos are already in this task; no duplicates were added.",
    "selected_summary": "JPG {jpg} · RAW {raw} · {total} photos total",
    "selected_summary_other": "JPG {jpg} · RAW {raw} · Other {other} · {total} photos total",
    "task_import_summary": "Added {added_total} · JPG {added_jpg} · RAW {added_raw} · {total} photos total",
    "task_import_summary_other": "Added {added_total} · JPG {added_jpg} · RAW {added_raw} · Other {added_other} · {total} photos total",
    "import_ready": "{total} photos imported\nSelect Start analysis to view the culling results",
    "raw_import_ready": "RAW {raw} · JPG {jpg} imported\nSelect Start analysis to load RAW previews and begin",
    "append_status": "Photos added. The next analysis will recalculate the complete task.",
    "eta_estimating": "Estimated time remaining: calculating…",
    "eta_remaining": "Estimated time remaining: {minutes}m {seconds}s",
    "eta_paused": "Estimated time remaining: paused",
    "eta_complete": "Complete ✅",
    "done": "Analysis complete: {count} total, {selected} selected, including {keep} 3★ picks. Review before writing XMP.",
    "export_log": "Export culling log",
    "export_log_title": "Export this culling log",
    "export_log_done": "Culling log exported: {path}",
    "export_log_failed": "Could not export the log: {error}",
    "calibration": "Selection preference tuning",
    "calibration_active": "Selection preference tuning: on",
    "calibration_idle": "Selection preference tuning",
    "calibration_update": "Update selection preference tuning",
    "calibration_intro": "Use your final picks to tune future recommendations locally",
    "calibration_privacy": "Local preference learning; never uploads or changes originals or Lightroom ratings",
    "sync_summary": "After writing XMP, read metadata from files in Lightroom to sync.",
    "sync_show": "Show Lightroom sync instructions",
    "sync_hide": "Hide Lightroom sync instructions",
    "calibration_prepare": "For the current task, choose only the final-picks folder",
    "calibration_log_item": "This culling log: automatically saved",
    "calibration_export_item": "Your final JPG / original folder (filenames only)",
    "calibration_source_item": "This task's originals: automatically linked",
    "calibration_log": "Choose a culling log (CSV or automatic JSON)",
    "calibration_export": "Choose the final JPG / original folder",
    "calibration_sources": "Choose calibration original folders",
    "calibration_running": "Learning your final selections locally; originals stay read-only…",
    "calibration_progress": "Extracting local visual features {current}/{total} · {name}",
    "calibration_apply": "Reviewing featured picks against your history {current}/{total} · {name}",
    "calibration_done": "Selection preference tuning is on: learned {samples} originals, with {positive} top-rated; group-separated validation {auc:.0%}.",
    "calibration_failed": "Selection preference tuning was not enabled: {error}",
    "calibration_apply_skipped": "Selection preference tuning was not applied: {error}. Base culling results were kept.",
})

if sys.platform == "win32":
    TEXT["zh"]["history_open_log"] = "打开日志文件夹"
    TEXT["en"]["history_open_log"] = "Open log folder"


class ToolTip:
    """Small native tooltip for icon-only controls."""

    def __init__(self, widget: ttk.Button, text_getter) -> None:
        self.widget = widget
        self.text_getter = text_getter
        self.popup: tk.Toplevel | None = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _event=None) -> None:
        if self.popup is not None:
            return
        self.popup = tk.Toplevel(self.widget)
        self.popup.wm_overrideredirect(True)
        self.popup.wm_geometry(
            f"+{self.widget.winfo_rootx()}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 6}"
        )
        tk.Label(
            self.popup,
            text=self.text_getter(),
            background="#2a2d30",
            foreground="#f0ede8",
            padx=8,
            pady=5,
            font=("Helvetica Neue", 10),
        ).pack()

    def hide(self, _event=None) -> None:
        if self.popup is not None:
            self.popup.destroy()
            self.popup = None


class EventCullApp(TkinterDnD.Tk):
    def __init__(self) -> None:
        super().__init__()
        # Tear-off menus are never useful in this app.  On macOS, leaving the
        # Tk option at its default can materialize menu internals as empty
        # floating windows (for example ``!menu``) in a bundled build.
        self.option_add("*tearOff", "0")
        self.language = os.environ.get("EVENT_CULL_LANGUAGE", "zh").lower()
        self.language = "zh" if self.language not in TEXT else self.language
        self.t = TEXT[self.language]
        self.product_title = PRODUCT_TITLE_ZH if self.language == "zh" else self.t["title"]
        self.title(self.product_title)
        self._fit_main_window_to_display()
        self.configure(bg="#101112")
        self.results: list[PhotoResult] = []
        self.imported_files: list[Path] = []
        self.last_import_directory = Path.home()
        self.selected: PhotoResult | None = None
        self.preview: ImageTk.PhotoImage | None = None
        self._active_preview_path: Path | None = None
        self.brand_logo: ImageTk.PhotoImage | None = None
        self._preview_memory: OrderedDict[Path, Image.Image] = OrderedDict()
        self._zoom_preview_memory: OrderedDict[Path, Image.Image] = OrderedDict()
        self._preview_lock = threading.RLock()
        self._preview_generation = 0
        self._preview_zoom_generation = 0
        self._preview_zoom_window: tk.Toplevel | None = None
        self._preview_zoom_canvas: tk.Canvas | None = None
        self._preview_zoom_source: Image.Image | None = None
        self._preview_zoom_photo: ImageTk.PhotoImage | None = None
        self._preview_zoom_path: Path | None = None
        self._preview_zoom_scale = 1.0
        self._preview_zoom_fit = True
        self.current_task_id: str | None = None
        self.current_task_name = ""
        self.current_task_created_at: str | None = None
        self.license_controller = LicenseController()
        self.folder = tk.StringVar(value=self.t["choose"])
        self.import_notice = tk.StringVar(value="")
        self.task_import_summary = tk.StringVar(value="")
        self.lr_instructions = (
            "Lightroom 星级同步\n\n"
            "前置操作：\n"
            "1. Lightroom 顶部菜单「元数据」→ 取消勾选\n"
            "   「仅显示目标照片的元数据」\n"
            "2. 右侧「元数据」面板改为「选定的照片」\n"
            "3. 关闭 Lightroom，开始使用本软件分析照片\n"
            "4. 分析完毕后，确认写入星级（RAW 写入 XMP；HEIC 写入文件内 XMP）\n"
            "5. 打开 Lightroom，在「图库」→「网格视图」（快捷键 G）\n"
            "   全选本次分析的照片\n"
            "6. 右键 →「元数据」→「从文件中读取元数据」，\n"
            "   即可同步完成。"
            if self.language == "zh"
            else
            "Lightroom rating sync\n\n"
            "Before you begin:\n"
            "1. In Lightroom Metadata, uncheck Show Metadata for Target Photo Only\n"
            "2. Set the Metadata panel to Selected Photos\n"
            "3. Quit Lightroom, then analyse photos in this app\n"
            "4. When analysis is complete, confirm writing XMP star ratings\n"
            "5. Open Lightroom, switch to Library Grid (G), and select this batch\n"
            "6. Right-click → Metadata → Read Metadata from Files to sync"
        )
        self.xmp_notice = tk.StringVar(value=self.lr_instructions)
        self.xmp_result_notice = tk.StringVar(value="")
        self.target = tk.StringVar(value="1000")
        self.target_max = tk.StringVar(value="1200")
        self.picks = tk.StringVar(value="2")
        self.picks_max = tk.StringVar(value="3")
        # Picker preferences live locally with the task UI.  They are mapped
        # to AnalysisOptions only when a new analysis starts, so changing a
        # setting never rewrites already-reviewed ratings.
        self.scene_type = tk.StringVar(value="通用")
        self.ai_selection_enabled = tk.BooleanVar(value=True)
        self.priority_more_faces = tk.BooleanVar(value=True)
        self.fill_with_similar = tk.BooleanVar(value=False)
        self.keep_existing_labels = tk.BooleanVar(value=True)
        self.manual_ratings: dict[Path, int] = {}
        self.detect_closed_eyes = tk.BooleanVar(value=True)
        self.detect_blur = tk.BooleanVar(value=True)
        self.detect_exposure = tk.BooleanVar(value=True)
        self.detect_duplicates = tk.BooleanVar(value=True)
        self.blur_sensitivity = tk.StringVar(value="标准")
        self.exposure_sensitivity = tk.StringVar(value="标准")
        self.duplicate_similarity = tk.StringVar(value="标准")
        self.keep_per_duplicate = tk.StringVar(value="2")
        # Each imported batch must have an explicit rule confirmation before
        # AI work can start.  Appending photos invalidates that confirmation:
        # the source changed, so the photographer reviews the rules again.
        self.picker_settings_confirmed = False
        self.active_result_index: int | None = None
        self.active_import_path: Path | None = None
        self._grid_render_scheduled = False
        self._grid_render_after: str | None = None
        self._grid_auto_load_scheduled = False
        self._thumbnail_reflow_after: str | None = None
        self._grid_photo_refs: list[ImageTk.PhotoImage] = []
        self._grid_thumbnail_cache: OrderedDict[tuple[Path, int, int], ImageTk.PhotoImage] = OrderedDict()
        self._virtual_grid_layout: tuple[int, int, int, int] | None = None
        self._virtual_grid_entries: list[tuple[int | None, Path, list[Path], str, int, list[str]]] = []
        self._virtual_grid_range: tuple[int, int] | None = None
        self._virtual_grid_rows: dict[int, list[tk.Widget]] = {}
        self._virtual_grid_render_after: str | None = None
        self._mousewheel_bound = False
        self.grid_visible_limit = 72
        self.thumbnail_scale_percent = tk.IntVar(value=100)
        self.thumbnail_scale_label = tk.StringVar(value="缩略图 100%")
        self.photo_view_mode = tk.StringVar(value="grid")
        self.workflow_stage = 1
        self._workflow_pulse = 0
        self._workflow_animation_running = False
        self._update_check_started = False
        self.stats_text = tk.StringVar(value=self.t["stats_empty"])
        self.legend_text = tk.StringVar(value="")
        self.backup_star = tk.StringVar(value="2★")
        self.featured_star = tk.StringVar(value="3★")
        self.rating_filter = tk.StringVar(value=self.t["filter_all"])
        self.rating_filters: dict[str, int | None] = {}
        self.write_mode = tk.StringVar(value=self.t["mode_safe"])
        self.write_modes = {
            self.t["mode_safe"]: "safe",
            self.t["mode_managed"]: "managed",
            self.t["mode_force"]: "force",
        }
        self.status = tk.StringVar(value=self.t["safe_note"])
        self.progress_text = tk.StringVar(value="")
        self.eta_text = tk.StringVar(value="")
        self.pause_event = threading.Event()
        self.cancel_event = threading.Event()
        self.export_cancel_event = threading.Event()
        self.export_running = False
        self.ui_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.is_paused = False
        self.analysis_running = False
        self.analysis_started_at: float | None = None
        self.analysis_pause_started_at: float | None = None
        self.analysis_paused_seconds = 0.0
        self.analysis_progress_current = 0
        self.analysis_progress_total = 0
        self.analysis_completed_at: str | None = None
        self.current_log_path: Path | None = None
        self.running_metadata_editors: tuple[str, ...] = ()
        self.calibration_profile: CalibrationProfile | None = load_profile()
        self.calibration_running = False
        self._build()
        self.after_idle(self._dismiss_startup_menu_tearoffs)
        self._register_drop_targets(self)
        self.after(50, self.process_ui_events)
        self.after(250, self.refresh_lightroom_guard)
        self.after(1600, self._check_for_updates_in_background)

    @staticmethod
    def _main_window_size_for_screen(screen_width: int, screen_height: int) -> tuple[int, int]:
        """Open a roomy review surface while retaining a desktop-safe margin."""
        width = min(1900, max(980, screen_width - 32))
        height = min(1180, max(680, screen_height - 64))
        return width, height

    def _fit_main_window_to_display(self) -> None:
        """Centre the first window and keep every persistent region reachable."""
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width, height = self._main_window_size_for_screen(screen_width, screen_height)
        x = max(12, (screen_width - width) // 2)
        y = max(24, (screen_height - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(min(1180, width), min(720, height))

    def _dismiss_startup_menu_tearoffs(self) -> None:
        """Unpost any Tk menu left mapped by an earlier bundled launch state."""
        for widget in self.winfo_children():
            if not isinstance(widget, tk.Menu):
                continue
            try:
                widget.configure(tearoff=False)
                if widget.winfo_ismapped():
                    widget.unpost()
            except tk.TclError:
                continue

    @staticmethod
    def _wheel_units(delta: int) -> int:
        """Normalize macOS trackpads and notched mouse wheels to canvas units."""
        if delta == 0:
            return 0
        magnitude = max(1, round(abs(delta) / 120)) if abs(delta) >= 120 else 1
        return -magnitude if delta > 0 else magnitude

    def _build(self) -> None:
        # The active interface is a photographer-first workbench.  The former
        # split-pane construction remains below temporarily as implementation
        # reference for the controls that were carried into this layout.
        self._build_workbench()
        return

        style = ttk.Style(self)
        style.theme_use("clam")
        colors = {
            "bg": "#101112", "surface": "#191b1d", "surface_2": "#222528",
            "border": "#303438", "text": "#f0ede8", "muted": "#999b9d",
            "copper": "#c98b37", "copper_hover": "#e0a24c", "safe": "#799686",
        }
        style.configure("TFrame", background=colors["bg"])
        style.configure("Panel.TFrame", background=colors["surface"])
        style.configure("TLabel", background=colors["bg"], foreground=colors["text"], font=("Helvetica Neue", 13))
        style.configure("Panel.TLabel", background=colors["surface"], foreground=colors["text"], font=("Helvetica Neue", 13))
        style.configure("Title.TLabel", font=("Helvetica Neue", 24, "bold"), foreground=colors["text"])
        style.configure("Section.TLabel", font=("Helvetica Neue", 12, "bold"), foreground=colors["text"])
        style.configure("Muted.TLabel", foreground=colors["muted"], font=("Helvetica Neue", 11))
        style.configure("PanelMuted.TLabel", background=colors["surface"], foreground=colors["muted"], font=("Helvetica Neue", 11))
        style.configure("DropHint.TLabel", background=colors["surface"], foreground="#c2c5c7", font=("Helvetica Neue", 15, "bold"))
        style.configure("Stats.TLabel", background=colors["surface"], foreground="#d8c29d", font=("Helvetica Neue", 10, "bold"))
        style.configure("Chip.TLabel", background="#23312c", foreground="#b9d3c2", padding=(9, 5), font=("Helvetica Neue", 10, "bold"))
        style.configure("ImportSuccess.TLabel", background="#173b2b", foreground="#d8f2df", padding=(9, 5), font=("Helvetica Neue", 11, "bold"))
        style.configure("XmpCard.TFrame", background=colors["surface"])
        style.configure("XmpResult.TLabel", background="#2f281e", foreground="#f6d69b", padding=(12, 9), font=("Helvetica Neue", 10, "bold"))
        style.configure("SyncTitle.TLabel", background=colors["surface"], foreground=colors["text"], font=("Helvetica Neue", 12, "bold"))
        style.configure("SyncStep.TLabel", background=colors["surface"], foreground="#d2d4d5", font=("Helvetica Neue", 10))
        style.configure("SyncIndex.TLabel", background=colors["copper"], foreground="#17120c", padding=(7, 4), font=("Helvetica Neue", 10, "bold"))
        style.configure("CalibrationCard.TFrame", background="#211f1b")
        style.configure("CalibrationTitle.TLabel", background="#211f1b", foreground=colors["text"], font=("Helvetica Neue", 12, "bold"))
        style.configure("CalibrationCopy.TLabel", background="#211f1b", foreground="#c9c3b8", font=("Helvetica Neue", 10))
        style.configure("CalibrationMuted.TLabel", background="#211f1b", foreground=colors["muted"], font=("Helvetica Neue", 9))
        style.configure("CalibrationIndex.TLabel", background="#303438", foreground="#efc987", padding=(5, 3), font=("Helvetica Neue", 9, "bold"))
        style.configure("GuardSafe.TLabel", background="#173b2b", foreground="#d8f2df", padding=(10, 8), font=("Helvetica Neue", 10, "bold"))
        style.configure("GuardWarning.TLabel", background="#4a2722", foreground="#ffd4ca", padding=(10, 8), font=("Helvetica Neue", 10, "bold"))
        style.configure("TButton", background=colors["surface_2"], foreground=colors["text"], borderwidth=0, padding=(14, 10), font=("Helvetica Neue", 12))
        style.map("TButton", background=[("active", "#303438"), ("disabled", "#202224")], foreground=[("disabled", "#717477")])
        style.configure("Accent.TButton", background=colors["copper"], foreground="#17120c", padding=(18, 11), font=("Helvetica Neue", 12, "bold"))
        style.map("Accent.TButton", background=[("active", colors["copper_hover"]), ("disabled", "#5b4b35")], foreground=[("disabled", "#93816a")])
        style.configure("Calibration.TButton", background="#2b241a", foreground="#efc987", borderwidth=1, relief="solid", padding=(13, 9), font=("Helvetica Neue", 11, "bold"))
        style.map("Calibration.TButton", background=[("active", "#3a2e1e"), ("disabled", "#25211b")], foreground=[("disabled", "#7f735d")])
        style.configure("Icon.TButton", background=colors["surface_2"], padding=(12, 11), borderwidth=0)
        style.map("Icon.TButton", background=[("active", "#303438"), ("disabled", "#202224")])
        style.configure("TEntry", fieldbackground="#141618", foreground=colors["text"], insertcolor=colors["text"], bordercolor=colors["border"], padding=(9, 7))
        style.configure("TSpinbox", fieldbackground="#141618", foreground=colors["text"], arrowcolor=colors["muted"], bordercolor=colors["border"], padding=(6, 5))
        style.configure("Treeview", background=colors["surface"], fieldbackground=colors["surface"], foreground=colors["text"], borderwidth=0, rowheight=36, font=("Helvetica Neue", 12))
        style.map("Treeview", background=[("selected", "#383126")], foreground=[("selected", "#fff7e8")])
        style.configure("Treeview.Heading", background=colors["surface"], foreground=colors["muted"], relief="flat", font=("Helvetica Neue", 10, "bold"), padding=(10, 10))
        style.configure("Vertical.TScrollbar", background=colors["surface_2"], troughcolor=colors["surface"], bordercolor=colors["surface"])
        style.configure("Culling.Horizontal.TProgressbar", troughcolor="#0e1011", background=colors["copper"], bordercolor=colors["border"], lightcolor=colors["copper_hover"], darkcolor=colors["copper"])

        self._build_menubar()

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        header = ttk.Frame(self, padding=(24, 18))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(2, weight=1)
        self.new_task_button = ttk.Button(header, text=self.t["new_task"], command=self.new_task)
        self.new_task_button.grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Label(header, text=self.t["title"], style="Title.TLabel").grid(row=0, column=1, sticky="w")
        self.choose_files_button = ttk.Button(header, text=self.t["choose_files"], command=self.choose_files)
        self.choose_files_button.grid(row=0, column=3, sticky="e", padx=(0, 8))
        self.choose_button = ttk.Button(header, text=self.t["choose"], command=self.choose_folder)
        self.choose_button.grid(row=0, column=4, sticky="e")
        self.license_button = ttk.Button(header, command=self.show_license_dialog)
        self.license_button.grid(row=0, column=2, sticky="e", padx=(0, 14))
        self._refresh_license_entry()

        controls = ttk.Frame(self, style="Panel.TFrame", padding=(20, 12))
        controls.grid(row=1, column=0, sticky="ew", padx=16)
        # Keep the two selection-range controls visually centred even when the
        # source summary on the left is empty or short.
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(6, weight=1)
        ttk.Label(controls, text=self.t["folder"], style="PanelMuted.TLabel").grid(row=0, column=0, sticky="w")
        self.import_badge = ttk.Label(controls, textvariable=self.import_notice, style="ImportSuccess.TLabel", wraplength=430)
        self.import_badge.grid(row=0, column=1, columnspan=2, sticky="w", padx=(12, 0))
        self.import_badge.grid_remove()
        ttk.Label(controls, textvariable=self.folder, style="Panel.TLabel").grid(row=1, column=0, columnspan=3, sticky="w", pady=(3, 0))
        ttk.Label(controls, text=self.t["per_group"], style="PanelMuted.TLabel").grid(row=0, column=3, sticky="w", padx=(64, 0))
        picks_input = ttk.Frame(controls, style="Panel.TFrame")
        picks_input.grid(row=1, column=3, sticky="w", padx=(64, 0), pady=(3, 0))
        self.picks_min_menu = ttk.Combobox(
            picks_input,
            textvariable=self.picks,
            values=tuple(str(value) for value in range(1, 11)),
            state="readonly",
            width=3,
        )
        self.picks_min_menu.pack(side="left")
        ttk.Label(
            picks_input,
            text=self.t["range_separator"],
            style="Panel.TLabel",
        ).pack(side="left", padx=4)
        self.picks_max_menu = ttk.Combobox(
            picks_input,
            textvariable=self.picks_max,
            values=tuple(str(value) for value in range(1, 11)),
            state="readonly",
            width=3,
        )
        self.picks_max_menu.pack(side="left")
        ttk.Label(picks_input, text=self.t["count_unit"], style="Panel.TLabel").pack(side="left", padx=(6, 0))
        ttk.Label(controls, text=self.t["target"], style="PanelMuted.TLabel").grid(row=0, column=4, sticky="w", padx=(18, 0))
        target_input = ttk.Frame(controls, style="Panel.TFrame")
        target_input.grid(row=1, column=4, sticky="w", padx=(18, 0), pady=(3, 0))
        ttk.Label(target_input, text="约" if self.language == "zh" else "~", style="Panel.TLabel").pack(side="left", padx=(0, 6))
        ttk.Spinbox(
            target_input,
            from_=1,
            to=50000,
            increment=10,
            width=8,
            textvariable=self.target,
        ).pack(side="left")
        ttk.Label(
            target_input,
            text=self.t["range_separator"],
            style="Panel.TLabel",
        ).pack(side="left", padx=4)
        ttk.Spinbox(
            target_input,
            from_=1,
            to=50000,
            increment=10,
            width=8,
            textvariable=self.target_max,
        ).pack(side="left")
        ttk.Label(target_input, text=self.t["count_unit"], style="Panel.TLabel").pack(side="left", padx=(6, 0))
        ttk.Label(
            controls,
            textvariable=self.eta_text,
            style="PanelMuted.TLabel",
        ).grid(row=0, column=6, rowspan=2, sticky="e", padx=(16, 16))
        self.analyze_button = ttk.Button(controls, text=self.t["analyze"], style="Accent.TButton", command=self.start_analysis)
        self.analyze_button.grid(row=0, column=7, rowspan=2, sticky="e")
        self.pause_icon = self._control_icon("pause")
        self.resume_icon = self._control_icon("resume")
        self.stop_icon = self._control_icon("stop")
        self.pause_button = ttk.Button(
            controls,
            image=self.pause_icon,
            style="Icon.TButton",
            command=self.toggle_pause,
            state="disabled",
            takefocus=True,
        )
        self.pause_button.grid(row=0, column=8, rowspan=2, sticky="e", padx=(8, 0))
        self.stop_button = ttk.Button(
            controls,
            image=self.stop_icon,
            style="Icon.TButton",
            command=self.stop_analysis,
            state="disabled",
            takefocus=True,
        )
        self.stop_button.grid(row=0, column=9, rowspan=2, sticky="e", padx=(8, 0))
        self.pause_tooltip = ToolTip(
            self.pause_button,
            lambda: self.t["resume"] if self.is_paused else self.t["pause"],
        )
        self.stop_tooltip = ToolTip(self.stop_button, lambda: self.t["stop"])
        self.progress = ttk.Progressbar(controls, style="Culling.Horizontal.TProgressbar", mode="determinate", maximum=1, value=0)
        self.progress.grid(row=2, column=0, columnspan=10, sticky="ew", pady=(14, 0))
        ttk.Label(controls, textvariable=self.progress_text, style="PanelMuted.TLabel").grid(row=3, column=0, columnspan=10, sticky="w", pady=(5, 0))

        # A native PanedWindow gives photographers direct control over their
        # review workspace: widen the preview for image judgement, or widen
        # the result list when reviewing filenames and reasons.  Unlike the
        # former fixed grid proportions, both dividers can be dragged at any
        # time and retain their position for the lifetime of the window.
        content = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            background=colors["bg"],
            borderwidth=0,
            relief="flat",
            sashwidth=10,
            sashpad=3,
            sashrelief="raised",
            opaqueresize=True,
            showhandle=False,
        )
        content.grid(row=2, column=0, sticky="nsew", padx=16, pady=(16, 8))
        self.main_panes = content
        left = ttk.Frame(content, style="Panel.TFrame", padding=(14, 14, 0, 14))
        result_header = ttk.Frame(left, style="Panel.TFrame")
        result_header.pack(fill="x", padx=8)
        ttk.Label(result_header, text=self.t["results"], style="Section.TLabel").pack(side="left")
        rating_scheme = ttk.Frame(result_header, style="Panel.TFrame")
        rating_scheme.pack(side="right")
        ttk.Label(
            rating_scheme,
            text="精选推荐" if self.language == "zh" else "Featured",
            style="PanelMuted.TLabel",
        ).pack(side="left", padx=(8, 4))
        self.featured_star_menu = ttk.Combobox(
            rating_scheme,
            textvariable=self.featured_star,
            values=("3★", "4★", "5★"),
            state="readonly",
            width=4,
        )
        self.featured_star_menu.pack(side="left")
        self.featured_star_menu.bind("<<ComboboxSelected>>", self._on_star_mapping_changed)
        ttk.Label(
            rating_scheme,
            text="优质备用" if self.language == "zh" else "Backup",
            style="PanelMuted.TLabel",
        ).pack(side="left", padx=(8, 4))
        self.backup_star_menu = ttk.Combobox(
            rating_scheme,
            textvariable=self.backup_star,
            values=("2★", "3★", "4★"),
            state="readonly",
            width=4,
        )
        self.backup_star_menu.pack(side="left")
        self.backup_star_menu.bind("<<ComboboxSelected>>", self._on_star_mapping_changed)
        self.rating_filter_menu = ttk.Combobox(
            result_header,
            textvariable=self.rating_filter,
            state="readonly",
            width=12,
        )
        self.rating_filter_menu.pack(side="right")
        self.rating_filter_menu.bind("<<ComboboxSelected>>", self.filter_results)
        ttk.Label(result_header, text=self.t["filter_label"], style="PanelMuted.TLabel").pack(side="right", padx=(0, 7))
        ttk.Label(left, textvariable=self.legend_text, style="PanelMuted.TLabel").pack(anchor="w", padx=8, pady=(4, 3))
        ttk.Label(left, textvariable=self.stats_text, style="Stats.TLabel").pack(anchor="w", padx=8, pady=(0, 8))
        self._refresh_rating_presentation()
        columns = ("group", "star", "file", "reason")
        self.table = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse")
        # The first two fields are compact identifiers, not prose.  Prevent
        # them from consuming a share of every resize; the file and reason
        # columns receive the reading space when the middle pane is widened.
        for name, title, width, minimum, stretch in (
            ("group", "组", 48, 42, False),
            ("star", "评级", 62, 56, False),
            ("file", "原片", 280, 180, True),
            ("reason", "分析原因", 440, 240, True),
        ):
            self.table.heading(name, text=title)
            self.table.column(
                name,
                width=width,
                minwidth=minimum,
                stretch=stretch,
                anchor="w",
            )
        self.table.pack(side="left", fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.show_selected)
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.table.yview)
        scroll.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=scroll.set)

        center = ttk.Frame(content, style="Panel.TFrame", padding=14)
        ttk.Label(center, text=self.t["preview"], style="Section.TLabel").pack(anchor="w")
        self.preview_label = ttk.Label(center, text=self.t["empty_preview"], style="DropHint.TLabel", justify="center", anchor="center")
        self.preview_label.pack(fill="both", expand=True, pady=(14, 0))
        content.add(center, minsize=260, stretch="always", width=390)
        content.add(left, minsize=500, stretch="always", width=830)

        # Keep this pane independently scrollable. It may still be resized
        # with the second divider without clipping its lower controls.
        right_shell = ttk.Frame(content, style="Panel.TFrame", width=300)
        right_shell.grid_propagate(False)
        right_shell.columnconfigure(0, weight=1)
        right_shell.rowconfigure(0, weight=1)
        content.add(right_shell, minsize=280, stretch="never", width=320)
        right_canvas = tk.Canvas(
            right_shell,
            background=colors["surface"],
            borderwidth=0,
            highlightthickness=0,
            yscrollincrement=18,
        )
        right_scroll = ttk.Scrollbar(right_shell, orient="vertical", command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right_canvas.grid(row=0, column=0, sticky="nsew")
        right_scroll.grid(row=0, column=1, sticky="ns")
        right = ttk.Frame(right_canvas, style="Panel.TFrame", padding=14)
        right_window = right_canvas.create_window((0, 0), window=right, anchor="nw")

        def update_right_scroll_region(_event=None) -> None:
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))

        def fit_right_content(event) -> None:
            right_canvas.itemconfigure(right_window, width=event.width)
            update_right_scroll_region()

        def pointer_is_in_right_panel() -> bool:
            widget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
            while widget is not None:
                if widget is right_shell:
                    return True
                widget = widget.master
            return False

        def scroll_right_panel(event) -> None:
            if not pointer_is_in_right_panel() or event.delta == 0:
                return
            right_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        right.bind("<Configure>", update_right_scroll_region)
        right_canvas.bind("<Configure>", fit_right_content)
        self.bind_all("<MouseWheel>", scroll_right_panel, add="+")
        ttk.Label(right, text=self.t["decision"], style="Section.TLabel").pack(anchor="w")
        self.detail = ttk.Label(right, text=self.t["empty_detail"], style="PanelMuted.TLabel", justify="left", wraplength=220)
        self.detail.pack(fill="x", pady=(6, 12))
        ttk.Label(right, text=self.t["manual"], style="PanelMuted.TLabel").pack(anchor="w", pady=(0, 8))
        stars = ttk.Frame(right, style="Panel.TFrame")
        stars.pack(fill="x")
        for column in range(4):
            stars.columnconfigure(column, weight=1, uniform="rating")
        self.manual_rating_buttons: dict[int, ttk.Button] = {}
        for rating in (0, 1, 2, 3):
            button = ttk.Button(
                stars,
                text=self.t["clear"] if rating == 0 else f"{self._output_rating(rating)}★",
                command=lambda value=rating: self.set_manual_rating(value),
            )
            button.grid(row=0, column=rating, sticky="ew", padx=(0 if rating == 0 else 4, 0))
            self.manual_rating_buttons[rating] = button
        self._refresh_rating_presentation()
        ttk.Separator(right).pack(fill="x", pady=14)
        self.lr_guard_slot = ttk.Frame(right, style="Panel.TFrame")
        self.lr_guard_slot.pack(fill="x")
        self.lr_guard_label = ttk.Label(self.lr_guard_slot, text="", style="GuardWarning.TLabel", justify="left", wraplength=220)
        ttk.Label(right, text=self.t["write_mode"], style="PanelMuted.TLabel").pack(anchor="w", pady=(0, 5))
        self.write_mode_menu = ttk.Combobox(
            right,
            textvariable=self.write_mode,
            values=tuple(self.write_modes),
            state="readonly",
            width=24,
        )
        self.write_mode_menu.pack(anchor="w", pady=(0, 8))
        self.write_button = ttk.Button(right, text=self.t["write"], style="Accent.TButton", command=self.confirm_write, state="disabled")
        self.write_button.pack(fill="x")
        self.restore_button = ttk.Button(right, text=self.t.get("restore", "Restore app ratings"), command=self.confirm_restore, state="disabled")
        self.restore_button.pack(fill="x", pady=(6, 0))
        self.export_selected_button = ttk.Button(
            right,
            text=self.t["export_selected"],
            command=self.export_selected_photos,
            state="disabled",
        )
        self.export_selected_button.pack(fill="x", pady=(6, 0))
        # Keep the Lightroom hand-off beside the write controls without
        # permanently occupying the narrow review column.  The detailed
        # checklist is intentionally opt-in; the action that matters stays
        # visible before the user expands it.
        self.xmp_card = ttk.Frame(right, style="XmpCard.TFrame")
        self.xmp_card.pack(fill="x", pady=(16, 0))
        self.xmp_result_label = ttk.Label(
            self.xmp_card,
            textvariable=self.xmp_result_notice,
            style="XmpResult.TLabel",
            justify="left",
            wraplength=220,
        )
        self.sync_heading = ttk.Label(
            self.xmp_card,
            text="Lightroom 星级同步" if self.language == "zh" else "Lightroom rating sync",
            style="SyncTitle.TLabel",
        )
        self.sync_heading.pack(anchor="w", pady=(0, 10))
        ttk.Label(
            self.xmp_card,
            text=self.t["sync_summary"],
            style="PanelMuted.TLabel",
            justify="left",
            wraplength=220,
        ).pack(anchor="w", fill="x", pady=(0, 9))
        self.sync_toggle_button = ttk.Button(
            self.xmp_card,
            text=self.t["sync_show"],
            command=self.toggle_lightroom_sync_instructions,
        )
        self.sync_toggle_button.pack(fill="x")
        self.sync_steps_frame = ttk.Frame(self.xmp_card, style="XmpCard.TFrame")
        ttk.Label(
            self.sync_steps_frame,
            text="前置操作：" if self.language == "zh" else "Before you begin:",
            style="PanelMuted.TLabel",
        ).pack(anchor="w", pady=(0, 8))
        sync_steps = (
            (
                "Lightroom 顶部菜单「元数据」→ 取消勾选「仅显示目标照片的元数据」",
                "Uncheck Show Metadata for Target Photo Only",
            ),
            (
                "右侧「元数据」面板改为「选定的照片」",
                "Set the Metadata panel to Selected Photos",
            ),
            (
                "关闭 Lightroom，开始使用本软件分析照片",
                "Quit Lightroom, then analyse photos in this app",
            ),
            (
                "分析完毕后，确认写入 XMP 星级信息",
                "When analysis is complete, confirm writing XMP star ratings",
            ),
            (
                "打开 Lightroom，在「图库」→「网格视图」（快捷键 G）全选本次分析的照片",
                "Open Lightroom, switch to Library Grid (G), and select this batch",
            ),
            (
                "右键 →「元数据」→「从文件中读取元数据」，即可同步完成。",
                "Right-click → Metadata → Read Metadata from Files to sync",
            ),
        )
        for number, step in enumerate(sync_steps, start=1):
            row = ttk.Frame(self.sync_steps_frame, style="XmpCard.TFrame")
            row.pack(fill="x", pady=(0, 6))
            ttk.Label(row, text=str(number), style="SyncIndex.TLabel", width=2, anchor="center").pack(side="left")
            ttk.Label(
                row,
                text=step[0] if self.language == "zh" else step[1],
                style="SyncStep.TLabel",
                justify="left",
                wraplength=188,
            ).pack(side="left", fill="x", expand=True, padx=(9, 0))

        # Calibration belongs at the end of the Lightroom hand-off: it is a
        # deliberate after-the-fact learning action, never part of importing
        # or starting this batch's analysis.
        self.calibration_card = ttk.Frame(right, style="CalibrationCard.TFrame", padding=(12, 12))
        self.calibration_card.pack(side="bottom", fill="x", pady=(14, 0))
        ttk.Label(self.calibration_card, text=self.t["calibration"], style="CalibrationTitle.TLabel").pack(anchor="w")
        ttk.Label(
            self.calibration_card,
            text=self.t["calibration_intro"],
            style="CalibrationCopy.TLabel",
            justify="left",
            wraplength=220,
        ).pack(anchor="w", fill="x", pady=(5, 10))
        ttk.Label(
            self.calibration_card,
            text=deep_status_label(self.language),
            style="CalibrationMuted.TLabel",
            justify="left",
            wraplength=220,
        ).pack(anchor="w", fill="x", pady=(0, 8))
        ttk.Label(self.calibration_card, text=self.t["calibration_prepare"], style="CalibrationMuted.TLabel").pack(anchor="w", pady=(0, 6))
        calibration_items = (
            self.t["calibration_log_item"],
            self.t["calibration_export_item"],
            self.t["calibration_source_item"],
        )
        for number, item in enumerate(calibration_items, start=1):
            row = ttk.Frame(self.calibration_card, style="CalibrationCard.TFrame")
            row.pack(fill="x", pady=(0, 5))
            ttk.Label(row, text=str(number), style="CalibrationIndex.TLabel", width=2, anchor="center").pack(side="left")
            ttk.Label(
                row,
                text=item,
                style="CalibrationCopy.TLabel",
                justify="left",
                wraplength=188,
            ).pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.calibration_button = ttk.Button(
            self.calibration_card,
            command=self.import_calibration,
            style="Calibration.TButton",
        )
        self.calibration_button.pack(fill="x", pady=(7, 7))
        ttk.Label(
            self.calibration_card,
            text=self.t["calibration_privacy"],
            style="CalibrationMuted.TLabel",
            justify="left",
            wraplength=220,
        ).pack(anchor="w", fill="x")
        self._refresh_calibration_button()

        footer = ttk.Frame(self, padding=(24, 9))
        footer.grid(row=3, column=0, sticky="ew")
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(side="left")

    def _build_workbench(self) -> None:
        """Build the culling workbench around pictures, not application chrome."""
        colors = {
            "canvas": "#111214",
            "paper": "#1b1d1f",
            "soft": "#272a2d",
            "line": "#3a3e42",
            "ink": "#f1f0ec",
            "muted": "#a8aaa9",
            "sidebar": "#151617",
            "sidebar_soft": "#242628",
            "copper": "#c98b37",
            "copper_deep": "#e0a24c",
            "green": "#799686",
        }
        self._workbench_colors = colors
        self.configure(bg=colors["canvas"])
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Workbench.TFrame", background=colors["canvas"])
        style.configure("Paper.TFrame", background=colors["paper"])
        style.configure("Soft.TFrame", background=colors["soft"])
        style.configure("Side.TFrame", background=colors["sidebar"])
        style.configure("TLabel", background=colors["canvas"], foreground=colors["ink"], font=("Helvetica Neue", 12))
        style.configure("Paper.TLabel", background=colors["paper"], foreground=colors["ink"], font=("Helvetica Neue", 12))
        style.configure("Soft.TLabel", background=colors["soft"], foreground=colors["ink"], font=("Helvetica Neue", 11))
        style.configure("Muted.TLabel", background=colors["canvas"], foreground=colors["muted"], font=("Helvetica Neue", 11))
        style.configure("PaperMuted.TLabel", background=colors["paper"], foreground=colors["muted"], font=("Helvetica Neue", 10))
        style.configure("Side.TLabel", background=colors["sidebar"], foreground="#f5f1e9", font=("Helvetica Neue", 12))
        style.configure("SideMuted.TLabel", background=colors["sidebar"], foreground="#b6b1a9", font=("Helvetica Neue", 10))
        # Tk accepts one family name, not a CSS-like fallback list.  The old
        # multi-family tuple made these two important headings disappear on
        # some macOS installations.
        style.configure("SideTitle.TLabel", background=colors["sidebar"], foreground="#fffaf0", font=("Songti SC", 26, "bold"))
        style.configure("TaskTitle.TLabel", background=colors["canvas"], foreground=colors["ink"], font=("Songti SC", 22, "bold"))
        style.configure("TaskStatus.TLabel", background=colors["canvas"], foreground="#c9c4ba", font=("Helvetica Neue", 11, "bold"))
        style.configure("Step.TLabel", background=colors["canvas"], foreground=colors["muted"], font=("Helvetica Neue", 12))
        style.configure("StepActive.TLabel", background=colors["canvas"], foreground=colors["ink"], font=("Helvetica Neue", 12, "bold"))
        style.configure("Toolbar.TLabel", background=colors["paper"], foreground=colors["muted"], font=("Helvetica Neue", 11))
        style.configure("CardName.TLabel", background=colors["paper"], foreground=colors["ink"], font=("Helvetica Neue", 10, "bold"))
        style.configure("CardMeta.TLabel", background=colors["paper"], foreground=colors["muted"], font=("Helvetica Neue", 9))
        style.configure("CardScore.TLabel", background=colors["paper"], foreground="#e5af60", font=("Helvetica Neue", 14, "bold"))
        style.configure("PanelTitle.TLabel", background=colors["paper"], foreground=colors["ink"], font=("Helvetica Neue", 14, "bold"))
        style.configure("PanelMuted.TLabel", background=colors["paper"], foreground=colors["muted"], font=("Helvetica Neue", 10))
        style.configure("Stats.TLabel", background=colors["paper"], foreground=colors["copper_deep"], font=("Helvetica Neue", 10, "bold"))
        style.configure("DropHint.TLabel", background=colors["paper"], foreground=colors["muted"], font=("Helvetica Neue", 14, "bold"))
        style.configure("Decision.TLabel", background=colors["soft"], foreground=colors["ink"], font=("Helvetica Neue", 10))
        style.configure("Export.TCheckbutton", background=colors["paper"], foreground=colors["ink"], font=("Helvetica Neue", 12), padding=(6, 7))
        style.map("Export.TCheckbutton", background=[("active", colors["paper"])], foreground=[("active", colors["ink"])])
        style.configure("XmpResult.TLabel", background="#3b2a20", foreground="#ffd49a", padding=(9, 7), font=("Helvetica Neue", 10, "bold"))
        style.configure("SyncTitle.TLabel", background=colors["paper"], foreground=colors["ink"], font=("Helvetica Neue", 12, "bold"))
        style.configure("SyncStep.TLabel", background=colors["paper"], foreground=colors["muted"], font=("Helvetica Neue", 9))
        style.configure("SyncIndex.TLabel", background="#d99135", foreground="#24170b", padding=(6, 3), font=("Helvetica Neue", 9, "bold"))
        style.configure("CalibrationCard.TFrame", background="#24201b")
        style.configure("CalibrationTitle.TLabel", background="#24201b", foreground=colors["ink"], font=("Helvetica Neue", 12, "bold"))
        style.configure("CalibrationCopy.TLabel", background="#24201b", foreground="#d0cac0", font=("Helvetica Neue", 10))
        style.configure("CalibrationMuted.TLabel", background="#24201b", foreground=colors["muted"], font=("Helvetica Neue", 9))
        style.configure("CalibrationIndex.TLabel", background="#34363a", foreground="#efc987", padding=(5, 3), font=("Helvetica Neue", 9, "bold"))
        style.configure("GuardSafe.TLabel", background="#173b2b", foreground="#d8f2df", padding=(9, 7), font=("Helvetica Neue", 10, "bold"))
        style.configure("GuardWarning.TLabel", background="#4a2722", foreground="#ffd4ca", padding=(9, 7), font=("Helvetica Neue", 10, "bold"))
        style.configure("TButton", background="#292c2f", foreground=colors["ink"], borderwidth=0, padding=(12, 8), font=("Helvetica Neue", 11))
        style.map("TButton", background=[("active", "#363a3d"), ("disabled", "#202224")], foreground=[("disabled", "#747778")])
        style.configure("Primary.TButton", background=colors["copper"], foreground="#17120c", padding=(19, 11), font=("Helvetica Neue", 12, "bold"))
        style.map("Primary.TButton", background=[("active", colors["copper_deep"]), ("disabled", "#5b4b35")], foreground=[("disabled", "#93816a")])
        style.configure("Outline.TButton", background=colors["paper"], foreground=colors["ink"], borderwidth=1, relief="solid", padding=(15, 10), font=("Helvetica Neue", 11, "bold"))
        style.map("Outline.TButton", background=[("active", "#292c2f")])
        style.configure("Nav.TButton", background=colors["sidebar"], foreground="#eee9e2", anchor="w", borderwidth=0, padding=(18, 13), font=("Helvetica Neue", 12, "bold"))
        style.map("Nav.TButton", background=[("active", colors["sidebar_soft"])])
        style.configure("NavActive.TButton", background="#2b2721", foreground="#fff7e8", anchor="w", borderwidth=0, padding=(18, 13), font=("Helvetica Neue", 12, "bold"))
        style.map("NavActive.TButton", background=[("active", "#373027")])
        # Task-level actions live together beneath "新建任务".  Keeping the
        # import paths here makes them easy to find without duplicating a
        # second toolbar in the workflow area.
        style.configure("SideHistory.TButton", background=colors["sidebar"], foreground="#e4e0d8", anchor="center", borderwidth=0, padding=(14, 10), font=("Helvetica Neue", 11, "bold"))
        style.map("SideHistory.TButton", background=[("active", "#222427")])
        style.configure("SideImport.TButton", background="#242628", foreground="#f3eee5", anchor="center", borderwidth=0, padding=(14, 10), font=("Helvetica Neue", 11, "bold"))
        style.map("SideImport.TButton", background=[("active", "#303336"), ("disabled", "#1d1f21")], foreground=[("disabled", "#777a79")])
        style.configure("ViewMode.TButton", background="#292c2f", foreground="#dfdbd3", borderwidth=0, padding=(10, 7), font=("Helvetica Neue", 10, "bold"))
        style.map("ViewMode.TButton", background=[("active", "#3a3d40")])
        style.configure("ViewModeActive.TButton", background="#4a3922", foreground="#ffe7b5", borderwidth=0, padding=(10, 7), font=("Helvetica Neue", 10, "bold"))
        style.map("ViewModeActive.TButton", background=[("active", "#614722")])
        style.configure("SideBuy.TButton", background=colors["copper"], foreground="#fffaf3", borderwidth=0, padding=(13, 10), font=("Helvetica Neue", 11, "bold"))
        style.map("SideBuy.TButton", background=[("active", colors["copper_deep"])])
        style.configure("SideTeam.TButton", background="#25211a", foreground="#f6d89e", borderwidth=1, padding=(11, 8), font=("Helvetica Neue", 10, "bold"))
        style.map("SideTeam.TButton", background=[("active", "#3b3020")], foreground=[("active", "#ffe8ba")])
        style.configure("TEntry", fieldbackground="#151719", foreground=colors["ink"], insertcolor=colors["ink"], bordercolor=colors["line"], padding=(8, 6))
        style.configure("TSpinbox", fieldbackground="#151719", foreground=colors["ink"], arrowcolor=colors["muted"], bordercolor=colors["line"], padding=(5, 4))
        # Selection controls deliberately remain white-on-black: photographers
        # scan these small numeric choices at a glance against the dark desk.
        style.configure("TCombobox", fieldbackground="#f7f6f2", background="#f7f6f2", foreground="#171717", arrowcolor="#333333", padding=(5, 3))
        style.configure("Treeview", background=colors["paper"], fieldbackground=colors["paper"], foreground=colors["ink"], borderwidth=0, rowheight=34, font=("Helvetica Neue", 11))
        style.map("Treeview", background=[("selected", "#393128")], foreground=[("selected", "#fff7e8")])
        style.configure("Treeview.Heading", background=colors["soft"], foreground=colors["muted"], relief="flat", font=("Helvetica Neue", 10, "bold"), padding=(9, 9))
        style.configure(
            "Culling.Horizontal.TProgressbar",
            troughcolor="#181a1c",
            background=colors["copper"],
            bordercolor=colors["line"],
            lightcolor=colors["copper_deep"],
            darkcolor=colors["copper"],
            thickness=8,
        )

        self._build_menubar()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        shell = ttk.Frame(self, style="Workbench.TFrame")
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(shell, style="Side.TFrame", width=184)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)
        self._build_brand_lockup(sidebar, colors)
        # Native bitmap icons keep the sidebar scannable without relying on
        # platform-dependent emoji or font glyphs.
        self.sidebar_icons = {
            "history": self._sidebar_icon("history"),
            "folder": self._sidebar_icon("folder"),
            "photos": self._sidebar_icon("photos"),
            "cull": self._sidebar_icon("cull"),
            "list": self._sidebar_icon("list"),
            "cull_active": self._sidebar_icon("cull", color=colors["copper"]),
            "export": self._sidebar_icon("export"),
            "export_active": self._sidebar_icon("export", color=colors["copper"]),
            "calibration": self._sidebar_icon("calibration"),
            "calibration_active": self._sidebar_icon("calibration", color=colors["copper"]),
        }
        self.new_task_button = ttk.Button(
            sidebar,
            text="＋ 新建任务",
            style="Primary.TButton",
            command=self.new_task,
        )
        self.new_task_button.pack(fill="x", padx=20, pady=(0, 8))

        def sidebar_action_button(
            text: str,
            icon: str,
            style_name: str,
            background: str,
            command,
            *,
            bottom_padding: int,
        ) -> ttk.Button:
            """Keep task-action text centered while its icon has a fixed left slot."""
            holder = tk.Frame(sidebar, bg=background, highlightthickness=0)
            holder.pack(fill="x", padx=20, pady=(0, bottom_padding))
            button = ttk.Button(holder, text=text, style=style_name, command=command)
            button.pack(fill="x")
            icon_label = tk.Label(holder, image=self.sidebar_icons[icon], bg=background, bd=0, takefocus=0)
            icon_label.place(x=12, rely=0.5, anchor="w")

            def invoke_button(_event: tk.Event) -> str | None:
                if str(button.cget("state")) != "disabled":
                    button.invoke()
                return "break"

            icon_label.bind("<Button-1>", invoke_button)
            return button

        self.history_button = sidebar_action_button(
            "历史记录",
            "history",
            "SideHistory.TButton",
            colors["sidebar"],
            self.show_task_history,
            bottom_padding=16,
        )
        ttk.Separator(sidebar).pack(fill="x", padx=20, pady=(0, 11))
        ttk.Label(sidebar, text="导入照片", style="SideMuted.TLabel").pack(anchor="w", padx=24, pady=(0, 7))
        self.choose_button = sidebar_action_button(
            "导入文件夹",
            "folder",
            "SideImport.TButton",
            "#242628",
            self.choose_folder,
            bottom_padding=7,
        )
        self.choose_files_button = sidebar_action_button(
            "选择照片",
            "photos",
            "SideImport.TButton",
            "#242628",
            self.choose_files,
            bottom_padding=18,
        )
        ttk.Separator(sidebar).pack(fill="x", padx=20, pady=(0, 7))
        self.nav_indicators: dict[str, tk.Frame] = {}

        def nav_row(key: str, text: str, icon: str, command, *, active: bool = False) -> ttk.Button:
            container = tk.Frame(sidebar, bg=colors["sidebar"], highlightthickness=0)
            container.pack(fill="x")
            indicator = tk.Frame(
                container,
                width=5,
                bg=colors["copper"] if active else colors["sidebar"],
            )
            indicator.pack(side="left", fill="y")
            indicator.pack_propagate(False)
            button = ttk.Button(
                container,
                text=text,
                image=self.sidebar_icons[f"{icon}_active" if active else icon],
                compound="left",
                style="NavActive.TButton" if active else "Nav.TButton",
                command=command,
            )
            button.pack(side="left", fill="x", expand=True)
            self.nav_indicators[key] = indicator
            return button

        self.cull_nav_button = nav_row("cull", "照片筛选", "cull", self.show_culling_workspace, active=True)
        self.export_nav_button = nav_row("export", "写入与导出", "export", self.show_export_workspace)
        self.calibration_nav_button = nav_row("calibration", "选片偏好校准", "calibration", self.show_calibration_workspace)
        sidebar_spacer = ttk.Frame(sidebar, style="Side.TFrame")
        sidebar_spacer.pack(fill="both", expand=True)
        ttk.Separator(sidebar).pack(fill="x", padx=25, pady=(0, 14))
        self.license_caption = ttk.Label(sidebar, text="试用版", style="SideMuted.TLabel")
        self.license_caption.pack(anchor="w", padx=28)
        self.license_button = ttk.Button(sidebar, style="SideBuy.TButton", command=self.show_license_dialog)
        self.license_button.pack(fill="x", padx=25, pady=(6, 22))
        ttk.Label(sidebar, text=f"v{APP_VERSION}", style="SideMuted.TLabel").pack(anchor="w", padx=28, pady=(0, 26))
        self._refresh_license_entry()

        body = ttk.Frame(shell, style="Workbench.TFrame", padding=(30, 18, 24, 18))
        body.grid(row=0, column=1, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)
        header = ttk.Frame(body, style="Workbench.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        # The header keeps task identity on the left and uses the right edge
        # for contextual status after an import, rather than duplicating more
        # permanent actions that already live in the sidebar and workflow.
        header.columnconfigure(0, weight=1)
        self.task_title = tk.StringVar(value="新建筛选任务 · 等待导入")
        ttk.Label(header, textvariable=self.task_title, style="TaskTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.task_status_label = ttk.Label(
            header,
            textvariable=self.task_import_summary,
            style="TaskStatus.TLabel",
            justify="right",
        )
        self.task_status_label.grid(row=0, column=1, sticky="e", padx=(24, 0))

        # The workflow owns a dedicated band instead of competing with the
        # import and analysis actions above it.  This makes its current stage
        # readable as a process state, not a second row of toolbar controls.
        workflow_band = tk.Frame(
            body,
            background=colors["soft"],
            highlightbackground=colors["line"],
            highlightthickness=1,
        )
        workflow_band.grid(row=1, column=0, sticky="ew", pady=(12, 20))
        workflow_band.columnconfigure(0, weight=1)
        steps = ttk.Frame(workflow_band, style="Soft.TFrame", padding=(22, 16, 22, 10))
        steps.grid(row=0, column=0, sticky="ew")
        steps.columnconfigure(0, weight=1)
        self.workflow_bubbles: list[tk.Label] = []
        self.workflow_labels: list[tk.Label] = []
        self.workflow_tracks: list[tk.Canvas] = []
        stage_row = tk.Frame(steps, bg=colors["soft"])
        stage_row.grid(row=0, column=0, sticky="ew")
        for index, label in enumerate(("导入照片", "确认规则", "AI 分析", "人工复核", "写入评级"), start=1):
            stage_column = (index - 1) * 2
            stage = tk.Frame(stage_row, bg=colors["soft"])
            stage.grid(row=0, column=stage_column, sticky="w")
            bubble = tk.Label(stage, text=str(index), width=3, bg="#303336", fg="#b8bbb9", font=("Helvetica Neue", 13, "bold"), padx=3, pady=5)
            bubble.pack(side="left")
            stage_label = tk.Label(stage, text=label, bg=colors["soft"], fg="#a6a8a7", font=("Helvetica Neue", 13, "bold"), padx=8)
            stage_label.pack(side="left")
            self.workflow_bubbles.append(bubble)
            self.workflow_labels.append(stage_label)
            if index < 5:
                track = tk.Canvas(stage_row, height=8, bg=colors["soft"], highlightthickness=0, bd=0)
                track_column = stage_column + 1
                stage_row.columnconfigure(track_column, weight=1, minsize=34)
                track.grid(row=0, column=track_column, sticky="ew", padx=(10, 14))
                track.bind("<Configure>", lambda _event, canvas=track: self._draw_workflow_track(canvas))
                self.workflow_tracks.append(track)
        checkpoint = tk.Frame(
            workflow_band,
            background="#202326",
            highlightbackground="#42464a",
            highlightthickness=1,
            padx=18,
            pady=15,
        )
        checkpoint.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 16))
        checkpoint.columnconfigure(1, weight=1)
        self.rule_heading = tk.StringVar(value="开始前：先导入照片")
        self.rule_summary = tk.StringVar(value="从左侧导入文件夹或选择照片，也可直接把照片拖进窗口。")
        tk.Label(
            checkpoint,
            textvariable=self.rule_heading,
            background="#202326",
            foreground="#fff8eb",
            font=("Helvetica Neue", 14, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            checkpoint,
            textvariable=self.rule_summary,
            background="#202326",
            foreground="#b6b7b5",
            font=("Helvetica Neue", 10),
        ).grid(row=0, column=1, sticky="w", padx=(22, 18))
        actions = tk.Frame(checkpoint, background="#202326")
        actions.grid(row=0, column=2, sticky="e")
        self.picker_settings_button = ttk.Button(actions, text="设置挑图规则", style="Primary.TButton", command=self.show_picker_settings)
        self.picker_settings_button.pack(side="left", padx=(0, 10))
        self.analyze_button = ttk.Button(actions, text="开始分析", style="Primary.TButton", command=self.start_analysis, state="disabled")
        self.analyze_button.pack(side="left")
        self.pause_icon = self._control_icon("pause")
        self.resume_icon = self._control_icon("resume")
        self.stop_icon = self._control_icon("stop")
        self.pause_button = ttk.Button(actions, image=self.pause_icon, command=self.toggle_pause, state="disabled", width=2)
        self.pause_button.pack(side="left", padx=(8, 4))
        self.stop_button = ttk.Button(actions, image=self.stop_icon, command=self.stop_analysis, state="disabled", width=2)
        self.stop_button.pack(side="left")
        self.pause_tooltip = ToolTip(self.pause_button, lambda: self.t["resume"] if self.is_paused else self.t["pause"])
        self.stop_tooltip = ToolTip(self.stop_button, lambda: self.t["stop"])
        self._set_analysis_controls_visible(False)
        self.progress = ttk.Progressbar(workflow_band, style="Culling.Horizontal.TProgressbar", mode="determinate", maximum=1, value=0)
        self.progress.grid(row=2, column=0, sticky="ew", padx=18)
        ttk.Label(workflow_band, textvariable=self.progress_text, style="Muted.TLabel").grid(row=3, column=0, sticky="w", padx=18, pady=(6, 12))
        ttk.Label(workflow_band, textvariable=self.eta_text, style="Muted.TLabel").grid(row=3, column=0, sticky="e", padx=18, pady=(6, 12))
        self._update_workflow_steps()
        self._refresh_picker_confirmation_ui()

        workspace = ttk.Frame(body, style="Paper.TFrame")
        workspace.grid(row=2, column=0, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.columnconfigure(1, weight=0)
        workspace.rowconfigure(0, weight=1)
        self.grid_panel = ttk.Frame(workspace, style="Paper.TFrame", padding=(14, 13, 14, 12))
        self.grid_panel.grid(row=0, column=0, sticky="nsew")
        self.grid_panel.columnconfigure(0, weight=1)
        self.grid_panel.rowconfigure(1, weight=1)
        self._build_grid_panel()
        self.table_panel = ttk.Frame(workspace, style="Paper.TFrame", padding=(14, 13, 14, 12))
        self.table_panel.grid(row=0, column=0, sticky="nsew")
        self.table_panel.columnconfigure(0, weight=1)
        self.table_panel.rowconfigure(2, weight=1)
        self._build_table_panel()
        self.table_panel.grid_remove()
        self._build_calibration_workspace(workspace)
        self.calibration_panel.grid_remove()
        self._build_decision_panel(workspace)
        self._install_mousewheel_routing()

        footer = ttk.Frame(body, style="Workbench.TFrame")
        footer.grid(row=3, column=0, sticky="ew", pady=(9, 0))
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(side="left")
        ttk.Label(footer, text="本地处理 · 确认后才写入星级元数据", style="Muted.TLabel").pack(side="right")

    def _build_brand_lockup(self, sidebar: ttk.Frame, colors: dict[str, str]) -> None:
        """Render the compact 秒选 mark and its two-line product name."""
        brand = tk.Frame(sidebar, background=colors["sidebar"], highlightthickness=0)
        brand.pack(fill="x", padx=22, pady=(27, 28))

        try:
            with Image.open(app_resource_path(BRAND_LOGO_RESOURCE)) as source:
                image = source.convert("RGBA")
                image.thumbnail((46, 46), Image.Resampling.LANCZOS)
            self.brand_logo = ImageTk.PhotoImage(image)
            tk.Label(
                brand,
                image=self.brand_logo,
                background=colors["sidebar"],
                borderwidth=0,
                highlightthickness=0,
            ).pack(side="left")
        except (FileNotFoundError, OSError):
            tk.Label(
                brand,
                text="✓",
                background="#262a28",
                foreground="#d5ff35",
                font=("Helvetica Neue", 21, "bold"),
                width=2,
            ).pack(side="left")

        copy = tk.Frame(brand, background=colors["sidebar"], highlightthickness=0)
        copy.pack(side="left", padx=(9, 0), fill="y")
        tk.Label(
            copy,
            text="秒选",
            background=colors["sidebar"],
            foreground="#fffaf0",
            font=("Songti SC", 19, "bold"),
        ).pack(anchor="w")
        tk.Label(
            copy,
            text="照片智能筛选器",
            background=colors["sidebar"],
            foreground="#b6b1a9",
            font=("PingFang SC", 9, "bold"),
        ).pack(anchor="w", pady=(0, 1))

    def _build_grid_panel(self) -> None:
        toolbar = ttk.Frame(self.grid_panel, style="Paper.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.grid_toolbar = toolbar
        self._grid_toolbar_reflow_after: str | None = None
        self._grid_toolbar_compact = False
        toolbar.columnconfigure(8, weight=1)
        ttk.Label(toolbar, text="全部", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.grid_count_label = ttk.Label(toolbar, text="0 张", style="Toolbar.TLabel")
        self.grid_count_label.grid(row=0, column=1, sticky="w", padx=(9, 22))
        ttk.Label(toolbar, text="显示照片", style="Toolbar.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.rating_filter_menu = ttk.Combobox(toolbar, textvariable=self.rating_filter, state="readonly", width=11)
        self.rating_filter_menu.grid(row=0, column=3, sticky="w")
        self.rating_filter_menu.bind("<<ComboboxSelected>>", self.filter_results)
        self.rating_filter_tooltip = ToolTip(
            self.rating_filter_menu,
            lambda: "1★ 为初选复核：已进入保留池，但建议人工确认；筛选不会改变推荐星级。",
        )
        ttk.Label(toolbar, text="备用评级", style="Toolbar.TLabel").grid(row=0, column=4, sticky="w", padx=(13, 6))
        self.backup_star_menu = ttk.Combobox(toolbar, textvariable=self.backup_star, values=("2★", "3★", "4★"), state="readonly", width=4)
        self.backup_star_menu.grid(row=0, column=5, sticky="w")
        self.backup_star_menu.bind("<<ComboboxSelected>>", self._on_star_mapping_changed)
        self.backup_star_tooltip = ToolTip(
            self.backup_star_menu,
            lambda: "把内部“优质备用”层级写入 Lightroom 时使用的星级。",
        )
        ttk.Label(toolbar, text="精选评级", style="Toolbar.TLabel").grid(row=0, column=6, sticky="w", padx=(9, 6))
        self.featured_star_menu = ttk.Combobox(toolbar, textvariable=self.featured_star, values=("3★", "4★", "5★"), state="readonly", width=4)
        self.featured_star_menu.grid(row=0, column=7, sticky="w")
        self.featured_star_menu.bind("<<ComboboxSelected>>", self._on_star_mapping_changed)
        self.featured_star_tooltip = ToolTip(
            self.featured_star_menu,
            lambda: "把内部“精选推荐”层级写入 Lightroom 时使用的星级。",
        )
        ttk.Label(toolbar, text="排序：拍摄时间（新→旧）", style="Toolbar.TLabel").grid(row=0, column=8, sticky="w", padx=(17, 0))
        view_controls = tk.Frame(toolbar, bg=self._workbench_colors["paper"])
        self.grid_view_controls = view_controls
        view_controls.grid(row=0, column=9, sticky="e", padx=(16, 0))
        tk.Label(
            view_controls,
            textvariable=self.thumbnail_scale_label,
            bg=self._workbench_colors["paper"],
            fg=self._workbench_colors["muted"],
            font=("Helvetica Neue", 10),
        ).pack(side="left", padx=(0, 5))
        self.thumbnail_scale = tk.Canvas(
            view_controls,
            width=104,
            height=25,
            bg=self._workbench_colors["paper"],
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.thumbnail_scale.pack(side="left")
        self.thumbnail_scale.bind("<Button-1>", self._set_thumbnail_scale_from_pointer)
        self.thumbnail_scale.bind("<B1-Motion>", self._set_thumbnail_scale_from_pointer)
        self.thumbnail_scale.bind("<ButtonRelease-1>", self._finish_thumbnail_scale_drag)
        self.thumbnail_scale.bind("<Configure>", lambda _event: self._draw_thumbnail_scale())
        self.thumbnail_scale.after_idle(self._draw_thumbnail_scale)
        self.grid_view_button = ttk.Button(
            view_controls,
            text="网格",
            image=self.sidebar_icons["cull"],
            compound="left",
            style="ViewModeActive.TButton",
            command=lambda: self._set_photo_view("grid"),
        )
        self.grid_view_button.pack(side="left", padx=(12, 3))
        self.list_view_button = ttk.Button(
            view_controls,
            text="列表",
            image=self.sidebar_icons["list"],
            compound="left",
            style="ViewMode.TButton",
            command=lambda: self._set_photo_view("list"),
        )
        self.list_view_button.pack(side="left")
        self.grid_hint = ttk.Label(toolbar, text="", style="Toolbar.TLabel")
        self.grid_hint.grid(row=1, column=0, columnspan=7, sticky="w", pady=(8, 0))
        self.import_badge = ttk.Label(toolbar, textvariable=self.import_notice, style="Toolbar.TLabel")
        self.import_badge.grid(row=1, column=7, sticky="e", pady=(8, 0))
        self.import_badge.grid_remove()
        toolbar.bind("<Configure>", self._schedule_grid_toolbar_reflow, add="+")
        grid_shell = ttk.Frame(self.grid_panel, style="Paper.TFrame")
        grid_shell.grid(row=1, column=0, sticky="nsew")
        grid_shell.columnconfigure(0, weight=1)
        grid_shell.rowconfigure(0, weight=1)
        self.photo_grid_canvas = tk.Canvas(grid_shell, background=self._workbench_colors["paper"], highlightthickness=0, borderwidth=0)
        self.photo_grid_canvas.grid(row=0, column=0, sticky="nsew")
        self.photo_grid_scroll = ttk.Scrollbar(grid_shell, orient="vertical", command=self.photo_grid_canvas.yview)
        self.photo_grid_scroll.grid(row=0, column=1, sticky="ns")
        self.photo_grid_canvas.configure(yscrollcommand=self._on_photo_grid_scroll)
        self.photo_grid_inner = tk.Frame(self.photo_grid_canvas, background=self._workbench_colors["paper"])
        self._photo_grid_window = self.photo_grid_canvas.create_window((0, 0), window=self.photo_grid_inner, anchor="nw")
        self.photo_grid_inner.bind("<Configure>", lambda _event: self.photo_grid_canvas.configure(scrollregion=self.photo_grid_canvas.bbox("all")))
        self.photo_grid_canvas.bind("<Configure>", self._on_grid_canvas_resize)

    @staticmethod
    def _grid_toolbar_is_compact(width: int) -> bool:
        """Wrap view controls before a narrow workbench would clip them."""
        return width < 960

    def _thumbnail_scale_text(self, percent: int) -> str:
        if getattr(self, "_grid_toolbar_compact", False):
            return f"{percent}%"
        return f"缩略图 {percent}%"

    def _schedule_grid_toolbar_reflow(self, event) -> None:
        self._grid_toolbar_width = max(0, int(event.width))
        if self._grid_toolbar_reflow_after is None:
            self._grid_toolbar_reflow_after = self.after_idle(self._reflow_grid_toolbar)

    def _reflow_grid_toolbar(self) -> None:
        """Keep size and view controls inside a reduced review workspace."""
        self._grid_toolbar_reflow_after = None
        if not hasattr(self, "grid_view_controls"):
            return
        compact = self._grid_toolbar_is_compact(getattr(self, "_grid_toolbar_width", 0))
        if compact:
            self.grid_view_controls.grid_configure(
                row=1, column=0, columnspan=10, sticky="w", padx=(0, 0), pady=(8, 0)
            )
            self.grid_hint.grid_configure(row=2, column=0, columnspan=7, pady=(8, 0))
            self.import_badge.grid_configure(row=2, column=7, columnspan=3, pady=(8, 0))
            self.thumbnail_scale.configure(width=78)
        else:
            self.grid_view_controls.grid_configure(
                row=0, column=9, columnspan=1, sticky="e", padx=(16, 0), pady=0
            )
            self.grid_hint.grid_configure(row=1, column=0, columnspan=7, pady=(8, 0))
            self.import_badge.grid_configure(row=1, column=7, columnspan=1, pady=(8, 0))
            self.thumbnail_scale.configure(width=104)
        self._grid_toolbar_compact = compact
        self.thumbnail_scale_label.set(
            self._thumbnail_scale_text(self.thumbnail_scale_percent.get())
        )
        self._draw_thumbnail_scale()

    def _build_table_panel(self) -> None:
        header = ttk.Frame(self.table_panel, style="Paper.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="写入与导出", style="PanelTitle.TLabel").pack(side="left")
        ttk.Button(header, text="← 返回照片筛选", style="Outline.TButton", command=self.show_culling_workspace).pack(side="right")
        ttk.Label(self.table_panel, textvariable=self.stats_text, style="Stats.TLabel").grid(row=1, column=0, sticky="w", pady=(7, 9))
        columns = ("group", "star", "file", "reason")
        self.table = ttk.Treeview(self.table_panel, columns=columns, show="headings", selectmode="browse")
        for name, title, width, minimum, stretch in (
            ("group", "组", 50, 42, False), ("star", "评级", 65, 56, False),
            ("file", "原片", 280, 160, True), ("reason", "分析原因", 520, 240, True),
        ):
            self.table.heading(name, text=title)
            self.table.column(name, width=width, minwidth=minimum, stretch=stretch, anchor="w")
        self.table.grid(row=2, column=0, sticky="nsew")
        self.table.bind("<<TreeviewSelect>>", self.show_selected)
        scroll = ttk.Scrollbar(self.table_panel, orient="vertical", command=self.table.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)

    def _build_calibration_workspace(self, workspace: ttk.Frame) -> None:
        """A dedicated local-learning page, separate from a frame's XMP work."""
        self.calibration_panel = ttk.Frame(workspace, style="Paper.TFrame", padding=(34, 30))
        self.calibration_panel.grid(row=0, column=0, sticky="nsew")
        self.calibration_panel.columnconfigure(0, weight=1)
        self.calibration_panel.columnconfigure(1, weight=1)
        ttk.Label(self.calibration_panel, text="选片偏好校准", style="TaskTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            self.calibration_panel,
            text="让软件逐步学会你在不同拍摄题材中，真正留下哪一张。所有比对与训练均在本机完成。",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(7, 24))
        loop = ttk.Frame(self.calibration_panel, style="Soft.TFrame", padding=(20, 18))
        loop.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(loop, text="本地反馈闭环", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(loop, text="每一次交付后的最终选片，都能成为下一次更贴近你审美的依据。", style="PanelMuted.TLabel").pack(anchor="w", pady=(4, 15))
        steps = (
            ("1", "保存推荐", "每个任务自动保存软件的建议、分组与评分。"),
            ("2", "选择最终交付", "只需选择 Lightroom 最终导出的照片文件夹。"),
            ("3", "自动对比", "按文件名识别选对、漏选与误选，优先学习同组连拍。"),
            ("4", "安全更新", "新偏好优于当前模型才启用；旧模型始终可一键回退。"),
        )
        loop_grid = ttk.Frame(loop, style="Soft.TFrame")
        loop_grid.pack(fill="x")
        for index, (number, title, copy) in enumerate(steps):
            loop_grid.columnconfigure(index, weight=1, uniform="learning")
            item = ttk.Frame(loop_grid, style="Soft.TFrame")
            item.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 12, 0))
            ttk.Label(item, text=number, style="SyncIndex.TLabel", width=2, anchor="center").pack(anchor="w")
            ttk.Label(item, text=title, style="Paper.TLabel").pack(anchor="w", pady=(8, 3))
            ttk.Label(item, text=copy, style="PanelMuted.TLabel", justify="left", wraplength=180).pack(anchor="w", fill="x")
        evidence = ttk.Frame(self.calibration_panel, style="Paper.TFrame", padding=(0, 24, 12, 0))
        evidence.grid(row=3, column=0, sticky="nsew")
        ttk.Label(evidence, text="下一次更新选片偏好校准", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(evidence, text="选择 Lightroom 最终交付文件夹后，软件会自动关联当前任务的本地筛选日志。", style="PanelMuted.TLabel", justify="left", wraplength=390).pack(anchor="w", pady=(7, 14))
        task_ready = "本次任务筛选日志：已就绪" if self.current_log_path else "本次任务筛选日志：完成一次分析后自动生成"
        self.calibration_task_status = ttk.Label(evidence, text=task_ready, style="Stats.TLabel")
        self.calibration_task_status.pack(anchor="w", pady=(0, 12))
        self.calibration_button = ttk.Button(evidence, text="选择最终交付文件夹并更新", style="Primary.TButton", command=self.import_calibration)
        self.calibration_button.pack(anchor="w")
        ttk.Label(evidence, text="不会上传、移动、删除或改写原片及 Lightroom 星级。", style="PanelMuted.TLabel").pack(anchor="w", pady=(10, 0))
        insight = ttk.Frame(self.calibration_panel, style="Soft.TFrame", padding=(20, 18))
        insight.grid(row=3, column=1, sticky="nsew", padx=(12, 0), pady=(24, 0))
        ttk.Label(insight, text="校准报告", style="PanelTitle.TLabel").pack(anchor="w")
        self.calibration_report = ttk.Label(insight, text=self._calibration_report_text(), style="PanelMuted.TLabel", justify="left", wraplength=340)
        self.calibration_report.pack(anchor="w", fill="x", pady=(9, 16))
        ttk.Label(insight, text="推荐的学习方式", style="Paper.TLabel").pack(anchor="w")
        ttk.Label(insight, text="• 按婚礼、写真、活动分开校准\n• 重点学习同组连拍最终留片\n• 只复核模型最没有把握的少量照片", style="PanelMuted.TLabel", justify="left", wraplength=340).pack(anchor="w", pady=(7, 0))
        self._refresh_calibration_button()

    def _calibration_report_text(self) -> str:
        if self.calibration_profile is None:
            return "尚未建立个人偏好。完成一轮分析并导入最终交付照片后，即可生成第一份本地校准报告。"
        return (
            f"已学习 {self.calibration_profile.training_examples:,} 个本地样本\n"
            f"验证准确度 {self.calibration_profile.validation_auc:.1%}\n"
            "当前偏好已启用；下次更新仅在验证结果更好时替换。"
        )

    def _build_decision_panel(self, workspace: ttk.Frame) -> None:
        shell = ttk.Frame(workspace, style="Paper.TFrame", width=360)
        shell.grid(row=0, column=1, sticky="nsew")
        shell.grid_propagate(False)
        shell.rowconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)
        # The decision rail may be shorter than its controls on compact
        # displays.  Keeping it independently scrollable means that write and
        # restore actions remain reachable instead of being clipped below the
        # window edge.
        self.decision_canvas = tk.Canvas(
            shell,
            bg=self._workbench_colors["paper"],
            bd=0,
            highlightthickness=0,
            yscrollincrement=12,
        )
        self.decision_canvas.grid(row=0, column=0, sticky="nsew")
        self.decision_scroll = ttk.Scrollbar(shell, orient="vertical", command=self.decision_canvas.yview)
        self.decision_scroll.grid(row=0, column=1, sticky="ns")
        self.decision_canvas.configure(yscrollcommand=self.decision_scroll.set)
        right = ttk.Frame(shell, style="Paper.TFrame", padding=(15, 10, 15, 12))
        self._decision_window = self.decision_canvas.create_window((0, 0), window=right, anchor="nw")

        def update_decision_scroll_region(_event=None) -> None:
            self.decision_canvas.configure(scrollregion=self.decision_canvas.bbox("all"))

        def fit_decision_content(event) -> None:
            self.decision_canvas.itemconfigure(self._decision_window, width=event.width)
            update_decision_scroll_region()

        right.bind("<Configure>", update_decision_scroll_region)
        self.decision_canvas.bind("<Configure>", fit_decision_content)
        ttk.Label(right, text="本张决策", style="PanelTitle.TLabel").pack(anchor="w")
        self.preview_label = ttk.Label(right, text=self.t["empty_preview"], style="DropHint.TLabel", justify="center", anchor="center")
        # The image already defines the preview's height.  Extra internal
        # padding used to add a large black band above and below every photo
        # and pushed the rating/write controls below the visible rail.
        self.preview_label.pack(fill="x", pady=(7, 6), ipady=4)
        self.preview_label.configure(cursor="hand2")
        self.preview_label.bind("<Button-1>", self._open_preview_zoom, add="+")
        self.preview_zoom_tooltip = ToolTip(
            self.preview_label,
            lambda: "点击放大预览；在放大窗口滚轮缩放，双击可适应窗口。",
        )
        self.detail = ttk.Label(right, text=self.t["empty_detail"], style="Decision.TLabel", justify="left", wraplength=280)
        self.detail.pack(fill="x", pady=(0, 8), ipadx=10, ipady=7)
        ttk.Label(right, text="手动评级", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 6))
        stars = ttk.Frame(right, style="Paper.TFrame")
        stars.pack(fill="x")
        for column in range(4):
            stars.columnconfigure(column, weight=1, uniform="rating")
        self.manual_rating_buttons: dict[int, ttk.Button] = {}
        for rating in (0, 1, 2, 3):
            button = ttk.Button(stars, text=self.t["clear"] if rating == 0 else f"{self._output_rating(rating)}★", command=lambda value=rating: self.set_manual_rating(value))
            button.grid(row=0, column=rating, sticky="ew", padx=(0 if rating == 0 else 4, 0))
            self.manual_rating_buttons[rating] = button
        ttk.Separator(right).pack(fill="x", pady=10)
        self.lr_guard_slot = ttk.Frame(right, style="Paper.TFrame")
        self.lr_guard_slot.pack(fill="x")
        self.lr_guard_label = ttk.Label(self.lr_guard_slot, text="", style="GuardWarning.TLabel", justify="left", wraplength=275)
        ttk.Label(right, text=self.t["write_mode"], style="PanelMuted.TLabel").pack(anchor="w", pady=(0, 5))
        self.write_mode_menu = ttk.Combobox(right, textvariable=self.write_mode, values=tuple(self.write_modes), state="readonly", width=25)
        self.write_mode_menu.pack(fill="x", pady=(0, 8))
        self.write_button = ttk.Button(right, text=self.t["write"], style="Primary.TButton", command=self.confirm_write, state="disabled")
        self.write_button.pack(fill="x")
        self.repair_legacy_heic_button = ttk.Button(
            right,
            text="修复旧版照片星级",
            style="Outline.TButton",
            command=self.repair_legacy_heic_ratings,
        )
        self.repair_legacy_tooltip = ToolTip(
            self.repair_legacy_heic_button,
            lambda: "待处理数字表示照片张数，不是星级。仅将旧版写在旁车的星级迁移到照片文件内。",
        )
        self.restore_button = ttk.Button(right, text=self.t.get("restore", "还原本次星级"), command=self.confirm_restore, state="disabled")
        self.restore_button.pack(fill="x", pady=(6, 0))
        self.export_selected_button = ttk.Button(right, text=self.t["export_selected"], command=self.export_selected_photos, state="disabled")
        self.export_selected_button.pack(fill="x", pady=(6, 0))
        self.xmp_card = ttk.Frame(right, style="Paper.TFrame")
        self.xmp_card.pack(fill="x", pady=(10, 0))
        self.xmp_result_label = ttk.Label(self.xmp_card, textvariable=self.xmp_result_notice, style="XmpResult.TLabel", justify="left", wraplength=275)
        self.sync_heading = ttk.Label(self.xmp_card, text="Lightroom 星级同步", style="SyncTitle.TLabel")
        self.sync_heading.pack(anchor="w", pady=(0, 7))
        ttk.Label(self.xmp_card, text=self.t["sync_summary"], style="PanelMuted.TLabel", justify="left", wraplength=275).pack(anchor="w", fill="x", pady=(0, 8))
        self.sync_toggle_button = ttk.Button(self.xmp_card, text=self.t["sync_show"], command=self.toggle_lightroom_sync_instructions)
        self.sync_toggle_button.pack(fill="x")
        self.sync_steps_frame = ttk.Frame(self.xmp_card, style="Paper.TFrame")
        ttk.Label(self.sync_steps_frame, text="前置操作：", style="PanelMuted.TLabel").pack(anchor="w", pady=(9, 7))
        sync_steps = (
            "Lightroom 顶部菜单「元数据」→ 取消勾选「仅显示目标照片的元数据」",
            "右侧「元数据」面板改为「选定的照片」",
            "关闭 Lightroom，开始使用本软件分析照片",
            "分析完毕后，确认写入 XMP 星级信息",
            "打开 Lightroom，在「图库」→「网格视图」（快捷键 G）全选本次分析的照片",
            "右键 →「元数据」→「从文件中读取元数据」，即可同步完成。",
        )
        for number, step in enumerate(sync_steps, start=1):
            row = ttk.Frame(self.sync_steps_frame, style="Paper.TFrame")
            row.pack(fill="x", pady=(0, 5))
            ttk.Label(row, text=str(number), style="SyncIndex.TLabel", width=2, anchor="center").pack(side="left")
            ttk.Label(row, text=step, style="SyncStep.TLabel", justify="left", wraplength=235).pack(side="left", fill="x", expand=True, padx=(8, 0))
        self._refresh_rating_presentation()
        self._refresh_legacy_heic_repair_button()

    def set_workflow_stage(self, stage: int) -> None:
        self.workflow_stage = max(1, min(5, stage))
        self._update_workflow_steps()

    def _draw_workflow_track(self, canvas: tk.Canvas) -> None:
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_line(0, height // 2, width, height // 2, fill="#383b3e", width=2)
        index = self.workflow_tracks.index(canvas) + 1 if canvas in self.workflow_tracks else 0
        if self.workflow_stage > index:
            canvas.create_line(0, height // 2, width, height // 2, fill="#c98b37", width=2)
        elif self.workflow_stage == index and self.analysis_running:
            pulse = 0.45 + (self._workflow_pulse / 10) * 0.4
            canvas.create_line(0, height // 2, int(width * pulse), height // 2, fill="#dca256", width=2)

    def _update_workflow_steps(self) -> None:
        if not hasattr(self, "workflow_bubbles"):
            return
        for index, (bubble, label) in enumerate(zip(self.workflow_bubbles, self.workflow_labels), start=1):
            if index < self.workflow_stage:
                bubble.configure(text="✓", bg="#5f705f", fg="#eef5ed")
                label.configure(fg="#d9ddd8")
            elif index == self.workflow_stage:
                bubble.configure(text=str(index), bg="#c98b37", fg="#17120c")
                label.configure(fg="#fff7e8")
            else:
                bubble.configure(text=str(index), bg="#303336", fg="#aeb0ae")
                label.configure(fg="#858886")
        for track in self.workflow_tracks:
            self._draw_workflow_track(track)
        if self.analysis_running and not self._workflow_animation_running:
            self._workflow_animation_running = True
            self.after(380, self._animate_workflow)

    def _animate_workflow(self) -> None:
        if not self.analysis_running:
            self._workflow_animation_running = False
            self._update_workflow_steps()
            return
        self._workflow_pulse = (self._workflow_pulse + 1) % 11
        for track in self.workflow_tracks:
            self._draw_workflow_track(track)
        self.after(380, self._animate_workflow)

    def _set_active_nav(self, active_key: str) -> None:
        """Synchronize the sidebar's selected row and its gold edge marker."""
        entries = {
            "cull": (self.cull_nav_button, "cull"),
            "export": (self.export_nav_button, "export"),
            "calibration": (self.calibration_nav_button, "calibration"),
        }
        for key, (button, icon) in entries.items():
            selected = key == active_key
            button.configure(
                style="NavActive.TButton" if selected else "Nav.TButton",
                image=self.sidebar_icons[f"{icon}_active" if selected else icon],
            )
            indicator = self.nav_indicators.get(key)
            if indicator is not None:
                indicator.configure(
                    bg=self._workbench_colors["copper"]
                    if selected
                    else self._workbench_colors["sidebar"]
                )

    def show_culling_workspace(self) -> None:
        """Return to the visual review wall without discarding table state."""
        self.table_panel.grid_remove()
        self.calibration_panel.grid_remove()
        self.grid_panel.grid()
        self._set_active_nav("cull")
        self._schedule_grid_render()

    def show_export_workspace(self) -> None:
        """Show the complete, keyboard-friendly table and XMP tools."""
        self.grid_panel.grid_remove()
        self.calibration_panel.grid_remove()
        self.table_panel.grid()
        self._set_active_nav("export")

    def show_calibration_workspace(self) -> None:
        self.grid_panel.grid_remove()
        self.table_panel.grid_remove()
        self.calibration_panel.grid()
        self._set_active_nav("calibration")
        if hasattr(self, "calibration_task_status"):
            task_ready = "本次任务筛选日志：已就绪" if self.current_log_path else "本次任务筛选日志：完成一次分析后自动生成"
            self.calibration_task_status.configure(text=task_ready)
            self.calibration_report.configure(text=self._calibration_report_text())

    @staticmethod
    def _pointer_is_over_widget(event: tk.Event, widget: tk.Widget) -> bool:
        """Check the pointer in screen coordinates so card children also scroll."""
        try:
            left = widget.winfo_rootx()
            top = widget.winfo_rooty()
            return (
                left <= event.x_root < left + widget.winfo_width()
                and top <= event.y_root < top + widget.winfo_height()
            )
        except tk.TclError:
            return False

    def _route_mousewheel(self, event: tk.Event) -> str | None:
        """Let the pointer's active review surface own wheel and trackpad input."""
        delta = int(getattr(event, "delta", 0) or 0)
        if delta == 0 and getattr(event, "num", None) in {4, 5}:
            delta = 1 if event.num == 4 else -1
        units = self._wheel_units(delta)
        if units == 0:
            return None
        if hasattr(self, "photo_grid_canvas") and self._pointer_is_over_widget(event, self.photo_grid_canvas):
            self.photo_grid_canvas.yview_scroll(units, "units")
            return "break"
        if hasattr(self, "decision_canvas") and self._pointer_is_over_widget(event, self.decision_canvas):
            self.decision_canvas.yview_scroll(units, "units")
            return "break"
        return None

    def _install_mousewheel_routing(self) -> None:
        """Install one root-level route so wheel input works over every grid card."""
        if self._mousewheel_bound:
            return
        self._mousewheel_bound = True
        self.bind_all("<MouseWheel>", self._route_mousewheel, add="+")
        self.bind_all("<Button-4>", lambda event: self._route_mousewheel(event), add="+")
        self.bind_all("<Button-5>", lambda event: self._route_mousewheel(event), add="+")

    def _on_grid_canvas_resize(self, event) -> None:
        self.photo_grid_canvas.itemconfigure(self._photo_grid_window, width=event.width)
        self._schedule_grid_render()

    def _on_photo_grid_scroll(self, first: str, last: str) -> None:
        """Page in the next review batch shortly before the visible end."""
        self.photo_grid_scroll.set(first, last)
        if self._virtual_grid_layout is not None:
            self._schedule_virtual_grid_render()
            return
        try:
            nearing_end = float(last) >= 0.78
        except (TypeError, ValueError):
            return
        if (
            nearing_end
            and self.grid_visible_limit < len(self._grid_photo_entries())
            and not self._grid_auto_load_scheduled
        ):
            self._grid_auto_load_scheduled = True
            self.after_idle(self._auto_load_more_grid_photos)

    def _schedule_virtual_grid_render(self) -> None:
        """Coalesce high-frequency scrollbar callbacks into one light redraw."""
        if self._virtual_grid_render_after is None:
            self._virtual_grid_render_after = self.after(16, self._render_virtual_grid_window)

    def _render_virtual_grid_window(self) -> None:
        self._virtual_grid_render_after = None
        if self._virtual_grid_layout is None:
            return
        columns, card_width, thumbnail_height, row_height = self._virtual_grid_layout
        entries = self._virtual_grid_entries
        if not entries:
            return
        total_rows = math.ceil(len(entries) / columns)
        top = self.photo_grid_canvas.canvasy(0)
        viewport_height = max(1, self.photo_grid_canvas.winfo_height())
        first_row = max(0, int(top // row_height) - 2)
        last_row = min(total_rows, math.ceil((top + viewport_height) / row_height) + 2)
        desired_range = (first_row, last_row)
        if desired_range == self._virtual_grid_range:
            return
        desired_rows = set(range(first_row, last_row))
        for row, cards in list(self._virtual_grid_rows.items()):
            if row not in desired_rows:
                for card in cards:
                    card.destroy()
                del self._virtual_grid_rows[row]
        self._virtual_grid_range = desired_range
        for row in range(first_row, last_row):
            if row in self._virtual_grid_rows:
                continue
            start = row * columns
            end = min(len(entries), start + columns)
            self._virtual_grid_rows[row] = self._append_grid_cards(
                entries[start:end],
                start=start,
                columns=columns,
                card_width=card_width,
                thumbnail_height=thumbnail_height,
                virtual_row_height=row_height,
            )
        self._grid_photo_refs = list(self._grid_thumbnail_cache.values())

    def _auto_load_more_grid_photos(self) -> None:
        self._grid_auto_load_scheduled = False
        entries = self._grid_photo_entries()
        total = len(entries)
        previous_limit = min(self.grid_visible_limit, total)
        if previous_limit >= total:
            return
        self.grid_visible_limit = min(total, previous_limit + 24)
        layout = getattr(self, "_grid_layout", None)
        if self.photo_view_mode.get() == "grid" and layout is not None:
            columns, card_width, thumbnail_height = layout
            self._append_grid_cards(
                entries[previous_limit : self.grid_visible_limit],
                start=previous_limit,
                columns=columns,
                card_width=card_width,
                thumbnail_height=thumbnail_height,
            )
            self.photo_grid_canvas.configure(scrollregion=self.photo_grid_canvas.bbox("all"))
            return
        self._render_photo_grid()

    def _on_thumbnail_scale_changed(self, value: str) -> None:
        """Keep dragging responsive; reflow only after the control settles."""
        percent = max(75, min(150, int(round(float(value) / 5) * 5)))
        if percent == self.thumbnail_scale_percent.get():
            return
        self.thumbnail_scale_percent.set(percent)
        self.thumbnail_scale_label.set(self._thumbnail_scale_text(percent))
        if hasattr(self, "thumbnail_scale"):
            self._draw_thumbnail_scale()
        self._schedule_thumbnail_reflow()

    def _schedule_thumbnail_reflow(self) -> None:
        """Debounce expensive Tk image/card rebuilding while the slider moves."""
        if self._grid_render_scheduled and self._grid_render_after is not None:
            self.after_cancel(self._grid_render_after)
            self._grid_render_scheduled = False
            self._grid_render_after = None
        if self._thumbnail_reflow_after is not None:
            self.after_cancel(self._thumbnail_reflow_after)
        self._thumbnail_reflow_after = self.after(180, self._apply_thumbnail_reflow)

    def _apply_thumbnail_reflow(self) -> None:
        self._thumbnail_reflow_after = None
        self._schedule_grid_render()

    def _finish_thumbnail_scale_drag(self, _event=None) -> str:
        """Commit the final size immediately when the user releases the knob."""
        if self._thumbnail_reflow_after is not None:
            self.after_cancel(self._thumbnail_reflow_after)
            self._thumbnail_reflow_after = None
        self._schedule_grid_render()
        return "break"

    def _draw_thumbnail_scale(self) -> None:
        """Draw a high-contrast slider; macOS Tk's native Scale is too faint here."""
        control = self.thumbnail_scale
        width = max(104, control.winfo_width())
        center_y = 13
        left, right = 10, width - 10
        fraction = (self.thumbnail_scale_percent.get() - 75) / 75
        knob_x = left + (right - left) * fraction
        control.delete("all")
        control.create_line(left, center_y, right, center_y, fill="#44484b", width=4, capstyle="round")
        control.create_line(left, center_y, knob_x, center_y, fill=self._workbench_colors["copper"], width=4, capstyle="round")
        control.create_oval(knob_x - 7, center_y - 7, knob_x + 7, center_y + 7, fill="#f6e4c1", outline="#151719", width=2)

    def _set_thumbnail_scale_from_pointer(self, event: tk.Event) -> str:
        control = self.thumbnail_scale
        left, right = 10, max(104, control.winfo_width()) - 10
        fraction = min(1.0, max(0.0, (event.x - left) / max(1, right - left)))
        self._on_thumbnail_scale_changed(str(75 + fraction * 75))
        return "break"

    def _set_photo_view(self, mode: str) -> None:
        """Switch between the review wall and a compact filename-led list."""
        if mode not in {"grid", "list"}:
            return
        self.photo_view_mode.set(mode)
        self.grid_view_button.configure(style="ViewModeActive.TButton" if mode == "grid" else "ViewMode.TButton")
        self.list_view_button.configure(style="ViewModeActive.TButton" if mode == "list" else "ViewMode.TButton")
        self._schedule_grid_render()

    def _schedule_grid_render(self) -> None:
        if self._grid_render_scheduled:
            return
        self._grid_render_scheduled = True
        self._grid_render_after = self.after(80, self._render_photo_grid)

    def _grid_photo_entries(self) -> list[tuple[int | None, Path, list[Path], str, int, list[str]]]:
        """Return logical captures for the picture wall, never decoding RAW here."""
        selected_rating = self.rating_filters.get(self.rating_filter.get())
        if self.results:
            return [
                (index, item.path, item.pair_paths, item.display_name, item.suggested_rating, item.reasons)
                for index, item in enumerate(self.results)
                if self._matches_rating_filter(item.suggested_rating, selected_rating)
            ]
        return [
            (None, primary, companions, primary.name, 0, ["已导入，等待开始分析"])
            for primary, companions in self._logical_import_rows(self.imported_files)
        ]

    @staticmethod
    def _grid_row_height(thumbnail_height: int, *, analyzed: bool) -> int:
        """Keep every review row on one consistent visual rhythm."""
        return thumbnail_height + (216 if analyzed else 70)

    def _render_photo_grid(self) -> None:
        self._grid_render_scheduled = False
        self._grid_render_after = None
        self._virtual_grid_layout = None
        self._virtual_grid_entries = []
        self._virtual_grid_range = None
        self._virtual_grid_rows = {}
        if not hasattr(self, "photo_grid_inner"):
            return
        entries = self._grid_photo_entries()
        self.photo_grid_canvas.delete("empty_hint")
        for child in self.photo_grid_inner.winfo_children():
            child.destroy()
        for column in range(12):
            self.photo_grid_inner.grid_columnconfigure(column, weight=0)
        self._grid_photo_refs.clear()
        if hasattr(self, "grid_count_label"):
            self.grid_count_label.configure(text=f"{len(entries):,} 张")
        if not entries:
            self.photo_grid_canvas.itemconfigure(self._photo_grid_window, height=0)
            empty = tk.Label(
                self.photo_grid_canvas,
                text="请在任意位置拖入文件或文件夹",
                bg=self._workbench_colors["paper"], fg=self._workbench_colors["muted"],
                font=("Helvetica Neue", 15, "bold"), justify="center",
            )
            self.photo_grid_canvas.create_window(
                max(1, self.photo_grid_canvas.winfo_width() // 2),
                max(1, self.photo_grid_canvas.winfo_height() // 2),
                window=empty, tags="empty_hint",
            )
            return
        if self.photo_view_mode.get() == "list":
            self.photo_grid_canvas.itemconfigure(self._photo_grid_window, height=0)
            self._render_photo_list(entries)
            self.photo_grid_canvas.configure(scrollregion=self.photo_grid_canvas.bbox("all"))
            return
        available = max(self.photo_grid_canvas.winfo_width(), 620)
        scale = self.thumbnail_scale_percent.get() / 100
        base_card_width = 218 if available >= 980 else 196 if available >= 760 else 174
        card_width = max(150, int(base_card_width * scale))
        thumbnail_height = max(100, int(144 * scale))
        columns = max(2, available // (card_width + 12))
        row_height = self._grid_row_height(thumbnail_height, analyzed=bool(self.results))
        total_height = max(
            self.photo_grid_canvas.winfo_height(),
            math.ceil(len(entries) / columns) * row_height + 10,
        )
        self._virtual_grid_entries = entries
        self._virtual_grid_layout = (columns, card_width, thumbnail_height, row_height)
        self.photo_grid_inner.configure(width=available, height=total_height)
        self.photo_grid_canvas.itemconfigure(
            self._photo_grid_window,
            width=available,
            height=total_height,
        )
        self.photo_grid_canvas.configure(scrollregion=(0, 0, available, total_height))
        self._render_virtual_grid_window()

    def _grid_thumbnail(self, source: Path, width: int, height: int) -> ImageTk.PhotoImage | None:
        """Reuse Tk image handles while cards enter and leave the viewport."""
        try:
            key = (source.resolve(), width, height)
        except OSError:
            return None
        cached = self._grid_thumbnail_cache.get(key)
        if cached is not None:
            self._grid_thumbnail_cache.move_to_end(key)
            self._grid_photo_refs.append(cached)
            return cached
        if not self._preview_is_cached(source):
            return None
        try:
            image = self._preview_image(source)
            image.thumbnail((width, height))
            thumbnail = ImageTk.PhotoImage(image)
        except Exception:
            return None
        self._grid_thumbnail_cache[key] = thumbnail
        self._grid_thumbnail_cache.move_to_end(key)
        while len(self._grid_thumbnail_cache) > GRID_THUMBNAIL_CACHE_LIMIT:
            self._grid_thumbnail_cache.popitem(last=False)
        self._grid_photo_refs.append(thumbnail)
        return thumbnail

    def _append_grid_cards(
        self,
        entries: list[tuple[int | None, Path, list[Path], str, int, list[str]]],
        *,
        start: int,
        columns: int,
        card_width: int,
        thumbnail_height: int,
        virtual_row_height: int | None = None,
    ) -> list[tk.Widget]:
        """Add only a new batch of cards; existing visible cards stay intact."""
        cards: list[tk.Widget] = []
        for position, (index, path, companions, name, rating, reasons) in enumerate(entries, start=start):
            row, column = divmod(position, columns)
            selected = index == self.active_result_index if index is not None else path == self.active_import_path
            border = self._workbench_colors["copper"] if selected else self._workbench_colors["line"]
            card = tk.Frame(self.photo_grid_inner, bg=self._workbench_colors["paper"], highlightbackground=border, highlightcolor=border, highlightthickness=2 if selected else 1, bd=0)
            cards.append(card)
            if virtual_row_height is None:
                card.grid(row=row, column=column, sticky="nsew", padx=GRID_CARD_GUTTER, pady=GRID_CARD_GUTTER)
                card_content_width = card_width - GRID_CARD_GUTTER * 2
            else:
                cell_width = max(1, self.photo_grid_canvas.winfo_width() // columns)
                card.place(
                    x=column * cell_width + GRID_CARD_GUTTER,
                    y=row * virtual_row_height + GRID_CARD_GUTTER,
                    width=max(1, cell_width - GRID_CARD_GUTTER * 2),
                    height=max(1, virtual_row_height - GRID_CARD_GUTTER * 2),
                )
                card_content_width = cell_width - GRID_CARD_GUTTER * 2
            source = self._preview_source(path, companions)
            thumbnail_slot = tk.Frame(card, bg="#2a2d30", height=thumbnail_height)
            thumbnail_slot.pack(fill="x", padx=5, pady=(5, 4))
            thumbnail_slot.pack_propagate(False)
            thumbnail = self._grid_thumbnail(source, max(100, card_content_width - 12), thumbnail_height)
            photo = (
                tk.Label(thumbnail_slot, image=thumbnail, bg="#2a2d30")
                if thumbnail is not None
                else tk.Label(thumbnail_slot, text="正在准备预览", bg="#2a2d30", fg=self._workbench_colors["muted"], font=("Helvetica Neue", 10))
            )
            photo.place(relx=0.5, rely=0.5, anchor="center")
            name_row = tk.Frame(card, bg=self._workbench_colors["paper"])
            name_row.pack(fill="x", padx=7)
            tk.Label(name_row, text=name, bg=self._workbench_colors["paper"], fg=self._workbench_colors["ink"], anchor="w", font=("Helvetica Neue", 10, "bold")).pack(side="left", fill="x", expand=True)
            if self.results and index is not None:
                item = self.results[index]
                score = int(round(min(100.0, item.score)))
                star = "—" if rating == 0 else f"{self._output_rating(rating)}★"
                tk.Label(name_row, text=star, bg=self._workbench_colors["paper"], fg=self._workbench_colors["copper"], font=("Helvetica Neue", 10, "bold")).pack(side="right")
                score_row = tk.Frame(card, bg=self._workbench_colors["paper"])
                score_row.pack(fill="x", padx=7, pady=(4, 1))
                tk.Label(score_row, text="AI 评分", bg=self._workbench_colors["paper"], fg=self._workbench_colors["muted"], font=("Helvetica Neue", 9)).pack(side="left")
                tk.Label(score_row, text=f"{score} / 100", bg=self._workbench_colors["paper"], fg=self._workbench_colors["copper"], font=("Helvetica Neue", 13, "bold")).pack(side="right")
                metric_values = (("构图与画面", score), ("表情与情感", min(100, 74 + item.smiling_face_count * 9 + item.open_eye_face_count * 3)), ("光线与色彩", max(0, score - int((item.highlight_clip or 0) * 50))), ("清晰度与技术", min(100, int((item.subject_sharpness or item.sharpness or 0) / 1.4))))
                for metric, value in metric_values:
                    metric_row = tk.Frame(card, bg=self._workbench_colors["paper"])
                    metric_row.pack(fill="x", padx=7, pady=(1, 0))
                    tk.Label(metric_row, text=metric, width=8, anchor="w", bg=self._workbench_colors["paper"], fg=self._workbench_colors["muted"], font=("Helvetica Neue", 8)).pack(side="left")
                    track = tk.Frame(metric_row, bg="#3b3f42", height=3)
                    track.pack(side="left", fill="x", expand=True, padx=(3, 4), pady=5)
                    tk.Frame(track, bg="#a98c66", height=3).place(relwidth=max(0.03, min(value, 100) / 100), relheight=1)
                    tk.Label(metric_row, text=str(value), bg=self._workbench_colors["paper"], fg=self._workbench_colors["muted"], font=("Helvetica Neue", 8)).pack(side="right")
                tk.Label(card, text=("；".join(reasons[-2:]) or self.t["best"]), bg=self._workbench_colors["paper"], fg=self._workbench_colors["muted"], justify="left", anchor="w", wraplength=max(100, card_content_width - 14), font=("Helvetica Neue", 8)).pack(fill="x", padx=7, pady=(5, 7))
            else:
                tk.Label(card, text="等待 AI 分析", bg=self._workbench_colors["paper"], fg=self._workbench_colors["muted"], anchor="w", font=("Helvetica Neue", 9)).pack(fill="x", padx=7, pady=(4, 8))
            select = (
                (lambda _event, value=index: self._select_result_index(value))
                if index is not None
                else (lambda _event, primary=path, pair_paths=companions: self._select_imported_photo(primary, pair_paths))
            )
            for widget in (card, thumbnail_slot, photo, name_row, *card.winfo_children()):
                widget.bind("<Button-1>", select, add="+")
        return cards

    def _render_photo_list(self, entries: list[tuple[int | None, Path, list[Path], str, int, list[str]]]) -> None:
        """Render the same filtered photos as compact rows for filename-first review."""
        self.photo_grid_inner.grid_columnconfigure(0, weight=1)
        shown_entries = entries[: self.grid_visible_limit]
        scale = self.thumbnail_scale_percent.get() / 100
        preview_size = (max(84, int(118 * scale)), max(60, int(80 * scale)))
        for row, (index, path, companions, name, rating, reasons) in enumerate(shown_entries):
            selected = (
                index == self.active_result_index
                if index is not None
                else path == self.active_import_path
            )
            border = self._workbench_colors["copper"] if selected else self._workbench_colors["line"]
            card = tk.Frame(
                self.photo_grid_inner,
                bg=self._workbench_colors["paper"], highlightbackground=border,
                highlightcolor=border, highlightthickness=2 if selected else 1, bd=0,
            )
            card.grid(row=row, column=0, sticky="ew", padx=5, pady=(0 if row == 0 else 5, 0))
            source = self._preview_source(path, companions)
            thumbnail = None
            if self._preview_is_cached(source):
                try:
                    image = self._preview_image(source)
                    image.thumbnail(preview_size)
                    thumbnail = ImageTk.PhotoImage(image)
                    self._grid_photo_refs.append(thumbnail)
                except Exception:
                    thumbnail = None
            if thumbnail is None:
                photo = tk.Label(card, text="正在准备预览", width=13, height=4, bg="#2a2d30", fg=self._workbench_colors["muted"], font=("Helvetica Neue", 9))
            else:
                photo = tk.Label(card, image=thumbnail, bg="#2a2d30")
            photo.pack(side="left", padx=7, pady=7)
            details = tk.Frame(card, bg=self._workbench_colors["paper"])
            details.pack(side="left", fill="both", expand=True, padx=(0, 9), pady=7)
            headline = tk.Frame(details, bg=self._workbench_colors["paper"])
            headline.pack(fill="x")
            tk.Label(headline, text=name, anchor="w", bg=self._workbench_colors["paper"], fg=self._workbench_colors["ink"], font=("Helvetica Neue", 11, "bold")).pack(side="left", fill="x", expand=True)
            star = "—" if rating == 0 else f"{self._output_rating(rating)}★"
            tk.Label(headline, text=star, bg=self._workbench_colors["paper"], fg=self._workbench_colors["copper"], font=("Helvetica Neue", 11, "bold")).pack(side="right")
            if self.results and index is not None:
                item = self.results[index]
                score = int(round(min(100.0, item.score)))
                summary = f"AI 评分 {score} / 100  ·  " + ("；".join(reasons[-2:]) or self.t["best"])
            else:
                summary = "已导入，等待开始 AI 分析"
            detail_label = tk.Label(details, text=summary, anchor="w", justify="left", wraplength=700, bg=self._workbench_colors["paper"], fg=self._workbench_colors["muted"], font=("Helvetica Neue", 9))
            detail_label.pack(fill="x", pady=(5, 0))
            if index is not None:
                select = lambda _event, value=index: self._select_result_index(value)
            else:
                select = lambda _event, primary=path, pair_paths=companions: self._select_imported_photo(primary, pair_paths)
            for widget in (card, photo, details, headline, detail_label, *headline.winfo_children()):
                widget.bind("<Button-1>", select, add="+")

    def _show_more_grid_photos(self) -> None:
        self.grid_visible_limit += 72
        self._render_photo_grid()

    def _select_result_index(self, index: int) -> None:
        if not 0 <= index < len(self.results):
            return
        self.active_result_index = index
        self.active_import_path = None
        item_id = str(index)
        if self.table.exists(item_id):
            self.table.selection_set(item_id)
            self.table.focus(item_id)
        self.selected = self.results[index]
        self.show_selected()
        self._render_photo_grid()

    def _select_imported_photo(self, path: Path, companions: list[Path]) -> None:
        """Allow pre-analysis review without inventing a rating for the photo."""
        if path not in self.imported_files:
            return
        self.active_result_index = None
        self.active_import_path = path
        self.selected = None
        self.preview_label.configure(image="", text="正在准备预览")
        self.detail.configure(text=f"{path.name}\n已导入，等待开始 AI 分析。")
        self._render_photo_grid()
        source = self._preview_source(path, companions)
        generation = self._preview_generation

        def prepare_preview() -> None:
            try:
                self._preview_image(source)
            except Exception:
                return
            self.ui_events.put(("import_selection_preview_ready", (generation, path, source)))

        threading.Thread(
            target=prepare_preview,
            name="event-cull-import-preview",
            daemon=True,
        ).start()

    @staticmethod
    def _picker_threshold(value: str, *, weak: float, standard: float, strong: float) -> float:
        return {"弱": weak, "标准": standard, "强": strong}.get(value, standard)

    @staticmethod
    def _scene_grouping_settings(scene_type: str, similarity_label: str) -> tuple[int, float]:
        """Allow a longer pose sequence to count as one male-portrait scene."""
        if scene_type == ATMOSPHERE_PORTRAIT_SCENE:
            return ({"严格": 22, "标准": 30, "宽松": 32}.get(similarity_label, 30), 60.0)
        return ({
            "严格": 10,
            "标准": 18,
            "宽松": 26,
            "重复": 10,
            "相似": 26,
        }.get(similarity_label, 18), 8.0)

    def _analysis_options_from_picker(
        self,
        *,
        picks_min: int,
        picks_max: int,
        target_min: int,
        target_max: int,
        duplicate_keep_limit: int,
    ) -> AnalysisOptions:
        # The picker uses human-facing grouping language while the engine
        # compares perceptual-hash distance.  Keep the old labels as a
        # compatibility fallback for an already-open legacy settings window.
        similarity, burst_gap = self._scene_grouping_settings(
            self.scene_type.get(), self.duplicate_similarity.get()
        )
        return AnalysisOptions(
            picks_per_group=picks_min,
            max_picks_per_group=picks_max,
            target_keep_count=target_min,
            target_keep_max_count=target_max,
            blur_floor=self._picker_threshold(self.blur_sensitivity.get(), weak=45, standard=55, strong=70),
            severe_clip_fraction=self._picker_threshold(self.exposure_sensitivity.get(), weak=0.18, standard=0.12, strong=0.08),
            similarity_threshold=int(similarity),
            burst_gap_seconds=burst_gap,
            detect_closed_eyes=self.detect_closed_eyes.get(),
            detect_blur=self.detect_blur.get(),
            detect_exposure=self.detect_exposure.get(),
            detect_duplicates=self.detect_duplicates.get(),
            ai_selection_enabled=self.ai_selection_enabled.get(),
            prefer_people=(
                self.priority_more_faces.get()
                and self.scene_type.get() != ATMOSPHERE_PORTRAIT_SCENE
            ),
            max_duplicate_picks_per_group=duplicate_keep_limit,
            fill_with_similar=self.fill_with_similar.get(),
            scene_type=self.scene_type.get(),
            language=self.language,
        )

    def _duplicate_keep_limit(self) -> int:
        """Read the editable duplicate-cluster cap with a safe UI range."""
        value = int(self.keep_per_duplicate.get().strip())
        if not 1 <= value <= 99:
            raise ValueError("duplicate keep limit must be between 1 and 99")
        return value

    def _target_selection_range(self) -> tuple[int, int]:
        """Read the required keep floor and the maximum selection quantity."""
        minimum = int(self.target.get().strip())
        maximum = int(self.target_max.get().strip())
        if minimum < 1 or maximum < minimum:
            raise ValueError("target selection range is invalid")
        return minimum, maximum

    def _reset_picker_settings(self) -> None:
        self.scene_type.set("通用")
        self.ai_selection_enabled.set(True)
        self.priority_more_faces.set(True)
        self.fill_with_similar.set(False)
        self.keep_existing_labels.set(True)
        self.detect_closed_eyes.set(True)
        self.detect_blur.set(True)
        self.detect_exposure.set(True)
        self.detect_duplicates.set(True)
        self.blur_sensitivity.set("标准")
        self.exposure_sensitivity.set("标准")
        self.duplicate_similarity.set("标准")
        self.keep_per_duplicate.set("2")
        self.picker_settings_confirmed = False
        self._refresh_picker_confirmation_ui()

    def _picker_rule_summary_text(self) -> str:
        """Return the compact, human-readable rule summary shown before AI runs."""
        range_text = self.picks.get()
        if self.picks_max.get() != self.picks.get():
            range_text = f"{range_text}–{self.picks_max.get()}"
        target_text = self.target.get()
        if self.target_max.get() != self.target.get():
            target_text = f"至少 {target_text}，最多 {self.target_max.get()}"
        enabled = []
        if self.detect_closed_eyes.get():
            enabled.append("闭眼")
        if self.detect_blur.get():
            enabled.append("模糊")
        if self.detect_exposure.get():
            enabled.append("曝光")
        if self.detect_duplicates.get():
            enabled.append("重复")
        detection = "、".join(enabled) if enabled else "未启用自动检测"
        return f"目标 {target_text} 张 · 每组优选 {range_text} 张 · {detection}检测"

    def _refresh_picker_confirmation_ui(self) -> None:
        """Keep the pre-analysis gate legible and enforceable in the main UI."""
        if not hasattr(self, "picker_settings_button"):
            return
        if not self.imported_files:
            self.rule_heading.set("开始前：先导入照片")
            self.rule_summary.set("从左侧导入文件夹或选择照片，也可直接把照片拖进窗口。")
            self.choose_button.configure(text="导入文件夹", state="normal")
            self.choose_files_button.configure(text="选择照片", state="normal")
            self.picker_settings_button.configure(
                text="设置挑图规则", style="Outline.TButton", state="disabled"
            )
            self.analyze_button.configure(state="disabled")
            return
        self.choose_button.configure(text="追加文件夹", state="normal")
        self.choose_files_button.configure(text="追加照片", state="normal")
        if self.picker_settings_confirmed:
            self.rule_heading.set("挑图规则已确认")
            self.rule_summary.set(self._picker_rule_summary_text() + " · 可以开始分析")
            self.picker_settings_button.configure(
                text="调整挑图规则", style="Outline.TButton", state="normal"
            )
            if not self.analysis_running and not self.calibration_running:
                self.analyze_button.configure(state="normal")
        else:
            self.rule_heading.set("开始前：确认挑图规则")
            self.rule_summary.set(self._picker_rule_summary_text() + " · 确认后才可开始分析")
            self.picker_settings_button.configure(
                text="设置挑图规则", style="Primary.TButton", state="normal"
            )
            self.analyze_button.configure(state="disabled")

    def show_picker_settings(self) -> None:
        """Show the selected single-column rules layout before AI work begins."""
        palette = {
            "window": "#17191b",
            "header": "#202326",
            "surface": "#1c1f21",
            "line": "#3b4043",
            "ink": "#fffaf0",
            "muted": "#b7b9b8",
            "copper": "#df9b2d",
            "copper_dark": "#9e6b25",
            "control": "#f7f5f0",
            "control_ink": "#191919",
        }
        dialog = tk.Toplevel(self)
        dialog.title("秒选 · AI 挑图设置")
        dialog.configure(bg=palette["window"])
        dialog.transient(self)
        dialog.grab_set()
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        # Prefer the generous one-page reference layout, but never let a
        # smaller desktop push the footer or any controls off-screen.
        width = min(1240, max(960, screen_width - 56))
        height = min(1000, max(720, screen_height - 64))
        x = self.winfo_rootx() + max(0, (self.winfo_width() - width) // 2)
        y = self.winfo_rooty() + max(0, (self.winfo_height() - height) // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        dialog.minsize(min(960, width), min(720, height))
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=1)

        picker_style = ttk.Style(dialog)
        picker_style.configure("Picker.TCombobox", fieldbackground=palette["control"], background=palette["control"], foreground=palette["control_ink"], arrowcolor=palette["control_ink"], padding=(8, 6))
        picker_style.map("Picker.TCombobox", fieldbackground=[("readonly", palette["control"])], foreground=[("readonly", palette["control_ink"])])
        picker_style.configure("PickerPrimary.TButton", background=palette["copper"], foreground="#1b1309", borderwidth=0, padding=(22, 12), font=("Helvetica Neue", 12, "bold"))
        picker_style.map("PickerPrimary.TButton", background=[("active", "#efb35a")])
        picker_style.configure("PickerQuiet.TButton", background="#303438", foreground=palette["ink"], borderwidth=1, padding=(20, 12), font=("Helvetica Neue", 11, "bold"))
        picker_style.map("PickerQuiet.TButton", background=[("active", "#3d4246")])
        dialog.option_add("*TCombobox*Listbox.background", palette["control"])
        dialog.option_add("*TCombobox*Listbox.foreground", palette["control_ink"])
        dialog.option_add("*TCombobox*Listbox.selectBackground", palette["copper"])
        dialog.option_add("*TCombobox*Listbox.selectForeground", "#17120c")

        header = tk.Frame(dialog, bg=palette["header"], highlightbackground=palette["line"], highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(header, text="秒选", bg=palette["header"], fg=palette["ink"], font=("Songti SC", 25, "bold")).pack(side="left", padx=(28, 12), pady=17)
        tk.Label(header, text="|", bg=palette["header"], fg="#888b8c", font=("Helvetica Neue", 19)).pack(side="left")
        tk.Label(header, text="AI 挑图设置", bg=palette["header"], fg=palette["ink"], font=("Helvetica Neue", 20, "bold")).pack(side="left", padx=14)
        tk.Label(header, text="设置完成后将开始本次 AI 分析", bg=palette["header"], fg=palette["muted"], font=("Helvetica Neue", 10)).pack(side="right", padx=28)

        # The header and actions deliberately stay pinned.  The form itself
        # scrolls only when the available display height is less than the
        # full reference layout, so a shorter laptop never hides duplicate
        # handling or the confirmation actions.
        form_shell = tk.Frame(dialog, bg=palette["window"])
        form_shell.grid(row=1, column=0, sticky="nsew")
        form_shell.columnconfigure(0, weight=1)
        form_shell.rowconfigure(0, weight=1)
        form_canvas = tk.Canvas(
            form_shell,
            bg=palette["window"],
            bd=0,
            highlightthickness=0,
            yscrollincrement=12,
        )
        form_scroll = ttk.Scrollbar(form_shell, orient="vertical", command=form_canvas.yview)
        form_canvas.configure(yscrollcommand=form_scroll.set)
        form_canvas.grid(row=0, column=0, sticky="nsew")
        form_scroll.grid(row=0, column=1, sticky="ns")
        content = tk.Frame(form_canvas, bg=palette["window"], padx=32, pady=20)
        content_window = form_canvas.create_window((0, 0), window=content, anchor="nw")

        def sync_form_scroll_region(_event=None) -> None:
            form_canvas.configure(scrollregion=form_canvas.bbox("all"))

        def resize_form_content(event) -> None:
            form_canvas.itemconfigure(content_window, width=event.width)

        def scroll_form(event) -> str:
            raw_delta = getattr(event, "delta", 0)
            if raw_delta:
                units = self._wheel_units(raw_delta)
            else:
                units = -1 if getattr(event, "num", 0) == 5 else 1
            form_canvas.yview_scroll(units, "units")
            return "break"

        content.bind("<Configure>", sync_form_scroll_region)
        form_canvas.bind("<Configure>", resize_form_content)
        form_canvas.bind("<MouseWheel>", scroll_form)
        form_canvas.bind("<Button-4>", scroll_form)
        form_canvas.bind("<Button-5>", scroll_form)

        def section(title: str, subtitle: str) -> tk.Frame:
            block = tk.Frame(content, bg=palette["window"])
            tk.Label(block, text=title, bg=palette["window"], fg=palette["ink"], font=("Songti SC", 20, "bold")).pack(anchor="w")
            tk.Label(block, text=subtitle, bg=palette["window"], fg=palette["muted"], font=("Helvetica Neue", 10), justify="left").pack(anchor="w", pady=(4, 10))
            return block

        def divider(parent: tk.Widget) -> None:
            tk.Frame(parent, bg=palette["line"], height=1).pack(fill="x", pady=(10, 0))

        def row(parent: tk.Widget, label: str, description: str) -> tuple[tk.Frame, tk.Frame]:
            line = tk.Frame(parent, bg=palette["window"])
            line.pack(fill="x", pady=(9, 0))
            copy = tk.Frame(line, bg=palette["window"])
            copy.pack(side="left", fill="x", expand=True)
            tk.Label(copy, text=label, bg=palette["window"], fg="#f2efe9", font=("Helvetica Neue", 12, "bold")).pack(anchor="w")
            tk.Label(copy, text=description, bg=palette["window"], fg=palette["muted"], font=("Helvetica Neue", 9)).pack(anchor="w", pady=(2, 0))
            controls = tk.Frame(line, bg=palette["window"])
            controls.pack(side="right", padx=(20, 0))
            return line, controls

        def spin(parent: tk.Widget, variable: tk.StringVar, *, width: int = 5, minimum: int = 1, maximum: int = 50000) -> tk.Spinbox:
            return tk.Spinbox(
                parent,
                from_=minimum,
                to=maximum,
                textvariable=variable,
                width=width,
                justify="center",
                bg=palette["control"],
                fg=palette["control_ink"],
                buttonbackground="#e5e2da",
                insertbackground=palette["control_ink"],
                relief="flat",
                highlightthickness=1,
                highlightbackground="#d5d1c8",
                highlightcolor=palette["copper"],
                font=("Helvetica Neue", 12, "bold"),
            )

        def toggle(parent: tk.Widget, variable: tk.BooleanVar, *, side: str = "left") -> tk.Canvas:
            """Render a compact, single-purpose rounded switch."""
            control = tk.Canvas(parent, width=60, height=30, bg=palette["window"], bd=0, highlightthickness=0, cursor="hand2")

            def redraw(_event=None) -> None:
                active = variable.get()
                track = palette["copper"] if active else "#53585b"
                control.delete("all")
                control.create_oval(2, 3, 28, 27, fill=track, outline=track)
                control.create_rectangle(15, 3, 45, 27, fill=track, outline=track)
                control.create_oval(32, 3, 58, 27, fill=track, outline=track)
                knob_x = 45 if active else 15
                control.create_oval(knob_x - 9, 6, knob_x + 9, 24, fill="#1c1e20" if active else "#e9e6df", outline="")

            def choose(_event=None) -> None:
                variable.set(not variable.get())
                redraw()

            control.bind("<Button-1>", choose)
            redraw()
            control.pack(side=side)
            return control

        def segments(parent: tk.Widget, values: tuple[str, ...], variable: tk.StringVar, *, item_width: int = 74) -> tk.Canvas:
            control = tk.Canvas(parent, width=item_width * len(values), height=34, bg=palette["window"], bd=0, highlightthickness=0, cursor="hand2")

            def redraw(_event=None) -> None:
                control.delete("all")
                for index, choice in enumerate(values):
                    selected = variable.get() == choice
                    left = index * item_width
                    fill = palette["copper"] if selected else "#34383b"
                    control.create_rectangle(left, 2, left + item_width - 2, 32, fill=fill, outline=palette["line"])
                    control.create_text(left + (item_width - 2) / 2, 17, text=choice, fill="#1c140a" if selected else "#f2efe9", font=("Helvetica Neue", 10, "bold"))

            def choose(event) -> None:
                index = min(len(values) - 1, max(0, event.x // item_width))
                variable.set(values[index])
                redraw()

            control.bind("<Button-1>", choose)
            redraw()
            control.pack(side="left")
            return control

        def preference(parent: tk.Widget, text: str, variable: tk.BooleanVar) -> tk.Frame:
            """Render one scan-friendly preference row with its own switch."""
            setting = tk.Frame(parent, bg=palette["window"])
            setting.pack(fill="x")
            setting_content = tk.Frame(setting, bg=palette["window"])
            setting_content.pack(fill="x", pady=2)
            tk.Label(
                setting_content,
                text=text,
                bg=palette["window"],
                fg="#f2efe9",
                anchor="w",
                font=("Helvetica Neue", 11, "bold"),
            ).pack(side="left", fill="x", expand=True, pady=7)
            toggle(setting_content, variable, side="right")
            tk.Frame(setting, bg=palette["line"], height=1).pack(fill="x", pady=(3, 0))
            return setting

        target_section = section("目标图片数量", "AI 选片不超过最多张数；明确命中的必选镜头可例外保留。")
        target_section.pack(fill="x")
        _, target_controls = row(target_section, "交付目标", "设置本次选片的最少和最多张数。")
        tk.Label(target_controls, text="至少", bg=palette["window"], fg=palette["muted"], font=("Helvetica Neue", 10)).pack(side="left", padx=(0, 7))
        spin(target_controls, self.target).pack(side="left")
        tk.Label(target_controls, text="～", bg=palette["window"], fg=palette["ink"], font=("Helvetica Neue", 15, "bold")).pack(side="left", padx=10)
        tk.Label(target_controls, text="最多", bg=palette["window"], fg=palette["muted"], font=("Helvetica Neue", 10)).pack(side="left", padx=(0, 7))
        spin(target_controls, self.target_max).pack(side="left")
        tk.Label(target_controls, text="张", bg=palette["window"], fg=palette["muted"], font=("Helvetica Neue", 10)).pack(side="left", padx=(7, 0))
        divider(target_section)
        _, picks_controls = row(target_section, "每组优选张数", "从每组相似图中优先挑出值得复核的照片。")
        spin(picks_controls, self.picks, width=4, maximum=9).pack(side="left")
        tk.Label(picks_controls, text="张 / 每组", bg=palette["window"], fg=palette["muted"], font=("Helvetica Neue", 10)).pack(side="left", padx=(8, 20))
        tk.Frame(picks_controls, bg=palette["line"], width=1, height=28).pack(side="left", padx=(0, 20))
        tk.Label(picks_controls, text="图片场景", bg=palette["window"], fg=palette["muted"], font=("Helvetica Neue", 10)).pack(side="left", padx=(0, 8))
        ttk.Combobox(picks_controls, textvariable=self.scene_type, values=SCENE_TYPES, state="readonly", width=14, style="Picker.TCombobox").pack(side="left")
        preferences = tk.Frame(target_section, bg=palette["window"])
        preferences.pack(fill="x", pady=(10, 0))
        preference(preferences, "启用 AI 精选", self.ai_selection_enabled)
        preference(preferences, "正脸与自然表情优先（氛围写真自动关闭）", self.priority_more_faces)
        preference(preferences, "数量不足时用重复照片补足", self.fill_with_similar)
        preference(preferences, "保留已手动设置的星级", self.keep_existing_labels)

        quality_section = section("画面质量检测", "自动识别并过滤不符合要求的照片，提升成片质量。")
        quality_section.pack(fill="x", pady=(20, 0))
        for index, (label, description, variable, sensitivity) in enumerate((
            ("闭眼检测", "人像中检测到多数人合眼时，会标记为不佳照片。", self.detect_closed_eyes, None),
            ("欠曝 / 过曝检测", "检测画面过暗或过亮的照片。", self.detect_exposure, self.exposure_sensitivity),
            ("模糊检测", "检测因抖动、对焦不准等造成的模糊照片。", self.detect_blur, self.blur_sensitivity),
        )):
            _, controls = row(quality_section, label, description)
            toggle(controls, variable)
            if sensitivity is not None:
                tk.Frame(controls, bg=palette["line"], width=1, height=26).pack(side="left", padx=(13, 15))
                tk.Label(controls, text="敏感度", bg=palette["window"], fg=palette["muted"], font=("Helvetica Neue", 9)).pack(side="left", padx=(0, 10))
                segments(controls, ("弱", "标准", "强"), sensitivity)
            if index < 2:
                divider(quality_section)

        duplicate_section = section("相似图处理", "识别相似的连拍或重复照片，并保留最值得复核的几张。")
        duplicate_section.pack(fill="x", pady=(20, 0))
        _, duplicate_controls = row(duplicate_section, "相似图聚类", "按连拍相似度自动分组，再保留最佳候选。")
        toggle(duplicate_controls, self.detect_duplicates)
        divider(duplicate_section)
        _, similarity_controls = row(
            duplicate_section,
            "分组严格度",
            "严格只合并几乎同一画面的连拍；宽松会把姿势或构图略有变化的连拍也归为一组。",
        )
        segments(similarity_controls, ("严格", "标准", "宽松"), self.duplicate_similarity, item_width=90)
        divider(duplicate_section)
        _, keep_controls = row(duplicate_section, "每组重复图片精选张数", "每组相似图中保留供复核的最佳照片数量。")
        spin(keep_controls, self.keep_per_duplicate, width=4, maximum=99).pack(side="left")
        tk.Label(keep_controls, text="张（可输入 1–99）", bg=palette["window"], fg=palette["muted"], font=("Helvetica Neue", 10)).pack(side="left", padx=(8, 0))

        # Events target the embedded frame's labels and controls, rather than
        # the surrounding canvas.  Bind them recursively so the wheel works
        # wherever the pointer is in the settings form, including on a switch
        # or an input field.
        def bind_form_wheel(widget: tk.Widget) -> None:
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(sequence, scroll_form, add="+")
            for child in widget.winfo_children():
                bind_form_wheel(child)

        bind_form_wheel(content)

        actions = tk.Frame(dialog, bg=palette["header"], highlightbackground=palette["line"], highlightthickness=1)
        actions.grid(row=2, column=0, sticky="ew")
        ttk.Button(actions, text="恢复默认", style="PickerQuiet.TButton", command=self._reset_picker_settings).pack(side="left", padx=28, pady=12)
        ttk.Button(actions, text="取消", style="PickerQuiet.TButton", command=dialog.destroy).pack(side="right", padx=(8, 28), pady=12)

        def confirm_rules() -> None:
            try:
                self._duplicate_keep_limit()
                self._target_selection_range()
            except ValueError:
                messagebox.showerror(
                    "参数不正确",
                    "目标图片数量请输入有效范围（例如 200～300）；每组重复图片精选数请输入 1–99 的整数。",
                    parent=dialog,
                )
                return
            self.picker_settings_confirmed = True
            self._refresh_picker_confirmation_ui()
            dialog.destroy()

        ttk.Button(actions, text="确认挑图规则", style="PickerPrimary.TButton", command=confirm_rules).pack(side="right", pady=12)

    def _build_menubar(self) -> None:
        """Set up shortcuts without allowing Tk to create detachable macOS menus."""
        # Recent Tk builds can turn *every* ``tk.Menu`` child into a detached
        # desktop window on macOS (titled ``!menu``, ``!menu2``, ...), even
        # when ``tearoff=False`` is supplied.  The workbench already exposes
        # every action in-window, so omit the native Tk menu entirely there.
        if sys.platform == "darwin":
            self.configure(menu="")
            self._bind_menu_shortcuts()
            return

        menu_bar = tk.Menu(self, tearoff=False)
        app_menu = tk.Menu(menu_bar, tearoff=False)
        app_menu.add_command(label=self.t["menu_about"], command=self.show_about)
        app_menu.add_separator()
        app_menu.add_command(label=self.t["menu_history"], command=self.show_task_history)
        app_menu.add_command(label=self.t["menu_license"], command=self.show_license_dialog)
        app_menu.add_separator()
        app_menu.add_command(label=self.t["menu_quit"], command=self.destroy)
        menu_bar.add_cascade(label=self.product_title, menu=app_menu)

        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(label=self.t["menu_new_task"], command=self.new_task)
        file_menu.add_separator()
        file_menu.add_command(label=self.t["menu_import_files"], command=self.choose_files)
        file_menu.add_command(label=self.t["menu_import_folders"], command=self.choose_folder)
        file_menu.add_separator()
        file_menu.add_command(label=self.t["menu_history"], command=self.show_task_history)
        file_menu.add_command(label=self.t["menu_license"], command=self.show_license_dialog)
        menu_bar.add_cascade(label=self.t["menu_file"], menu=file_menu)

        edit_menu = tk.Menu(menu_bar, tearoff=False)
        edit_menu.add_command(label=self.t["menu_copy"], command=lambda: self.event_generate("<<Copy>>"))
        menu_bar.add_cascade(label=self.t["menu_edit"], menu=edit_menu)

        # Tk's custom-drawn workbench is not fully exposed by macOS Accessibility.
        # Mirror every primary workflow action in the native menu bar so
        # VoiceOver and keyboard-only users can still operate the complete flow.
        workflow_menu = tk.Menu(menu_bar, tearoff=False)
        shortcut = "⌘" if sys.platform == "darwin" else "Ctrl+"
        workflow_menu.add_command(label=self.t["menu_culling"], command=self.show_culling_workspace, accelerator=f"{shortcut}1")
        workflow_menu.add_command(label=self.t["menu_export"], command=self.show_export_workspace, accelerator=f"{shortcut}2")
        workflow_menu.add_command(label=self.t["menu_calibration"], command=self.show_calibration_workspace, accelerator=f"{shortcut}3")
        workflow_menu.add_separator()
        workflow_menu.add_command(label=self.t["menu_picker_settings"], command=self.show_picker_settings, accelerator=f"{shortcut},")
        workflow_menu.add_command(label=self.t["menu_start_analysis"], command=self.start_analysis, accelerator=f"{shortcut}Enter")
        workflow_menu.add_command(label=self.t["menu_pause_resume"], command=self._menu_toggle_pause, accelerator=f"{shortcut}P")
        workflow_menu.add_command(label=self.t["menu_stop_analysis"], command=self._menu_stop_analysis, accelerator=f"{shortcut}.")
        workflow_menu.add_command(label=self.t["menu_write_ratings"], command=self._menu_confirm_write, accelerator=f"{shortcut}Shift+W")
        menu_bar.add_cascade(label=self.t["menu_workflow"], menu=workflow_menu)

        window_menu = tk.Menu(menu_bar, tearoff=False)
        window_menu.add_command(label=self.t["menu_minimize"], command=self.iconify)
        menu_bar.add_cascade(label=self.t["menu_window"], menu=window_menu)

        help_menu = tk.Menu(menu_bar, tearoff=False)
        help_menu.add_command(label=self.t["menu_help_content"], command=self.show_about)
        menu_bar.add_cascade(label=self.t["menu_help"], menu=help_menu)
        self.menu_bar = menu_bar
        self.configure(menu=menu_bar)
        self._bind_menu_shortcuts()

    def _bind_menu_shortcuts(self) -> None:
        modifier = "Command" if sys.platform == "darwin" else "Control"
        self.bind_all(f"<{modifier}-Key-1>", lambda _event: self.show_culling_workspace())
        self.bind_all(f"<{modifier}-Key-2>", lambda _event: self.show_export_workspace())
        self.bind_all(f"<{modifier}-Key-3>", lambda _event: self.show_calibration_workspace())
        self.bind_all(f"<{modifier}-comma>", lambda _event: self.show_picker_settings())
        self.bind_all(f"<{modifier}-Return>", lambda _event: self.start_analysis())
        self.bind_all(f"<{modifier}-Key-p>", lambda _event: self._menu_toggle_pause())
        self.bind_all(f"<{modifier}-period>", lambda _event: self._menu_stop_analysis())
        self.bind_all(f"<{modifier}-Shift-Key-W>", lambda _event: self._menu_confirm_write())

    def _menu_toggle_pause(self) -> None:
        if self.analysis_running:
            self.toggle_pause()

    def _menu_stop_analysis(self) -> None:
        if self.analysis_running:
            self.stop_analysis()

    def _menu_confirm_write(self) -> None:
        if str(self.write_button.cget("state")) != "disabled":
            self.confirm_write()

    def show_about(self) -> None:
        messagebox.showinfo(self.t["menu_about"], self.t["about_content"])

    def _refresh_license_entry(self) -> None:
        """Keep an always-visible, accurate entitlement entry in the header."""
        if not hasattr(self, "license_button"):
            return
        try:
            decision = self.license_controller.decision
            self.license_button.configure(text=license_label(decision))
            if hasattr(self, "license_caption"):
                lease = decision.lease
                if lease is not None and lease.team_name:
                    self.license_caption.configure(text="团队永久版")
                elif decision.plan.value == "lifetime":
                    self.license_caption.configure(text="永久版")
                elif decision.plan.value == "developer":
                    self.license_caption.configure(text="内部测试版")
                else:
                    self.license_caption.configure(text="试用版")
        except Exception:
            self.license_button.configure(text="授权与订阅")

    def show_license_dialog(self) -> None:
        # The authorization sheet now contains team verification inline. A
        # second Tk modal window can crash macOS input methods while a cull is
        # still rendering HEIC previews, so never open a nested dialog here.
        LicenseDialog(self, self.license_controller, self._refresh_license_entry)

    def show_task_history(self) -> None:
        """Show the compact, local-only index of imports and completed culls."""
        records = load_history()
        window = tk.Toplevel(self)
        window.title(self.t["history_title"])
        window.geometry("880x460")
        window.minsize(680, 320)
        window.configure(bg="#101112")
        window.transient(self)

        shell = ttk.Frame(window, padding=18)
        shell.pack(fill="both", expand=True)
        if not records:
            ttk.Label(shell, text=self.t["history_empty"], style="Muted.TLabel").pack(
                anchor="center", expand=True
            )
            return

        columns = ("name", "photos", "selected", "featured", "status")
        table = ttk.Treeview(shell, columns=columns, show="headings", height=11)
        for column, label in zip(columns, self.t["history_columns"]):
            table.heading(column, text=label)
        table.column("name", width=360, minwidth=240, anchor="w")
        table.column("photos", width=85, minwidth=65, anchor="center")
        table.column("selected", width=85, minwidth=65, anchor="center")
        table.column("featured", width=75, minwidth=60, anchor="center")
        table.column("status", width=120, minwidth=90, anchor="center")
        scrollbar = ttk.Scrollbar(shell, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scrollbar.set)
        table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="left", fill="y")

        record_by_id: dict[str, dict[str, object]] = {}
        for position, record in enumerate(records):
            item_id = str(record.get("id") or f"history-{position}")
            record_by_id[item_id] = record
            counts = record.get("counts") if isinstance(record.get("counts"), dict) else {}
            table.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    str(record.get("name") or "—"),
                    str(record.get("logical_photo_count") or 0),
                    str(record.get("selected_total") or 0),
                    str(counts.get("3") or 0),
                    str(record.get("status") or "—"),
                ),
            )

        footer = ttk.Frame(window, padding=(18, 0, 18, 18))
        footer.pack(fill="x")
        detail = tk.StringVar(value="")
        ttk.Label(
            footer,
            textvariable=detail,
            style="Muted.TLabel",
            justify="left",
            wraplength=610,
        ).pack(side="left", fill="x", expand=True)

        def selected_record() -> dict[str, object] | None:
            selection = table.selection()
            return record_by_id.get(selection[0]) if selection else None

        def refresh_detail(_event=None) -> None:
            record = selected_record()
            if record is None:
                detail.set("")
                reveal.configure(state="disabled")
                return
            folders = record.get("source_folders")
            folder_text = "、".join(Path(str(value)).name for value in folders) if isinstance(folders, list) else "—"
            groups = record.get("group_count") or 0
            if self.language == "zh":
                detail.set(f"来源：{folder_text}\n分组：{groups}　建立：{record.get('created_at') or '—'}")
            else:
                detail.set(f"Sources: {folder_text}\nGroups: {groups}   Created: {record.get('created_at') or '—'}")
            reveal.configure(state="normal" if record.get("log_path") else "disabled")

        def reveal_log() -> None:
            record = selected_record()
            if record is None or not record.get("log_path"):
                return
            log_path = Path(str(record["log_path"]))
            if not log_path.exists():
                messagebox.showinfo(
                    self.t["history_title"],
                    "找不到此任务的本地日志文件。" if self.language == "zh" else "The local log file for this task is no longer available.",
                    parent=window,
                )
                return
            if sys.platform == "darwin":
                subprocess.run(["open", "-R", str(log_path)], check=False)
            elif sys.platform == "win32":
                os.startfile(str(log_path.parent))
            else:
                subprocess.run(["xdg-open", str(log_path.parent)], check=False)

        reveal = ttk.Button(footer, text=self.t["history_open_log"], command=reveal_log, state="disabled")
        reveal.pack(side="right", padx=(12, 0))
        table.bind("<<TreeviewSelect>>", refresh_detail)
        if table.get_children():
            table.selection_set(table.get_children()[0])
            refresh_detail()

    def _control_icon(self, kind: str) -> tk.PhotoImage:
        """Draw crisp transport icons without font or emoji substitutions."""
        image = tk.PhotoImage(width=18, height=18)
        color = "#f0ede8"
        if kind == "pause":
            image.put(color, to=(4, 3, 8, 15))
            image.put(color, to=(11, 3, 15, 15))
        elif kind == "resume":
            for x in range(5, 14):
                half_height = min(x - 5, 13 - x)
                image.put(color, to=(x, 8 - half_height, x + 1, 10 + half_height))
        elif kind == "stop":
            image.put(color, to=(4, 4, 14, 14))
        return image

    def _sidebar_icon(self, kind: str, *, color: str = "#e8e5de") -> tk.PhotoImage:
        """Render a native symbol, or a legible platform-independent fallback."""
        symbols = {
            "history": "clock.arrow.circlepath",
            "folder": "folder",
            "photos": "photo",
            "cull": "square.grid.2x2",
            "list": "list.bullet",
            "export": "square.and.pencil",
            "calibration": "person.crop.circle",
        }
        symbol_name = symbols[kind]
        if NSImage is None:
            canvas = Image.new("RGBA", (18, 18))
            draw = ImageDraw.Draw(canvas)
            if kind in {"folder", "photos", "export"}:
                draw.rectangle((2, 5, 15, 14), outline=color, width=2)
                draw.line((3, 4, 8, 4), fill=color, width=2)
            elif kind == "history":
                draw.ellipse((2, 2, 15, 15), outline=color, width=2)
                draw.line((9, 4, 9, 9, 13, 11), fill=color, width=2)
            elif kind == "cull":
                for left in (2, 10):
                    for top in (2, 10):
                        draw.rectangle((left, top, left + 5, top + 5), outline=color)
            elif kind == "list":
                for top in (4, 8, 12):
                    draw.line((4, top, 15, top), fill=color, width=2)
            else:
                draw.ellipse((5, 2, 12, 9), outline=color, width=2)
                draw.arc((2, 7, 15, 17), 180, 360, fill=color, width=2)
            return ImageTk.PhotoImage(canvas)
        native_image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol_name, None)
        if native_image is None:
            return tk.PhotoImage(width=18, height=18)
        try:
            source = Image.open(BytesIO(bytes(native_image.TIFFRepresentation()))).convert("RGBA")
            tinted = Image.new("RGBA", source.size, color)
            tinted.putalpha(source.getchannel("A"))
            tinted.thumbnail((18, 18), Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", (18, 18))
            x_offset = (18 - tinted.width) // 2
            y_offset = (18 - tinted.height) // 2
            canvas.alpha_composite(tinted, (x_offset, y_offset))
            return ImageTk.PhotoImage(canvas)
        except Exception:  # pragma: no cover - a missing system symbol is non-fatal.
            return tk.PhotoImage(width=18, height=18)

    def toggle_lightroom_sync_instructions(self) -> None:
        """Show optional Lightroom detail without making the review rail scroll."""
        # Keep the lightweight pre-window behavior available to the isolated
        # safety test harness.  A real Tk root always has ``tk`` in __dict__.
        if "tk" not in self.__dict__:
            if self.sync_steps_frame.winfo_manager():
                self.sync_steps_frame.pack_forget()
                self.sync_toggle_button.configure(text=self.t["sync_show"])
            else:
                self.sync_steps_frame.pack(fill="x", pady=(10, 0))
                self.sync_toggle_button.configure(text=self.t["sync_hide"])
            return
        dialog = tk.Toplevel(self)
        dialog.title("Lightroom 星级同步")
        dialog.geometry("560x510")
        dialog.minsize(500, 430)
        dialog.configure(bg=self._workbench_colors["canvas"])
        dialog.transient(self)
        body = ttk.Frame(dialog, style="Workbench.TFrame", padding=(24, 22))
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Lightroom 星级同步", style="TaskTitle.TLabel").pack(anchor="w")
        ttk.Label(
            body,
            text="写入 XMP 后，在 Lightroom 中读取元数据即可看到本次星级。",
            style="Muted.TLabel",
            justify="left",
            wraplength=490,
        ).pack(anchor="w", pady=(5, 18))
        for number, step in enumerate(self._lightroom_sync_steps(), start=1):
            row = ttk.Frame(body, style="Paper.TFrame", padding=(10, 8))
            row.pack(fill="x", pady=(0, 7))
            ttk.Label(row, text=str(number), style="SyncIndex.TLabel", width=2, anchor="center").pack(side="left")
            ttk.Label(row, text=step, style="Paper.TLabel", justify="left", wraplength=420).pack(side="left", fill="x", expand=True, padx=(10, 0))
        ttk.Button(body, text="完成", style="Outline.TButton", command=dialog.destroy).pack(anchor="e", pady=(6, 0))

    @staticmethod
    def _lightroom_sync_steps() -> tuple[str, ...]:
        return (
            "在 Lightroom 顶部菜单「元数据」中，取消勾选「仅显示目标照片的元数据」。",
            "将右侧「元数据」面板改为「选定的照片」。",
            "完成分析后，确认写入 XMP 星级信息。",
            "打开 Lightroom，在「图库」→「网格视图」（快捷键 G）中选中本次照片。",
            "右键 →「元数据」→「从文件中读取元数据」，即可同步完成。",
        )

    def _register_drop_targets(self, widget: tk.Misc) -> None:
        """Make every visible part of the main window an accepting file target."""
        try:
            widget.drop_target_register(DND_FILES)
            # TkDND requires an explicit action on macOS.  Returning ``break``
            # makes the Finder reject the drag before ``<<Drop>>`` is emitted,
            # which is why drops only appeared to work in some panels.
            widget.dnd_bind("<<DropEnter>>", self._accept_drop_target)
            widget.dnd_bind("<<DropPosition>>", self._accept_drop_target)
            widget.dnd_bind("<<DropLeave>>", self._restore_drop_hint)
            widget.dnd_bind("<<Drop>>", self.accept_drop)
        except (AttributeError, tk.TclError):
            # A platform-native child may not expose TkDND commands. Its
            # nearest registered parent remains available as a drop target.
            pass
        for child in widget.winfo_children():
            self._register_drop_targets(child)

    def _accept_drop_target(self, _event=None) -> str:
        """Acknowledge Finder drops for every registered panel on macOS."""
        if self.analysis_running or self.calibration_running:
            return REFUSE_DROP
        if not self.results and not self.analysis_running:
            self.preview_label.configure(text=self.t["empty_preview"], style="DropHint.TLabel")
        return COPY

    def _restore_drop_hint(self, _event=None) -> None:
        if not self.results and not self.imported_files:
            self.preview_label.configure(text=self.t["empty_preview"], style="DropHint.TLabel")

    @staticmethod
    def _default_target_count(logical_count: int) -> int:
        """Use one third of logical photos, rounded up to a multiple of 10."""
        if logical_count <= 0:
            return 0
        return ((logical_count + 29) // 30) * 10

    @staticmethod
    def _default_target_max_count(target_minimum: int) -> int:
        """Leave a modest, predictable scene-coverage allowance above the target."""
        if target_minimum <= 0:
            return 0
        allowance = (target_minimum * 12 + 9) // 10
        return max(target_minimum, ((allowance + 9) // 10) * 10)

    def _set_default_target(self, files: list[Path]) -> int:
        logical_count = self._source_photo_counts(files)["total"]
        target_minimum = self._default_target_count(logical_count)
        self.target.set(str(target_minimum))
        self.target_max.set(str(self._default_target_max_count(target_minimum)))
        return logical_count

    @staticmethod
    def _source_photo_counts(files: list[Path]) -> dict[str, int]:
        """Count logical captures, preferring a RAW when a RAW+JPG pair exists."""
        grouped_extensions: dict[tuple[Path, str], set[str]] = {}
        for path in files:
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            key = (path.parent.resolve(), path.stem.casefold())
            grouped_extensions.setdefault(key, set()).add(path.suffix.lower())

        raw_count = 0
        jpg_count = 0
        other_count = 0
        for extensions in grouped_extensions.values():
            if extensions & RAW_EXTENSIONS:
                raw_count += 1
            elif extensions & JPEG_EXTENSIONS:
                jpg_count += 1
            else:
                other_count += 1
        return {
            "jpg": jpg_count,
            "raw": raw_count,
            "other": other_count,
            "total": raw_count + jpg_count + other_count,
        }

    @staticmethod
    def _source_file_counts(files: list[Path]) -> dict[str, int]:
        """Count imported files by format without collapsing RAW+JPG pairs."""
        supported = [
            path for path in files
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        raw_count = sum(path.suffix.lower() in RAW_EXTENSIONS for path in supported)
        jpg_count = sum(path.suffix.lower() in JPEG_EXTENSIONS for path in supported)
        return {
            "jpg": jpg_count,
            "raw": raw_count,
            "other": len(supported) - raw_count - jpg_count,
            "total": len(supported),
        }

    def _source_count_text(self, counts: dict[str, int], *, imported: bool = False) -> str:
        key = "added" if imported else "selected_summary"
        if counts["other"]:
            key += "_other"
        return self.t[key].format(**counts)

    def _task_import_summary_text(
        self,
        added_counts: dict[str, int],
        total_counts: dict[str, int],
    ) -> str:
        """Describe the latest import and the full task total in one line."""
        key = "task_import_summary"
        if added_counts["other"] or total_counts["other"]:
            key += "_other"
        return self.t[key].format(
            added_total=added_counts["total"],
            added_jpg=added_counts["jpg"],
            added_raw=added_counts["raw"],
            added_other=added_counts["other"],
            total=total_counts["total"],
        )

    @staticmethod
    def _analysis_header_status_text(
        current: int,
        total: int,
        eta: str,
        *,
        language: str,
    ) -> str:
        """Keep task progress concise enough for the top-right status slot."""
        progress = (
            f"AI 分析 {current} / {total}"
            if language == "zh"
            else f"AI analysis {current} / {total}"
        )
        return f"{progress} · {eta}" if eta else progress

    def _refresh_analysis_header_status(self) -> None:
        """Reflect current analysis progress in the contextual task header."""
        self.task_import_summary.set(
            self._analysis_header_status_text(
                self.analysis_progress_current,
                self.analysis_progress_total,
                self.eta_text.get(),
                language=self.language,
            )
        )

    def _set_analysis_controls_visible(self, visible: bool) -> None:
        """Only expose pause and stop while there is an analysis to control."""
        if not hasattr(self, "pause_button") or not hasattr(self, "stop_button"):
            return
        if visible:
            if not self.pause_button.winfo_manager():
                self.pause_button.pack(side="left", padx=(8, 4))
            if not self.stop_button.winfo_manager():
                self.stop_button.pack(side="left")
            return
        self.pause_button.pack_forget()
        self.stop_button.pack_forget()

    @staticmethod
    def _merge_unique_paths(existing: list[Path], added: list[Path]) -> tuple[list[Path], int]:
        """Append real file paths in order while suppressing repeat imports."""
        merged = list(existing)
        known = {str(path.resolve()) for path in existing}
        added_count = 0
        for path in added:
            resolved = path.resolve()
            key = str(resolved)
            if key in known:
                continue
            known.add(key)
            merged.append(resolved)
            added_count += 1
        return merged, added_count

    @staticmethod
    def _files_in_folders(folders: list[Path]) -> list[Path]:
        """Find supported originals below every selected source folder."""
        return scan_source_images(folders)

    @staticmethod
    def _choose_folders_macos(
        initial_directory: Path,
        title: str,
        language: str = "zh",
    ) -> list[Path]:
        """Use the native macOS picker because Tk's folder picker is single-select."""
        script = r'''
on run argv
    set promptText to item 1 of argv
    set startFolder to POSIX file (item 2 of argv) as alias
    set chosenFolders to choose folder with prompt promptText default location startFolder with multiple selections allowed
    set outputPaths to {}
    repeat with chosenFolder in chosenFolders
        set end of outputPaths to POSIX path of chosenFolder
    end repeat
    set AppleScript's text item delimiters to character id 30
    return outputPaths as text
end run
'''
        picker_environment = os.environ.copy()
        if language == "zh":
            picker_environment.update(
                {"AppleLanguages": "(zh_CN)", "AppleLocale": "zh_CN"}
            )
        completed = subprocess.run(
            ["/usr/bin/osascript", "-e", script, "--", title, str(initial_directory)],
            capture_output=True,
            text=True,
            check=False,
            env=picker_environment,
        )
        if completed.returncode != 0:
            return []
        return [Path(value) for value in completed.stdout.strip().split("\x1e") if value]

    def _has_active_task(self) -> bool:
        return bool(self.results or self.imported_files)

    def _refresh_calibration_button(self) -> None:
        self.calibration_button.configure(text=self.t["calibration_update"])

    def _set_xmp_result(self, message: str = "") -> None:
        self.xmp_result_notice.set(message)
        if message:
            if not self.xmp_result_label.winfo_manager():
                self.xmp_result_label.pack(fill="x", before=self.sync_heading, pady=(0, 10))
        else:
            self.xmp_result_label.pack_forget()

    def _confirm_replace_task(self) -> bool:
        if not self._has_active_task():
            return True
        return messagebox.askyesno(
            self.t["new_task_confirm"],
            self.t["new_task_confirm_detail"],
        )

    def _reset_task_state(self) -> None:
        """Clear in-memory task state without touching photos, XMP, or Lightroom."""
        self.results = []
        self.imported_files = []
        self.analysis_completed_at = None
        self.current_log_path = None
        self.current_task_id = None
        self.current_task_name = ""
        self.current_task_created_at = None
        self.manual_ratings.clear()
        self.selected = None
        self.active_result_index = None
        self.active_import_path = None
        self.grid_visible_limit = 72
        self.preview = None
        self._active_preview_path = None
        self._close_preview_zoom()
        self._preview_generation += 1
        with self._preview_lock:
            self._preview_memory.clear()
            self._zoom_preview_memory.clear()
        self._grid_thumbnail_cache.clear()
        self.table.delete(*self.table.get_children())
        self.preview_label.configure(image="", text=self.t["empty_preview"])
        self.detail.configure(text=self.t["empty_detail"])
        self.folder.set(self.t["choose"])
        self.import_notice.set("")
        self.task_import_summary.set("")
        self.import_badge.grid_remove()
        self.picker_settings_confirmed = False
        self.target.set("1000")
        self.target_max.set("1200")
        self.picks.set("2")
        self.picks_max.set("3")
        self.stats_text.set(self.t["stats_empty"])
        self.rating_filter.set(self.t["filter_all"])
        self.write_mode.set(self.t["mode_safe"])
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=1, value=0)
        self.progress_text.set("")
        self.eta_text.set("")
        self.analysis_progress_current = 0
        self.analysis_progress_total = 0
        self.xmp_notice.set(self.lr_instructions)
        self._set_xmp_result()
        self.status.set(self.t["safe_note"])
        self.set_workflow_stage(1)
        self.write_button.configure(state="disabled")
        self.restore_button.configure(state="disabled")
        self._refresh_legacy_heic_repair_button()
        self.export_selected_button.configure(state="disabled")
        self.task_title.set("新建筛选任务 · 等待导入")
        self._set_analysis_controls_visible(False)
        self._refresh_picker_confirmation_ui()
        self._schedule_grid_render()
        self.refresh_lightroom_guard(schedule_next=False)

    def new_task(self) -> None:
        if self.__dict__.get("export_running", False):
            messagebox.showinfo("正在导出", "请等待已选图片导出完成，或点击“取消导出”。")
            return
        if not self._confirm_replace_task():
            return
        self._reset_task_state()

    def choose_folder(self) -> None:
        if self.__dict__.get("export_running", False):
            return
        if sys.platform == "darwin":
            selected = self._choose_folders_macos(
                self.last_import_directory,
                self.t["pick_folder"],
                self.language,
            )
        else:
            value = filedialog.askdirectory(
                title=self.t["pick_folder"],
                initialdir=str(self.last_import_directory),
            )
            selected = [Path(value)] if value else []
        if not selected:
            return
        self.last_import_directory = selected[0].parent
        self._load_selected_files(self._files_in_folders(selected))

    def import_calibration(self) -> None:
        """Learn from final-kept filenames without writing either source."""
        if self.analysis_running or self.calibration_running:
            return
        # Current tasks already have an automatic JSON log and known original
        # folders. The normal workflow therefore asks for only the final picks.
        if self.current_log_path is not None and self.current_log_path.is_file() and self.imported_files:
            exported_value = filedialog.askdirectory(
                title=self.t["calibration_export"],
                initialdir=str(self.last_import_directory),
            )
            if not exported_value:
                return
            sources = sorted({path.parent for path in self.imported_files})
            self._start_calibration(self.current_log_path, Path(exported_value), sources)
            return

        # Historical tasks still accept the old portable CSV export as well as
        # the automatically saved JSON log.
        log_value = filedialog.askopenfilename(
            title=self.t["calibration_log"],
            initialdir=str(self.last_import_directory),
            filetypes=[
                ("Culling logs", "*.csv *.json"),
                ("CSV", "*.csv"),
                ("Automatic log", "*.json"),
            ],
        )
        if not log_value:
            return
        exported_value = filedialog.askdirectory(
            title=self.t["calibration_export"],
            initialdir=str(Path(log_value).parent),
        )
        if not exported_value:
            return
        if sys.platform == "darwin":
            sources = self._choose_folders_macos(
                Path(log_value).parent,
                self.t["calibration_sources"],
                self.language,
            )
        else:
            value = filedialog.askdirectory(
                title=self.t["calibration_sources"], initialdir=str(Path(log_value).parent)
            )
            sources = [Path(value)] if value else []
        if not sources:
            return
        self._start_calibration(Path(log_value), Path(exported_value), sources)

    def _start_calibration(
        self,
        log_path: Path,
        exported_folder: Path,
        sources: list[Path],
    ) -> None:
        self.calibration_running = True
        self.status.set(self.t["calibration_running"])
        self.calibration_button.configure(state="disabled")
        self.new_task_button.configure(state="disabled")
        self.choose_files_button.configure(state="disabled")
        self.choose_button.configure(state="disabled")
        self.picker_settings_button.configure(state="disabled")
        self.analyze_button.configure(state="disabled")
        self.progress.configure(mode="indeterminate", maximum=100, value=0)
        self.progress.start(12)
        self.progress_text.set(self.t["calibration_running"])
        threading.Thread(
            target=self._build_calibration,
            args=(log_path, exported_folder, sources),
            daemon=True,
        ).start()

    def _build_calibration(self, log_path: Path, exported_folder: Path, sources: list[Path]) -> None:
        try:
            def on_progress(current: int, total: int, path: Path) -> None:
                self.ui_events.put(("calibration_progress", (current, total, path)))

            profile = build_profile(log_path, exported_folder, sources, progress_callback=on_progress)
            save_profile(profile)
            self.ui_events.put(("calibration_completed", profile))
        except CalibrationError as exc:
            self.ui_events.put(("calibration_failed", str(exc)))
        except Exception as exc:
            self.ui_events.put(("calibration_failed", str(exc)))

    def _finish_calibration(self) -> None:
        self.calibration_running = False
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=1, value=0)
        self.new_task_button.configure(state="normal")
        self.calibration_button.configure(state="normal")
        self._refresh_picker_confirmation_ui()
        self._refresh_calibration_button()

    def _load_selected_files(self, files: list[Path]) -> bool:
        selected = [path for path in files if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not selected:
            messagebox.showerror(self.t["pick_needed"], self.t["drop_error"])
            return False
        merged, added_count = self._merge_unique_paths(self.imported_files, selected)
        if added_count == 0:
            self.status.set(self.t["already_added"])
            self._render_imported_files_table(self.imported_files)
            self._show_import_preview(self.imported_files)
            return False
        self._clear_results_for_source_change()
        added_paths = merged[len(self.imported_files):]
        self.imported_files = merged
        self._refresh_legacy_heic_repair_button()
        self.picker_settings_confirmed = False
        self._ensure_task_identity(merged)
        self.set_workflow_stage(2)
        source_counts = self._source_file_counts(merged)
        added_counts = self._source_file_counts(added_paths)
        self.folder.set(self._source_count_text(source_counts))
        self.status.set(self.t["append_status"])
        self._set_default_target(merged)
        self.import_notice.set(self._source_count_text(source_counts, imported=True))
        self.task_import_summary.set(
            self._task_import_summary_text(added_counts, source_counts)
        )
        self.import_badge.grid()
        self._render_imported_files_table(merged)
        self._start_preview_warmup(merged)
        self._show_import_preview(merged)
        self._refresh_picker_confirmation_ui()
        self._write_task_history("已导入" if self.language == "zh" else "Imported")
        return True

    def _ensure_task_identity(self, files: list[Path]) -> None:
        """Give an imported task a stable, human-readable local identity."""
        if self.current_task_id is None:
            created = datetime.now()
            self.current_task_created_at = created.isoformat(timespec="seconds")
        else:
            try:
                created = datetime.fromisoformat(str(self.current_task_created_at))
            except ValueError:
                created = datetime.now()
        self.current_task_name = history_task_name(files, created)
        if self.current_task_id is None:
            identity = f"{self.current_task_created_at}|{self.current_task_name}"
            self.current_task_id = (
                f"{created:%Y%m%d-%H%M%S}-"
                f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:8]}"
            )
        self.task_title.set(self.current_task_name)

    def _task_history_record(self, status: str) -> dict[str, object]:
        counts = {
            str(rating): sum(item.suggested_rating == rating for item in self.results)
            for rating in range(4)
        }
        selected_total = int(counts["1"]) + int(counts["2"]) + int(counts["3"])
        return {
            "id": self.current_task_id,
            "name": self.current_task_name,
            "created_at": self.current_task_created_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "source_folders": sorted({str(path.parent) for path in self.imported_files}),
            "source_file_count": len(self.imported_files),
            "logical_photo_count": len(self._logical_import_rows(self.imported_files)),
            "group_count": len({item.group_id for item in self.results if item.group_id is not None}),
            "counts": counts,
            "selected_total": selected_total,
            "log_path": str(self.current_log_path) if self.current_log_path else None,
        }

    def _write_task_history(self, status: str) -> None:
        if self.current_task_id is None or not self.imported_files:
            return
        try:
            upsert_task(self._task_history_record(status))
        except OSError:
            # History is helpful metadata; a read-only home directory must not
            # prevent normal culling or local review.
            pass

    @staticmethod
    def _logical_import_rows(files: list[Path]) -> list[tuple[Path, list[Path]]]:
        """Return one display row per capture, preferring RAW over a paired JPG."""
        grouped: dict[tuple[Path, str], list[Path]] = {}
        for path in files:
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            grouped.setdefault(
                (path.parent.resolve(), path.stem.casefold()), []
            ).append(path)
        rows: list[tuple[Path, list[Path]]] = []
        for paths in grouped.values():
            ordered = sorted(
                paths,
                key=lambda path: (
                    path.suffix.lower() not in RAW_EXTENSIONS,
                    path.name.casefold(),
                ),
            )
            rows.append((ordered[0], ordered))
        return sorted(
            rows,
            key=lambda row: (str(row[0].parent).casefold(), row[0].name.casefold()),
        )

    def _render_imported_files_table(self, files: list[Path]) -> None:
        """Show accepted source files before the analysis creates ratings."""
        self.table.delete(*self.table.get_children())
        for index, (primary, companions) in enumerate(
            self._logical_import_rows(files), start=1
        ):
            paired_jpg = any(
                path.suffix.lower() in JPEG_EXTENSIONS for path in companions
            )
            paired_raw = any(
                path.suffix.lower() in RAW_EXTENSIONS for path in companions
            )
            if self.language == "zh":
                note = (
                    "已导入 · RAW + JPG，等待开始分析"
                    if paired_raw and paired_jpg
                    else "已导入，等待开始分析"
                )
            else:
                note = (
                    "Imported · RAW + JPG · waiting to analyse"
                    if paired_raw and paired_jpg
                    else "Imported · waiting to analyse"
                )
            self.table.insert(
                "",
                "end",
                iid=f"import-{index}",
                values=("", "—", primary.name, note),
            )
        self.grid_visible_limit = 72
        self._schedule_grid_render()

    def _show_import_preview(self, files: list[Path]) -> None:
        """Make an accepted import visible before analysis populates the results table."""
        source_counts = self._source_photo_counts(files)
        preview_path = next(
            (path for path in files if path.suffix.lower() in JPEG_EXTENSIONS),
            None,
        )
        if preview_path is None:
            for path in files:
                if path.suffix.lower() not in RAW_EXTENSIONS:
                    continue
                preview_path = next(
                    (
                        sibling
                        for sibling in sorted(path.parent.glob(f"{path.stem}.*"))
                        if sibling.is_file()
                        and sibling.suffix.lower() in JPEG_EXTENSIONS
                    ),
                    None,
                )
                if preview_path is not None:
                    break
        if preview_path is None:
            preview_path = next(
                (path for path in files if path.suffix.lower() in RAW_EXTENSIONS),
                None,
            )
        # RAW rendering can take seconds for a camera without an embedded
        # thumbnail.  The warmup worker will notify the UI once its 320px
        # cache entry exists; never hold the import UI hostage to that work.
        if preview_path is not None and self._preview_is_cached(preview_path):
            if self._set_preview_from_path(preview_path):
                return
        if preview_path is None:
            self.preview = None
            self.preview_label.configure(
                image="",
                text=self.t["empty_preview"],
                style="DropHint.TLabel",
            )
            return
        message_key = "raw_import_ready" if source_counts["raw"] else "import_ready"
        self.preview = None
        self.preview_label.configure(
            image="",
            text=self.t[message_key].format(**source_counts),
            style="DropHint.TLabel",
        )

    def _preview_is_cached(self, path: Path) -> bool:
        key = path.resolve()
        with self._preview_lock:
            if key in self._preview_memory:
                return True
        try:
            return self._preview_cache_path(path).is_file()
        except OSError:
            return False

    def _set_preview_from_path(self, path: Path) -> bool:
        """Show a cached 320px preview without blocking on a full RAW decode."""
        try:
            image = self._preview_image(path)
            self._active_preview_path = path
            self.preview = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview, text="")
            if self._preview_zoom_is_open():
                self._set_preview_zoom_path(path, fallback=image)
            return True
        except Exception:
            return False

    @staticmethod
    def _clamp_preview_zoom(scale: float) -> float:
        """Keep a magnified preview useful without allocating huge Tk images."""
        return max(0.10, min(4.0, scale))

    @staticmethod
    def _preview_fit_scale(
        image_size: tuple[int, int], viewport_size: tuple[int, int]
    ) -> float:
        """Return the scale that shows a complete image in its viewport."""
        image_width, image_height = image_size
        viewport_width, viewport_height = viewport_size
        if min(image_width, image_height, viewport_width, viewport_height) <= 0:
            return 1.0
        return EventCullApp._clamp_preview_zoom(
            min(viewport_width / image_width, viewport_height / image_height)
        )

    def _preview_zoom_is_open(self) -> bool:
        window = self._preview_zoom_window
        try:
            return window is not None and bool(window.winfo_exists())
        except tk.TclError:
            return False

    def _open_preview_zoom(self, _event=None) -> str:
        """Open the selected photo in a separate, high-resolution review view."""
        path = self._active_preview_path
        if path is None:
            return "break"
        if not self._preview_zoom_is_open():
            window = tk.Toplevel(self)
            self._preview_zoom_window = window
            window.title("放大预览")
            window.configure(bg="#111214")
            window.minsize(520, 420)
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            width = min(1120, max(620, screen_width - 220))
            height = min(860, max(480, screen_height - 180))
            x = max(24, (screen_width - width) // 2)
            y = max(32, (screen_height - height) // 2)
            window.geometry(f"{width}x{height}+{x}+{y}")
            window.protocol("WM_DELETE_WINDOW", self._close_preview_zoom)
            window.bind("<Escape>", lambda _event: self._close_preview_zoom())

            toolbar = tk.Frame(window, bg="#1b1d1f", padx=14, pady=10)
            toolbar.pack(fill="x")
            self.preview_zoom_name = tk.Label(
                toolbar,
                bg="#1b1d1f",
                fg="#f1f0ec",
                anchor="w",
                font=("Helvetica Neue", 12, "bold"),
            )
            self.preview_zoom_name.pack(side="left", fill="x", expand=True)
            self.preview_zoom_percent = tk.Label(
                toolbar,
                text="",
                bg="#1b1d1f",
                fg="#b7b3aa",
                font=("Helvetica Neue", 10),
            )
            self.preview_zoom_percent.pack(side="left", padx=(10, 8))
            for text, command, tooltip in (
                ("−", lambda: self._change_preview_zoom(1 / 1.25), "缩小"),
                ("适应", self._fit_preview_zoom, "适应窗口"),
                ("100%", self._actual_preview_zoom, "按原始预览大小显示"),
                ("＋", lambda: self._change_preview_zoom(1.25), "放大"),
            ):
                button = ttk.Button(toolbar, text=text, command=command, style="ViewMode.TButton")
                button.pack(side="left", padx=(0, 4))
                ToolTip(button, lambda copy=tooltip: copy)

            canvas = tk.Canvas(
                window,
                bg="#101112",
                bd=0,
                highlightthickness=0,
                cursor="fleur",
            )
            canvas.pack(fill="both", expand=True)
            self._preview_zoom_canvas = canvas
            canvas.bind("<Configure>", self._on_preview_zoom_canvas_resize, add="+")
            canvas.bind("<ButtonPress-1>", lambda event: canvas.scan_mark(event.x, event.y))
            canvas.bind("<B1-Motion>", lambda event: canvas.scan_dragto(event.x, event.y, gain=1))
            canvas.bind("<Double-Button-1>", self._fit_preview_zoom)
            canvas.bind("<MouseWheel>", self._zoom_preview_from_wheel)
            canvas.bind("<Button-4>", self._zoom_preview_from_wheel)
            canvas.bind("<Button-5>", self._zoom_preview_from_wheel)
        else:
            self._preview_zoom_window.deiconify()
            self._preview_zoom_window.lift()
            self._preview_zoom_window.focus_force()

        fallback = None
        try:
            fallback = self._preview_image(path)
        except Exception:
            pass
        self._set_preview_zoom_path(path, fallback=fallback)
        return "break"

    def _close_preview_zoom(self) -> str:
        window = self._preview_zoom_window
        self._preview_zoom_window = None
        self._preview_zoom_canvas = None
        self._preview_zoom_source = None
        self._preview_zoom_photo = None
        self._preview_zoom_path = None
        self._preview_zoom_generation += 1
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass
        return "break"

    def _set_preview_zoom_path(self, path: Path, *, fallback: Image.Image | None) -> None:
        """Display an immediate lightweight frame while the sharper preview loads."""
        if not self._preview_zoom_is_open():
            return
        self._preview_zoom_path = path
        self._preview_zoom_generation += 1
        generation = self._preview_zoom_generation
        self._preview_zoom_fit = True
        if fallback is not None:
            self._preview_zoom_source = fallback.copy()
            self._render_preview_zoom()
        self.preview_zoom_name.configure(text=path.name)
        try:
            key = path.resolve()
        except OSError:
            key = path
        with self._preview_lock:
            cached = self._zoom_preview_memory.get(key)
            if cached is not None:
                self._zoom_preview_memory.move_to_end(key)
                image = cached.copy()
            else:
                image = None
        if image is not None:
            self._receive_zoom_preview(generation, path, image)
            return

        def prepare_high_resolution_preview() -> None:
            try:
                image = self._zoom_preview_image(path)
            except Exception:
                return
            self.ui_events.put(("zoom_preview_ready", (generation, path, image)))

        threading.Thread(
            target=prepare_high_resolution_preview,
            name="event-cull-zoom-preview",
            daemon=True,
        ).start()

    def _remember_zoom_preview(self, path: Path, image: Image.Image) -> None:
        try:
            key = path.resolve()
        except OSError:
            key = path
        with self._preview_lock:
            self._zoom_preview_memory[key] = image.copy()
            self._zoom_preview_memory.move_to_end(key)
            while len(self._zoom_preview_memory) > ZOOM_PREVIEW_MEMORY_CACHE_LIMIT:
                self._zoom_preview_memory.popitem(last=False)

    def _zoom_preview_image(self, path: Path) -> Image.Image:
        """Read one larger, disposable preview off the UI thread for close review."""
        try:
            key = path.resolve()
        except OSError:
            key = path
        with self._preview_lock:
            cached = self._zoom_preview_memory.get(key)
            if cached is not None:
                self._zoom_preview_memory.move_to_end(key)
                return cached.copy()
        if path.suffix.lower() not in RAW_EXTENSIONS | HEIC_EXTENSIONS:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB").copy()
        else:
            frame = _load_preview(path, maximum_dimension=ZOOM_PREVIEW_MAX_DIMENSION)
            if frame is None or frame.size == 0:
                raise RuntimeError("No high-resolution display preview was generated")
            image = Image.fromarray(frame[:, :, ::-1].copy())
        image.thumbnail(
            (ZOOM_PREVIEW_MAX_DIMENSION, ZOOM_PREVIEW_MAX_DIMENSION),
            Image.Resampling.LANCZOS,
        )
        self._remember_zoom_preview(path, image)
        return image.copy()

    def _receive_zoom_preview(
        self, generation: int, path: Path, image: Image.Image
    ) -> None:
        if (
            generation != self._preview_zoom_generation
            or path != self._preview_zoom_path
            or not self._preview_zoom_is_open()
        ):
            return
        self._preview_zoom_source = image.copy()
        self._render_preview_zoom()

    def _on_preview_zoom_canvas_resize(self, _event=None) -> None:
        if self._preview_zoom_fit:
            self._render_preview_zoom()

    def _change_preview_zoom(self, multiplier: float) -> None:
        if self._preview_zoom_source is None:
            return
        current = self._preview_effective_scale()
        self._preview_zoom_scale = self._clamp_preview_zoom(current * multiplier)
        self._preview_zoom_fit = False
        self._render_preview_zoom()

    def _actual_preview_zoom(self) -> None:
        if self._preview_zoom_source is None:
            return
        self._preview_zoom_scale = 1.0
        self._preview_zoom_fit = False
        self._render_preview_zoom()

    def _fit_preview_zoom(self, _event=None) -> str:
        self._preview_zoom_fit = True
        self._render_preview_zoom()
        return "break"

    def _zoom_preview_from_wheel(self, event: tk.Event) -> str:
        delta = int(getattr(event, "delta", 0) or 0)
        if delta == 0 and getattr(event, "num", None) in {4, 5}:
            delta = 1 if event.num == 4 else -1
        if delta:
            self._change_preview_zoom(1.15 if delta > 0 else 1 / 1.15)
        return "break"

    def _preview_effective_scale(self) -> float:
        source = self._preview_zoom_source
        canvas = self._preview_zoom_canvas
        if source is None or canvas is None:
            return 1.0
        if self._preview_zoom_fit:
            return self._preview_fit_scale(
                source.size, (canvas.winfo_width(), canvas.winfo_height())
            )
        return self._preview_zoom_scale

    def _render_preview_zoom(self) -> None:
        canvas = self._preview_zoom_canvas
        source = self._preview_zoom_source
        if canvas is None or source is None:
            return
        try:
            viewport_width = max(1, canvas.winfo_width())
            viewport_height = max(1, canvas.winfo_height())
            scale = self._preview_effective_scale()
            rendered_width = max(1, int(round(source.width * scale)))
            rendered_height = max(1, int(round(source.height * scale)))
            rendered = source.resize(
                (rendered_width, rendered_height), Image.Resampling.LANCZOS
            )
            self._preview_zoom_photo = ImageTk.PhotoImage(rendered)
            canvas.delete("preview")
            offset_x = max(0, (viewport_width - rendered_width) // 2)
            offset_y = max(0, (viewport_height - rendered_height) // 2)
            canvas.create_image(
                offset_x, offset_y, anchor="nw", image=self._preview_zoom_photo,
                tags="preview",
            )
            canvas.configure(
                scrollregion=(
                    0,
                    0,
                    max(viewport_width, rendered_width),
                    max(viewport_height, rendered_height),
                )
            )
            if rendered_width > viewport_width or rendered_height > viewport_height:
                canvas.xview_moveto(0.5)
                canvas.yview_moveto(0.5)
            else:
                canvas.xview_moveto(0)
                canvas.yview_moveto(0)
            if hasattr(self, "preview_zoom_percent"):
                self.preview_zoom_percent.configure(text=f"{round(scale * 100)}%")
        except tk.TclError:
            return

    @staticmethod
    def _preview_cache_path(path: Path) -> Path:
        stat = path.stat()
        identity = f"preview-v1|{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return application_data_dir() / "preview-cache" / f"{digest}.jpg"

    def _remember_preview(self, path: Path, image: Image.Image) -> None:
        key = path.resolve()
        with self._preview_lock:
            self._preview_memory[key] = image.copy()
            self._preview_memory.move_to_end(key)
            while len(self._preview_memory) > MEMORY_PREVIEW_CACHE_LIMIT:
                self._preview_memory.popitem(last=False)

    def _preview_image(self, path: Path) -> Image.Image:
        """Load one small preview from memory, disk, or a read-only RAW decode."""
        key = path.resolve()
        with self._preview_lock:
            cached = self._preview_memory.get(key)
            if cached is not None:
                self._preview_memory.move_to_end(key)
                return cached.copy()

        destination = self._preview_cache_path(path)
        try:
            with Image.open(destination) as cached_file:
                image = cached_file.convert("RGB").copy()
        except (OSError, ValueError):
            frame = _load_preview(path, maximum_dimension=DISPLAY_PREVIEW_MAX_DIMENSION)
            if frame is None or frame.size == 0:
                raise RuntimeError("No display preview was generated")
            image = Image.fromarray(frame[:, :, ::-1].copy())
            image.thumbnail((DISPLAY_PREVIEW_MAX_DIMENSION, DISPLAY_PREVIEW_MAX_DIMENSION))
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(
                    f"{destination.stem}-{threading.get_ident()}.tmp"
                )
                image.save(temporary, format="JPEG", quality=82, optimize=True)
                os.replace(temporary, destination)
            except OSError:
                # Disk cache failures (for example a full drive) must not stop
                # a photo from being reviewed during this session.
                pass
        self._remember_preview(path, image)
        return image.copy()

    @staticmethod
    def _preview_source(primary: Path, companions: list[Path]) -> Path:
        """A paired camera JPG is fastest; otherwise use the logical RAW."""
        return next(
            (path for path in companions if path.suffix.lower() in JPEG_EXTENSIONS),
            primary,
        )

    def _start_preview_warmup(self, files: list[Path]) -> None:
        """Build all small previews in one low-impact background queue on import."""
        self._preview_generation += 1
        generation = self._preview_generation
        sources = [
            self._preview_source(primary, companions)
            for primary, companions in self._logical_import_rows(files)
        ]

        def warm() -> None:
            for index, source in enumerate(sources):
                if generation != self._preview_generation:
                    return
                try:
                    self._preview_image(source)
                    if index == 0 and generation == self._preview_generation:
                        self.ui_events.put(("preview_ready", (generation, source)))
                except Exception:
                    continue

        threading.Thread(target=warm, name="event-cull-preview-warmup", daemon=True).start()

    def _clear_results_for_source_change(self) -> None:
        """Invalidate stale analysis while preserving the accumulating source list."""
        self.results = []
        self.selected = None
        self.active_result_index = None
        self.active_import_path = None
        self.grid_visible_limit = 72
        self.analysis_completed_at = None
        self.current_log_path = None
        self.preview = None
        self._active_preview_path = None
        self._close_preview_zoom()
        self._preview_generation += 1
        with self._preview_lock:
            self._preview_memory.clear()
            self._zoom_preview_memory.clear()
        self._grid_thumbnail_cache.clear()
        self.table.delete(*self.table.get_children())
        self.preview_label.configure(image="", text=self.t["empty_preview"])
        self.detail.configure(text=self.t["empty_detail"])
        self.stats_text.set(self.t["stats_empty"])
        self.rating_filter.set(self.t["filter_all"])
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=1, value=0)
        self.progress_text.set("")
        self.eta_text.set("")
        self.write_button.configure(state="disabled")
        self.restore_button.configure(state="disabled")
        self.export_selected_button.configure(state="disabled")
        self._schedule_grid_render()

    def choose_files(self) -> None:
        if self.__dict__.get("export_running", False):
            return
        selected = filedialog.askopenfilenames(
            title=self.t["pick_files"],
            initialdir=str(self.last_import_directory),
            filetypes=[
                ("照片文件", "*.ARW *.arw *.CR2 *.cr2 *.CR3 *.cr3 *.DNG *.dng *.NEF *.nef *.ORF *.orf *.RAF *.raf *.RW2 *.rw2 *.JPG *.jpg *.JPEG *.jpeg *.PNG *.png *.TIF *.tif *.TIFF *.tiff *.HEIC *.heic *.HEIF *.heif"),
                ("所有文件", "*"),
            ],
        )
        if selected:
            paths = [Path(value) for value in selected]
            self.last_import_directory = paths[0].parent
            self._load_selected_files(paths)

    def accept_drop(self, event) -> str:
        if self.analysis_running or self.calibration_running or self.__dict__.get("export_running", False):
            messagebox.showinfo(self.t["drop_busy_title"], self.t["drop_busy_detail"])
            return REFUSE_DROP
        paths = [Path(value) for value in self.tk.splitlist(event.data)]
        folders = [path for path in paths if path.is_dir()]
        files = [path for path in paths if path.is_file()]
        if folders:
            self.last_import_directory = folders[0].parent
            self._load_selected_files(self._files_in_folders(folders) + files)
        elif files:
            self.last_import_directory = files[0].parent
            self._load_selected_files(files)
        else:
            messagebox.showerror(self.t["pick_needed"], self.t["drop_error"])
            return REFUSE_DROP
        return COPY

    def start_analysis(self) -> None:
        if self.__dict__.get("export_running", False):
            messagebox.showinfo("正在导出", "请等待已选图片导出完成，或点击“取消导出”。")
            return
        if not self.imported_files:
            messagebox.showerror(self.t["pick_needed"], self.t["pick_error"])
            return
        if not self.picker_settings_confirmed:
            self.show_picker_settings()
            return
        if not self.license_controller.decision.allows_analysis:
            messagebox.showinfo(
                "需要授权",
                self.license_controller.decision.message,
            )
            self.show_license_dialog()
            return
        try:
            picks_min = int(self.picks.get())
            picks_max = int(self.picks_max.get())
            duplicate_keep_limit = self._duplicate_keep_limit()
            target_min, target_max = self._target_selection_range()
            if (
                picks_min < 1
                or picks_max < picks_min
            ):
                raise ValueError("invalid selection range")
            options = self._analysis_options_from_picker(
                picks_min=picks_min,
                picks_max=picks_max,
                target_min=target_min,
                target_max=target_max,
                duplicate_keep_limit=duplicate_keep_limit,
            )
        except ValueError:
            messagebox.showerror(self.t["bad_settings"], self.t["bad_settings_detail"])
            return
        self.status.set(self.t["analyzing"])
        self.xmp_notice.set(self.lr_instructions)
        self._set_xmp_result()
        self.pause_event.clear()
        self.cancel_event.clear()
        self.is_paused = False
        self.analysis_running = True
        self.set_workflow_stage(3)
        self.analysis_started_at = time.monotonic()
        self.analysis_pause_started_at = None
        self.analysis_paused_seconds = 0.0
        self.analysis_completed_at = None
        self.current_log_path = None
        self.eta_text.set(self.t["eta_estimating"])
        self.analysis_progress_current = 0
        self.analysis_progress_total = len(self.imported_files)
        self._refresh_analysis_header_status()
        self.analyze_button.configure(state="disabled")
        self.picker_settings_button.configure(state="disabled")
        self.new_task_button.configure(state="disabled")
        self.choose_files_button.configure(state="disabled")
        self.choose_button.configure(state="disabled")
        self.calibration_button.configure(state="disabled")
        self._set_analysis_controls_visible(True)
        self.pause_button.configure(image=self.pause_icon, state="normal")
        self.stop_button.configure(state="normal")
        self.progress.configure(mode="indeterminate", maximum=100, value=0)
        self.progress.start(12)
        self.progress_text.set(self.t["scanning"])
        self.write_button.configure(state="disabled")
        self.export_selected_button.configure(state="disabled")
        source: list[Path] = self.imported_files[:]
        preserved_manual_ratings = (
            dict(self.manual_ratings) if self.keep_existing_labels.get() else {}
        )
        threading.Thread(
            target=self._analyze,
            args=(source, options, preserved_manual_ratings),
            daemon=True,
        ).start()

    def toggle_pause(self) -> None:
        if self.is_paused:
            if self.analysis_pause_started_at is not None:
                self.analysis_paused_seconds += time.monotonic() - self.analysis_pause_started_at
            self.analysis_pause_started_at = None
            self.pause_event.clear()
            self.is_paused = False
            self.pause_button.configure(image=self.pause_icon)
            self.status.set(self.t["analyzing"])
            self._refresh_analysis_header_status()
        else:
            self.pause_event.set()
            self.is_paused = True
            self.analysis_pause_started_at = time.monotonic()
            self.pause_button.configure(image=self.resume_icon)
            self.status.set(self.t["paused"])
            self.eta_text.set(self.t["eta_paused"])
            if self.language == "zh":
                self.task_import_summary.set(
                    f"AI 分析已暂停 · {self.analysis_progress_current} / {self.analysis_progress_total}"
                )
            else:
                self.task_import_summary.set(
                    f"AI analysis paused · {self.analysis_progress_current} / {self.analysis_progress_total}"
                )

    def stop_analysis(self) -> None:
        self.cancel_event.set()
        self.pause_event.clear()
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status.set(self.t["stopping"])
        if self.language == "zh":
            self.task_import_summary.set(
                f"正在停止分析 · {self.analysis_progress_current} / {self.analysis_progress_total}"
            )
        else:
            self.task_import_summary.set(
                f"Stopping analysis · {self.analysis_progress_current} / {self.analysis_progress_total}"
            )

    @staticmethod
    def _restore_manual_ratings(
        results: list[PhotoResult], ratings: dict[Path, int], language: str
    ) -> None:
        """Restore only photographer-set ratings after a re-analysis."""
        if not ratings:
            return
        reason = "保留手动评级" if language == "zh" else "Manual rating preserved"
        for item in results:
            if item.path in ratings:
                item.suggested_rating = ratings[item.path]
                if reason not in item.reasons:
                    item.reasons.append(reason)

    def _analyze(
        self,
        source: Path | list[Path],
        options: AnalysisOptions,
        preserved_manual_ratings: dict[Path, int] | None = None,
    ) -> None:
        try:
            def on_progress(current: int, total: int, path: Path) -> None:
                # Tk must only be called from its main thread on macOS.
                self.ui_events.put(("progress", (current, total, path)))
            if isinstance(source, list):
                results = analyze_paths(source, options, progress_callback=on_progress, pause_event=self.pause_event, cancel_event=self.cancel_event)
            else:
                results = analyze_folder(source, options, progress_callback=on_progress, pause_event=self.pause_event, cancel_event=self.cancel_event)
            self._restore_manual_ratings(
                results, preserved_manual_ratings or {}, self.language
            )
            if self.calibration_profile is not None:
                try:
                    def on_calibration_progress(current: int, total: int, path: Path) -> None:
                        self.ui_events.put(("calibration_apply_progress", (current, total, path)))

                    apply_profile(results, self.calibration_profile, progress_callback=on_calibration_progress)
                except CalibrationError as exc:
                    self.ui_events.put(("calibration_apply_skipped", str(exc)))
            self.ui_events.put(("completed", results))
        except AnalysisCancelled:
            self.ui_events.put(("cancelled", None))
        except Exception as exc:
            self.ui_events.put(("error", str(exc)))

    def _check_for_updates_in_background(self) -> None:
        if self._update_check_started:
            return
        self._update_check_started = True

        def check() -> None:
            update = check_for_update(APP_VERSION)
            if update is not None:
                self.ui_events.put(("update_available", update))

        threading.Thread(target=check, name="miaoxuan-update-check", daemon=True).start()

    def _offer_update(self, update: UpdateInfo) -> None:
        notes = f"\n\n更新内容：\n{update.notes}" if update.notes else ""
        accepted = messagebox.askyesno(
            "发现新版本",
            f"当前版本 v{APP_VERSION}，可更新至 v{update.version}。{notes}\n\n现在下载更新包吗？",
        )
        if not accepted:
            return
        self.status.set(f"正在下载 v{update.version} 更新包并校验…")

        def download() -> None:
            try:
                self.ui_events.put(("update_downloaded", download_update(update)))
            except Exception as exc:
                self.ui_events.put(("update_download_failed", str(exc)))

        threading.Thread(target=download, name="miaoxuan-update-download", daemon=True).start()

    def _show_downloaded_update(self, path: Path) -> None:
        self.status.set(f"更新包已下载并校验：{path.name}")
        messagebox.showinfo(
            "更新包已准备好",
            f"已完成 SHA-256 校验。请在“下载”文件夹中打开并安装：\n{path.name}",
        )

    def process_ui_events(self) -> None:
        """Run all UI mutations in the Tk main thread."""
        try:
            while True:
                event, payload = self.ui_events.get_nowait()
                if event == "progress":
                    current, total, path = payload
                    self.update_progress(current, total, path)
                elif event == "calibration_progress":
                    current, total, path = payload
                    self.update_calibration_progress(current, total, path, apply=False)
                elif event == "calibration_apply_progress":
                    current, total, path = payload
                    self.update_calibration_progress(current, total, path, apply=True)
                elif event == "calibration_completed":
                    self.calibration_profile = payload
                    self._finish_calibration()
                    message = self.t["calibration_done"].format(
                        samples=payload.training_examples,
                        positive=payload.positive_examples,
                        auc=payload.validation_auc,
                    )
                    self.status.set(message)
                    messagebox.showinfo(self.t["calibration"], message)
                elif event == "calibration_failed":
                    self._finish_calibration()
                    message = self.t["calibration_failed"].format(error=payload)
                    self.status.set(message)
                    messagebox.showerror(self.t["calibration"], message)
                elif event == "calibration_apply_skipped":
                    self.status.set(self.t["calibration_apply_skipped"].format(error=payload))
                elif event == "preview_ready":
                    generation, path = payload
                    if (
                        generation == self._preview_generation
                        and not self.results
                        and path in self.imported_files
                    ):
                        self._set_preview_from_path(path)
                elif event == "import_selection_preview_ready":
                    generation, path, source = payload
                    if (
                        generation == self._preview_generation
                        and not self.results
                        and path == self.active_import_path
                    ):
                        self._set_preview_from_path(source)
                elif event == "zoom_preview_ready":
                    generation, path, image = payload
                    self._receive_zoom_preview(generation, path, image)
                elif event == "update_available":
                    self._offer_update(payload)
                elif event == "update_downloaded":
                    self._show_downloaded_update(payload)
                elif event == "update_download_failed":
                    self.status.set("更新下载失败；当前版本仍可正常使用。")
                    messagebox.showerror("更新下载失败", str(payload))
                elif event == "selected_export_progress":
                    current, total, path = payload
                    self.progress.configure(mode="determinate", maximum=total, value=current)
                    self.progress_text.set(f"正在导出已选图片 {current}/{total} · {path.name}")
                elif event == "selected_export_completed":
                    self._finish_selected_export()
                    folders = "\n".join(str(folder) for folder in payload.folders)
                    message = f"已复制 {payload.photo_count} 张已选照片，共 {payload.file_count} 个原文件/旁车文件。"
                    self.status.set(message)
                    self.progress_text.set("已选图片导出完成")
                    messagebox.showinfo("导出完成", f"{message}\n\n保存位置：\n{folders}")
                elif event == "selected_export_cancelled":
                    self._finish_selected_export()
                    self.status.set("已取消导出；未更改原片，未保留未完成的导出文件夹。")
                    self.progress_text.set("已取消导出")
                elif event == "selected_export_failed":
                    self._finish_selected_export()
                    self.status.set("已选图片导出失败；原片未更改。")
                    self.progress_text.set("导出失败")
                    messagebox.showerror("导出已选图片失败", str(payload))
                elif event == "completed":
                    self.analysis_completed_at = datetime.now().isoformat(timespec="seconds")
                    self.show_results(payload)
                    try:
                        self._write_automatic_log()
                    except OSError as exc:
                        self.status.set("自动保存筛选日志失败；本次结果仍在界面中。")
                        messagebox.showwarning("自动日志保存失败", str(exc))
                    self._write_task_history("已完成" if self.language == "zh" else "Completed")
                elif event == "cancelled":
                    self.analysis_cancelled()
                elif event == "error":
                    self.analysis_cancelled()
                    self.status.set(f"{self.t['analysis_failed']}：{payload}")
                    messagebox.showerror(self.t["analysis_failed"], str(payload))
        except queue.Empty:
            pass
        self.after(50, self.process_ui_events)

    def analysis_cancelled(self) -> None:
        self.analysis_running = False
        self.progress.stop()
        self.pause_event.clear()
        self.is_paused = False
        self.new_task_button.configure(state="normal")
        self.calibration_button.configure(state="normal")
        self._refresh_picker_confirmation_ui()
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self._set_analysis_controls_visible(False)
        self.status.set(self.t["cancelled"])
        self.eta_text.set("")
        if self.language == "zh":
            self.task_import_summary.set(
                f"分析已中止 · {self.analysis_progress_current} / {self.analysis_progress_total}"
            )
        else:
            self.task_import_summary.set(
                f"Analysis stopped · {self.analysis_progress_current} / {self.analysis_progress_total}"
            )

    @staticmethod
    def _remaining_seconds(elapsed_seconds: float, current: int, total: int) -> int | None:
        """Estimate remaining work only after enough samples are available."""
        if current >= total and total > 0:
            return 0
        if current < 3 or elapsed_seconds <= 0:
            return None
        return max(0, math.ceil((elapsed_seconds / current) * (total - current)))

    def _update_eta(self, current: int, total: int) -> None:
        if self.is_paused:
            self.eta_text.set(self.t["eta_paused"])
            return
        if self.analysis_started_at is None:
            self.eta_text.set(self.t["eta_estimating"])
            return
        elapsed = max(
            0.0,
            time.monotonic() - self.analysis_started_at - self.analysis_paused_seconds,
        )
        remaining = self._remaining_seconds(elapsed, current, total)
        if remaining is None:
            self.eta_text.set(self.t["eta_estimating"])
            return
        minutes, seconds = divmod(remaining, 60)
        self.eta_text.set(
            self.t["eta_remaining"].format(minutes=minutes, seconds=seconds)
        )

    def update_progress(self, current: int, total: int, path: Path) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=max(total, 1), value=current)
        self.progress_text.set(self.t["progress"].format(current=current, total=total, name=path.name))
        self._update_eta(current, total)
        self.analysis_progress_current = current
        self.analysis_progress_total = total
        self._refresh_analysis_header_status()

    def update_calibration_progress(self, current: int, total: int, path: Path, *, apply: bool) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=max(total, 1), value=current)
        key = "calibration_apply" if apply else "calibration_progress"
        self.progress_text.set(self.t[key].format(current=current, total=total, name=path.name))

    def filter_results(self, _event=None) -> None:
        self._render_results_table()

    @staticmethod
    def _matches_rating_filter(rating: int, selected_rating: int | None) -> bool:
        return selected_rating is None or rating == selected_rating

    @staticmethod
    def _mapped_rating(tier: int, backup: int, featured: int) -> int:
        """Map presentation/XMP ratings without changing the three internal tiers."""
        if tier == 0:
            return 0
        if tier == 1:
            return 1
        if tier == 2:
            return backup
        if tier == 3:
            return featured
        raise ValueError(f"Unknown internal rating tier: {tier}")

    @staticmethod
    def _star_value(value: str) -> int:
        return int(value.rstrip("★"))

    def _output_rating(self, tier: int) -> int:
        return self._mapped_rating(
            tier,
            self._star_value(self.backup_star.get()),
            self._star_value(self.featured_star.get()),
        )

    def _star_mapping(self) -> dict[str, int]:
        return {
            "1": self._output_rating(1),
            "2": self._output_rating(2),
            "3": self._output_rating(3),
        }

    def _on_star_mapping_changed(self, _event=None) -> None:
        backup = self._star_value(self.backup_star.get())
        featured = self._star_value(self.featured_star.get())
        if featured <= backup:
            self.featured_star.set(f"{min(5, backup + 1)}★")
        self._refresh_rating_presentation()
        if self.results:
            self._render_results_table()
            counts = self._set_stats_text()
            completed_prefix = "分析完成：" if self.language == "zh" else "Analysis complete:"
            if self.status.get().startswith(completed_prefix):
                self.status.set(self._completion_status(counts))

    def _refresh_rating_presentation(self) -> None:
        """Refresh only labels/outputs; the culling engine still stores tiers 0–3."""
        backup = self._star_value(self.backup_star.get())
        featured = self._star_value(self.featured_star.get())
        if featured <= backup:
            featured = min(5, backup + 1)
            self.featured_star.set(f"{featured}★")
        self.featured_star_menu.configure(values=tuple(f"{value}★" for value in range(backup + 1, 6)))
        previous_tier = self.rating_filters.get(self.rating_filter.get())
        if self.language == "zh":
            self.legend_text.set(f"1★ 初选复核　{backup}★ 优质备用　{featured}★ 精选推荐　— 未入选")
            rating_hint = (
                f"评级说明：1★ 初选复核（已选，建议人工确认） · "
                f"{backup}★ 优质备用 · {featured}★ 精选推荐"
            )
            filters = {
                self.t["filter_all"]: None,
                self.t["filter_one"]: 1,
                f"仅 {backup}★": 2,
                f"仅 {featured}★": 3,
                self.t["filter_zero"]: 0,
            }
        else:
            self.legend_text.set(f"1★ Review  ·  {backup}★ Strong backup  ·  {featured}★ Best  ·  — Not selected")
            rating_hint = (
                f"Ratings: 1★ review (selected; confirm manually) · "
                f"{backup}★ backup · {featured}★ best"
            )
            filters = {
                self.t["filter_all"]: None,
                self.t["filter_one"]: 1,
                f"{backup}★ only": 2,
                f"{featured}★ only": 3,
                self.t["filter_zero"]: 0,
            }
        self.rating_filters = filters
        self.rating_filter_menu.configure(values=tuple(filters))
        if hasattr(self, "grid_hint"):
            self.grid_hint.configure(text=rating_hint)
        selected_label = next(
            (label for label, tier in filters.items() if tier == previous_tier),
            self.t["filter_all"],
        )
        self.rating_filter.set(selected_label)
        for tier, button in getattr(self, "manual_rating_buttons", {}).items():
            button.configure(text=self.t["clear"] if tier == 0 else f"{self._output_rating(tier)}★")

    def _set_stats_text(self) -> dict[int, int]:
        counts = {rating: sum(item.suggested_rating == rating for item in self.results) for rating in range(4)}
        backup = self._output_rating(2)
        featured = self._output_rating(3)
        if self.language == "zh":
            self.stats_text.set(
                f"1★ 初选复核 {counts[1]}　{backup}★ 优质备用 {counts[2]}　"
                f"{featured}★ 精选推荐 {counts[3]}　已选合计 "
                f"{counts[1] + counts[2] + counts[3]}　未入选 {counts[0]}　共 {len(self.results)} 张"
            )
        else:
            self.stats_text.set(
                f"1★ Review {counts[1]}   {backup}★ Backup {counts[2]}   "
                f"{featured}★ Featured {counts[3]}   Selected "
                f"{counts[1] + counts[2] + counts[3]}   Not selected {counts[0]}   Total {len(self.results)}"
            )
        return counts

    def _completion_status(self, counts: dict[int, int]) -> str:
        selected = counts[1] + counts[2] + counts[3]
        featured = self._output_rating(3)
        if self.language == "zh":
            return (
                f"分析完成：共 {len(self.results)} 张，已选 {selected} 张，"
                f"其中 {featured}★ 精选 {counts[3]} 张；请复核后再写入 XMP。"
            )
        return (
            f"Analysis complete: {len(self.results)} total, {selected} selected, "
            f"including {counts[3]} featured {featured}★ picks. Review before writing XMP."
        )

    def _render_results_table(self) -> None:
        selected_rating = self.rating_filters.get(self.rating_filter.get())
        self.table.delete(*self.table.get_children())
        for index, item in enumerate(self.results):
            if not self._matches_rating_filter(item.suggested_rating, selected_rating):
                continue
            star = "—" if item.suggested_rating == 0 else f"{self._output_rating(item.suggested_rating)}★"
            reasons = "；".join(item.reasons) if item.reasons else self.t["best"]
            self.table.insert(
                "",
                "end",
                iid=str(index),
                values=(item.group_id, star, item.display_name, reasons),
            )
        self.grid_visible_limit = max(72, min(self.grid_visible_limit, max(len(self.results), 72)))
        self._schedule_grid_render()

    def show_results(self, results: list[PhotoResult]) -> None:
        self.analysis_running = False
        self.set_workflow_stage(4)
        self.results = results
        self.selected = None
        self.active_result_index = None
        self.active_import_path = None
        self._render_results_table()
        counts = self._set_stats_text()
        self.progress.stop()
        self.progress.configure(maximum=max(len(results), 1), value=len(results))
        self.eta_text.set(self.t["eta_complete"])
        self.status.set(self._completion_status(counts))
        self.new_task_button.configure(state="normal")
        self.calibration_button.configure(state="normal")
        self._refresh_picker_confirmation_ui()
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self._set_analysis_controls_visible(False)
        selected = counts[1] + counts[2] + counts[3]
        featured = self._output_rating(3)
        if self.language == "zh":
            self.task_import_summary.set(f"已选 {selected} 张 · {featured}★ 精选 {counts[3]} 张")
        else:
            self.task_import_summary.set(f"Selected {selected} · {counts[3]} featured {featured}★")
        self.export_selected_button.configure(state="normal")
        self.refresh_lightroom_guard(schedule_next=False)
        if results:
            self._select_result_index(0)

    @staticmethod
    def _log_record(item: PhotoResult, index: int) -> dict[str, object]:
        return {
            "index": index,
            "folder": str(item.path.parent),
            "file": item.path.name,
            "group": item.group_id,
            "rating": item.suggested_rating,
            "score": round(item.score, 2),
            "frame_sharpness": None if item.sharpness is None else round(item.sharpness, 2),
            "subject_sharpness": None if item.subject_sharpness is None else round(item.subject_sharpness, 2),
            "frontal_faces": item.frontal_face_count,
            "smiling_faces": item.smiling_face_count,
            "open_eye_faces": item.open_eye_face_count,
            "face_coverage": round(item.face_coverage, 4),
            "closest_face_distance": None if item.closest_face_distance is None else round(item.closest_face_distance, 3),
            "eye_detection_mode": item.eye_detection_mode,
            "eye_penalty": round(item.eye_penalty, 2),
            "must_keep_id": item.must_keep_id,
            "median_luminance": None if item.median_luminance is None else round(item.median_luminance, 2),
            "relative_exposure_ev": None if item.relative_exposure_ev is None else round(item.relative_exposure_ev, 2),
            "calibration_confidence": None if item.calibration_confidence is None else round(item.calibration_confidence, 4),
            "scene_softness_protected": item.scene_protected,
            "scene_coverage_protected": item.coverage_protected,
            "reasons": "；".join(item.reasons),
        }

    def _mapped_log_record(self, item: PhotoResult, index: int) -> dict[str, object]:
        record = self._log_record(item, index)
        record["xmp_rating"] = self._output_rating(item.suggested_rating)
        return record

    def _analysis_log_payload(self) -> dict[str, object]:
        counts = {
            str(rating): sum(item.suggested_rating == rating for item in self.results)
            for rating in range(4)
        }
        target_text = self.target.get().strip()
        return {
            "version": 5,
            "selection_policy": "scene-minimum-3plus-dynamic-density-v7-scene-rules",
            "task": {
                "id": self.current_task_id,
                "name": self.current_task_name,
                "created_at": self.current_task_created_at,
            },
            "completed_at": self.analysis_completed_at,
            "settings": {
                "similar_photos_kept": {
                    "min": int(self.picks.get()),
                    "max": int(self.picks_max.get()),
                },
                "target_selection_count": {
                    "min": int(target_text) if target_text else None,
                    "max": int(self.target_max.get()) if self.target_max.get().strip() else None,
                },
                "scene_type": self.scene_type.get(),
            },
            "source_folders": sorted({str(path.parent) for path in self.imported_files}),
            "counts": counts,
            "tier_to_xmp_rating": self._star_mapping(),
            "personal_calibration": None if self.calibration_profile is None else {
                "version": 1,
                "training_examples": self.calibration_profile.training_examples,
                "validation_auc": round(self.calibration_profile.validation_auc, 4),
            },
            "selected_total": int(counts["1"]) + int(counts["2"]) + int(counts["3"]),
            "photos": [
                self._mapped_log_record(item, index)
                for index, item in enumerate(self.results, start=1)
            ],
        }

    def _write_automatic_log(self) -> None:
        if not self.results:
            return
        if self.current_log_path is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_folder = application_data_dir() / "logs"
            log_folder.mkdir(parents=True, exist_ok=True)
            self.current_log_path = log_folder / f"analysis-{stamp}.json"
        temporary = self.current_log_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._analysis_log_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.current_log_path)

    def _choose_export_options(self) -> tuple[frozenset[int], Path | None] | None:
        """Choose rating tiers and destination in one compact dialog."""
        dialog = tk.Toplevel(self)
        dialog.title("导出已选图片")
        dialog.configure(bg=self._workbench_colors["paper"])
        dialog.resizable(False, False)
        dialog.transient(self)
        width, height = 430, 390
        x = max(0, self.winfo_rootx() + (self.winfo_width() - width) // 2)
        y = max(0, self.winfo_rooty() + (self.winfo_height() - height) // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        content = ttk.Frame(dialog, style="Paper.TFrame", padding=22)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="导出已选图片", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(content, text="选择星级（可多选）", style="PanelMuted.TLabel").pack(anchor="w", pady=(5, 12))
        counts = {
            tier: sum(item.suggested_rating == tier for item in self.results)
            for tier in (1, 2, 3)
        }
        choices = {tier: tk.BooleanVar(master=dialog, value=True) for tier in (1, 2, 3)}
        summary = tk.StringVar(master=dialog)

        def update_summary() -> None:
            count = sum(counts[tier] for tier in choices if choices[tier].get())
            summary.set(f"将导出 {count} 张照片" if count else "请至少勾选一种星级")

        for tier, name in ((1, "初选复核"), (2, "优质备用"), (3, "精选推荐")):
            ttk.Checkbutton(
                content,
                text=f"{self._output_rating(tier)}★  {name}  ·  {counts[tier]} 张",
                variable=choices[tier],
                command=update_summary,
                style="Export.TCheckbutton",
            ).pack(anchor="w", fill="x")
        update_summary()
        ttk.Label(content, textvariable=summary, style="Stats.TLabel").pack(anchor="w", pady=(8, 8))
        ttk.Label(content, text="都会新建“秒选”文件夹；不会改动原片。", style="PanelMuted.TLabel").pack(anchor="w", pady=(0, 10))
        chosen: list[tuple[frozenset[int], Path | None]] = []

        def confirm(*, choose_directory: bool) -> None:
            selected = frozenset(tier for tier, value in choices.items() if value.get())
            if not selected:
                update_summary()
                return
            if not any(counts[tier] for tier in selected):
                summary.set("所选星级没有照片，请选择其他星级")
                return
            custom_parent = None
            if choose_directory:
                selected_parent = filedialog.askdirectory(
                    parent=dialog,
                    title="选择保存位置（将新建“秒选”文件夹）",
                    initialdir=str(self.last_import_directory),
                )
                if not selected_parent:
                    return
                custom_parent = Path(selected_parent)
            chosen.append((selected, custom_parent))
            dialog.destroy()

        ttk.Button(
            content, text="导出到原照片所在文件夹", style="Primary.TButton",
            command=lambda: confirm(choose_directory=False),
        ).pack(fill="x", pady=(0, 7))
        ttk.Button(
            content, text="选择指定目录…",
            command=lambda: confirm(choose_directory=True),
        ).pack(fill="x")
        ttk.Button(content, text="取消", command=dialog.destroy).pack(anchor="e", pady=(8, 0))
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.grab_set()
        self.wait_window(dialog)
        return chosen[0] if chosen else None

    def export_selected_photos(self) -> None:
        """Offer rating and destination choices for copied original photos."""
        if self.__dict__.get("export_running", False):
            self.export_cancel_event.set()
            self.status.set("正在取消导出…")
            return
        if not self.results:
            return
        export_options = self._choose_export_options()
        if export_options is None:
            return
        included_tiers, custom_parent = export_options
        try:
            selection = collect_selected_files(self.results, included_tiers)
        except OSError as exc:
            messagebox.showerror("无法准备导出", str(exc))
            return
        if selection.photo_count == 0:
            messagebox.showinfo("没有匹配照片", "当前选择的星级没有照片，请返回选择其他星级。")
            return
        self.export_running = True
        self.export_cancel_event.clear()
        self._export_prior_states = [
            (widget, widget.cget("state"))
            for widget in (
                self.new_task_button, self.choose_button, self.choose_files_button,
                self.analyze_button, self.calibration_button, self.picker_settings_button,
                self.write_button, self.restore_button, self.repair_legacy_heic_button,
                *self.manual_rating_buttons.values(),
            )
        ]
        for widget, _state in self._export_prior_states:
            widget.configure(state="disabled")
        self.export_selected_button.configure(text=self.t["export_cancel"], state="normal")
        self.progress.configure(mode="determinate", maximum=selection.file_count, value=0)
        self.progress_text.set(f"正在准备导出 {selection.photo_count} 张已选照片…")
        self.status.set("正在复制已选原片；原文件保持不变。")

        def copy_in_background() -> None:
            try:
                result = export_selected_files(
                    selection,
                    custom_parent=custom_parent,
                    cancel_event=self.export_cancel_event,
                    progress_callback=lambda current, total, path: self.ui_events.put(
                        ("selected_export_progress", (current, total, path))
                    ),
                )
                self.ui_events.put(("selected_export_completed", result))
            except ExportCancelled:
                self.ui_events.put(("selected_export_cancelled", None))
            except (OSError, ValueError) as exc:
                self.ui_events.put(("selected_export_failed", str(exc)))

        threading.Thread(
            target=copy_in_background,
            name="event-cull-selected-export",
            daemon=True,
        ).start()

    def _finish_selected_export(self) -> None:
        self.export_running = False
        self.export_cancel_event.clear()
        for widget, state in self._export_prior_states:
            widget.configure(state=state)
        self._export_prior_states = []
        self.export_selected_button.configure(
            text=self.t["export_selected"],
            state="normal" if self.results else "disabled",
        )

    def export_log(self) -> None:
        if not self.results:
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        selected = filedialog.asksaveasfilename(
            title=self.t["export_log_title"],
            initialdir=str(self.last_import_directory),
            initialfile=f"筛选日志-{stamp}.csv" if self.language == "zh" else f"culling-log-{stamp}.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not selected:
            return
        destination = Path(selected)
        fieldnames = [
            "index", "folder", "file", "group", "rating", "xmp_rating", "score",
            "frame_sharpness", "subject_sharpness", "frontal_faces",
            "smiling_faces", "open_eye_faces", "face_coverage", "closest_face_distance",
            "eye_detection_mode", "eye_penalty", "must_keep_id", "median_luminance",
            "relative_exposure_ev", "calibration_confidence", "scene_softness_protected", "scene_coverage_protected", "reasons",
        ]
        try:
            with destination.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    self._mapped_log_record(item, index)
                    for index, item in enumerate(self.results, start=1)
                )
            self.last_import_directory = destination.parent
            self.status.set(self.t["export_log_done"].format(path=destination.name))
        except OSError as exc:
            messagebox.showerror(
                self.t["export_log_title"],
                self.t["export_log_failed"].format(error=exc),
            )

    @staticmethod
    def _metadata_editors_running() -> tuple[str, ...]:
        """Return active photo editors that could hold RAW/XMP metadata open.

        Only application main executables are considered. In particular,
        Lightroom's crash reporter must not lock a task after Lightroom itself
        has quit, so framework helpers are deliberately ignored.
        """
        if sys.platform == "win32":
            try:
                process_names = running_executable_names()
            except OSError:
                return ()
            matches = (
                ("lightroom.exe", "Lightroom Classic"),
                ("photoshop.exe", "Photoshop"),
                ("pixelcake.exe", "像素蛋糕"),
            )
            return tuple(label for executable, label in matches if executable in process_names)
        try:
            result = subprocess.run(
                ["ps", "-axo", "comm="],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            commands = str(getattr(result, "stdout", "")).splitlines()
        except (OSError, subprocess.SubprocessError):
            return ()
        labels: list[str] = []
        for command in commands:
            process = command.strip().casefold()
            # Framework plug-ins and crash reporters are not the interactive
            # app process and do not hold a catalog or an XMP sidecar lock.
            if ".app/contents/macos/" not in process:
                continue
            if "adobe lightroom classic.app/contents/macos/" in process:
                label = "Lightroom Classic"
            elif "adobe photoshop" in process:
                label = "Photoshop"
            elif "像素蛋糕" in process or "pixelcake" in process or "pixel cake" in process:
                label = "像素蛋糕"
            else:
                continue
            if label not in labels:
                labels.append(label)
        return tuple(labels)

    def refresh_lightroom_guard(self, schedule_next: bool = True) -> None:
        self.running_metadata_editors = self._metadata_editors_running()
        can_write = bool(self.results) and not self.running_metadata_editors
        state = "normal" if can_write else "disabled"
        for button in (self.write_button, self.restore_button):
            button.configure(state=state)
        if self.running_metadata_editors:
            names = "、".join(self.running_metadata_editors)
            self.lr_guard_label.configure(
                text=(
                    f"⚠ 检测到 {names} 正在运行\n"
                    "为保护 RAW 目录与 XMP，退出这些软件后才能修改 XMP。"
                    if self.language == "zh"
                    else f"⚠ {', '.join(self.running_metadata_editors)} is running\n"
                    "Quit these apps before changing XMP."
                ),
                style="GuardWarning.TLabel",
            )
            if not self.lr_guard_label.winfo_manager():
                self.lr_guard_label.pack(fill="x", pady=(0, 8))
        else:
            self.lr_guard_label.pack_forget()
        if schedule_next:
            self.after(2000, self.refresh_lightroom_guard)

    def ensure_metadata_editors_closed(self) -> bool:
        editors = self._metadata_editors_running()
        if not editors:
            return True
        names = "、".join(editors)
        messagebox.showwarning(
            "请先退出照片编辑软件",
            f"检测到 {names} 正在运行。\n\n"
            "为保护 RAW 目录和侧车元数据，请完全退出这些软件后再写入或还原 XMP。",
        )
        self.refresh_lightroom_guard(schedule_next=False)
        return False

    def show_selected(self, _event=None) -> None:
        selection = self.table.selection()
        if not selection or not self.results:
            return
        try:
            self.selected = self.results[int(selection[0])]
        except (TypeError, ValueError, IndexError):
            return
        item = self.selected
        # Use the paired JPG when present, otherwise the engine's LibRaw /
        # Quick Look chain so RAW files receive a genuine thumbnail too.
        preview_path = next(
            (path for path in item.pair_paths if path.suffix.lower() in {".jpg", ".jpeg"}),
            item.path,
        )
        if not self._set_preview_from_path(preview_path):
            self.preview_label.configure(image="", text=self.t["raw_preview"])
        sharpness = item.subject_sharpness if item.subject_sharpness is not None else (item.sharpness or 0)
        score_label = self.t["score_subject"] if item.subject_sharpness is not None else self.t["score_frame"]
        details = f"{item.path.name}\n{score_label.format(score=item.score, sharpness=sharpness)}\n" + ("\n".join(item.reasons) or self.t["best"])
        self.detail.configure(text=details)

    def set_manual_rating(self, rating: int) -> None:
        if self.export_running:
            return
        if self.selected is None:
            return
        self.selected.suggested_rating = rating
        self.manual_ratings[self.selected.path] = rating
        self.show_results(self.results)
        self.status.set(self.t["not_written"])
        try:
            self._write_automatic_log()
        except OSError as exc:
            self.status.set(f"评级已在界面更新，但自动日志保存失败：{exc}")
        self._write_task_history("已人工调整" if self.language == "zh" else "Manually adjusted")

    def confirm_write(self) -> None:
        if self.export_running:
            return
        if not self.ensure_metadata_editors_closed():
            return
        mode = self.write_modes[self.write_mode.get()]
        ratings = {
            item.path: self._output_rating(item.suggested_rating)
            for item in self.results
            if mode != "safe" or item.suggested_rating > 0
        }
        if not ratings:
            messagebox.showinfo(self.t["nothing"], self.t["nothing_detail"])
            return
        if mode == "safe":
            detail = self.t["confirm_detail"].format(count=len(ratings))
        elif mode == "managed":
            detail = (
                f"将检查列表中的 {len(ratings)} 张照片，只更新本软件此前写入的星级。\n"
                "Lightroom 或其他软件原有的非零星级仍会跳过；0 星结果会清除本软件旧星级。\n"
                "写入前会自动备份现有 XMP。"
                if self.language == "zh"
                else f"Check {len(ratings)} photos and update only ratings previously written by this app.\n"
                "Existing ratings from Lightroom or other apps remain protected. Existing XMP is backed up first."
            )
        else:
            detail = (
                f"将按当前结果覆盖列表中 {len(ratings)} 张照片的已有星级。\n"
                "这可能替换你在 Lightroom 中手动设置的星级；写入前会自动备份现有 XMP。"
                if self.language == "zh"
                else f"Overwrite existing ratings for {len(ratings)} photos with the current results.\n"
                "This may replace manual Lightroom ratings. Existing XMP is backed up first."
            )
        embedded_count = sum(path.suffix.lower() in EMBEDDED_EXTENSIONS for path in ratings)
        if embedded_count:
            detail += (
                f"\n\n其中 {embedded_count} 张 JPG、DNG、TIFF 或 HEIC：为了让 Lightroom 读取星级，"
                "软件会更新文件内的 XMP 元数据。写入前会完整备份每张照片到同目录的 "
                ".event-cull-backups；照片像素不会重新编码。"
                if self.language == "zh"
                else f"\n\n{embedded_count} JPEG, DNG, TIFF or HEIC file(s) will receive embedded XMP Rating metadata so Lightroom can read it. "
                "Each original is fully backed up in .event-cull-backups before writing; pixels are not re-encoded."
            )
        if not messagebox.askyesno(self.t["confirm"], detail):
            return
        if mode == "force" and not messagebox.askyesno(
            "再次确认强制覆盖" if self.language == "zh" else "Confirm force overwrite",
            "确定覆盖所有已有星级吗？相机 RAW 会改写旁车 XMP；JPG、DNG、TIFF、HEIC 会更新文件内元数据，并先自动完整备份。"
            if self.language == "zh"
            else "Overwrite every existing rating? Camera RAW sidecars and embedded JPEG, DNG, TIFF or HEIC ratings will be replaced after automatic backup.",
        ):
            return
        try:
            outcomes = write_ratings(ratings, mode=mode)
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("星级写入失败" if self.language == "zh" else "Rating write failed", str(exc))
            return
        updated = sum(value in {"created", "updated", "embedded-updated"} for value in outcomes.values())
        skipped = sum(value == "skipped-existing-rating" for value in outcomes.values())
        message = (
            f"星级写入完成：更新 {updated} 张，保留已有星级 {skipped} 张。"
            if self.language == "zh"
            else f"Rating write complete: updated {updated}, preserved existing ratings {skipped}."
        )
        self.status.set(message)
        self._set_xmp_result(f"✓ {message}")
        self.xmp_notice.set(self.lr_instructions)
        self.set_workflow_stage(5)
        self._write_task_history("已写入 XMP" if self.language == "zh" else "XMP written")

    @staticmethod
    def _legacy_repair_button_text(count: int) -> str:
        return f"修复旧版星级 · 待处理 {count} 张照片"

    def _refresh_legacy_heic_repair_button(self) -> None:
        """Expose migration when a format has an obsolete app-managed sidecar."""
        if not hasattr(self, "repair_legacy_heic_button"):
            return
        count = len(legacy_managed_embedded_photos(self.imported_files))
        if count:
            self.repair_legacy_heic_button.configure(
                text=self._legacy_repair_button_text(count)
            )
            if not self.repair_legacy_heic_button.winfo_manager():
                self.repair_legacy_heic_button.pack(fill="x", pady=(6, 0))
            return
        self.repair_legacy_heic_button.pack_forget()

    def repair_legacy_heic_ratings(self) -> None:
        """Migrate obsolete sidecar ratings without a new AI pass."""
        if self.export_running:
            return
        if not self.ensure_metadata_editors_closed():
            return
        candidates = legacy_managed_embedded_photos(self.imported_files)
        if not candidates:
            self._refresh_legacy_heic_repair_button()
            return
        if not messagebox.askyesno(
            "修复旧版照片星级",
            f"检测到 {len(candidates)} 张 JPG、DNG、TIFF 或 HEIC 使用了旧版旁边的 .xmp 星级。\n\n"
            "将把这些已有星级写入照片文件内的 XMP，让 Lightroom 正常读取；"
            "每张原文件会先完整备份到 .event-cull-backups，照片像素不会重新编码。",
        ):
            return
        try:
            outcomes = migrate_legacy_embedded_ratings(candidates)
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("照片星级修复失败", str(exc))
            return
        updated = sum(value == "embedded-updated" for value in outcomes.values())
        skipped = sum(value == "skipped-existing-rating" for value in outcomes.values())
        message = f"照片星级修复完成：已写入 {updated} 张，保留已有嵌入星级 {skipped} 张。"
        self.status.set(message)
        self._set_xmp_result(f"✓ {message}\n回到 Lightroom 后执行“从文件中读取元数据”。")
        self._refresh_legacy_heic_repair_button()

    def confirm_restore(self) -> None:
        if self.export_running:
            return
        if not self.ensure_metadata_editors_closed():
            return
        if not messagebox.askyesno(self.t["restore_confirm"], self.t["restore_confirm_detail"]):
            return
        outcomes = restore_managed_ratings([item.path for item in self.results])
        restored = sum(value in {"removed-sidecar", "restored"} for value in outcomes.values())
        skipped = len(outcomes) - restored
        if restored == 0:
            messagebox.showinfo(self.t["restore_confirm"], self.t["restore_nothing"])
            return
        message = self.t["restored"].format(restored=restored, skipped=skipped)
        self.status.set(message)
        self._set_xmp_result(f"✓ {message}")
        self.xmp_notice.set(self.lr_instructions)
        self._write_task_history("已还原 XMP" if self.language == "zh" else "XMP restored")

def _run_packaging_smoke_test() -> None:
    """Exercise the frozen Windows shell and its primary navigation in CI."""
    trace_path = os.environ.get("MIAOXUAN_SMOKE_TRACE")
    fault_stream = None
    if trace_path:
        import faulthandler
        fault_stream = Path(trace_path).with_suffix(".faults.txt").open("w", encoding="utf-8")
        globals()["_SMOKE_FAULT_STREAM"] = fault_stream
        faulthandler.enable(file=fault_stream, all_threads=True)

    def mark(stage: str) -> None:
        if trace_path:
            with Path(trace_path).open("a", encoding="utf-8") as stream:
                stream.write(stage + "\n")

    def verify_button_wiring(parent: tk.Misc) -> int:
        checked = 0
        stack = [parent]
        while stack:
            widget = stack.pop()
            stack.extend(widget.winfo_children())
            if isinstance(widget, (tk.Button, ttk.Button)):
                checked += 1
                if not str(widget.cget("command")):
                    raise RuntimeError(f"Unbound Windows button: {widget}")
        return checked

    mark("starting")
    app = EventCullApp()
    mark("app-created")
    from tempfile import TemporaryDirectory
    source_dir = TemporaryDirectory(prefix="MiaoXuan-Smoke-Source-")
    try:
        imported_photo = Path(source_dir.name) / "import.jpg"
        Image.new("RGB", (32, 32), "#4682b4").save(imported_photo)
        if not app._load_selected_files([imported_photo]):
            raise RuntimeError("Windows photo import failed")
        mark("photo-imported")
        app.update_idletasks()
        mark("idle-updated")
        required = (
            "new_task_button", "choose_button", "choose_files_button", "history_button",
            "cull_nav_button", "export_nav_button", "calibration_nav_button",
            "license_button", "picker_settings_button", "analyze_button",
            "write_button", "restore_button", "export_selected_button",
            "grid_view_button", "list_view_button", "calibration_button",
        )
        for name in required:
            widget = getattr(app, name)
            if not widget.winfo_exists() or not str(widget.cget("command")):
                raise RuntimeError(f"Missing or unbound Windows button: {name}")
        mark(f"controls-verified-{verify_button_wiring(app)}")
        for name in ("export_nav_button", "calibration_nav_button", "cull_nav_button", "history_button"):
            before = set(app.winfo_children())
            getattr(app, name).invoke()
            mark(f"invoked-{name}")
            app.update()
            if name == "history_button":
                windows = [child for child in app.winfo_children() if isinstance(child, tk.Toplevel) and child not in before]
                if len(windows) != 1:
                    raise RuntimeError("History button did not open its window")
                mark(f"history-buttons-verified-{verify_button_wiring(windows[0])}")
                windows[0].destroy()
                mark("history-closed")
                app.update()
        for name in ("picker_settings_button", "license_button"):
            before = set(app.winfo_children())
            getattr(app, name).invoke()
            mark(f"invoked-{name}")
            app.update()
            windows = [child for child in app.winfo_children() if isinstance(child, tk.Toplevel) and child not in before]
            if len(windows) != 1:
                raise RuntimeError(f"{name} did not open its window")
            mark(f"{name}-buttons-verified-{verify_button_wiring(windows[0])}")
            windows[0].destroy()
            mark(f"closed-{name}")
            app.update()
        from event_cull.xmp import _read_embedded_rating, restore_managed_rating, write_rating
        with TemporaryDirectory() as directory:
            photo = Path(directory) / "测试照片.jpg"
            Image.new("RGB", (32, 32), "#4682b4").save(photo)
            mark("writing-jpeg-rating")
            write_rating(photo, 3)
            mark("reading-jpeg-rating")
            if _read_embedded_rating(photo) != 3:
                raise RuntimeError("Bundled Windows ExifTool rating round-trip failed")
            mark("restoring-jpeg-rating")
            restore_managed_rating(photo)
        mark("exiftool-verified")
        print("MiaoXuan Windows packaging smoke passed")
    finally:
        mark("destroying")
        app.destroy()
        mark("destroyed")
        source_dir.cleanup()


if __name__ == "__main__":
    if "--packaging-smoke-test" in sys.argv:
        _run_packaging_smoke_test()
    else:
        EventCullApp().mainloop()
