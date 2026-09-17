"""Small local review UI for the culling MVP; run with ``python -m event_cull.app``."""

from __future__ import annotations

import csv
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

from PIL import Image, ImageOps, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

from event_cull.calibration import (
    CalibrationError,
    CalibrationProfile,
    apply_profile,
    build_profile,
    load_profile,
    save_profile,
)
from event_cull.engine import AnalysisCancelled, AnalysisOptions, PhotoResult, analyze_folder, analyze_paths
from event_cull.xmp import restore_managed_ratings, write_ratings


SUPPORTED_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2",
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif",
}


TEXT = {
    "zh": {"title": "照片智能筛选器", "safe": "本地分析 · 原图只读", "choose": "选择原片文件夹", "folder": "原片来源", "per_group": "每组优选", "target": "目标 3★ 数量（约）", "analyze": "分析这一组", "pause": "暂停", "resume": "继续", "stop": "中止", "paused": "已暂停；可继续或中止。", "stopping": "正在中止；会在当前图片处理完成后停止。", "cancelled": "分析已中止；没有写入 XMP，原图未改动。", "results": "筛选结果", "legend": "3★ 推荐保留　2★ 可选　1★ 复核　— 不标星", "preview": "预览", "empty_preview": "拖入文件夹或照片\n在这里查看细节与清晰度", "decision": "本张决策", "xmp_hint": "星级只在确认写入时生成 XMP", "empty_detail": "从左侧选择一张照片\n查看评分与原因", "manual": "手动评级", "clear": "清除", "write": "确认写入 XMP 星级", "repair": "修复 LR 同步", "repair_confirm": "重写本次评分为 Lightroom 兼容 XMP？", "repair_confirm_detail": "将重写本次有星级的 XMP 封包，仅更新评级字段，不会移动、删除或改写 ARW/JPG。\n用于修复旧版本侧车无法被 Lightroom 读取的问题。", "repaired": "已重写 {count} 个 Lightroom 兼容 XMP。请回到 Lightroom 执行“元数据 → 从文件读取元数据”。", "restore": "还原本次星级", "restore_confirm": "还原本次写入的星级？", "restore_confirm_detail": "只撤回本 App 写入且之后未被 Lightroom 改动的星级。\n不会删除或修改你原有的 Lightroom 星级、照片或其他 XMP 信息。", "restore_nothing": "没有可还原的本次星级。", "restored": "已还原 {restored} 张本次写入的星级；跳过 {skipped} 张后续被修改或非本 App 写入的照片。请在 Lightroom 执行“元数据 → 从文件读取元数据”（不是“同步元数据”）。", "safe_note": "不会移动、删除或改写原图", "pick_folder": "选择一个或多个原片文件夹", "pick_needed": "需要照片来源", "pick_error": "请先选择或拖入一个或多个原片文件夹/照片文件。", "bad_settings": "参数不正确", "bad_settings_detail": "优选张数和目标数量都必须是正整数。", "analyzing": "正在本地分析；RAW 只读取预览，不修改原图…", "scanning": "正在扫描文件并准备第一张照片预览…", "progress": "正在检测 {current}/{total} · {name}", "analysis_failed": "分析失败", "best": "本组优选", "done": "分析完成：{count} 个逻辑原片，推荐保留 {keep} 张 3★；请复核后再写入 XMP。", "raw_preview": "RAW 预览请在 Lightroom 中核对", "score": "评分 {score:.0f}  清晰度 {sharpness:.0f}", "not_written": "已修改建议星级；尚未写入磁盘。", "nothing": "没有可写入内容", "nothing_detail": "目前没有带星级的照片。", "confirm": "确认写入 XMP", "confirm_detail": "将为 {count} 张照片创建或合并相邻 .xmp 文件。\n不会移动、删除或改写任何原图。\n已有非零星级会跳过。", "written": "XMP 写入完成：写入 {updated}，保留现有星级 {skipped}。请在 Lightroom 执行“元数据 → 从文件读取元数据”（不是“同步元数据”）。", "drop": "已导入：{folder}", "drop_error": "请拖入一个或多个原片文件夹/照片文件。"},
    "en": {"title": "Event Cull", "safe": "LOCAL · ORIGINALS READ-ONLY", "choose": "Choose source folders", "folder": "PHOTO SOURCES", "per_group": "PICKS PER GROUP", "target": "TARGET 3★ COUNT", "analyze": "Start analysis", "pause": "Pause", "resume": "Resume", "stop": "Stop", "paused": "Paused. Resume or stop when ready.", "stopping": "Stopping after the current photo finishes.", "cancelled": "Analysis stopped. No XMP was written and originals are unchanged.", "results": "CULL RESULTS", "legend": "3★ Keep  ·  2★ Optional  ·  1★ Review  ·  — Unrated", "preview": "PREVIEW", "empty_preview": "Drop folders or photo files here\nto review detail and sharpness", "decision": "THIS FRAME", "xmp_hint": "Ratings are written to XMP only when confirmed", "empty_detail": "Select a frame on the left\nto review its score and reasons", "manual": "MANUAL RATING", "clear": "Clear", "write": "Write ratings to XMP", "safe_note": "Never moves, deletes, or rewrites originals", "pick_folder": "Choose one or more source folders", "pick_needed": "Choose photo sources", "pick_error": "Choose or drop one or more source folders or photo files first.", "bad_settings": "Invalid settings", "bad_settings_detail": "Picks per group and target count must be positive integers.", "analyzing": "Analyzing locally; RAW previews are read-only…", "scanning": "Scanning files and reading the first RAW preview…", "progress": "Analyzing {current}/{total} · {name}", "analysis_failed": "Analysis failed", "best": "Group pick", "done": "Analysis complete: {count} logical photos, {keep} recommended 3★ picks. Review before writing XMP.", "raw_preview": "Review RAW preview in Lightroom", "score": "Score {score:.0f}  Sharpness {sharpness:.0f}", "not_written": "Rating updated locally; not written to disk yet.", "nothing": "Nothing to write", "nothing_detail": "No frames have a star rating yet.", "confirm": "Write XMP ratings", "confirm_detail": "This creates or merges adjacent .xmp files for {count} photos.\nIt will not move, delete, or rewrite any original.\nExisting non-zero ratings are skipped.", "written": "XMP complete: wrote {updated}, preserved existing ratings {skipped}. In Lightroom, choose Read Metadata from File.", "drop": "Imported: {folder}", "drop_error": "Drop one or more source folders or photo files."},
}

TEXT["zh"].update({
    "empty_preview": "",
    "new_task": "新建任务",
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
    "filter_one": "仅 1★",
    "filter_zero": "仅未入选",
    "write_mode": "写入方式",
    "per_group": "相似图保留",
    "count_unit": "张",
    "drop_busy_title": "正在分析",
    "drop_busy_detail": "请先中止当前分析，等待停止后再拖入新照片。",
    "mode_safe": "安全写入｜跳过已有星级",
    "mode_managed": "更新本软件上次写入的星级",
    "mode_force": "强制覆盖全部星级",
    "score_subject": "评分 {score:.0f}  主体清晰度 {sharpness:.0f}",
    "score_frame": "评分 {score:.0f}  整图清晰度 {sharpness:.0f}",
    "pick_folder": "选择一个或多个原片文件夹",
    "added": "✓ 本次新增 {added} 个文件 · 当前共 {files} 个文件 · {folders} 个文件夹 · {photos} 组照片",
    "already_added": "所选照片已在当前任务中，没有重复添加。",
    "selected_summary": "已导入 {files} 个文件 · {folders} 个文件夹",
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
    "calibration": "个人校准",
    "calibration_active": "个人校准：已启用",
    "calibration_idle": "个人校准",
    "calibration_update": "更新个人校准",
    "calibration_intro": "用你在 Lightroom 的最终选择，调整本机筛选偏好",
    "calibration_privacy": "全程本地处理；不上传、不修改原片或 Lightroom 星级",
    "calibration_prepare": "更新前请准备",
    "calibration_log_item": "本软件导出的筛选日志",
    "calibration_export_item": "Lightroom 导出带星级的 JPG（勾选所有元数据）",
    "calibration_source_item": "对应的原片文件夹（可多选）",
    "calibration_log": "选择筛选日志 CSV",
    "calibration_export": "选择 Lightroom 导出带星级 JPG 文件夹",
    "calibration_sources": "选择校准原片文件夹（可多选）",
    "calibration_running": "正在本地学习你的 Lightroom 选片偏好；不会修改原图…",
    "calibration_progress": "正在提取本地画面特征 {current}/{total} · {name}",
    "calibration_apply": "正在按你的历史选片偏好复核 3★ {current}/{total} · {name}",
    "calibration_done": "个人校准已启用：学习 {samples} 张原图，其中 {positive} 张为最高星级；分组验证区分度 {auc:.0%}。",
    "calibration_failed": "个人校准未启用：{error}",
    "calibration_apply_skipped": "个人校准未应用到本次结果：{error}；已保留基础筛选结果。",
})
TEXT["en"].update({
    "empty_preview": "",
    "new_task": "New task",
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
    "filter_one": "1★ only",
    "filter_zero": "Not selected",
    "write_mode": "WRITE MODE",
    "per_group": "SIMILAR PHOTOS KEPT",
    "count_unit": "photos",
    "drop_busy_title": "Analysis in progress",
    "drop_busy_detail": "Stop the current analysis and wait for it to finish before dropping new photos.",
    "mode_safe": "Safe · preserve existing ratings",
    "mode_managed": "Update ratings written by this app",
    "mode_force": "Force overwrite all ratings",
    "score_subject": "Score {score:.0f}  Subject sharpness {sharpness:.0f}",
    "score_frame": "Score {score:.0f}  Frame sharpness {sharpness:.0f}",
    "pick_folder": "Choose one or more source folders",
    "added": "✓ Added {added} files · {files} total · {folders} folders · {photos} photos",
    "already_added": "Those photos are already in this task; no duplicates were added.",
    "selected_summary": "{files} files imported · {folders} folders",
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
    "calibration": "Personal calibration",
    "calibration_active": "Personal calibration: on",
    "calibration_idle": "Personal calibration",
    "calibration_update": "Update personal calibration",
    "calibration_intro": "Use your final Lightroom picks to tune local culling preferences",
    "calibration_privacy": "Runs locally; never uploads or changes originals or Lightroom ratings",
    "calibration_prepare": "Prepare before updating",
    "calibration_log_item": "Culling log exported by this app",
    "calibration_export_item": "Star-rated JPGs exported from Lightroom (All Metadata selected)",
    "calibration_source_item": "Matching original photo folders (multiple allowed)",
    "calibration_log": "Choose a culling-log CSV",
    "calibration_export": "Choose the Lightroom star-rated JPG folder",
    "calibration_sources": "Choose calibration original folders",
    "calibration_running": "Learning your Lightroom choices locally; originals stay read-only…",
    "calibration_progress": "Extracting local visual features {current}/{total} · {name}",
    "calibration_apply": "Reviewing featured picks against your history {current}/{total} · {name}",
    "calibration_done": "Personal calibration is on: learned {samples} originals, with {positive} top-rated; group-separated validation {auc:.0%}.",
    "calibration_failed": "Personal calibration was not enabled: {error}",
    "calibration_apply_skipped": "Personal calibration was not applied: {error}. Base culling results were kept.",
})


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
        self.language = os.environ.get("EVENT_CULL_LANGUAGE", "zh").lower()
        self.language = "zh" if self.language not in TEXT else self.language
        self.t = TEXT[self.language]
        self.title(self.t["title"])
        self.geometry("1700x1050")
        self.minsize(980, 680)
        self.configure(bg="#101112")
        self.results: list[PhotoResult] = []
        self.imported_files: list[Path] = []
        self.last_import_directory = Path.home()
        self.selected: PhotoResult | None = None
        self.preview: ImageTk.PhotoImage | None = None
        self.folder = tk.StringVar(value=self.t["choose"])
        self.import_notice = tk.StringVar(value="")
        self.lr_instructions = (
            "Lightroom 星级同步\n\n"
            "1. 顶部导航栏「元数据」→\n"
            "   「仅显示目标照片的元数据」，取消勾选\n"
            "2. 打开右上角「图库」，按 G 进入网格视图，\n"
            "   全选本次分析的照片\n"
            "3. 右侧元数据面板选择「选定的照片」\n"
            "4. 右键单击图库网格中的任意照片 →\n"
            "   「元数据」→「从文件中读取元数据」"
            if self.language == "zh"
            else
            "Lightroom rating sync\n\n"
            "1. In Metadata, uncheck Show Metadata for Target Photo Only\n"
            "2. Open Library, press G for Grid, and select all analyzed photos\n"
            "3. In the Metadata panel, choose Selected Photos\n"
            "4. Right-click a grid photo → Metadata → Read Metadata from Files"
        )
        self.xmp_notice = tk.StringVar(value=self.lr_instructions)
        self.xmp_result_notice = tk.StringVar(value="")
        self.target = tk.StringVar(value="1000")
        self.picks = tk.StringVar(value="2")
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
        self.ui_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.is_paused = False
        self.analysis_running = False
        self.analysis_started_at: float | None = None
        self.analysis_pause_started_at: float | None = None
        self.analysis_paused_seconds = 0.0
        self.analysis_completed_at: str | None = None
        self.current_log_path: Path | None = None
        self.lightroom_running = False
        self.calibration_profile: CalibrationProfile | None = load_profile()
        self.calibration_running = False
        self._build()
        self._register_drop_targets(self)
        self.after(50, self.process_ui_events)
        self.after(250, self.refresh_lightroom_guard)

    def _build(self) -> None:
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

        controls = ttk.Frame(self, style="Panel.TFrame", padding=(20, 12))
        controls.grid(row=1, column=0, sticky="ew", padx=16)
        controls.columnconfigure(6, weight=1)
        ttk.Label(controls, text=self.t["folder"], style="PanelMuted.TLabel").grid(row=0, column=0, sticky="w")
        self.import_badge = ttk.Label(controls, textvariable=self.import_notice, style="ImportSuccess.TLabel", wraplength=430)
        self.import_badge.grid(row=0, column=1, columnspan=2, sticky="w", padx=(12, 0))
        self.import_badge.grid_remove()
        ttk.Label(controls, textvariable=self.folder, style="Panel.TLabel").grid(row=1, column=0, columnspan=3, sticky="w", pady=(3, 0))
        ttk.Label(controls, text=self.t["per_group"], style="PanelMuted.TLabel").grid(row=0, column=3, sticky="w", padx=(64, 0))
        picks_input = ttk.Frame(controls, style="Panel.TFrame")
        picks_input.grid(row=1, column=3, sticky="w", padx=(64, 0), pady=(3, 0))
        ttk.Spinbox(picks_input, from_=1, to=10, width=5, textvariable=self.picks).pack(side="left")
        ttk.Label(picks_input, text=self.t["count_unit"], style="Panel.TLabel").pack(side="left", padx=(6, 0))
        ttk.Label(controls, text=self.t["target"], style="PanelMuted.TLabel").grid(row=0, column=4, sticky="w", padx=(18, 0))
        target_input = ttk.Frame(controls, style="Panel.TFrame")
        target_input.grid(row=1, column=4, sticky="w", padx=(18, 0), pady=(3, 0))
        ttk.Label(target_input, text="约" if self.language == "zh" else "~", style="Panel.TLabel").pack(side="left", padx=(0, 6))
        ttk.Entry(target_input, width=8, textvariable=self.target).pack(side="left")
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

        content = ttk.Frame(self, padding=(16, 16, 16, 8))
        content.grid(row=2, column=0, sticky="nsew")
        content.columnconfigure(0, weight=22, minsize=260)
        content.columnconfigure(1, weight=58, minsize=500)
        content.columnconfigure(2, weight=20, minsize=300)
        content.rowconfigure(0, weight=1)
        left = ttk.Frame(content, style="Panel.TFrame", padding=(14, 14, 0, 14))
        left.grid(row=0, column=1, sticky="nsew", padx=8)
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
        for name, title, width in (("group", "组", 42), ("star", "评级", 62), ("file", "原片", 152), ("reason", "分析原因", 230)):
            self.table.heading(name, text=title)
            self.table.column(name, width=width, anchor="w")
        self.table.pack(side="left", fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.show_selected)
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.table.yview)
        scroll.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=scroll.set)

        center = ttk.Frame(content, style="Panel.TFrame", padding=14)
        center.grid(row=0, column=0, sticky="nsew")
        ttk.Label(center, text=self.t["preview"], style="Section.TLabel").pack(anchor="w")
        self.preview_label = ttk.Label(center, text=self.t["empty_preview"], style="PanelMuted.TLabel", justify="center", anchor="center")
        self.preview_label.pack(fill="both", expand=True, pady=(14, 0))

        # Keep this column narrow and independently scrollable.  The prior
        # fixed frame let its packed content request a very wide column and
        # silently clipped the lower Lightroom/calibration instructions.
        right_shell = ttk.Frame(content, style="Panel.TFrame", width=300)
        right_shell.grid(row=0, column=2, sticky="nsew")
        right_shell.grid_propagate(False)
        right_shell.columnconfigure(0, weight=1)
        right_shell.rowconfigure(0, weight=1)
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
        self.export_log_button = ttk.Button(
            right,
            text=self.t["export_log"],
            command=self.export_log,
            state="disabled",
        )
        self.export_log_button.pack(fill="x", pady=(6, 0))
        # Keep the Lightroom hand-off beside the write controls, but let the
        # four actions scan as a short checklist instead of a large text card.
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
        sync_steps = (
            (
                "取消勾选「仅显示目标照片的元数据」",
                "Uncheck Show Metadata for Target Photo Only",
            ),
            (
                "打开「图库」，按 G 网格视图全选本次照片",
                "Open Library, press G, and select all analyzed photos",
            ),
            (
                "元数据面板选择「选定的照片」",
                "Choose Selected Photos in the Metadata panel",
            ),
            (
                "右键照片 →「元数据」→「从文件中读取元数据」",
                "Right-click → Metadata → Read Metadata from Files",
            ),
        )
        for number, step in enumerate(sync_steps, start=1):
            row = ttk.Frame(self.xmp_card, style="XmpCard.TFrame")
            row.pack(fill="x", pady=(0, 7))
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

    def _register_drop_targets(self, widget: tk.Misc) -> None:
        """Route drops from every existing child widget to one import handler."""
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self.accept_drop)
        except (AttributeError, tk.TclError):
            # A platform-native child may not expose TkDND commands. Its
            # nearest registered parent remains available as a drop target.
            pass
        for child in widget.winfo_children():
            self._register_drop_targets(child)

    @staticmethod
    def _default_target_count(logical_count: int) -> int:
        """Use one third of logical photos, rounded up to a multiple of 10."""
        if logical_count <= 0:
            return 0
        return ((logical_count + 29) // 30) * 10

    def _set_default_target(self, files: list[Path]) -> int:
        logical_count = len({(path.parent, path.stem.lower()) for path in files})
        self.target.set(str(self._default_target_count(logical_count)))
        return logical_count

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
        files: list[Path] = []
        for folder in folders:
            try:
                files.extend(
                    path
                    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower())
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                )
            except OSError:
                continue
        return files

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
        self.selected = None
        self.preview = None
        self.table.delete(*self.table.get_children())
        self.preview_label.configure(image="", text=self.t["empty_preview"])
        self.detail.configure(text=self.t["empty_detail"])
        self.folder.set(self.t["choose"])
        self.import_notice.set("")
        self.import_badge.grid_remove()
        self.target.set("1000")
        self.stats_text.set(self.t["stats_empty"])
        self.rating_filter.set(self.t["filter_all"])
        self.write_mode.set(self.t["mode_safe"])
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=1, value=0)
        self.progress_text.set("")
        self.eta_text.set("")
        self.xmp_notice.set(self.lr_instructions)
        self._set_xmp_result()
        self.status.set(self.t["safe_note"])
        self.write_button.configure(state="disabled")
        self.restore_button.configure(state="disabled")
        self.export_log_button.configure(state="disabled")
        self.analyze_button.configure(state="normal")
        self.refresh_lightroom_guard(schedule_next=False)

    def new_task(self) -> None:
        if not self._confirm_replace_task():
            return
        self._reset_task_state()

    def choose_folder(self) -> None:
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
        """Collect a past app log and Lightroom grades without writing either."""
        if self.analysis_running or self.calibration_running:
            return
        log_value = filedialog.askopenfilename(
            title=self.t["calibration_log"],
            initialdir=str(self.last_import_directory),
            filetypes=[("CSV", "*.csv")],
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
        self.calibration_running = True
        self.status.set(self.t["calibration_running"])
        self.calibration_button.configure(state="disabled")
        self.new_task_button.configure(state="disabled")
        self.choose_files_button.configure(state="disabled")
        self.choose_button.configure(state="disabled")
        self.analyze_button.configure(state="disabled")
        self.progress.configure(mode="indeterminate", maximum=100, value=0)
        self.progress.start(12)
        self.progress_text.set(self.t["calibration_running"])
        threading.Thread(
            target=self._build_calibration,
            args=(Path(log_value), Path(exported_value), sources),
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
        self.choose_files_button.configure(state="normal")
        self.choose_button.configure(state="normal")
        self.analyze_button.configure(state="normal")
        self.calibration_button.configure(state="normal")
        self._refresh_calibration_button()

    def _load_selected_files(self, files: list[Path]) -> bool:
        selected = [path for path in files if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not selected:
            messagebox.showerror(self.t["pick_needed"], self.t["drop_error"])
            return False
        merged, added_count = self._merge_unique_paths(self.imported_files, selected)
        if added_count == 0:
            self.status.set(self.t["already_added"])
            return False
        self._clear_results_for_source_change()
        self.imported_files = merged
        folder_count = len({path.parent for path in merged})
        self.folder.set(self.t["selected_summary"].format(files=len(merged), folders=folder_count))
        self.status.set(self.t["append_status"])
        logical_count = self._set_default_target(merged)
        self.import_notice.set(
            self.t["added"].format(
                added=added_count,
                files=len(merged),
                folders=folder_count,
                photos=logical_count,
            )
        )
        self.import_badge.grid()
        return True

    def _clear_results_for_source_change(self) -> None:
        """Invalidate stale analysis while preserving the accumulating source list."""
        self.results = []
        self.selected = None
        self.analysis_completed_at = None
        self.current_log_path = None
        self.preview = None
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
        self.export_log_button.configure(state="disabled")

    def choose_files(self) -> None:
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
        if self.analysis_running or self.calibration_running:
            messagebox.showinfo(self.t["drop_busy_title"], self.t["drop_busy_detail"])
            return "break"
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
            return "break"
        return "break"

    def start_analysis(self) -> None:
        if not self.imported_files:
            messagebox.showerror(self.t["pick_needed"], self.t["pick_error"])
            return
        try:
            options = AnalysisOptions(picks_per_group=int(self.picks.get()), target_keep_count=int(self.target.get()) if self.target.get().strip() else None, language=self.language)
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
        self.analysis_started_at = time.monotonic()
        self.analysis_pause_started_at = None
        self.analysis_paused_seconds = 0.0
        self.analysis_completed_at = None
        self.current_log_path = None
        self.eta_text.set(self.t["eta_estimating"])
        self.analyze_button.configure(state="disabled")
        self.new_task_button.configure(state="disabled")
        self.choose_files_button.configure(state="disabled")
        self.choose_button.configure(state="disabled")
        self.calibration_button.configure(state="disabled")
        self.pause_button.configure(image=self.pause_icon, state="normal")
        self.stop_button.configure(state="normal")
        self.progress.configure(mode="indeterminate", maximum=100, value=0)
        self.progress.start(12)
        self.progress_text.set(self.t["scanning"])
        self.write_button.configure(state="disabled")
        self.export_log_button.configure(state="disabled")
        source: list[Path] = self.imported_files[:]
        threading.Thread(target=self._analyze, args=(source, options), daemon=True).start()

    def toggle_pause(self) -> None:
        if self.is_paused:
            if self.analysis_pause_started_at is not None:
                self.analysis_paused_seconds += time.monotonic() - self.analysis_pause_started_at
            self.analysis_pause_started_at = None
            self.pause_event.clear()
            self.is_paused = False
            self.pause_button.configure(image=self.pause_icon)
            self.status.set(self.t["analyzing"])
        else:
            self.pause_event.set()
            self.is_paused = True
            self.analysis_pause_started_at = time.monotonic()
            self.pause_button.configure(image=self.resume_icon)
            self.status.set(self.t["paused"])
            self.eta_text.set(self.t["eta_paused"])

    def stop_analysis(self) -> None:
        self.cancel_event.set()
        self.pause_event.clear()
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status.set(self.t["stopping"])

    def _analyze(self, source: Path | list[Path], options: AnalysisOptions) -> None:
        try:
            def on_progress(current: int, total: int, path: Path) -> None:
                # Tk must only be called from its main thread on macOS.
                self.ui_events.put(("progress", (current, total, path)))
            if isinstance(source, list):
                results = analyze_paths(source, options, progress_callback=on_progress, pause_event=self.pause_event, cancel_event=self.cancel_event)
            else:
                results = analyze_folder(source, options, progress_callback=on_progress, pause_event=self.pause_event, cancel_event=self.cancel_event)
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
                elif event == "completed":
                    self.analysis_completed_at = datetime.now().isoformat(timespec="seconds")
                    self.show_results(payload)
                    try:
                        self._write_automatic_log()
                    except OSError:
                        pass
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
        self.analyze_button.configure(state="normal")
        self.new_task_button.configure(state="normal")
        self.choose_files_button.configure(state="normal")
        self.choose_button.configure(state="normal")
        self.calibration_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status.set(self.t["cancelled"])
        self.eta_text.set("")

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
            filters = {
                self.t["filter_all"]: None,
                self.t["filter_one"]: 1,
                f"仅 {backup}★": 2,
                f"仅 {featured}★": 3,
                self.t["filter_zero"]: 0,
            }
        else:
            self.legend_text.set(f"1★ Review  ·  {backup}★ Strong backup  ·  {featured}★ Best  ·  — Not selected")
            filters = {
                self.t["filter_all"]: None,
                self.t["filter_one"]: 1,
                f"{backup}★ only": 2,
                f"{featured}★ only": 3,
                self.t["filter_zero"]: 0,
            }
        self.rating_filters = filters
        self.rating_filter_menu.configure(values=tuple(filters))
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

    def show_results(self, results: list[PhotoResult]) -> None:
        self.analysis_running = False
        self.results = results
        self.selected = None
        self._render_results_table()
        counts = self._set_stats_text()
        self.progress.stop()
        self.progress.configure(maximum=max(len(results), 1), value=len(results))
        self.eta_text.set(self.t["eta_complete"])
        self.status.set(self._completion_status(counts))
        self.analyze_button.configure(state="normal")
        self.new_task_button.configure(state="normal")
        self.choose_files_button.configure(state="normal")
        self.choose_button.configure(state="normal")
        self.calibration_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.export_log_button.configure(state="normal")
        self.refresh_lightroom_guard(schedule_next=False)

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
            "version": 3,
            "selection_policy": "scene-minimum-3plus-dynamic-density-v5-personal-calibration",
            "completed_at": self.analysis_completed_at,
            "settings": {
                "similar_photos_kept": int(self.picks.get()),
                "target_selection_count": int(target_text) if target_text else None,
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
            log_folder = Path.home() / "Library" / "Application Support" / "EventCull" / "logs"
            log_folder.mkdir(parents=True, exist_ok=True)
            self.current_log_path = log_folder / f"analysis-{stamp}.json"
        temporary = self.current_log_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._analysis_log_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.current_log_path)

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
            "smiling_faces", "open_eye_faces", "median_luminance",
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
    def _lightroom_is_running() -> bool:
        """Detect Lightroom without opening or querying its catalog."""
        try:
            result = subprocess.run(
                ["pgrep", "-x", "Adobe Lightroom Classic"],
                capture_output=True,
                timeout=1,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def refresh_lightroom_guard(self, schedule_next: bool = True) -> None:
        self.lightroom_running = self._lightroom_is_running()
        can_write = bool(self.results) and not self.lightroom_running
        state = "normal" if can_write else "disabled"
        for button in (self.write_button, self.restore_button):
            button.configure(state=state)
        if self.lightroom_running:
            self.lr_guard_label.configure(
                text=(
                    "⚠ Lightroom 正在运行\n为保护目录，退出 Lightroom 后才能修改 XMP。"
                    if self.language == "zh"
                    else "⚠ Lightroom is running\nQuit Lightroom before changing XMP."
                ),
                style="GuardWarning.TLabel",
            )
            if not self.lr_guard_label.winfo_manager():
                self.lr_guard_label.pack(fill="x", pady=(0, 8))
        else:
            self.lr_guard_label.pack_forget()
        if schedule_next:
            self.after(2000, self.refresh_lightroom_guard)

    def ensure_lightroom_closed(self) -> bool:
        if not self._lightroom_is_running():
            return True
        messagebox.showwarning(
            "请先退出 Lightroom",
            "检测到 Lightroom Classic 正在运行。\n\n为保护目录和侧车元数据，完全退出 Lightroom 后再写入或还原 XMP。",
        )
        self.refresh_lightroom_guard(schedule_next=False)
        return False

    def show_selected(self, _event=None) -> None:
        selection = self.table.selection()
        if not selection:
            return
        self.selected = self.results[int(selection[0])]
        item = self.selected
        try:
            # The analysis engine uses this paired JPEG when available; show
            # the identical visual reference to make review trustworthy.
            preview_path = next(
                (path for path in item.pair_paths if path.suffix.lower() in {".jpg", ".jpeg"}),
                item.path,
            )
            with Image.open(preview_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((350, 460))
                self.preview = ImageTk.PhotoImage(image)
                self.preview_label.configure(image=self.preview, text="")
        except Exception:
            self.preview_label.configure(image="", text=self.t["raw_preview"])
        sharpness = item.subject_sharpness if item.subject_sharpness is not None else (item.sharpness or 0)
        score_label = self.t["score_subject"] if item.subject_sharpness is not None else self.t["score_frame"]
        details = f"{item.path.name}\n{score_label.format(score=item.score, sharpness=sharpness)}\n" + ("\n".join(item.reasons) or self.t["best"])
        self.detail.configure(text=details)

    def set_manual_rating(self, rating: int) -> None:
        if self.selected is None:
            return
        self.selected.suggested_rating = rating
        self.show_results(self.results)
        self.status.set(self.t["not_written"])
        try:
            self._write_automatic_log()
        except OSError:
            pass

    def confirm_write(self) -> None:
        if not self.ensure_lightroom_closed():
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
        if not messagebox.askyesno(self.t["confirm"], detail):
            return
        if mode == "force" and not messagebox.askyesno(
            "再次确认强制覆盖" if self.language == "zh" else "Confirm force overwrite",
            "确定覆盖所有已有星级吗？此操作不会修改原图，但会改写 XMP 中的评级字段。"
            if self.language == "zh"
            else "Overwrite every existing rating? Originals stay unchanged, but XMP rating fields will be replaced.",
        ):
            return
        outcomes = write_ratings(ratings, mode=mode)
        updated = sum(value in {"created", "updated"} for value in outcomes.values())
        skipped = sum(value == "skipped-existing-rating" for value in outcomes.values())
        message = (
            f"XMP 处理完成：更新星级 {updated} 张，保留已有星级 {skipped} 张。"
            if self.language == "zh"
            else f"XMP complete: updated {updated}, preserved existing ratings {skipped}."
        )
        self.status.set(message)
        self._set_xmp_result(f"✓ {message}")
        self.xmp_notice.set(self.lr_instructions)

    def confirm_restore(self) -> None:
        if not self.ensure_lightroom_closed():
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

if __name__ == "__main__":
    EventCullApp().mainloop()
