"""Tk authorization, checkout, and three-device self-service UI."""

from __future__ import annotations

import os
import threading
import webbrowser
from collections.abc import Callable
from typing import Any

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import ImageTk

from event_cull.licensing import (
    TRIAL_DAYS,
    CheckoutOrder,
    LicenseController,
    LicenseError,
    LicensePlan,
    license_label,
)


def payment_actions_enabled() -> bool:
    """Whether real purchase checkout has explicitly been enabled for a release."""
    return os.environ.get("EVENT_CULL_ENABLE_PAYMENT", "").strip() == "1"


class DeviceManagementDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, controller: LicenseController) -> None:
        super().__init__(parent)
        self.controller = controller
        self.title("管理设备")
        self.geometry("650x430")
        self.minsize(570, 330)
        self.transient(parent)
        self.grab_set()

        body = ttk.Frame(self, padding=20)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="管理设备", style="Title.TLabel").pack(anchor="w")
        self.summary = tk.StringVar(value="正在读取已激活设备…")
        ttk.Label(body, textvariable=self.summary, style="Muted.TLabel").pack(anchor="w", pady=(6, 12))
        columns = ("name", "last_seen", "current")
        self.table = ttk.Treeview(body, columns=columns, show="headings", height=8)
        for name, title, width in (
            ("name", "设备", 300), ("last_seen", "最近验证", 190), ("current", "状态", 80),
        ):
            self.table.heading(name, text=title)
            self.table.column(name, width=width, anchor="w" if name == "name" else "center")
        self.table.pack(fill="both", expand=True)
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(12, 0))
        ttk.Button(actions, text="刷新", command=self.refresh).pack(side="left")
        self.remove_button = ttk.Button(actions, text="移除所选设备", command=self.remove_selected, state="disabled")
        self.remove_button.pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="关闭", command=self.destroy).pack(side="right")
        self.table.bind("<<TreeviewSelect>>", lambda _event: self._update_remove_state())
        self._devices: dict[str, dict[str, Any]] = {}
        self.refresh()

    def _run(self, operation: Callable[[], Any], finished: Callable[[Any], None]) -> None:
        def worker() -> None:
            try:
                value = operation()
            except LicenseError as error:
                detail = str(error)
                self.after(0, lambda detail=detail: messagebox.showwarning("设备管理", detail, parent=self))
            else:
                self.after(0, lambda: finished(value))
        threading.Thread(target=worker, daemon=True).start()

    def refresh(self) -> None:
        self.summary.set("正在读取已激活设备…")
        self._run(self.controller.service.list_devices, self._render)

    def _render(self, payload: dict[str, Any]) -> None:
        self.table.delete(*self.table.get_children())
        devices = [device for device in payload.get("devices", []) if isinstance(device, dict)]
        current = str(payload.get("current_device_id", ""))
        maximum = int(payload.get("max_devices", 3))
        self._devices = {str(device.get("device_id", "")): device for device in devices}
        self.summary.set(f"已使用 {len(devices)} / {maximum} 台设备。移除旧设备后，名额会立即释放。")
        for device_id, device in self._devices.items():
            last_seen = str(device.get("last_seen_at", "")).replace("T", " ").replace("Z", "")[:19]
            self.table.insert(
                "", "end", iid=device_id,
                values=(str(device.get("device_name", "未知设备")), last_seen, "本机" if device_id == current else ""),
            )
        self._update_remove_state()

    def _update_remove_state(self) -> None:
        self.remove_button.configure(state="normal" if self.table.selection() else "disabled")

    def remove_selected(self) -> None:
        selection = self.table.selection()
        if not selection:
            return
        device_id = selection[0]
        current = self.controller.service.credential()
        is_current = current is not None and current.device_id == device_id
        detail = "移除后，该设备下一次联网验证将无法继续使用。"
        if is_current:
            detail = "退出后，本机授权会立即清除；如需继续使用，需要重新激活。"
        if not messagebox.askyesno("确认移除设备", detail, parent=self):
            return

        def complete(_result: Any) -> None:
            if is_current:
                self.controller.lease = None
                messagebox.showinfo("已退出", "本机授权已移除。", parent=self)
                self.destroy()
            else:
                self.refresh()

        self._run(lambda: self.controller.service.remove_device(device_id), complete)


class TeamAuthorizationDialog(tk.Toplevel):
    """A small, focused entry point for a member's team authorization.

    The desktop client only ever manages the signed-in member's own devices.
    Cross-member invitations and revocations stay in the owner-only web
    console, which prevents a cached desktop credential from becoming an
    administrator credential.
    """

    def __init__(
        self,
        parent: tk.Misc,
        controller: LicenseController,
        on_changed: Callable[[], None],
        *,
        initial_email: str = "",
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.on_changed = on_changed
        self.title("团队授权")
        self.geometry("580x430")
        self.minsize(520, 390)
        self.configure(background="#101112")
        self.transient(parent)
        self.grab_set()
        self.after_idle(self._center_on_parent)
        self._configure_styles()

        outer = ttk.Frame(self, style="Team.Root.TFrame", padding=(28, 24))
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="团队授权", style="Team.Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="团队成员在这里验证邀请邮箱，并管理自己已绑定的设备。照片始终只在本机处理。",
            style="Team.Subtitle.TLabel", wraplength=500, justify="left",
        ).pack(anchor="w", pady=(6, 18))

        lease = self.controller.lease
        if lease is not None and lease.team_name:
            card = ttk.Frame(outer, style="Team.Active.TFrame", padding=(20, 18))
            card.pack(fill="x")
            ttk.Label(card, text="团队永久版已启用", style="Team.ActiveTitle.TLabel").pack(anchor="w")
            ttk.Label(card, text=lease.team_name, style="Team.Name.TLabel").pack(anchor="w", pady=(6, 0))
            ttk.Label(
                card,
                text="当前 Mac 已绑定。团队主人移除这台设备或撤销成员后，设备将在下次联网验证时失效。",
                style="Team.ActiveCopy.TLabel", wraplength=450, justify="left",
            ).pack(anchor="w", pady=(8, 0))
            actions = ttk.Frame(outer, style="Team.Root.TFrame")
            actions.pack(fill="x", pady=(18, 0))
            ttk.Button(actions, text="管理我的设备", command=self.manage_devices, style="Team.Primary.TButton").pack(side="left")
            ttk.Button(actions, text="关闭", command=self.destroy, style="Team.Quiet.TButton").pack(side="right")
            return

        card = ttk.Frame(outer, style="Team.Card.TFrame", padding=(18, 16))
        card.pack(fill="x")
        ttk.Label(card, text="使用团队邀请邮箱", style="Team.Section.TLabel").pack(anchor="w")
        ttk.Label(
            card,
            text="团队主人先在管理页邀请你的邮箱；收到验证码后，即可为当前 Mac 开通团队永久授权。",
            style="Team.Copy.TLabel", wraplength=470, justify="left",
        ).pack(anchor="w", pady=(4, 13))
        ttk.Label(card, text="邀请邮箱", style="Team.Field.TLabel").pack(anchor="w")
        self.email = tk.StringVar(value=initial_email.strip())
        ttk.Entry(card, textvariable=self.email).pack(fill="x", pady=(3, 10))
        request_row = ttk.Frame(card, style="Team.Card.TFrame")
        request_row.pack(fill="x")
        self.send_button = ttk.Button(
            request_row, text="发送验证码", command=self.request_code, style="Team.Secondary.TButton"
        )
        self.send_button.pack(side="left")
        self.note = tk.StringVar(value="验证码会发送至受邀请的团队邮箱。")
        ttk.Label(request_row, textvariable=self.note, style="Team.Note.TLabel").pack(side="left", padx=(10, 0))
        ttk.Label(card, text="验证码", style="Team.Field.TLabel").pack(anchor="w", pady=(15, 0))
        verify_row = ttk.Frame(card, style="Team.Card.TFrame")
        verify_row.pack(fill="x", pady=(3, 0))
        self.code = tk.StringVar()
        ttk.Entry(verify_row, textvariable=self.code, width=16).pack(side="left")
        self.activate_button = ttk.Button(
            verify_row, text="确认团队授权", command=self.activate, style="Team.Primary.TButton"
        )
        self.activate_button.pack(side="right")
        footer = ttk.Frame(outer, style="Team.Root.TFrame")
        footer.pack(fill="x", pady=(18, 0))
        ttk.Label(footer, text="团队主人请在 cull.lhxx.online/team 管理成员与设备。", style="Team.Note.TLabel").pack(side="left")
        ttk.Button(footer, text="关闭", command=self.destroy, style="Team.Quiet.TButton").pack(side="right")

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.configure("Team.Root.TFrame", background="#101112")
        style.configure("Team.Card.TFrame", background="#1a1d20")
        style.configure("Team.Active.TFrame", background="#1c2924")
        style.configure("Team.Title.TLabel", background="#101112", foreground="#f5f7f8", font=("Helvetica Neue", 25, "bold"))
        style.configure("Team.Subtitle.TLabel", background="#101112", foreground="#9da6ad", font=("Helvetica Neue", 11))
        style.configure("Team.Section.TLabel", background="#1a1d20", foreground="#f2f3f3", font=("Helvetica Neue", 14, "bold"))
        style.configure("Team.Copy.TLabel", background="#1a1d20", foreground="#b6bdc2", font=("Helvetica Neue", 10))
        style.configure("Team.Field.TLabel", background="#1a1d20", foreground="#aab2b8", font=("Helvetica Neue", 10))
        style.configure("Team.Note.TLabel", background="#101112", foreground="#8f989e", font=("Helvetica Neue", 10))
        style.configure("Team.ActiveTitle.TLabel", background="#1c2924", foreground="#f4f7f3", font=("Helvetica Neue", 19, "bold"))
        style.configure("Team.Name.TLabel", background="#1c2924", foreground="#9bd4ae", font=("Helvetica Neue", 12, "bold"))
        style.configure("Team.ActiveCopy.TLabel", background="#1c2924", foreground="#c4d2c8", font=("Helvetica Neue", 10))
        style.configure("Team.Primary.TButton", background="#d99536", foreground="#18130c", borderwidth=0, padding=(13, 9), font=("Helvetica Neue", 11, "bold"))
        style.map("Team.Primary.TButton", background=[("active", "#e9ab50"), ("disabled", "#625035")])
        style.configure("Team.Secondary.TButton", background="#292e33", foreground="#edf0f2", borderwidth=0, padding=(12, 9), font=("Helvetica Neue", 11, "bold"))
        style.map("Team.Secondary.TButton", background=[("active", "#363d44"), ("disabled", "#23272b")])
        style.configure("Team.Quiet.TButton", background="#101112", foreground="#aab2b8", borderwidth=0, padding=(10, 8), font=("Helvetica Neue", 11))

    def _center_on_parent(self) -> None:
        self.update_idletasks()
        width, height = self.winfo_width(), self.winfo_height()
        x = self.master.winfo_rootx() + max(0, (self.master.winfo_width() - width) // 2)
        y = self.master.winfo_rooty() + max(0, (self.master.winfo_height() - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _run(self, operation: Callable[[], Any], completed: Callable[[Any], None]) -> None:
        controls = [control for control in (getattr(self, "send_button", None), getattr(self, "activate_button", None)) if control]
        for control in controls:
            control.configure(state="disabled")

        def worker() -> None:
            try:
                value = operation()
            except (LicenseError, ValueError) as error:
                detail = str(error)
                self.after(0, lambda detail=detail: self._failed(detail, controls))
            except Exception:
                self.after(0, lambda: self._failed("操作暂时无法完成，请检查网络后重试。", controls))
            else:
                self.after(0, lambda: self._completed(value, completed, controls))
        threading.Thread(target=worker, daemon=True).start()

    def _failed(self, detail: str, controls: list[ttk.Button]) -> None:
        for control in controls:
            control.configure(state="normal")
        messagebox.showwarning("团队授权", detail, parent=self)

    def _completed(self, value: Any, completed: Callable[[Any], None], controls: list[ttk.Button]) -> None:
        completed(value)
        for control in controls:
            control.configure(state="normal")

    def request_code(self) -> None:
        self._run(
            lambda: self.controller.request_team_activation_code(self.email.get()),
            lambda _result: self.note.set("验证码已发送，请在 10 分钟内完成验证。"),
        )

    def activate(self) -> None:
        def activated(_lease: Any) -> None:
            self.on_changed()
            messagebox.showinfo("团队授权", "当前 Mac 已获得团队永久授权。", parent=self)
            self.destroy()
        self._run(lambda: self.controller.activate_team(self.email.get(), self.code.get()), activated)

    def manage_devices(self) -> None:
        DeviceManagementDialog(self, self.controller)


class LicenseDialog(tk.Toplevel):
    """All billing actions use server-created orders and verified callbacks."""

    def __init__(self, parent: tk.Misc, controller: LicenseController, on_changed: Callable[[], None]) -> None:
        super().__init__(parent)
        self.controller = controller
        self.on_changed = on_changed
        self.internal_build = self.controller.decision.plan is LicensePlan.DEVELOPER
        self.payment_actions_available = payment_actions_enabled()
        self.pending_order: CheckoutOrder | None = None
        self._qr_image: ImageTk.PhotoImage | None = None
        self.title("授权与订阅")
        self.geometry("680x750")
        self.minsize(620, 680)
        self.configure(background="#101112")
        self.transient(parent)
        self.grab_set()
        self.after_idle(self._center_on_parent)
        self._configure_subscription_styles()

        viewport = ttk.Frame(self, style="LicenseRoot.TFrame")
        viewport.pack(fill="both", expand=True)
        self._scroll_canvas = tk.Canvas(
            viewport,
            background="#101112",
            highlightthickness=0,
            borderwidth=0,
            yscrollincrement=28,
        )
        self._scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=self._scroll_canvas.yview)
        scrollbar.pack(side="right", fill="y", padx=(0, 6), pady=8)
        self._scroll_canvas.configure(yscrollcommand=scrollbar.set)
        outer = ttk.Frame(self._scroll_canvas, style="LicenseRoot.TFrame", padding=(28, 22))
        self._scroll_window = self._scroll_canvas.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", self._sync_scroll_region)
        self._scroll_canvas.bind("<Configure>", self._fit_scroll_width)
        self.bind("<MouseWheel>", self._scroll_license_content, add="+")
        hero = ttk.Frame(outer, style="LicenseRoot.TFrame")
        hero.pack(fill="x", pady=(0, 14))
        ttk.Label(hero, text="授权与订阅", style="LicenseTitle.TLabel").pack(anchor="w")

        if self.internal_build:
            internal = ttk.Frame(outer, style="LicenseTrial.TFrame", padding=(20, 18))
            internal.pack(fill="x", pady=(8, 0))
            ttk.Label(internal, text="开发者内测版 · 已解锁", style="LicenseTrialTitle.TLabel").pack(anchor="w")
            ttk.Label(
                internal,
                text="当前内部构建已直接开放全部本地功能，无需领取试用、填写邮箱或进行支付。正式发布包仍会使用服务器授权。",
                style="LicenseTrialCopy.TLabel",
                wraplength=590,
                justify="left",
            ).pack(anchor="w", pady=(8, 0))
            ttk.Button(
                outer,
                text="关闭",
                command=self.destroy,
                style="LicenseQuiet.TButton",
            ).pack(anchor="e", pady=(18, 0))
            return

        # First use needs one clear answer.  Keep this card before every
        # email-related field so photographers can try the workflow directly.
        trial = ttk.Frame(outer, style="LicenseTrial.TFrame", padding=(20, 17))
        trial.pack(fill="x")
        trial.columnconfigure(0, weight=1)
        ttk.Label(trial, text="首次使用", style="LicenseTrialEyebrow.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(trial, text=f"直接领取 {TRIAL_DAYS} 天试用", style="LicenseTrialTitle.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Label(trial, text="无需邮箱", style="LicenseTrialChip.TLabel").grid(row=0, column=1, sticky="e")
        trial_actions = ttk.Frame(trial, style="LicenseTrial.TFrame")
        trial_actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(9, 0))
        self.trial_start_button = ttk.Button(
            trial_actions,
            text=f"领取 {TRIAL_DAYS} 天试用",
            command=self.start_trial,
            style="LicenseTrial.TButton",
        )
        self.trial_start_button.pack(side="left")
        ttk.Label(trial_actions, text="每台 Mac 限领一次", style="LicenseTrialHint.TLabel").pack(side="left", padx=(12, 0))

        card = ttk.Frame(outer, style="LicenseStatus.TFrame", padding=(15, 11))
        card.pack(fill="x", pady=(10, 0))
        card.columnconfigure(0, weight=1)
        self.status = tk.StringVar()
        self.detail = tk.StringVar()
        ttk.Label(card, textvariable=self.status, style="LicenseStatusTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(card, text="最多 3 台设备", style="LicenseChip.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Label(
            card,
            textvariable=self.detail,
            style="LicenseStatusCopy.TLabel",
            wraplength=540,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        team_access = ttk.Frame(outer, style="LicenseTeam.TFrame", padding=(15, 12))
        team_access.pack(fill="x", pady=(10, 0))
        team_access.columnconfigure(0, weight=1)
        ttk.Label(team_access, text="我是团队成员 / 团队主人", style="LicenseTeamTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            team_access,
            text="已受邀请的邮箱不需要购买或输入个人激活码；验证邮箱即可将团队永久授权绑定到当前 Mac。",
            style="LicenseTeamCopy.TLabel", wraplength=400, justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.team_button = ttk.Button(
            team_access,
            text="使用团队邮箱验证",
            command=self.manage_team,
            style="LicensePrimary.TButton",
        )
        self.team_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=(14, 0))

        secondary_heading = ttk.Frame(outer, style="LicenseRoot.TFrame")
        secondary_heading.pack(fill="x", pady=(15, 7))
        ttk.Label(secondary_heading, text="个人授权，或准备购买", style="LicenseSecondaryHeading.TLabel").pack(side="left")

        offers = ttk.Frame(outer, style="LicenseRoot.TFrame")
        offers.pack(fill="x")
        offers.columnconfigure(0, weight=6)
        offers.columnconfigure(1, weight=4)

        account = ttk.Frame(offers, style="LicenseCard.TFrame", padding=(17, 14))
        account.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Label(account, text="个人购买激活", style="LicenseSection.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(account, text="个人购买 / 激活邮箱", style="LicenseField.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.email = tk.StringVar()
        ttk.Entry(account, textvariable=self.email).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 7))
        ttk.Label(account, text="已有激活码", style="LicenseField.TLabel").grid(row=3, column=0, sticky="w")
        self.code = tk.StringVar()
        ttk.Entry(account, textvariable=self.code).grid(row=4, column=0, sticky="ew", pady=(2, 0))
        self.activate_button = ttk.Button(account, text="激活", command=self.activate, style="LicenseSecondary.TButton")
        self.activate_button.grid(row=4, column=1, sticky="e", padx=(10, 0))
        account.columnconfigure(0, weight=1)

        purchase = ttk.Frame(offers, style="LicenseCard.TFrame", padding=(17, 14))
        purchase.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ttk.Label(purchase, text="购买或续订", style="LicenseSection.TLabel").pack(anchor="w")
        self.plan = tk.StringVar(value="monthly")
        plan_row = ttk.Frame(purchase, style="LicenseCard.TFrame")
        plan_row.pack(fill="x", pady=(9, 8))
        self.monthly_plan = ttk.Radiobutton(
            plan_row, text="月度 ¥4.99", value="monthly", variable=self.plan, style="LicensePlan.TRadiobutton",
        )
        self.monthly_plan.pack(side="left")
        self.lifetime_plan = ttk.Radiobutton(
            plan_row, text="永久 ¥39", value="lifetime", variable=self.plan, style="LicensePlan.TRadiobutton",
        )
        self.lifetime_plan.pack(side="right")
        payment_actions = ttk.Frame(purchase, style="LicenseCard.TFrame")
        payment_actions.pack(fill="x")
        self.alipay_button = ttk.Button(payment_actions, text="支付宝", command=lambda: self.begin_purchase("alipay"), style="LicenseSecondary.TButton")
        self.alipay_button.pack(side="left")
        self.wechat_button = ttk.Button(payment_actions, text="微信支付", command=lambda: self.begin_purchase("wechat"), style="LicensePrimary.TButton")
        self.wechat_button.pack(side="right")

        payment_feedback = ttk.Frame(outer, style="LicenseRoot.TFrame")
        payment_feedback.pack(fill="x", pady=(11, 0))
        self.payment_note = tk.StringVar()
        ttk.Label(payment_feedback, textvariable=self.payment_note, style="LicenseSubtitle.TLabel", wraplength=590, justify="left").pack(anchor="w", fill="x")
        self.order_status_button = ttk.Button(
            payment_feedback,
            text="完成付款后刷新授权",
            command=self.refresh_order,
            state="disabled",
            style="LicensePrimary.TButton",
        )
        self.order_status_button.pack(anchor="w", pady=(8, 0))
        self.qr = ttk.Label(payment_feedback, style="LicenseCard.TLabel")
        self.qr.pack(anchor="w", pady=(10, 0))

        footer = ttk.Frame(outer, style="LicenseRoot.TFrame")
        footer.pack(fill="x", pady=(12, 0))
        self.devices_button = ttk.Button(footer, text="管理设备", command=self.manage_devices, style="LicenseSecondary.TButton")
        self.devices_button.pack(side="left")
        self.refresh_button = ttk.Button(footer, text="刷新授权", command=self.refresh_license, style="LicenseSecondary.TButton")
        self.refresh_button.pack(side="left", padx=(8, 0))
        ttk.Button(footer, text="关闭", command=self.destroy, style="LicenseQuiet.TButton").pack(side="right")
        self._controls = [
            self.trial_start_button, self.alipay_button,
            self.wechat_button, self.order_status_button, self.activate_button,
            self.refresh_button, self.devices_button, self.team_button, self.monthly_plan, self.lifetime_plan,
        ]

        self.refresh_view()

    def _center_on_parent(self) -> None:
        """Keep the subscription sheet visible beside a full-screen workbench."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        parent_width = self.master.winfo_width()
        parent_height = self.master.winfo_height()
        x = self.master.winfo_rootx() + max(0, (parent_width - width) // 2)
        y = self.master.winfo_rooty() + max(0, (parent_height - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _sync_scroll_region(self, _event: tk.Event | None = None) -> None:
        self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

    def _fit_scroll_width(self, event: tk.Event) -> None:
        self._scroll_canvas.itemconfigure(self._scroll_window, width=event.width)

    def _scroll_license_content(self, event: tk.Event) -> str:
        if event.delta:
            self._scroll_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def _configure_subscription_styles(self) -> None:
        """Use a calm subscription hierarchy instead of one large control stack."""
        style = ttk.Style(self)
        style.configure("LicenseRoot.TFrame", background="#101112")
        style.configure("LicenseCard.TFrame", background="#1a1d20")
        style.configure("LicenseTrial.TFrame", background="#1c2924")
        style.configure("LicenseStatus.TFrame", background="#17191b")
        style.configure("LicenseTeam.TFrame", background="#1c2924")
        style.configure("LicenseTitle.TLabel", background="#101112", foreground="#f5f7f8", font=("Helvetica Neue", 25, "bold"))
        style.configure("LicenseSubtitle.TLabel", background="#101112", foreground="#9da6ad", font=("Helvetica Neue", 11))
        style.configure("LicensePreview.TLabel", background="#101112", foreground="#d5a348", font=("Helvetica Neue", 10, "bold"))
        style.configure("LicenseTrialEyebrow.TLabel", background="#1c2924", foreground="#96c7ab", font=("Helvetica Neue", 10, "bold"))
        style.configure("LicenseTrialTitle.TLabel", background="#1c2924", foreground="#f4f7f3", font=("Helvetica Neue", 21, "bold"))
        style.configure("LicenseTrialCopy.TLabel", background="#1c2924", foreground="#c4d2c8", font=("Helvetica Neue", 11))
        style.configure("LicenseTrialHint.TLabel", background="#1c2924", foreground="#9cb3a5", font=("Helvetica Neue", 10))
        style.configure("LicenseTrialChip.TLabel", background="#28583f", foreground="#e4f6e9", padding=(9, 5), font=("Helvetica Neue", 10, "bold"))
        style.configure("LicenseStatusTitle.TLabel", background="#17191b", foreground="#edf0ee", font=("Helvetica Neue", 13, "bold"))
        style.configure("LicenseStatusCopy.TLabel", background="#17191b", foreground="#aeb8b1", font=("Helvetica Neue", 10))
        style.configure("LicenseTeamTitle.TLabel", background="#1c2924", foreground="#e4f6e9", font=("Helvetica Neue", 12, "bold"))
        style.configure("LicenseTeamCopy.TLabel", background="#1c2924", foreground="#c4d2c8", font=("Helvetica Neue", 10))
        style.configure("LicenseChip.TLabel", background="#255c43", foreground="#e2f5e8", padding=(9, 5), font=("Helvetica Neue", 10, "bold"))
        style.configure("LicenseSecondaryHeading.TLabel", background="#101112", foreground="#e4e6e6", font=("Helvetica Neue", 12, "bold"))
        style.configure("LicenseSecondaryHint.TLabel", background="#101112", foreground="#8f989e", font=("Helvetica Neue", 10))
        style.configure("LicenseSection.TLabel", background="#1a1d20", foreground="#f2f3f3", font=("Helvetica Neue", 13, "bold"))
        style.configure("LicenseCardCopy.TLabel", background="#1a1d20", foreground="#b6bdc2", font=("Helvetica Neue", 10))
        style.configure("LicenseField.TLabel", background="#1a1d20", foreground="#aab2b8", font=("Helvetica Neue", 10))
        style.configure("LicenseCard.TLabel", background="#1a1d20", foreground="#e8eaeb")
        style.configure("LicensePlan.TRadiobutton", background="#1a1d20", foreground="#e4e7e9", font=("Helvetica Neue", 10, "bold"))
        style.map("LicensePlan.TRadiobutton", background=[("active", "#1a1d20")], foreground=[("disabled", "#70777d")])
        style.configure("LicensePrimary.TButton", background="#3478e5", foreground="#ffffff", borderwidth=0, padding=(13, 9), font=("Helvetica Neue", 11, "bold"))
        style.map("LicensePrimary.TButton", background=[("active", "#4a8bf0"), ("disabled", "#35465f")], foreground=[("disabled", "#98a4b5")])
        style.configure("LicenseTrial.TButton", background="#d99536", foreground="#18130c", borderwidth=0, padding=(18, 10), font=("Helvetica Neue", 12, "bold"))
        style.map("LicenseTrial.TButton", background=[("active", "#e9ab50"), ("disabled", "#625035")], foreground=[("disabled", "#b7a98f")])
        style.configure("LicenseSecondary.TButton", background="#292e33", foreground="#edf0f2", borderwidth=0, padding=(12, 9), font=("Helvetica Neue", 11, "bold"))
        style.map("LicenseSecondary.TButton", background=[("active", "#363d44"), ("disabled", "#23272b")], foreground=[("disabled", "#70777d")])
        style.configure("LicenseQuiet.TButton", background="#101112", foreground="#aab2b8", borderwidth=0, padding=(10, 8), font=("Helvetica Neue", 11))
        style.map("LicenseQuiet.TButton", background=[("active", "#202326")], foreground=[("disabled", "#70777d")])

    def _run(self, operation: Callable[[], Any], finished: Callable[[Any], None]) -> None:
        for control in self._controls:
            control.configure(state="disabled")

        def worker() -> None:
            try:
                value = operation()
            except (LicenseError, ValueError) as error:
                detail = str(error)
                self.after(0, lambda detail=detail: self._failed(detail))
            except Exception:
                self.after(0, lambda: self._failed("操作暂时无法完成，请检查网络后重试。"))
            else:
                self.after(0, lambda: self._finished(value, finished))
        threading.Thread(target=worker, daemon=True).start()

    def _failed(self, detail: str) -> None:
        self.refresh_view()
        messagebox.showwarning("授权与订阅", detail, parent=self)

    def _finished(self, value: Any, finished: Callable[[Any], None]) -> None:
        finished(value)
        self.on_changed()
        self.refresh_view()

    def refresh_view(self) -> None:
        decision = self.controller.decision
        self.status.set(license_label(decision))
        team_name = decision.lease.team_name if decision.lease is not None else None
        self.detail.set(
            f"{team_name} · 团队永久授权已启用。团队主人可在网站管理成员和设备。"
            if team_name else decision.message
        )
        self.team_button.configure(text="管理团队授权设备" if team_name else "使用团队邮箱验证")
        has_device = self.controller.service.credential() is not None
        self.devices_button.configure(state="normal" if has_device else "disabled")
        self.refresh_button.configure(state="normal" if has_device else "disabled")
        if self.pending_order is not None and self.payment_actions_available:
            self.order_status_button.configure(state="normal")
        else:
            self.order_status_button.configure(state="disabled")
        self.activate_button.configure(state="normal")
        self.alipay_button.configure(state="normal" if self.payment_actions_available else "disabled")
        self.wechat_button.configure(state="normal" if self.payment_actions_available else "disabled")
        self.trial_start_button.configure(state="disabled" if has_device else "normal")

    def start_trial(self) -> None:
        self._run(self.controller.start_device_trial, self._trial_activated)

    def _trial_activated(self, _lease: Any) -> None:
        self.payment_note.set(f"{TRIAL_DAYS} 天试用已开通，并已绑定当前 Mac。授权凭据已保存到系统钥匙串。")
        self.qr.configure(image="")
        self._qr_image = None

    def activate(self) -> None:
        if not self.code.get().strip():
            self.manage_team(initial_email=self.email.get())
            return
        self._run(lambda: self.controller.activate(self.email.get(), self.code.get()), self._activated)

    def _activated(self, _lease: Any) -> None:
        self.code.set("")
        self.payment_note.set("当前设备已完成授权。你可在“管理设备”中自行移除旧设备。")
        self.qr.configure(image="")
        self._qr_image = None

    def refresh_license(self) -> None:
        self._run(self.controller.refresh, lambda _lease: self.payment_note.set("授权已刷新。"))

    def begin_purchase(self, provider: str) -> None:
        if not self.payment_actions_available:
            return
        plan = self.plan.get()
        if plan not in {"monthly", "lifetime"} or provider not in {"alipay", "wechat"}:
            return

        def created(order: CheckoutOrder) -> None:
            self.pending_order = order
            self.order_status_button.configure(state="normal")
            price = f"¥{order.amount_cents / 100:.2f}"
            if order.payment_type == "redirect":
                webbrowser.open(order.payment_value)
                self.payment_note.set(f"支付宝支付页已在浏览器打开（{price}）。完成付款后回到这里点击“我已付款，刷新授权”。")
                self.qr.configure(image="")
                self._qr_image = None
                return
            try:
                import qrcode
                image = qrcode.make(order.payment_value).convert("RGB")
                image.thumbnail((230, 230))
                self._qr_image = ImageTk.PhotoImage(image)
                self.qr.configure(image=self._qr_image, text="")
                self.payment_note.set(f"请使用微信扫描二维码完成支付（{price}）。支付后点击“我已付款，刷新授权”。")
            except Exception:
                self.payment_note.set("微信订单已创建；请在微信商户侧完成支付后点击“我已付款，刷新授权”。")

        self._run(
            lambda: self.controller.service.create_checkout(email=self.email.get(), plan=plan, provider=provider),
            created,
        )

    def refresh_order(self) -> None:
        if not self.payment_actions_available:
            return
        order = self.pending_order
        if order is None:
            return

        def checked(status: dict[str, Any]) -> None:
            if status.get("status") != "paid":
                self.payment_note.set("订单仍在等待支付平台回调确认；请稍后再刷新。")
                return
            code = status.get("activation_code")
            if not isinstance(code, str):
                self.payment_note.set("付款已确认，但激活码暂未生成，请稍后刷新。")
                return
            self.code.set(code)
            self._run(lambda: self.controller.activate(self.email.get(), code), self._activated)

        self._run(lambda: self.controller.service.checkout_status(order), checked)

    def manage_devices(self) -> None:
        DeviceManagementDialog(self, self.controller)

    def manage_team(self, *, initial_email: str = "") -> None:
        def completed_team_authorization() -> None:
            self.on_changed()
            self.destroy()

        TeamAuthorizationDialog(
            self,
            self.controller,
            completed_team_authorization,
            initial_email=initial_email,
        )
