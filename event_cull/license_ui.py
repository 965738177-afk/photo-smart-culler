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

from event_cull.licensing import CheckoutOrder, LicenseController, LicenseError, LicensePlan, license_label


def payment_actions_enabled() -> bool:
    """Keep checkout safely disconnected while the subscription UI is reviewed.

    A release build must opt in explicitly.  This avoids accidentally sending
    trial, activation, order, or device-management requests while the product
    team is still approving the customer-facing experience.
    """
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


class LicenseDialog(tk.Toplevel):
    """All billing actions use server-created orders and verified callbacks."""

    def __init__(self, parent: tk.Misc, controller: LicenseController, on_changed: Callable[[], None]) -> None:
        super().__init__(parent)
        self.controller = controller
        self.on_changed = on_changed
        self.internal_build = self.controller.decision.plan is LicensePlan.DEVELOPER
        self.design_preview = not payment_actions_enabled() and not self.internal_build
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

        outer = ttk.Frame(self, style="LicenseRoot.TFrame", padding=(28, 22))
        outer.pack(fill="both", expand=True)
        hero = ttk.Frame(outer, style="LicenseRoot.TFrame")
        hero.pack(fill="x", pady=(0, 14))
        ttk.Label(hero, text="授权与订阅", style="LicenseTitle.TLabel").pack(anchor="w")
        ttk.Label(
            hero,
            text="先试用，再决定是否购买。授权仅绑定设备，不会上传你的照片。",
            style="LicenseSubtitle.TLabel",
        ).pack(anchor="w", pady=(5, 0))
        if self.design_preview:
            ttk.Label(
                hero,
                text="开发预览：上线包会连接授权服务；这里不发起真实请求。",
                style="LicensePreview.TLabel",
            ).pack(anchor="w", pady=(7, 0))

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
        ttk.Label(trial, text="直接领取 1 天试用", style="LicenseTrialTitle.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Label(trial, text="无需邮箱", style="LicenseTrialChip.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Label(
            trial,
            text="联网签发并仅绑定当前 Mac。可完整体验 AI 挑图、人工复核与 Lightroom 写入；设备凭据会安全保存在系统钥匙串。",
            style="LicenseTrialCopy.TLabel",
            wraplength=590,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 13))
        trial_actions = ttk.Frame(trial, style="LicenseTrial.TFrame")
        trial_actions.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.trial_start_button = ttk.Button(
            trial_actions,
            text="领取 1 天试用",
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

        secondary_heading = ttk.Frame(outer, style="LicenseRoot.TFrame")
        secondary_heading.pack(fill="x", pady=(15, 7))
        ttk.Label(secondary_heading, text="已有授权，或准备购买", style="LicenseSecondaryHeading.TLabel").pack(side="left")

        offers = ttk.Frame(outer, style="LicenseRoot.TFrame")
        offers.pack(fill="x")
        offers.columnconfigure(0, weight=6)
        offers.columnconfigure(1, weight=4)

        account = ttk.Frame(offers, style="LicenseCard.TFrame", padding=(17, 14))
        account.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Label(account, text="正式激活", style="LicenseSection.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            account,
            text="购买后收到激活码，或输入已有激活码继续使用。",
            style="LicenseCardCopy.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 8))
        ttk.Label(account, text="购买 / 激活邮箱", style="LicenseField.TLabel").grid(row=2, column=0, sticky="w")
        self.email = tk.StringVar()
        ttk.Entry(account, textvariable=self.email).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 7))
        ttk.Label(account, text="已有激活码（可选）", style="LicenseField.TLabel").grid(row=4, column=0, sticky="w")
        self.code = tk.StringVar()
        ttk.Entry(account, textvariable=self.code).grid(row=5, column=0, sticky="ew", pady=(2, 0))
        self.activate_button = ttk.Button(account, text="激活", command=self.activate, style="LicenseSecondary.TButton")
        self.activate_button.grid(row=5, column=1, sticky="e", padx=(10, 0))
        account.columnconfigure(0, weight=1)

        purchase = ttk.Frame(offers, style="LicenseCard.TFrame", padding=(17, 14))
        purchase.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ttk.Label(purchase, text="购买或续订", style="LicenseSection.TLabel").pack(anchor="w")
        ttk.Label(purchase, text="填写左侧邮箱后，选择套餐与支付方式。", style="LicenseCardCopy.TLabel", wraplength=215, justify="left").pack(anchor="w", pady=(3, 8))
        self.plan = tk.StringVar(value="monthly")
        plan_row = ttk.Frame(purchase, style="LicenseCard.TFrame")
        plan_row.pack(fill="x", pady=(0, 8))
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
        self.payment_note = tk.StringVar(value="免费试用无需邮箱；购买或正式激活时再填写邮箱。")
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
            self.refresh_button, self.devices_button, self.monthly_plan, self.lifetime_plan,
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

    def _configure_subscription_styles(self) -> None:
        """Use a calm subscription hierarchy instead of one large control stack."""
        style = ttk.Style(self)
        style.configure("LicenseRoot.TFrame", background="#101112")
        style.configure("LicenseCard.TFrame", background="#1a1d20")
        style.configure("LicenseTrial.TFrame", background="#1c2924")
        style.configure("LicenseStatus.TFrame", background="#17191b")
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
        if self.design_preview:
            self.status.set("开发者内测版 · 已解锁")
            self.detail.set("当前只展示授权界面。试用、激活、付款回调和设备管理将在正式发布前统一启用。")
            self.payment_note.set("支付入口为界面预览，不会创建订单、跳转支付或请求授权服务。")
            for control in self._controls:
                control.configure(state="disabled")
            return
        decision = self.controller.decision
        self.status.set(license_label(decision))
        self.detail.set(decision.message)
        has_device = self.controller.service.credential() is not None
        self.devices_button.configure(state="normal" if has_device else "disabled")
        self.refresh_button.configure(state="normal" if has_device else "disabled")
        if self.pending_order is not None:
            self.order_status_button.configure(state="normal")
        for control in (self.alipay_button, self.wechat_button, self.activate_button):
            if str(control.cget("state")) == "disabled":
                control.configure(state="normal")
        self.trial_start_button.configure(state="disabled" if has_device else "normal")

    def start_trial(self) -> None:
        if self.design_preview:
            return
        self._run(self.controller.start_device_trial, self._trial_activated)

    def _trial_activated(self, _lease: Any) -> None:
        self.payment_note.set("1 天试用已开通，并已绑定当前 Mac。授权凭据已保存到系统钥匙串。")
        self.qr.configure(image="")
        self._qr_image = None

    def activate(self) -> None:
        if self.design_preview:
            return
        self._run(lambda: self.controller.activate(self.email.get(), self.code.get()), self._activated)

    def _activated(self, _lease: Any) -> None:
        self.code.set("")
        self.payment_note.set("当前设备已完成授权。你可在“管理设备”中自行移除旧设备。")
        self.qr.configure(image="")
        self._qr_image = None

    def refresh_license(self) -> None:
        if self.design_preview:
            return
        self._run(self.controller.refresh, lambda _lease: self.payment_note.set("授权已刷新。"))

    def begin_purchase(self, provider: str) -> None:
        if self.design_preview:
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
        if self.design_preview:
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
        if self.design_preview:
            return
        DeviceManagementDialog(self, self.controller)
