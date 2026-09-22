"""Compact account, activation, and self-service device-management dialogs."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.license_access import LicenseAccessController
from core.license_policy import LicenseEnvironment
from core.license_service import LicenseServiceError


class DeviceManagementDialog(QDialog):
    """Let an activated customer reclaim a device slot without support tickets."""

    def __init__(self, controller: LicenseAccessController, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("秒选 · 管理设备")
        self.setModal(True)
        self.setMinimumSize(520, 360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 22)
        layout.setSpacing(12)
        heading = QLabel("管理设备")
        heading.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(heading)
        self.summary = QLabel("正在读取已激活设备…")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #a9aea8;")
        layout.addWidget(self.summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        self.device_list = QVBoxLayout(content)
        self.device_list.setContentsMargins(0, 0, 0, 0)
        self.device_list.setSpacing(8)
        self.device_list.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        footer = QHBoxLayout()
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._refresh)
        footer.addWidget(refresh)
        footer.addStretch()
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        footer.addWidget(close)
        layout.addLayout(footer)
        self._refresh()

    def _clear_devices(self) -> None:
        while self.device_list.count():
            item = self.device_list.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _refresh(self) -> None:
        self._clear_devices()
        try:
            payload = self.controller.list_devices()
            devices = payload.get("devices", [])
            max_devices = int(payload.get("max_devices", 3))
            current_device_id = str(payload.get("current_device_id", ""))
        except (LicenseServiceError, ValueError) as error:
            self.summary.setText(str(error))
            return

        self.summary.setText(f"已使用 {len(devices)} / {max_devices} 台设备。移除旧设备后，名额会立即释放。")
        for device in devices:
            if not isinstance(device, dict):
                continue
            device_id = str(device.get("device_id", ""))
            card = QFrame()
            card.setFrameShape(QFrame.Shape.StyledPanel)
            row = QHBoxLayout(card)
            row.setContentsMargins(14, 12, 14, 12)
            details = QVBoxLayout()
            name = QLabel(str(device.get("device_name", "未知设备")))
            name.setStyleSheet("font-weight: 650;")
            details.addWidget(name)
            last_seen = QLabel(f"最近验证：{str(device.get('last_seen_at', '')).replace('T', ' ')[:19]}")
            last_seen.setStyleSheet("color: #a9aea8; font-size: 12px;")
            details.addWidget(last_seen)
            row.addLayout(details, 1)
            is_current = device_id == current_device_id
            if is_current:
                current = QLabel("本机")
                current.setStyleSheet("color: #c7ff00; font-weight: 650;")
                row.addWidget(current)
            remove = QPushButton("退出此设备" if is_current else "移除")
            remove.clicked.connect(lambda _checked=False, target=device_id, current=is_current: self._remove(target, current))
            row.addWidget(remove)
            self.device_list.addWidget(card)
        self.device_list.addStretch()

    def _remove(self, device_id: str, is_current: bool) -> None:
        message = "移除后，该设备下一次联网验证将无法继续使用。"
        if is_current:
            message = "退出后，本机授权会立即清除；如需继续使用，需要重新激活。"
        answer = QMessageBox.question(self, "确认移除设备", message, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if answer is not QMessageBox.StandardButton.Yes:
            return
        try:
            self.controller.remove_device(device_id)
        except LicenseServiceError as error:
            QMessageBox.warning(self, "未能移除", str(error))
            return
        if is_current:
            QMessageBox.information(self, "已退出", "本机授权已移除。")
            self.accept()
            return
        self._refresh()


class LicenseDialog(QDialog):
    """Keep activation clear while placing routine device recovery with the user."""

    def __init__(self, controller: LicenseAccessController, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("秒选 · 登录与激活")
        self.setModal(True)
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 24)
        layout.setSpacing(14)
        heading = QLabel("登录与激活")
        heading.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(heading)
        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setStyleSheet("color: #a9aea8; line-height: 1.45;")
        layout.addWidget(self.description)

        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(6)
        self.status = QLabel()
        self.status.setStyleSheet("font-size: 16px; font-weight: 650;")
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: #a9aea8;")
        card_layout.addWidget(self.status)
        card_layout.addWidget(self.detail)
        layout.addWidget(card)

        self.activation_form = QFrame()
        form = QVBoxLayout(self.activation_form)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("购买时填写的邮箱")
        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("试用验证码或付款后的激活码")
        self.code_input.setTextMargins(4, 0, 4, 0)
        self.activate_button = QPushButton("激活此设备")
        self.activate_button.clicked.connect(self._activate_purchase)
        self.request_trial_code_button = QPushButton("获取试用验证码")
        self.request_trial_code_button.clicked.connect(self._request_trial_code)
        self.start_trial_button = QPushButton("开始 1 天试用")
        self.start_trial_button.clicked.connect(self._start_trial)
        form.addWidget(self.email_input)
        form.addWidget(self.code_input)
        form.addWidget(self.request_trial_code_button)
        form.addWidget(self.start_trial_button)
        form.addWidget(self.activate_button)
        layout.addWidget(self.activation_form)

        self.test_button = QPushButton("模拟月订阅开通")
        self.test_button.clicked.connect(self._activate_test_subscription)
        layout.addWidget(self.test_button)
        self.manage_button = QPushButton("管理已激活设备")
        self.manage_button.clicked.connect(self._manage_devices)
        layout.addWidget(self.manage_button)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #818780; font-size: 12px;")
        layout.addWidget(self.hint)

        buttons = QHBoxLayout()
        buttons.addStretch()
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self._refresh()

    def _activate_test_subscription(self) -> None:
        self.controller.activate_test_monthly_subscription(account_id="qa@example.com")
        self._refresh()

    def _activate_purchase(self) -> None:
        try:
            self.controller.activate_purchase(
                email=self.email_input.text(), activation_code=self.code_input.text()
            )
        except (LicenseServiceError, PermissionError) as error:
            QMessageBox.warning(self, "未能激活", str(error))
            return
        self.code_input.clear()
        self._refresh()

    def _request_trial_code(self) -> None:
        try:
            self.controller.request_trial_code(email=self.email_input.text())
        except (LicenseServiceError, PermissionError) as error:
            QMessageBox.warning(self, "未能发送验证码", str(error))
            return
        QMessageBox.information(self, "验证码已发送", "请查看购买邮箱，并在 10 分钟内完成试用。")

    def _start_trial(self) -> None:
        try:
            self.controller.start_one_day_trial(
                email=self.email_input.text(), code=self.code_input.text()
            )
        except (LicenseServiceError, PermissionError) as error:
            QMessageBox.warning(self, "未能开始试用", str(error))
            return
        self.code_input.clear()
        self._refresh()

    def _manage_devices(self) -> None:
        DeviceManagementDialog(self.controller, self).exec()
        self._refresh()

    def _refresh(self) -> None:
        decision = self.controller.decision
        self.status.setText("已可使用" if decision.allows_analysis else "需要激活")
        self.detail.setText(decision.message)
        environment = self.controller.environment
        if environment is LicenseEnvironment.DEVELOPMENT:
            self.description.setText("开发环境已启用本地开发授权。你可以直接测试原片筛选，无需登录或付款。")
            self.activation_form.hide()
            self.test_button.hide()
            self.manage_button.hide()
            self.hint.setText("开发授权不会被打包进正式版。")
        elif environment is LicenseEnvironment.STAGING:
            self.description.setText("内测环境使用测试订单验证激活、设备限制和离线宽限；不会向真实用户收费。")
            self.activation_form.hide()
            self.test_button.show()
            self.manage_button.hide()
            self.hint.setText("正式支付接入后，此按钮会替换为购买激活。")
        else:
            self.description.setText("可领取 1 天试用，需验证邮箱且同一邮箱/同一台 Mac 仅限一次。付款核验完成后，也可在此输入购买邮箱和激活码开通。")
            self.activation_form.show()
            self.test_button.hide()
            self.manage_button.setVisible(self.controller.has_activated_device)
            self.hint.setText("你可以随时移除旧设备，名额会立即释放。")
