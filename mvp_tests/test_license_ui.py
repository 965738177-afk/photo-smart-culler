from __future__ import annotations

import time
from urllib.parse import parse_qs, urlparse

from event_cull import license_ui
from event_cull.licensing import LicenseError


def test_checkout_uses_the_official_https_site_with_no_payment_secrets_in_the_client():
    url = license_ui.CHECKOUT_PAGE_URL
    parsed = urlparse(url)

    assert parsed.scheme == "https"
    assert parsed.netloc == "cull.lhxx.online"
    assert not parse_qs(parsed.query)


def test_payment_buttons_have_distinct_provider_styles_and_equal_grid_columns():
    source = (license_ui.__file__ and open(license_ui.__file__, encoding="utf-8").read())

    assert 'style="LicenseAlipay.TButton"' in source
    assert 'style="LicenseWeChat.TButton"' in source
    assert 'columnconfigure(0, weight=1, uniform="payment")' in source
    assert 'columnconfigure(1, weight=1, uniform="payment")' in source
    assert 'LicenseAlipay.TButton", background="#1677ff"' in source
    assert 'LicenseWeChat.TButton", background="#07c160"' in source


def test_purchase_buttons_open_the_official_checkout_with_the_selected_plan_and_provider(monkeypatch):
    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    dialog = object.__new__(license_ui.LicenseDialog)
    dialog.plan = Value("lifetime")
    dialog.email = Value("photographer@example.com")
    dialog.payment_note = Value("")
    opened = []
    monkeypatch.setattr(license_ui.webbrowser, "open_new_tab", opened.append)

    dialog.begin_purchase("wechat")

    assert len(opened) == 1
    parsed = urlparse(opened[0])
    assert f"{parsed.scheme}://{parsed.netloc}/" == license_ui.CHECKOUT_PAGE_URL
    assert parse_qs(parsed.query) == {
        "checkout": ["1"], "plan": ["lifetime"], "provider": ["wechat"],
        "email": ["photographer@example.com"],
    }
    assert "官网支付页" in dialog.payment_note.get()


def test_license_error_survives_until_tk_callback_runs():
    dialog = object.__new__(license_ui.LicenseDialog)
    dialog._controls = []
    callbacks = []
    failures = []
    dialog.after = lambda _delay, callback: callbacks.append(callback)
    dialog._failed = failures.append

    def fail():
        raise LicenseError("server unavailable")

    dialog._run(fail, lambda _value: None)
    deadline = time.monotonic() + 1
    while not callbacks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert callbacks

    callbacks[0]()

    assert failures == ["server unavailable"]


def test_team_email_verification_expands_inline_without_a_second_tk_window():
    class Value:
        def __init__(self, value=""):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    class Panel:
        visible = False
        packed = None

        def winfo_manager(self):
            return "pack" if self.visible else ""

        def pack(self, **kwargs):
            self.visible = True
            self.packed = kwargs

        def pack_forget(self):
            self.visible = False

    class Button:
        def __init__(self):
            self.text = ""

        def configure(self, **kwargs):
            self.text = kwargs.get("text", self.text)

    dialog = object.__new__(license_ui.LicenseDialog)
    dialog.controller = type("Controller", (), {"lease": None})()
    dialog.team_email = Value("")
    dialog.team_verify = Panel()
    dialog.team_button = Button()
    dialog.secondary_heading = object()
    dialog._sync_scroll_region = lambda: None

    dialog.manage_team(initial_email="member@example.com")

    assert dialog.team_email.get() == "member@example.com"
    assert dialog.team_verify.visible is True
    assert dialog.team_verify.packed["before"] is dialog.secondary_heading
    assert dialog.team_button.text == "收起团队邮箱验证"


def test_team_code_response_reenables_confirmation_button():
    class Value:
        def __init__(self, value=""):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    class Control:
        def __init__(self):
            self.state = "normal"

        def configure(self, **kwargs):
            self.state = kwargs.get("state", self.state)

    class Panel:
        def winfo_manager(self):
            return "pack"

    class Service:
        def credential(self):
            return None

    class Controller:
        lease = None
        service = Service()

        def request_team_activation_code(self, _email):
            return None

        @property
        def decision(self):
            from event_cull.licensing import AccessState, LicenseDecision, LicensePlan

            return LicenseDecision(AccessState.ACTIVATION_REQUIRED, LicensePlan.NONE, "等待授权")

    dialog = object.__new__(license_ui.LicenseDialog)
    dialog.controller = Controller()
    dialog.team_email = Value("member@example.com")
    dialog.team_note = Value()
    dialog.payment_note = Value()
    dialog.status = Value()
    dialog.detail = Value()
    dialog.team_access = Panel()
    dialog.team_verify = Panel()
    dialog.on_changed = lambda: None
    for name in (
        "team_button", "team_send_button", "team_activate_button", "monthly_plan",
        "lifetime_plan", "devices_button", "refresh_button", "activate_button",
        "alipay_button", "wechat_button", "trial_start_button",
    ):
        setattr(dialog, name, Control())
    dialog._controls = [dialog.team_send_button, dialog.team_activate_button, dialog.team_button]
    callbacks = []
    dialog.after = lambda _delay, callback: callbacks.append(callback)

    dialog.request_team_code()
    deadline = time.monotonic() + 1
    while not callbacks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert callbacks
    assert dialog.team_activate_button.state == "disabled"

    callbacks[0]()

    assert dialog.team_activate_button.state == "normal"
    assert dialog.team_send_button.state == "normal"
    assert dialog.team_note.get().startswith("验证码已发送")
