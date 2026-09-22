from __future__ import annotations

import time

from event_cull import license_ui
from event_cull.licensing import LicenseError


def test_subscription_actions_remain_off_without_explicit_release_opt_in(monkeypatch):
    monkeypatch.delenv("EVENT_CULL_ENABLE_PAYMENT", raising=False)

    assert license_ui.payment_actions_enabled() is False


def test_subscription_actions_require_an_explicit_release_opt_in(monkeypatch):
    monkeypatch.setenv("EVENT_CULL_ENABLE_PAYMENT", "1")

    assert license_ui.payment_actions_enabled() is True


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
