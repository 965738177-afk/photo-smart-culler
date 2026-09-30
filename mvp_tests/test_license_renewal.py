"""Automatic renewals must not block Tk or extend unsigned offline access."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import queue

import pytest

from event_cull import app as app_module
from event_cull.app import EventCullApp
from event_cull.licensing import (
    LicenseController, LicenseError, LicenseLease, LicensePlan, LEASE_ACCOUNT,
)


def controller(*, offline_hours=48, age_hours=0, frozen=True):
    now = datetime.now(UTC)
    lease = LicenseLease("member@example.com", "device", LicensePlan.LIFETIME,
                         now - timedelta(hours=age_hours),
                         now + timedelta(hours=offline_hours), None, "Team")
    service = SimpleNamespace(cached_lease=lambda: lease)
    return LicenseController(frozen=frozen, service=service)


@pytest.mark.parametrize("offline_hours,age_hours,expected", [
    (48, 0, False), (3, 0, True), (-1, 0, True), (48, 25, True),
])
def test_renew_before_server_expiry_or_after_one_day(offline_hours, age_hours, expected):
    assert controller(offline_hours=offline_hours, age_hours=age_hours).refresh_due() is expected


def test_source_build_does_not_refresh_real_account():
    assert not controller(offline_hours=-1, frozen=False).refresh_due()


@pytest.mark.parametrize("status_code", [None, 500, 401, 403])
def test_transient_failure_preserves_lease_but_server_revocation_clears_it(status_code):
    subject = controller()
    writes = []

    def fail():
        raise LicenseError("refresh failed", status_code=status_code)

    subject.service.refresh = fail
    subject.service._write_secret = lambda *args: writes.append(args)
    with pytest.raises(LicenseError):
        subject.refresh()
    revoked = status_code in {401, 403}
    assert (subject.lease is None) is revoked
    assert writes == ([(LEASE_ACCOUNT, "")] if revoked else [])


def desktop():
    subject = object.__new__(EventCullApp)
    subject.license_controller = controller(offline_hours=-1)
    subject._license_refresh_in_flight = False
    subject._license_next_refresh_attempt = 0
    subject._license_resume_context = None
    subject.current_task_id = "task"
    subject.imported_files = ["photo.heic"]
    subject.status = SimpleNamespace(set=lambda value: None)
    subject.ui_events = queue.Queue()
    subject._refresh_license_entry = lambda: None
    subject.winfo_children = lambda: []
    subject.analysis_running = False
    subject.picker_settings_confirmed = True
    return subject


def test_pending_analysis_reuses_background_request():
    subject = desktop()
    subject._license_refresh_in_flight = True
    subject._begin_license_refresh(resume_analysis=True)
    assert subject._license_resume_context == ("task", ("photo.heic",))
    assert subject.ui_events.empty()


def test_worker_uses_queue_and_never_touches_tk(monkeypatch):
    subject = desktop()
    subject.license_controller.refresh = lambda: None
    worker = []
    monkeypatch.setattr(app_module.threading, "Thread",
                        lambda **kwargs: SimpleNamespace(start=lambda: worker.append(kwargs["target"])))
    subject._begin_license_refresh()
    assert subject._license_refresh_in_flight
    worker[0]()
    assert subject.ui_events.get_nowait() == ("license_refresh_finished", None)


@pytest.mark.parametrize("task_changed", [False, True])
def test_resume_only_same_task_after_successful_renewal(task_changed):
    subject = desktop()
    subject._license_resume_context = ("task", ("photo.heic",))
    subject.license_controller.lease = controller().lease
    started = []
    subject.start_analysis = lambda: started.append(True)
    if task_changed:
        subject.current_task_id = "new-task"
    subject._finish_license_refresh(None)
    assert bool(started) is not task_changed
    assert subject._license_resume_context is None


def test_failed_periodic_refresh_backs_off_without_opening_dialog():
    subject = desktop()
    subject._finish_license_refresh("Network unavailable")
    assert subject._license_next_refresh_attempt > 0
    assert not subject._license_refresh_in_flight
