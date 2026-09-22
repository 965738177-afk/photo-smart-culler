from datetime import UTC, datetime, timedelta

import pytest

from core.license_access import LicenseAccessController, resolve_license_environment
from core.license_policy import LicenseAccess, LicenseEnvironment, LicenseLease, LicensePlan


NOW = datetime(2026, 9, 21, 8, tzinfo=UTC)


def test_source_build_defaults_to_local_developer_access(monkeypatch):
    monkeypatch.delenv("MIAOXUAN_LICENSE_ENV", raising=False)
    controller = LicenseAccessController(frozen=False, now_provider=lambda: NOW)

    assert controller.environment is LicenseEnvironment.DEVELOPMENT
    assert controller.permits_analysis()
    assert controller.decision.plan is LicensePlan.DEVELOPER


def test_packaged_build_cannot_be_demoted_to_development(monkeypatch):
    monkeypatch.setenv("MIAOXUAN_LICENSE_ENV", "development")

    assert resolve_license_environment(frozen=True) is LicenseEnvironment.PRODUCTION
    controller = LicenseAccessController(frozen=True, now_provider=lambda: NOW)
    assert not controller.permits_analysis()
    assert controller.decision.access is LicenseAccess.ACTIVATION_REQUIRED


def test_staging_test_subscription_matches_successful_paid_monthly_access():
    controller = LicenseAccessController(
        environment=LicenseEnvironment.STAGING, frozen=False, now_provider=lambda: NOW
    )

    lease = controller.activate_test_monthly_subscription(account_id="qa@example.com")

    assert lease.plan is LicensePlan.MONTHLY
    assert controller.permits_analysis()
    assert controller.decision.access is LicenseAccess.ACTIVE


def test_production_refuses_client_created_test_subscriptions():
    controller = LicenseAccessController(
        environment=LicenseEnvironment.PRODUCTION, frozen=True, now_provider=lambda: NOW
    )

    with pytest.raises(PermissionError):
        controller.activate_test_monthly_subscription()


def test_expired_verified_lease_blocks_analysis_in_production():
    controller = LicenseAccessController(
        environment=LicenseEnvironment.PRODUCTION,
        frozen=True,
        now_provider=lambda: NOW,
    )
    controller.install_verified_lease(
        LicenseLease(
            account_id="customer",
            device_id="mac-1",
            plan=LicensePlan.LIFETIME,
            issued_at=NOW - timedelta(days=6),
            offline_until=NOW - timedelta(seconds=1),
        )
    )

    assert not controller.permits_analysis()
    assert controller.decision.access is LicenseAccess.OFFLINE_GRACE_EXPIRED
