from datetime import UTC, datetime, timedelta

from core.license_policy import (
    LicenseAccess,
    LicenseEnvironment,
    LicenseLease,
    LicensePlan,
    evaluate_license,
)


NOW = datetime(2026, 9, 21, 12, tzinfo=UTC)


def _lease(*, offline_days: int = 5, subscription_days: int | None = None):
    return LicenseLease(
        account_id="account_01",
        device_id="device_01",
        plan=LicensePlan.MONTHLY,
        issued_at=NOW,
        offline_until=NOW + timedelta(days=offline_days),
        subscription_until=(
            NOW + timedelta(days=subscription_days)
            if subscription_days is not None
            else None
        ),
    )


def test_developer_bypass_is_limited_to_explicit_development_environment():
    decision = evaluate_license(
        None,
        now=NOW,
        environment=LicenseEnvironment.DEVELOPMENT,
        developer_bypass_enabled=True,
    )

    assert decision.allows_analysis
    assert decision.plan is LicensePlan.DEVELOPER

    production = evaluate_license(
        None,
        now=NOW,
        environment=LicenseEnvironment.PRODUCTION,
        developer_bypass_enabled=True,
    )
    assert production.access is LicenseAccess.ACTIVATION_REQUIRED


def test_paid_license_stays_available_during_offline_grace_period():
    decision = evaluate_license(
        _lease(), now=NOW + timedelta(days=4), environment=LicenseEnvironment.PRODUCTION
    )

    assert decision.access is LicenseAccess.REFRESH_RECOMMENDED
    assert decision.allows_analysis


def test_license_stops_after_server_issued_offline_deadline():
    decision = evaluate_license(
        _lease(),
        now=NOW + timedelta(days=5, seconds=1),
        environment=LicenseEnvironment.PRODUCTION,
    )

    assert decision.access is LicenseAccess.OFFLINE_GRACE_EXPIRED
    assert not decision.allows_analysis


def test_expired_subscription_cannot_be_extended_by_offline_grace():
    decision = evaluate_license(
        _lease(offline_days=5, subscription_days=-1),
        now=NOW,
        environment=LicenseEnvironment.PRODUCTION,
    )

    assert decision.access is LicenseAccess.SUBSCRIPTION_EXPIRED
