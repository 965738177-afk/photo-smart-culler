"""Shared, fail-closed licensing rules for 秒选.

The desktop app never decides whether a payment succeeded.  The licensing API
returns a signed lease after it has verified the payment-provider webhook.  This
module evaluates the lease in one place so every workflow can use identical
offline-grace rules.

``development`` is intentionally opt-in and is for source builds only.  A
release build must use ``production`` and can never derive an entitlement from a
local environment variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum


OFFLINE_GRACE_DAYS = 5
REFRESH_INTERVAL_HOURS = 24


class LicenseEnvironment(str, Enum):
    """Deployment channels with deliberately different authentication rules."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LicensePlan(str, Enum):
    NONE = "none"
    MONTHLY = "monthly"
    LIFETIME = "lifetime"
    DEVELOPER = "developer"


class LicenseAccess(str, Enum):
    ACTIVE = "active"
    REFRESH_RECOMMENDED = "refresh_recommended"
    ACTIVATION_REQUIRED = "activation_required"
    OFFLINE_GRACE_EXPIRED = "offline_grace_expired"
    SUBSCRIPTION_EXPIRED = "subscription_expired"


@dataclass(frozen=True)
class LicenseLease:
    """A verified authorization returned by the licensing service.

    ``offline_until`` is issued by the server and is never extended locally.
    This prevents moving a system clock or editing a settings file from creating
    an indefinite offline entitlement.
    """

    account_id: str
    device_id: str
    plan: LicensePlan
    issued_at: datetime
    offline_until: datetime | None
    subscription_until: datetime | None = None


@dataclass(frozen=True)
class LicenseDecision:
    access: LicenseAccess
    plan: LicensePlan
    message: str

    @property
    def allows_analysis(self) -> bool:
        return self.access in {
            LicenseAccess.ACTIVE,
            LicenseAccess.REFRESH_RECOMMENDED,
        }


def evaluate_license(
    lease: LicenseLease | None,
    *,
    now: datetime,
    environment: LicenseEnvironment,
    developer_bypass_enabled: bool = False,
) -> LicenseDecision:
    """Return the feature-access decision without performing network I/O.

    A developer bypass is valid only for an explicitly configured development
    build.  Staging has to exercise the same activation flow as production,
    using test payment orders and test accounts.
    """

    if now.tzinfo is None:
        raise ValueError("License evaluation requires a timezone-aware timestamp.")

    if environment is LicenseEnvironment.DEVELOPMENT and developer_bypass_enabled:
        return LicenseDecision(
            LicenseAccess.ACTIVE,
            LicensePlan.DEVELOPER,
            "开发环境本地授权已启用。此授权不能用于测试或正式版。",
        )

    if lease is None:
        return LicenseDecision(
            LicenseAccess.ACTIVATION_REQUIRED,
            LicensePlan.NONE,
            "请登录购买时使用的账号以激活秒选。",
        )

    if lease.subscription_until is not None and now > lease.subscription_until:
        return LicenseDecision(
            LicenseAccess.SUBSCRIPTION_EXPIRED,
            lease.plan,
            "订阅已到期，请联网续订或选择永久版。",
        )

    if lease.offline_until is not None and now > lease.offline_until:
        return LicenseDecision(
            LicenseAccess.OFFLINE_GRACE_EXPIRED,
            lease.plan,
            "离线验证已超过 5 天，请联网后继续使用。",
        )

    if (
        lease.offline_until is not None
        and (lease.offline_until - now).total_seconds()
        <= (OFFLINE_GRACE_DAYS * 24 - REFRESH_INTERVAL_HOURS) * 3600
    ):
        return LicenseDecision(
            LicenseAccess.REFRESH_RECOMMENDED,
            lease.plan,
            "授权可用；建议联网刷新验证。",
        )

    return LicenseDecision(LicenseAccess.ACTIVE, lease.plan, "授权有效。")


def utc_now() -> datetime:
    """Small seam for callers and tests that need a consistent clock."""

    return datetime.now(UTC)
