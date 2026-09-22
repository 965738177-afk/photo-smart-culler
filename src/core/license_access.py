"""Runtime owner for licensing decisions inside the desktop application.

This module owns the one shared decision used by all compute-heavy workflows.
It purposefully has no payment code: production leases must arrive from the
licensing API after a verified payment webhook, rather than from this client.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from core.license_policy import (
    OFFLINE_GRACE_DAYS,
    LicenseAccess,
    LicenseDecision,
    LicenseEnvironment,
    LicenseLease,
    LicensePlan,
    evaluate_license,
)
from core.runtime_paths import is_frozen_runtime
from core.license_service import LicenseService, LicenseServiceError


def resolve_license_environment(*, frozen: bool | None = None) -> LicenseEnvironment:
    """Resolve a safe channel without allowing a packaged app to self-demote.

    Source runs default to development so the developer can work offline.  A
    frozen application is always production, even if a user supplies an
    environment variable such as ``MIAOXUAN_LICENSE_ENV=development``.
    """

    frozen = is_frozen_runtime() if frozen is None else frozen
    if frozen:
        return LicenseEnvironment.PRODUCTION

    configured = os.environ.get("MIAOXUAN_LICENSE_ENV", "development").lower()
    try:
        return LicenseEnvironment(configured)
    except ValueError:
        return LicenseEnvironment.DEVELOPMENT


class LicenseAccessController:
    """Single in-process source of truth for entitlement checks.

    The temporary test-subscription helper is explicitly unavailable in a
    frozen build.  It exists only to let development and staging validate the
    same five-day lease logic before payment credentials are connected.
    """

    def __init__(
        self,
        *,
        environment: LicenseEnvironment | None = None,
        frozen: bool | None = None,
        now_provider: Callable[[], datetime] | None = None,
        license_service: LicenseService | None = None,
    ) -> None:
        self._frozen = is_frozen_runtime() if frozen is None else frozen
        self.environment = environment or resolve_license_environment(frozen=self._frozen)
        self._now = now_provider or (lambda: datetime.now(UTC))
        self._lease: LicenseLease | None = None
        self._license_service = license_service
        # Source builds stay fully offline in developer mode. A frozen release
        # restores only a signature-verified cached lease from the Keychain.
        if self._license_service is None and self._frozen:
            self._license_service = LicenseService()
        if self._license_service is not None:
            try:
                self._lease = self._license_service.cached_lease()
            except LicenseServiceError:
                # The activation dialog explains a missing/unavailable Keychain
                # without preventing the rest of the packaged app from opening.
                self._lease = None

    @property
    def is_local_developer_mode(self) -> bool:
        return (
            not self._frozen
            and self.environment is LicenseEnvironment.DEVELOPMENT
        )

    @property
    def decision(self) -> LicenseDecision:
        return evaluate_license(
            self._lease,
            now=self._now(),
            environment=self.environment,
            developer_bypass_enabled=self.is_local_developer_mode,
        )

    def permits_analysis(self) -> bool:
        return self.decision.allows_analysis

    def install_verified_lease(self, lease: LicenseLease) -> None:
        """Install a lease after the API signature verifier has accepted it."""

        if lease.plan is LicensePlan.NONE:
            raise ValueError("A verified lease must grant a paid plan.")
        self._lease = lease

    @property
    def has_activated_device(self) -> bool:
        if self._license_service is None:
            return False
        try:
            return self._license_service.credential() is not None
        except LicenseServiceError:
            return False

    def activate_purchase(self, *, email: str, activation_code: str) -> LicenseLease:
        """Activate with a paid order code; payment itself is never client-side."""
        if self.is_local_developer_mode:
            raise PermissionError("开发模式不需要激活购买授权。")
        if self._license_service is None:
            self._license_service = LicenseService()
        lease = self._license_service.activate(email=email, activation_code=activation_code)
        self.install_verified_lease(lease)
        return lease

    def request_trial_code(self, *, email: str) -> None:
        if self.is_local_developer_mode:
            raise PermissionError("开发模式不需要领取试用。")
        if self._license_service is None:
            self._license_service = LicenseService()
        self._license_service.request_trial_code(email=email)

    def start_one_day_trial(self, *, email: str, code: str) -> LicenseLease:
        if self.is_local_developer_mode:
            raise PermissionError("开发模式不需要领取试用。")
        if self._license_service is None:
            self._license_service = LicenseService()
        lease = self._license_service.start_trial(email=email, code=code)
        self.install_verified_lease(lease)
        return lease

    def refresh_from_server(self) -> LicenseLease:
        if self._license_service is None:
            raise LicenseServiceError("请先完成购买激活。")
        lease = self._license_service.refresh()
        self.install_verified_lease(lease)
        return lease

    def list_devices(self) -> dict[str, object]:
        if self._license_service is None:
            raise LicenseServiceError("请先完成购买激活。")
        return self._license_service.list_devices()

    def remove_device(self, device_id: str) -> dict[str, object]:
        if self._license_service is None:
            raise LicenseServiceError("请先完成购买激活。")
        result = self._license_service.remove_device(device_id)
        if self._license_service.credential() is None:
            self._lease = None
        return result

    def activate_test_monthly_subscription(self, *, account_id: str = "test-user") -> LicenseLease:
        """Create a staging-only fixture matching a successful monthly payment.

        Production payment status is never created by the desktop client.  The
        real API will invoke ``install_verified_lease`` only after it validates
        WeChat Pay or Alipay's server-to-server notification.
        """

        if self._frozen or self.environment is LicenseEnvironment.PRODUCTION:
            raise PermissionError("Test subscriptions are unavailable in production.")

        now = self._now()
        lease = LicenseLease(
            account_id=account_id,
            device_id="test-device",
            plan=LicensePlan.MONTHLY,
            issued_at=now,
            offline_until=now + timedelta(days=OFFLINE_GRACE_DAYS),
            subscription_until=now + timedelta(days=30),
        )
        self.install_verified_lease(lease)
        return lease

    def clear_for_test(self) -> None:
        """Drop an in-memory lease; useful only for tests and staging demos."""

        self._lease = None

    def blocked_message(self, feature_name: str = "开始分析") -> str:
        decision = self.decision
        if decision.access is LicenseAccess.ACTIVATION_REQUIRED:
            return f"{feature_name}需要激活秒选授权。"
        return decision.message
