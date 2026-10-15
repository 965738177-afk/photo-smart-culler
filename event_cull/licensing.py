"""Server-verified licensing for the packaged desktop application.

Payment secrets never enter this module.  The API verifies an Alipay or WeChat
server callback, then returns a short, Ed25519-signed lease to this client.
The client only verifies that signature and keeps its credential in the native
macOS Keychain or Windows Credential Locker through ``keyring``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from event_cull.build_config import INTERNAL_DEVELOPER_BUILD


LICENSE_API_URL = "https://cull.lhxx.online/api"
KEYRING_SERVICE = "online.lhxx.miaoxuan.license"
LEASE_ACCOUNT = "verified-lease"
CREDENTIAL_ACCOUNT = "device-credential"
DEVICE_ID_ACCOUNT = "device-id"
OFFLINE_GRACE_DAYS = 5
TRIAL_DAYS = 7

# This key is public by design.  It can verify an API-issued lease but cannot
# create one.  Its private counterpart exists only on the authorization host.
LICENSE_SIGNING_PUBLIC_KEY = b"""-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEA1CAsOl2e38HRIYgTtYoRNbbw2ovbZWBUgjff1txk82o=
-----END PUBLIC KEY-----
"""


class LicenseError(RuntimeError):
    """A concise error that is safe to show in the desktop UI."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AccessState(str, Enum):
    ACTIVE = "active"
    REFRESH_RECOMMENDED = "refresh_recommended"
    ACTIVATION_REQUIRED = "activation_required"
    OFFLINE_GRACE_EXPIRED = "offline_grace_expired"
    SUBSCRIPTION_EXPIRED = "subscription_expired"


class LicensePlan(str, Enum):
    MONTHLY = "monthly"
    LIFETIME = "lifetime"
    DEVELOPER = "developer"
    NONE = "none"


@dataclass(frozen=True)
class LicenseLease:
    account_id: str
    device_id: str
    plan: LicensePlan
    issued_at: datetime
    offline_until: datetime | None
    subscription_until: datetime | None
    team_name: str | None = None


@dataclass(frozen=True)
class LicenseDecision:
    state: AccessState
    plan: LicensePlan
    message: str
    lease: LicenseLease | None = None

    @property
    def allows_analysis(self) -> bool:
        return self.state in {AccessState.ACTIVE, AccessState.REFRESH_RECOMMENDED}


@dataclass(frozen=True)
class DeviceCredential:
    email: str
    device_id: str
    device_token: str


@dataclass(frozen=True)
class CheckoutOrder:
    order_id: str
    access_token: str
    plan: Literal["monthly", "lifetime"]
    amount_cents: int
    payment_type: Literal["redirect", "native_qr"]
    payment_value: str


def _parse_datetime(value: Any, *, allow_none: bool = False) -> datetime | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected an ISO time.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Expected a timezone-aware timestamp.")
    return parsed


def verify_lease(payload: Any) -> LicenseLease:
    """Reject unsigned, altered, expired-shape, or wrong-plan lease payloads."""
    if not isinstance(payload, dict):
        raise LicenseError("授权服务返回无效的授权信息。")
    required = {
        "account_id", "device_id", "plan", "issued_at", "offline_until",
        "subscription_until", "signature",
    }
    optional = {"team_name"}
    if (
        not required.issubset(payload)
        or set(payload) - required - optional
        or not isinstance(payload.get("signature"), str)
        or ("team_name" in payload and not isinstance(payload["team_name"], str))
    ):
        raise LicenseError("授权信息不完整。")
    signed = {key: value for key, value in payload.items() if key != "signature"}
    content = json.dumps(signed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        from cryptography.hazmat.primitives import serialization

        key = serialization.load_pem_public_key(LICENSE_SIGNING_PUBLIC_KEY)
        key.verify(base64.b64decode(payload["signature"]), content)
        plan = LicensePlan(str(payload["plan"]))
        issued_at = _parse_datetime(payload["issued_at"])
        offline_until = _parse_datetime(payload["offline_until"])
        subscription_until = _parse_datetime(payload["subscription_until"], allow_none=True)
    except LicenseError:
        raise
    except Exception as exc:
        raise LicenseError("授权签名无效或授权信息已损坏。") from exc
    if plan not in {LicensePlan.MONTHLY, LicensePlan.LIFETIME}:
        raise LicenseError("授权套餐无效。")
    return LicenseLease(
        account_id=str(payload["account_id"]),
        device_id=str(payload["device_id"]),
        plan=plan,
        issued_at=issued_at,  # type: ignore[arg-type]
        offline_until=offline_until,
        subscription_until=subscription_until,
        team_name=(str(payload["team_name"]).strip() or None) if "team_name" in payload else None,
    )


def evaluate_lease(lease: LicenseLease | None, *, now: datetime | None = None) -> LicenseDecision:
    now = now or datetime.now(UTC)
    if lease is None:
        return LicenseDecision(
            AccessState.ACTIVATION_REQUIRED,
            LicensePlan.NONE,
            f"尚未激活。可领取 {TRIAL_DAYS} 天试用，或购买后使用邮箱与激活码开通。",
        )
    if lease.subscription_until is not None and now > lease.subscription_until:
        return LicenseDecision(
            AccessState.SUBSCRIPTION_EXPIRED,
            lease.plan,
            "订阅已到期，请联网续订或选择永久版。",
            lease,
        )
    if lease.offline_until is not None and now > lease.offline_until:
        return LicenseDecision(
            AccessState.OFFLINE_GRACE_EXPIRED,
            lease.plan,
            "离线验证宽限已到期，请联网刷新授权后继续使用。",
            lease,
        )
    if lease.offline_until is not None and (lease.offline_until - now) <= timedelta(days=4):
        return LicenseDecision(
            AccessState.REFRESH_RECOMMENDED,
            lease.plan,
            "授权有效；建议联网刷新验证。",
            lease,
        )
    return LicenseDecision(AccessState.ACTIVE, lease.plan, "授权有效。", lease)


def license_label(decision: LicenseDecision) -> str:
    """Short status for the always-visible desktop authorization entry."""
    lease = decision.lease
    # A cached lifetime entitlement remains activated when its offline window
    # expires; verification is still required by the unchanged access gate.
    if lease is not None and lease.plan is LicensePlan.LIFETIME:
        if decision.allows_analysis:
            return "已激活"
        if decision.state is AccessState.OFFLINE_GRACE_EXPIRED:
            return "已激活 · 待验证"
    if not decision.allows_analysis or lease is None:
        return "开始试用 / 激活"
    if lease.team_name:
        return "团队授权"
    if lease.subscription_until is None:
        return "授权有效"
    remaining = max(0, int((lease.subscription_until - datetime.now(UTC)).total_seconds()))
    days, remainder = divmod(remaining, 86400)
    hours = remainder // 3600
    if (lease.subscription_until - lease.issued_at) <= timedelta(days=TRIAL_DAYS):
        if days == 0:
            return f"试用剩余 {max(1, hours)} 小时"
        return f"试用剩余 {days} 天 {hours} 小时"
    return f"订阅剩余 {days} 天"


class LicenseService:
    """HTTPS and Keychain boundary; it never determines payment success."""

    def __init__(self, *, api_url: str | None = None) -> None:
        configured = os.environ.get("MIAOXUAN_LICENSE_API_URL", "")
        self.api_url = (api_url or configured or LICENSE_API_URL).rstrip("/")
        self._secret_cache: dict[str, str | None] = {}
        self._secret_lock = threading.RLock()

    @staticmethod
    def _keyring():
        try:
            import keyring
            from keyring.errors import KeyringError
        except ImportError as exc:
            raise LicenseError("授权组件不完整，请重新安装照片智能筛选器。") from exc
        backend = keyring.get_keyring()
        backend_module = backend.__class__.__module__
        native_backend = (
            (platform.system() == "Darwin" and backend_module == "keyring.backends.macOS")
            or (platform.system() == "Windows" and backend_module == "keyring.backends.Windows")
        )
        if not native_backend:
            raise LicenseError("系统安全凭据存储不可用，请检查当前用户的系统账户权限。")
        return keyring, KeyringError

    def _read_secret(self, account: str) -> str | None:
        with self._secret_lock:
            if account in self._secret_cache:
                return self._secret_cache[account]
            keyring, keyring_error = self._keyring()
            try:
                value = keyring.get_password(KEYRING_SERVICE, account)
            except keyring_error as exc:
                raise LicenseError("无法读取系统安全凭据，请检查当前用户的系统账户权限。") from exc
            self._secret_cache[account] = value
            return value

    def _write_secret(self, account: str, value: str) -> None:
        with self._secret_lock:
            keyring, keyring_error = self._keyring()
            try:
                if platform.system() == "Darwin":
                    # keyring's macOS setter deletes/recreates the item,
                    # resetting its ACL; SecItemUpdate preserves that ACL.
                    from event_cull.secure_store import update_macos_secret
                    update_macos_secret(KEYRING_SERVICE, account, value)
                else:
                    keyring.set_password(KEYRING_SERVICE, account, value)
            except (keyring_error, RuntimeError) as exc:
                raise LicenseError("无法写入系统安全凭据，请检查当前用户的系统账户权限。") from exc
            self._secret_cache[account] = value

    def device_id(self) -> str:
        existing = self._read_secret(DEVICE_ID_ACCOUNT)
        if existing:
            return existing
        device_id = str(uuid4())
        self._write_secret(DEVICE_ID_ACCOUNT, device_id)
        return device_id

    def cached_lease(self) -> LicenseLease | None:
        serialized = self._read_secret(LEASE_ACCOUNT)
        if not serialized:
            return None
        try:
            return verify_lease(json.loads(serialized))
        except (LicenseError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def credential(self) -> DeviceCredential | None:
        serialized = self._read_secret(CREDENTIAL_ACCOUNT)
        if not serialized:
            return None
        try:
            value = json.loads(serialized)
            credential = DeviceCredential(
                email=str(value["email"]),
                device_id=str(value["device_id"]),
                device_token=str(value["device_token"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return credential if len(credential.device_token) >= 32 else None

    def request_trial_code(self, *, email: str) -> None:
        self._post("/v1/auth/trial-code", {"email": _email(email)})

    def start_trial(self, *, email: str, code: str) -> LicenseLease:
        device_id = self.device_id()
        response = self._post(
            "/v1/trials/start",
            {
                "email": _email(email),
                "code": code.strip(),
                "device_id": device_id,
                "device_name": self._device_name(),
                "machine_fingerprint": self._machine_fingerprint(),
            },
        )
        return self._accept_activation(email=email, device_id=device_id, response=response)

    def start_device_trial(self) -> LicenseLease:
        """Start the one-time, device-bound trial without collecting email."""
        device_id = self.device_id()
        response = self._post(
            "/v1/trials/device",
            {
                "device_id": device_id,
                "device_name": self._device_name(),
                "machine_fingerprint": self._machine_fingerprint(),
            },
        )
        account = str(response.get("trial_account", ""))
        if not account:
            raise LicenseError("试用服务未返回设备凭据。")
        return self._accept_activation(email=account, device_id=device_id, response=response)

    def request_team_activation_code(self, *, email: str) -> None:
        self._post("/v1/team/activation/request-code", {"email": _email(email)})

    def activate_team(self, *, email: str, code: str) -> LicenseLease:
        normalized_email = _email(email)
        device_id = self.device_id()
        response = self._post(
            "/v1/team/activation/complete",
            {
                "email": normalized_email,
                "code": code.strip(),
                "device_id": device_id,
                "device_name": self._device_name(),
            },
        )
        return self._accept_activation(email=normalized_email, device_id=device_id, response=response)

    def activate(self, *, email: str, activation_code: str) -> LicenseLease:
        device_id = self.device_id()
        response = self._post(
            "/v1/licenses/activate",
            {
                "email": _email(email),
                "activation_code": activation_code.strip().upper(),
                "device_id": device_id,
                "device_name": self._device_name(),
            },
        )
        return self._accept_activation(email=email, device_id=device_id, response=response)

    def refresh(self) -> LicenseLease:
        credential = self._session()
        response = self._post("/v1/licenses/refresh", self._session_payload(credential))
        lease = verify_lease(response.get("lease"))
        if lease.account_id != credential.email or lease.device_id != credential.device_id:
            raise LicenseError("授权服务返回的设备信息不匹配。")
        self._write_secret(LEASE_ACCOUNT, json.dumps(response["lease"], separators=(",", ":")))
        return lease

    def create_checkout(
        self, *, email: str, plan: Literal["monthly", "lifetime"], provider: Literal["alipay", "wechat"]
    ) -> CheckoutOrder:
        response = self._post(
            "/v1/checkout/orders", {"email": _email(email), "plan": plan, "provider": provider}
        )
        payment = response.get("payment")
        if not isinstance(payment, dict):
            raise LicenseError("支付服务没有返回付款信息。")
        payment_type = str(payment.get("type"))
        value_key = "url" if payment_type == "redirect" else "code_url"
        payment_value = payment.get(value_key)
        if payment_type not in {"redirect", "native_qr"} or not isinstance(payment_value, str):
            raise LicenseError("支付服务返回的付款信息无效。")
        order_id, token = response.get("order_id"), response.get("order_access_token")
        if not isinstance(order_id, str) or not isinstance(token, str):
            raise LicenseError("支付订单创建失败。")
        return CheckoutOrder(
            order_id=order_id,
            access_token=token,
            plan=plan,
            amount_cents=int(response.get("amount_cents", 0)),
            payment_type=payment_type,  # type: ignore[arg-type]
            payment_value=payment_value,
        )

    def checkout_status(self, order: CheckoutOrder) -> dict[str, Any]:
        query = urllib.parse.urlencode({"token": order.access_token})
        return self._get(f"/v1/checkout/orders/{urllib.parse.quote(order.order_id)}?{query}")

    def list_devices(self) -> dict[str, Any]:
        return self._post("/v1/licenses/devices/list", self._session_payload(self._session()))

    def remove_device(self, target_device_id: str) -> dict[str, Any]:
        credential = self._session()
        payload = self._session_payload(credential)
        payload["target_device_id"] = target_device_id
        result = self._post("/v1/licenses/devices/remove", payload)
        if target_device_id == credential.device_id:
            self._write_secret(LEASE_ACCOUNT, "")
            self._write_secret(CREDENTIAL_ACCOUNT, "")
        return result

    def _accept_activation(self, *, email: str, device_id: str, response: dict[str, Any]) -> LicenseLease:
        lease_payload = response.get("lease")
        lease = verify_lease(lease_payload)
        if lease.account_id != _email(email) or lease.device_id != device_id:
            raise LicenseError("授权服务返回的设备信息不匹配。")
        token = response.get("device_token")
        if not isinstance(token, str) or len(token) < 32:
            raise LicenseError("授权服务返回的设备凭据无效。")
        self._write_secret(LEASE_ACCOUNT, json.dumps(lease_payload, separators=(",", ":")))
        self._write_secret(
            CREDENTIAL_ACCOUNT,
            json.dumps({"email": lease.account_id, "device_id": device_id, "device_token": token}, separators=(",", ":")),
        )
        return lease

    def _session(self) -> DeviceCredential:
        credential = self.credential()
        if credential is None:
            raise LicenseError("请先完成当前设备的激活。")
        return credential

    @staticmethod
    def _session_payload(credential: DeviceCredential) -> dict[str, str]:
        return {"email": credential.email, "device_id": credential.device_id, "device_token": credential.device_token}

    def _post(self, path: str, payload: dict[str, str]) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_url}{path}", data=body, method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        return self._send(request)

    def _get(self, path: str) -> dict[str, Any]:
        return self._send(urllib.request.Request(f"{self.api_url}{path}", headers={"Accept": "application/json"}))

    @staticmethod
    def _send(request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("detail")
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = None
            raise LicenseError(str(detail or "授权服务暂时不可用，请稍后重试。"), status_code=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LicenseError("无法连接授权服务，请检查网络后重试。") from exc
        if not isinstance(data, dict):
            raise LicenseError("授权服务返回无效响应。")
        return data

    @staticmethod
    def _device_name() -> str:
        computer = platform.node().strip() or "Desktop"
        return f"{computer} · {platform.system().strip() or 'Desktop'}"[:120]

    @staticmethod
    def _machine_fingerprint() -> str:
        """Hash local traits; the raw serial/UUID is never uploaded."""
        traits = [platform.system(), platform.machine(), platform.processor()]
        if platform.system() == "Darwin":
            try:
                output = subprocess.run(
                    ["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    capture_output=True, text=True, check=False, timeout=3,
                ).stdout
                match = re.search(r'"IOPlatformUUID" = "([^"]+)"', output)
                if match:
                    traits.append(match.group(1))
            except (OSError, subprocess.SubprocessError):
                pass
        elif platform.system() == "Windows":
            try:
                import winreg

                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Cryptography",
                    0,
                    winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
                ) as key:
                    machine_guid, _kind = winreg.QueryValueEx(key, "MachineGuid")
                if isinstance(machine_guid, str) and machine_guid.strip():
                    traits.append(machine_guid.strip())
            except (ImportError, OSError):
                pass
        if len(traits) == 3:
            traits.append(platform.node())
        return hashlib.sha256("\x1f".join(traits).encode("utf-8")).hexdigest()


class LicenseController:
    """A fail-closed gate used by every packaged analysis entry point."""

    def __init__(self, *, frozen: bool | None = None, service: LicenseService | None = None) -> None:
        self.frozen = bool(getattr(__import__("sys"), "frozen", False)) if frozen is None else frozen
        self.service = service or LicenseService()
        self._refresh_lock = threading.Lock()
        try:
            self.lease = self.service.cached_lease()
        except LicenseError:
            self.lease = None

    @property
    def decision(self) -> LicenseDecision:
        if not self.frozen or INTERNAL_DEVELOPER_BUILD:
            return LicenseDecision(AccessState.ACTIVE, LicensePlan.DEVELOPER, "开发版本地授权已启用。")
        return evaluate_lease(self.lease)

    def request_trial_code(self, email: str) -> None:
        self.service.request_trial_code(email=email)

    def start_trial(self, email: str, code: str) -> LicenseLease:
        self.lease = self.service.start_trial(email=email, code=code)
        return self.lease

    def start_device_trial(self) -> LicenseLease:
        self.lease = self.service.start_device_trial()
        return self.lease

    def activate(self, email: str, activation_code: str) -> LicenseLease:
        self.lease = self.service.activate(email=email, activation_code=activation_code)
        return self.lease

    def request_team_activation_code(self, email: str) -> None:
        self.service.request_team_activation_code(email=email)

    def activate_team(self, email: str, code: str) -> LicenseLease:
        self.lease = self.service.activate_team(email=email, code=code)
        return self.lease

    def refresh(self) -> LicenseLease:
        # Manual and scheduled renewals share the same boundary. A temporary
        # network failure must not erase a still-valid signed offline lease.
        with self._refresh_lock:
            try:
                self.lease = self.service.refresh()
            except LicenseError as exc:
                if exc.status_code in {401, 403}:
                    self.lease = None
                    self.service._write_secret(LEASE_ACCOUNT, "")
                raise
            return self.lease

    def refresh_due(self, *, now: datetime | None = None) -> bool:
        """Renew signed leases daily or before a shorter server window ends."""
        if not self.frozen or INTERNAL_DEVELOPER_BUILD or self.lease is None:
            return False
        now = now or datetime.now(UTC)
        return (
            now - self.lease.issued_at >= timedelta(hours=24)
            or (self.lease.offline_until is not None
                and self.lease.offline_until - now <= timedelta(hours=4))
        )


def _email(value: str) -> str:
    email = value.strip().lower()
    if not email or "@" not in email or email.startswith("@") or email.endswith("@"):
        raise LicenseError("请输入有效的邮箱地址。")
    return email
