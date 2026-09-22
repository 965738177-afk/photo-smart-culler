"""Client for server-issued 秒选 licenses.

The desktop application holds only a device credential and a *server-signed*
short offline lease.  It never receives payment-provider keys and cannot mint
an entitlement on its own.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from core.license_policy import LicenseLease, LicensePlan
from core.runtime_paths import is_frozen_runtime


LICENSE_API_URL = "https://api.cull.lhxx.online"
KEYRING_SERVICE = "online.lhxx.miaoxuan.license"
LEASE_ACCOUNT = "verified-lease"
CREDENTIAL_ACCOUNT = "device-credential"
DEVICE_ID_ACCOUNT = "device-id"

# Public by design: it verifies, but cannot produce, a valid authorization.
LICENSE_SIGNING_PUBLIC_KEY = b"""-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEA1CAsOl2e38HRIYgTtYoRNbbw2ovbZWBUgjff1txk82o=
-----END PUBLIC KEY-----
"""


class LicenseServiceError(RuntimeError):
    """A concise, safe-to-show error from the activation service."""


@dataclass(frozen=True)
class DeviceCredential:
    email: str
    device_id: str
    device_token: str


class LicenseService:
    """HTTP + Keychain boundary for production license operations."""

    def __init__(self, *, api_url: str | None = None) -> None:
        # An installed app must use the real service. Source staging may point
        # to a dedicated test server without weakening the release build.
        configured = os.environ.get("MIAOXUAN_LICENSE_API_URL", "")
        self.api_url = (api_url or (LICENSE_API_URL if is_frozen_runtime() else configured or LICENSE_API_URL)).rstrip("/")

    @staticmethod
    def _keyring():
        try:
            import keyring
            from keyring.errors import KeyringError
        except ImportError as exc:
            raise LicenseServiceError("授权组件不完整，请重新安装秒选。") from exc
        return keyring, KeyringError

    def _read_secret(self, account: str) -> str | None:
        keyring, keyring_error = self._keyring()
        try:
            return keyring.get_password(KEYRING_SERVICE, account)
        except keyring_error as exc:
            raise LicenseServiceError("无法读取系统钥匙串，请检查钥匙串访问权限。") from exc

    def _write_secret(self, account: str, value: str) -> None:
        keyring, keyring_error = self._keyring()
        try:
            keyring.set_password(KEYRING_SERVICE, account, value)
        except keyring_error as exc:
            raise LicenseServiceError("无法写入系统钥匙串，请检查钥匙串访问权限。") from exc

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
            payload = json.loads(serialized)
            return self._verified_lease(payload)
        except (ValueError, TypeError, LicenseServiceError):
            return None

    def credential(self) -> DeviceCredential | None:
        serialized = self._read_secret(CREDENTIAL_ACCOUNT)
        if not serialized:
            return None
        try:
            payload = json.loads(serialized)
            credential = DeviceCredential(
                email=str(payload["email"]),
                device_id=str(payload["device_id"]),
                device_token=str(payload["device_token"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        return credential if credential.device_token else None

    def activate(self, *, email: str, activation_code: str) -> LicenseLease:
        device_id = self.device_id()
        response = self._post(
            "/v1/licenses/activate",
            {
                "email": email.strip().lower(),
                "activation_code": activation_code.strip().upper(),
                "device_id": device_id,
                "device_name": self._device_name(),
            },
        )
        lease_payload = response.get("lease")
        lease = self._verified_lease(lease_payload)
        if lease.device_id != device_id or lease.account_id != email.strip().lower():
            raise LicenseServiceError("授权服务返回的设备信息不匹配。")
        device_token = response.get("device_token")
        if not isinstance(device_token, str) or len(device_token) < 32:
            raise LicenseServiceError("授权服务返回的设备凭据无效。")
        self._store_verified_lease(lease_payload)
        self._store_device_credential(lease=lease, device_token=device_token)
        return lease

    def request_trial_code(self, *, email: str) -> None:
        self._post("/v1/auth/trial-code", {"email": email.strip().lower()})

    def start_trial(self, *, email: str, code: str) -> LicenseLease:
        device_id = self.device_id()
        response = self._post(
            "/v1/trials/start",
            {
                "email": email.strip().lower(),
                "code": code.strip(),
                "device_id": device_id,
                "device_name": self._device_name(),
                "machine_fingerprint": self._machine_fingerprint(),
            },
        )
        lease_payload = response.get("lease")
        lease = self._verified_lease(lease_payload)
        if lease.device_id != device_id or lease.account_id != email.strip().lower():
            raise LicenseServiceError("试用服务返回的设备信息不匹配。")
        device_token = response.get("device_token")
        if not isinstance(device_token, str) or len(device_token) < 32:
            raise LicenseServiceError("试用服务返回的设备凭据无效。")
        self._store_verified_lease(lease_payload)
        self._store_device_credential(lease=lease, device_token=device_token)
        return lease

    def refresh(self) -> LicenseLease:
        credential = self.credential()
        if credential is None:
            raise LicenseServiceError("请先使用购买邮箱和激活码完成激活。")
        response = self._post(
            "/v1/licenses/refresh",
            {"email": credential.email, "device_id": credential.device_id, "device_token": credential.device_token},
        )
        lease_payload = response.get("lease")
        lease = self._verified_lease(lease_payload)
        if lease.device_id != credential.device_id or lease.account_id != credential.email:
            raise LicenseServiceError("授权服务返回的设备信息不匹配。")
        self._store_verified_lease(lease_payload)
        return lease

    def list_devices(self) -> dict[str, Any]:
        response = self._post("/v1/licenses/devices/list", self._device_session_payload())
        devices = response.get("devices")
        if not isinstance(devices, list):
            raise LicenseServiceError("设备列表响应无效，请稍后重试。")
        return response

    def remove_device(self, target_device_id: str) -> dict[str, Any]:
        payload = self._device_session_payload()
        payload["target_device_id"] = target_device_id
        result = self._post("/v1/licenses/devices/remove", payload)
        credential = self.credential()
        if credential is not None and target_device_id == credential.device_id:
            # The server has revoked this Mac; avoid presenting a stale local lease.
            self._write_secret(LEASE_ACCOUNT, "")
            self._write_secret(CREDENTIAL_ACCOUNT, "")
        return result

    def _device_session_payload(self) -> dict[str, str]:
        credential = self.credential()
        if credential is None:
            raise LicenseServiceError("请先完成当前设备的激活。")
        return {
            "email": credential.email,
            "device_id": credential.device_id,
            "device_token": credential.device_token,
        }

    @staticmethod
    def _device_name() -> str:
        computer = platform.node().strip() or "Mac"
        system = platform.system().strip() or "Desktop"
        return f"{computer} · {system}"[:120]

    def _store_verified_lease(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise LicenseServiceError("授权缓存写入失败。")
        self._write_secret(LEASE_ACCOUNT, json.dumps(payload, separators=(",", ":")))

    def _store_device_credential(self, *, lease: LicenseLease, device_token: str) -> None:
        self._write_secret(
            CREDENTIAL_ACCOUNT,
            json.dumps(
                {"email": lease.account_id, "device_id": lease.device_id, "device_token": device_token},
                separators=(",", ":"),
            ),
        )

    def _post(self, path: str, payload: dict[str, str]) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_url}{path}",
            data=body,
            method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("detail")
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = None
            raise LicenseServiceError(str(detail or "授权服务暂时不可用，请稍后重试。")) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LicenseServiceError("无法连接授权服务，请检查网络后重试。") from exc
        if not isinstance(data, dict):
            raise LicenseServiceError("授权服务返回无效响应。")
        return data

    @staticmethod
    def _machine_fingerprint() -> str:
        """Hash stable local traits; raw serial/platform identifiers never leave the Mac."""
        traits = [platform.system(), platform.machine(), platform.processor()]
        if platform.system() == "Darwin":
            try:
                output = subprocess.run(
                    ["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=3,
                ).stdout
                match = re.search(r'"IOPlatformUUID" = "([^"]+)"', output)
                if match:
                    traits.append(match.group(1))
            except (OSError, subprocess.SubprocessError):
                pass
        if len(traits) == 3:
            # A degraded fingerprint still provides an anti-reinstall layer;
            # the Keychain device ID remains the primary local persistence key.
            traits.append(platform.node())
        return hashlib.sha256("\x1f".join(traits).encode("utf-8")).hexdigest()

    @staticmethod
    def _verified_lease(payload: Any) -> LicenseLease:
        if not isinstance(payload, dict):
            raise LicenseServiceError("授权服务返回无效的授权信息。")
        required = {"account_id", "device_id", "plan", "issued_at", "offline_until", "subscription_until", "signature"}
        if set(payload) != required or not isinstance(payload["signature"], str):
            raise LicenseServiceError("授权信息不完整。")
        signed_payload = {key: value for key, value in payload.items() if key != "signature"}
        content = json.dumps(signed_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        try:
            from cryptography.hazmat.primitives import serialization

            public_key = serialization.load_pem_public_key(LICENSE_SIGNING_PUBLIC_KEY)
            public_key.verify(base64.b64decode(payload["signature"]), content)
            plan = LicensePlan(payload["plan"])
            issued_at = _parse_datetime(payload["issued_at"])
            offline_until = _parse_datetime(payload["offline_until"])
            subscription_until = _parse_datetime(payload["subscription_until"], allow_none=True)
        except (ImportError, ValueError, TypeError) as exc:
            raise LicenseServiceError("授权信息校验失败。") from exc
        except Exception as exc:  # Invalid signatures have package-specific types.
            raise LicenseServiceError("授权签名无效。") from exc
        if plan not in {LicensePlan.MONTHLY, LicensePlan.LIFETIME}:
            raise LicenseServiceError("授权套餐无效。")
        return LicenseLease(
            account_id=str(payload["account_id"]),
            device_id=str(payload["device_id"]),
            plan=plan,
            issued_at=issued_at,
            offline_until=offline_until,
            subscription_until=subscription_until,
        )


def _parse_datetime(value: Any, *, allow_none: bool = False) -> datetime | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected ISO timestamp.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must be timezone aware.")
    return parsed
