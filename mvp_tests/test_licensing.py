import base64
import json
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

import event_cull.licensing as licensing


def _lease(*, plan=licensing.LicensePlan.MONTHLY, expires_in_days=30, offline_hours=24):
    moment = datetime.now(UTC)
    return licensing.LicenseLease(
        account_id="buyer@example.com",
        device_id="device-0000000001",
        plan=plan,
        issued_at=moment,
        offline_until=moment + timedelta(hours=offline_hours),
        subscription_until=None if plan is licensing.LicensePlan.LIFETIME else moment + timedelta(days=expires_in_days),
    )


def test_policy_blocks_expired_subscription_even_if_offline_lease_remains():
    lease = _lease(expires_in_days=-1, offline_hours=48)

    decision = licensing.evaluate_lease(lease)

    assert decision.state is licensing.AccessState.SUBSCRIPTION_EXPIRED
    assert not decision.allows_analysis


def test_policy_allows_lifetime_lease_until_server_offline_window_expires():
    decision = licensing.evaluate_lease(_lease(plan=licensing.LicensePlan.LIFETIME))

    assert decision.allows_analysis
    assert decision.plan is licensing.LicensePlan.LIFETIME


def test_signed_lease_is_verified_and_tampering_is_rejected(monkeypatch):
    private = ed25519.Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    monkeypatch.setattr(licensing, "LICENSE_SIGNING_PUBLIC_KEY", public)
    moment = datetime.now(UTC)
    payload = {
        "account_id": "buyer@example.com",
        "device_id": "device-0000000001",
        "plan": "monthly",
        "issued_at": moment.isoformat(),
        "offline_until": (moment + timedelta(days=5)).isoformat(),
        "subscription_until": (moment + timedelta(days=30)).isoformat(),
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["signature"] = base64.b64encode(private.sign(content)).decode("ascii")

    verified = licensing.verify_lease(payload)
    assert verified.account_id == "buyer@example.com"

    payload["plan"] = "lifetime"
    with pytest.raises(licensing.LicenseError):
        licensing.verify_lease(payload)


def test_checkout_response_never_interprets_provider_data_as_an_entitlement(monkeypatch):
    service = licensing.LicenseService(api_url="https://licenses.example.test")
    monkeypatch.setattr(
        service,
        "_post",
        lambda _path, _payload: {
            "order_id": "MX123",
            "order_access_token": "order-token",
            "amount_cents": 499,
            "payment": {"type": "redirect", "url": "https://provider.example/pay"},
        },
    )

    order = service.create_checkout(email="buyer@example.com", plan="monthly", provider="alipay")

    assert order.order_id == "MX123"
    assert order.payment_type == "redirect"
    assert order.payment_value.startswith("https://provider.example/")


def test_device_trial_collects_no_email_and_uses_the_server_issued_internal_account(monkeypatch):
    service = licensing.LicenseService(api_url="https://licenses.example.test")
    calls = []
    accepted = {}
    expected = _lease()
    monkeypatch.setattr(service, "device_id", lambda: "device-0000000001")
    monkeypatch.setattr(service, "_device_name", lambda: "QA Mac · Darwin")
    monkeypatch.setattr(service, "_machine_fingerprint", lambda: "a" * 64)
    monkeypatch.setattr(
        service,
        "_post",
        lambda path, payload: calls.append((path, payload)) or {"trial_account": "trial+opaque@device.local"},
    )
    monkeypatch.setattr(
        service,
        "_accept_activation",
        lambda **kwargs: accepted.update(kwargs) or expected,
    )

    assert service.start_device_trial() is expected
    assert calls == [(
        "/v1/trials/device",
        {
            "device_id": "device-0000000001",
            "device_name": "QA Mac · Darwin",
            "machine_fingerprint": "a" * 64,
        },
    )]
    assert accepted["email"] == "trial+opaque@device.local"
    assert accepted["device_id"] == "device-0000000001"


def test_source_build_keeps_developer_flow_unblocked_without_a_fake_paid_lease():
    class Service:
        def cached_lease(self):
            return None

    controller = licensing.LicenseController(frozen=False, service=Service())

    assert controller.decision.plan is licensing.LicensePlan.DEVELOPER
    assert controller.decision.allows_analysis


def test_controlled_internal_package_can_use_developer_entitlement(monkeypatch):
    class Service:
        def cached_lease(self):
            return None

    monkeypatch.setattr(licensing, "INTERNAL_DEVELOPER_BUILD", True)
    controller = licensing.LicenseController(frozen=True, service=Service())

    assert controller.decision.allows_analysis is True
    assert controller.decision.plan is licensing.LicensePlan.DEVELOPER
