import base64
import json
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

import event_cull.licensing as licensing
from event_cull.build_config import is_internal_developer_build


def _lease(*, plan=licensing.LicensePlan.MONTHLY, expires_in_days=30, offline_hours=24, team_name=None):
    moment = datetime.now(UTC)
    return licensing.LicenseLease(
        account_id="buyer@example.com",
        device_id="device-0000000001",
        plan=plan,
        issued_at=moment,
        offline_until=moment + timedelta(hours=offline_hours),
        subscription_until=None if plan is licensing.LicensePlan.LIFETIME else moment + timedelta(days=expires_in_days),
        team_name=team_name,
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


def test_team_lifetime_lease_has_a_clear_sidebar_label():
    decision = licensing.evaluate_lease(
        _lease(plan=licensing.LicensePlan.LIFETIME, team_name="秒选摄影团队")
    )

    assert decision.allows_analysis
    assert licensing.license_label(decision) == "团队授权"


def test_three_day_trial_copy_and_remaining_time_are_not_described_as_a_subscription():
    moment = datetime.now(UTC)
    lease = licensing.LicenseLease(
        account_id="trial+opaque@device.local",
        device_id="device-0000000001",
        plan=licensing.LicensePlan.MONTHLY,
        issued_at=moment,
        offline_until=moment + timedelta(hours=4),
        subscription_until=moment + timedelta(days=licensing.TRIAL_DAYS),
    )

    assert licensing.TRIAL_DAYS == 3
    assert "3 天试用" in licensing.evaluate_lease(None).message
    assert licensing.license_label(licensing.evaluate_lease(lease)) == "试用剩余 2 天 23 小时"


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


def test_signed_team_lease_preserves_the_server_selected_team_name(monkeypatch):
    private = ed25519.Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    monkeypatch.setattr(licensing, "LICENSE_SIGNING_PUBLIC_KEY", public)
    moment = datetime.now(UTC)
    payload = {
        "account_id": "member@example.com",
        "device_id": "device-0000000001",
        "plan": "lifetime",
        "issued_at": moment.isoformat(),
        "offline_until": (moment + timedelta(hours=24)).isoformat(),
        "subscription_until": None,
        "team_name": "秒选摄影团队",
    }
    payload["signature"] = base64.b64encode(
        private.sign(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    ).decode("ascii")

    assert licensing.verify_lease(payload).team_name == "秒选摄影团队"


def test_team_activation_uses_only_a_verified_team_email_code(monkeypatch):
    service = licensing.LicenseService(api_url="https://licenses.example.test")
    calls = []
    accepted = {}
    expected = _lease(plan=licensing.LicensePlan.LIFETIME, team_name="秒选摄影团队")
    monkeypatch.setattr(service, "device_id", lambda: "device-0000000001")
    monkeypatch.setattr(service, "_device_name", lambda: "QA Mac · Darwin")
    monkeypatch.setattr(
        service,
        "_post",
        lambda path, payload: calls.append((path, payload)) or {"lease": {}, "device_token": "x" * 36},
    )
    monkeypatch.setattr(service, "_accept_activation", lambda **kwargs: accepted.update(kwargs) or expected)

    service.request_team_activation_code(email="member@example.com")
    assert service.activate_team(email="member@example.com", code="123456") is expected
    assert calls == [
        ("/v1/team/activation/request-code", {"email": "member@example.com"}),
        (
            "/v1/team/activation/complete",
            {
                "email": "member@example.com", "code": "123456",
                "device_id": "device-0000000001", "device_name": "QA Mac · Darwin",
            },
        ),
    ]
    assert accepted["email"] == "member@example.com"


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


def test_public_package_cannot_enable_developer_access_with_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MIAOXUAN_INTERNAL_DEVELOPER_BUILD", "1")
    monkeypatch.setenv("MIAOXUAN_INTERNAL_BUILD", "1")

    assert not is_internal_developer_build(frozen=True, bundle_root=tmp_path)

    marker = tmp_path / "event_cull" / "internal_build_marker.txt"
    marker.parent.mkdir()
    marker.write_text("private package")
    assert is_internal_developer_build(frozen=True, bundle_root=tmp_path)
    assert not is_internal_developer_build(frozen=False, bundle_root=tmp_path)
