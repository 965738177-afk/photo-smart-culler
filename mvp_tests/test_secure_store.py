import sys
from types import SimpleNamespace

import pytest

from event_cull import secure_store, licensing


def security(monkeypatch, statuses, *, add_status=0):
    calls = []
    fake = SimpleNamespace(
        kSecClass="class", kSecClassGenericPassword="generic",
        kSecAttrService="service", kSecAttrAccount="account", kSecValueData="data",
        errSecItemNotFound=-25300, errSecDuplicateItem=-25299, errSecSuccess=0,
    )
    iterator = iter(statuses)
    fake.SecItemUpdate = lambda query, changes: calls.append(("update", query, changes)) or next(iterator)
    fake.SecItemAdd = lambda query, output: calls.append(("add", query)) or (add_status, None)
    monkeypatch.setitem(sys.modules, "Security", fake)
    return calls


def test_existing_item_is_updated_without_delete_or_recreate(monkeypatch):
    calls = security(monkeypatch, [0])
    secure_store.update_macos_secret("license", "lease", "new signed lease")
    assert [call[0] for call in calls] == ["update"]
    assert calls[0][2] == {"data": b"new signed lease"}


def test_missing_item_is_created_once(monkeypatch):
    calls = security(monkeypatch, [-25300])
    secure_store.update_macos_secret("license", "lease", "value")
    assert [call[0] for call in calls] == ["update", "add"]


def test_duplicate_creation_race_retries_in_place(monkeypatch):
    calls = security(monkeypatch, [-25300, 0], add_status=-25299)
    secure_store.update_macos_secret("license", "lease", "value")
    assert [call[0] for call in calls] == ["update", "add", "update"]


def test_denied_update_never_deletes_existing_item(monkeypatch):
    calls = security(monkeypatch, [-25293])
    with pytest.raises(RuntimeError, match="钥匙串写入失败"):
        secure_store.update_macos_secret("license", "lease", "value")
    assert [call[0] for call in calls] == ["update"]


def test_runtime_secret_cache_avoids_repeat_keychain_reads_and_tracks_writes(monkeypatch):
    reads = []
    service = licensing.LicenseService()
    backend = SimpleNamespace(get_password=lambda *_args: reads.append(True) or "old")
    monkeypatch.setattr(service, "_keyring", lambda: (backend, RuntimeError))
    monkeypatch.setattr(licensing.platform, "system", lambda: "Darwin")
    updates = []
    monkeypatch.setattr(secure_store, "update_macos_secret", lambda *args: updates.append(args))
    assert service._read_secret("lease") == "old"
    assert service._read_secret("lease") == "old"
    service._write_secret("lease", "new")
    assert service._read_secret("lease") == "new"
    assert len(reads) == len(updates) == 1


def test_failed_keychain_write_does_not_cache_unsaved_secret(monkeypatch):
    service = licensing.LicenseService()
    service._secret_cache["lease"] = "old"
    monkeypatch.setattr(service, "_keyring", lambda: (object(), RuntimeError))
    monkeypatch.setattr(licensing.platform, "system", lambda: "Darwin")

    def fail(*args):
        raise RuntimeError("denied")

    monkeypatch.setattr(secure_store, "update_macos_secret", fail)
    with pytest.raises(licensing.LicenseError):
        service._write_secret("lease", "new")
    assert service._read_secret("lease") == "old"
