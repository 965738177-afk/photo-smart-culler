"""Update macOS secrets in place so existing keychain access rules survive."""
from __future__ import annotations


def update_macos_secret(service: str, account: str, value: str) -> None:
    import Security

    query = {
        Security.kSecClass: Security.kSecClassGenericPassword,
        Security.kSecAttrService: service,
        Security.kSecAttrAccount: account,
    }
    changes = {Security.kSecValueData: value.encode("utf-8")}
    status = Security.SecItemUpdate(query, changes)
    if status == Security.errSecItemNotFound:
        result = Security.SecItemAdd({**query, **changes}, None)
        status = result[0] if isinstance(result, tuple) else result
        if status == Security.errSecDuplicateItem:
            status = Security.SecItemUpdate(query, changes)
    if status != Security.errSecSuccess:
        raise RuntimeError(f"系统钥匙串写入失败（状态 {status}）。")
