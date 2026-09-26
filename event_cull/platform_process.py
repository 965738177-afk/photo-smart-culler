"""Platform flags for command-line helpers launched by the windowed app."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes


def no_console_creationflags() -> int:
    """Keep Windows helpers from briefly opening a console window."""
    if sys.platform != "win32":
        return 0
    # The fallback lets tests exercise Windows behavior on a non-Windows host.
    return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


class _ProcessEntry32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


def running_executable_names() -> set[str]:
    """List Windows process names without starting a console or blocking on tasklist."""
    if sys.platform != "win32":
        return set()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    snapshot = kernel32.CreateToolhelp32Snapshot
    snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    snapshot.restype = wintypes.HANDLE
    first = kernel32.Process32FirstW
    first.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W))
    first.restype = wintypes.BOOL
    following = kernel32.Process32NextW
    following.argtypes = first.argtypes
    following.restype = wintypes.BOOL
    close = kernel32.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL

    handle = snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    if handle == ctypes.c_void_p(-1).value:
        return set()
    try:
        entry = _ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        names: set[str] = set()
        if first(handle, ctypes.byref(entry)):
            while True:
                names.add(entry.szExeFile.casefold())
                if not following(handle, ctypes.byref(entry)):
                    break
        return names
    finally:
        close(handle)
