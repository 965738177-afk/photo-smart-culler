"""Build-time product flags for the separately packaged internal app.

The public app must never accept a user-controlled environment variable as
proof that it is an internal build.  Only the private packaging target bundles
the marker file below.
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_internal_developer_build(
    *, frozen: bool | None = None, bundle_root: Path | None = None
) -> bool:
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    if bundle_root is None:
        bundle_root = Path(getattr(sys, "_MEIPASS", ""))
    return frozen and (bundle_root / "event_cull" / "internal_build_marker.txt").is_file()


INTERNAL_DEVELOPER_BUILD = is_internal_developer_build()
