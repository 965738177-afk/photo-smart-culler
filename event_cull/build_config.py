"""Build-time product flags.

Public packages remain fail-closed.  The private build uses a PyInstaller
runtime hook to set this environment flag before the application imports any
licensing code, so source files never need to be manually flipped before a
release.
"""

from __future__ import annotations

import os


INTERNAL_DEVELOPER_BUILD = os.environ.get("MIAOXUAN_INTERNAL_DEVELOPER_BUILD", "") == "1"
