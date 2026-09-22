"""Mark a package assembled for the owner's private, fully unlocked build."""

from __future__ import annotations

import os


os.environ["MIAOXUAN_INTERNAL_DEVELOPER_BUILD"] = "1"
