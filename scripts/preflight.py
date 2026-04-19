#!/usr/bin/env python3
"""Pre-flight check — enumerate every model path in config.yaml and
report which ones exist. Run this before the first `python main.py` on
a new device to surface missing downloads.

    python3 scripts/preflight.py

Exit code is 0 when all required files are present, 1 otherwise.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from companion.core.config import load_config  # noqa: E402
from companion.core.readiness import check_all  # noqa: E402


def main() -> int:
    cfg = load_config(os.path.join(_ROOT, "config.yaml"))
    report = check_all(cfg)
    report.print()
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
