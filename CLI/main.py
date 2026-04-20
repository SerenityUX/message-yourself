#!/usr/bin/env python3
"""Shim — use ``python3 cli.py``."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cli import main

if __name__ == "__main__":
    main()
