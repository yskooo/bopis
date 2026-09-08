"""Entry point for ``python -m bopis``."""

from __future__ import annotations

import sys

from bopis.cli import main

if __name__ == "__main__":
    sys.exit(main())
