"""Compatibility wrapper — prefer: python scripts/run_baseline.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    from run_baseline import main as run_main

    return run_main(["--train-rows", "all"] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())