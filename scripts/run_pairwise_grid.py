#!/usr/bin/env python3
"""Canonical CLI entrypoint for pairwise cold-start experiments."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.pairwise_grid import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
