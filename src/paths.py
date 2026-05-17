"""Shared path helpers for cold-start experiment workflows."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
PAIRWISE_ARTIFACTS_DIR = ARTIFACTS_ROOT / "pairwise_alignment"
PAPER_REPRO_ARTIFACTS_DIR = ARTIFACTS_ROOT / "paper_repro"
PAPER_COMPARISON_ARTIFACTS_DIR = ARTIFACTS_ROOT / "paper_comparison"
