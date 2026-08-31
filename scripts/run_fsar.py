#!/usr/bin/env python3
"""Stable entry point for the episodic Task-Adapter++ innovation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fsar.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

