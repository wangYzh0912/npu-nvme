#!/usr/bin/env python3
"""Unified frozen training, benchmark, restart and inspection entry."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'python'))
from experiments.baselines.repro.entry import main
if __name__ == '__main__':
    raise SystemExit(main())
