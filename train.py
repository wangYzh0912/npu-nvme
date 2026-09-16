#!/usr/bin/env python3
"""Unified, explicitly configured training and fresh-process recovery entry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'python'))
from npu_nvme.cli.qwen_training import main
if __name__ == '__main__':
    raise SystemExit(main())
