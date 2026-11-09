#!/usr/bin/env python3
"""Unified, explicitly configured training and fresh-process recovery entry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'python'))
if len(sys.argv) > 1 and sys.argv[1].startswith('incremental-'):
    from npu_nvme.cli.incremental_feasibility import main as incremental_main
    command = sys.argv.pop(1).removeprefix('incremental-')
    raise SystemExit(incremental_main([command, *sys.argv[1:]]))

from npu_nvme.cli.qwen_training import main
if __name__ == '__main__':
    raise SystemExit(main())
