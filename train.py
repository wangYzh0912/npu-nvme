#!/usr/bin/env python3
"""Unified, explicitly configured training and fresh-process recovery entry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'python'))
from npu_nvme.cli.training import main
if __name__ == '__main__':
    if sys.argv[1:2]==['qwen-d2']:
        from tools.run_qwen_d2 import main as qwen_main
        raise SystemExit(qwen_main(sys.argv[2:]))
    raise SystemExit(main())
