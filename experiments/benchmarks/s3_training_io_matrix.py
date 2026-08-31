#!/usr/bin/env python3
"""Stage-3 single-card FULL training/I/O matrix orchestrator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "benchmarks" / "run_single_card_full.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gpt2", "gpt2_xl"), default="gpt2_xl")
    parser.add_argument("--modes", nargs="+", choices=("none", "serial", "queue", "async"),
                        default=["none", "serial", "queue", "async"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--intervals", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--total-steps", type=int, default=110)
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--chunks", nargs="+", type=int, default=[4 * 1024**2])
    parser.add_argument("--depths", nargs="+", type=int, default=[4])
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=18700)
    parser.add_argument("--output-root", default=str(ROOT / "results" / "stage3-training-io"))
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    config_index = 0
    for seed in args.seeds:
        for interval in args.intervals:
            checkpoints = list(range(interval, args.total_steps, interval))
            if not checkpoints:
                parser.error("each interval must leave a checkpoint and continuation step")
            for chunk in args.chunks:
                for depth in args.depths:
                    for mode in args.modes:
                        run_dir = root / (
                            f"{args.model}_seed{seed}_i{interval}_{mode}_"
                            f"c{chunk}_d{depth}")
                        command = [
                            sys.executable, str(RUNNER), "--model", args.model,
                            "--mode", mode, "--checkpoint-steps",
                            *[str(step) for step in checkpoints],
                            "--total-steps", str(args.total_steps),
                            "--seq-len", str(args.seq_len), "--seed", str(seed),
                            "--npu", str(args.npu), "--pci", args.pci,
                            "--chunk-size", str(chunk), "--pipeline-depth", str(depth),
                            "--shm-id", str(args.shm_id + config_index),
                            "--timeout", str(args.timeout), "--run-dir", str(run_dir),
                        ]
                        proc = subprocess.run(command, text=True, capture_output=True)
                        result_path = run_dir / "result.json"
                        if proc.returncode == 0 and result_path.exists():
                            result = json.loads(result_path.read_text(encoding="utf-8"))
                            records.append({"seed": seed, "interval": interval,
                                            "mode": mode, "chunk": chunk,
                                            "depth": depth, "result": result})
                        else:
                            failures.append({"seed": seed, "interval": interval,
                                             "mode": mode, "chunk": chunk,
                                             "depth": depth, "returncode": proc.returncode,
                                             "stdout": proc.stdout[-4000:],
                                             "stderr": proc.stderr[-4000:]})
                        config_index += 1
                        if failures and args.fail_fast:
                            break
                    if failures and args.fail_fast:
                        break
                if failures and args.fail_fast:
                    break
            if failures and args.fail_fast:
                break
        if failures and args.fail_fast:
            break
    summary = {"status": "pass" if not failures else "fail",
               "records": records, "failures": failures}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "passed": len(records),
                      "failed": len(failures), "output": str(root)}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
