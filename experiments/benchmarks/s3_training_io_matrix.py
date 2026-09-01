#!/usr/bin/env python3
"""Stage-3 single-card FULL training/I/O matrix orchestrator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
RUNNER = ROOT / "experiments" / "benchmarks" / "run_single_card_full.py"

from experiments.benchmarks.longrun_utils import (atomic_json, checked_stdout,
                                                   completed_result, open_campaign,
                                                   update_entry)
from ppt_evidence import command


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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    commit_id = checked_stdout(
        command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]), "git commit")
    campaign_config = {
        "model": args.model, "modes": args.modes, "seeds": args.seeds,
        "intervals": args.intervals, "total_steps": args.total_steps,
        "seq_len": args.seq_len, "chunks": args.chunks,
        "depths": args.depths, "npu": args.npu, "pci": args.pci,
        "shm_id": args.shm_id, "timeout": args.timeout,
    }
    campaign_path = root / "campaign.json"
    campaign = open_campaign(campaign_path, commit_id, campaign_config,
                             resume=args.resume)
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
                        command_line = [
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
                        result_path = run_dir / "result.json"
                        key = run_dir.name
                        prior = completed_result(campaign, key, result_path)
                        if prior is not None:
                            records.append({"seed": seed, "interval": interval,
                                            "mode": mode, "chunk": chunk,
                                            "depth": depth, "result": prior})
                            config_index += 1
                            continue
                        update_entry(campaign_path, campaign, key, "running",
                                     run_dir=str(run_dir), shm_id=args.shm_id + config_index)
                        run_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            proc = subprocess.run(command_line, text=True,
                                                  capture_output=True,
                                                  timeout=args.timeout * 3)
                        except subprocess.TimeoutExpired as error:
                            proc = subprocess.CompletedProcess(
                                error.cmd, 124, error.stdout or "", error.stderr or "")
                        (run_dir / "orchestrator.stdout.log").write_text(
                            proc.stdout or "", encoding="utf-8")
                        (run_dir / "orchestrator.stderr.log").write_text(
                            proc.stderr or "", encoding="utf-8")
                        if proc.returncode == 0 and result_path.exists():
                            result = json.loads(result_path.read_text(encoding="utf-8"))
                            if (result.get("status") == "pass" and
                                    (mode == "none" or
                                     (result.get("persisted") is True and
                                      result.get("restore_verified") is True and
                                      result.get("loss_allclose") is True and
                                      result.get("loaded_state_byte_exact") is True))):
                                records.append({"seed": seed, "interval": interval,
                                                "mode": mode, "chunk": chunk,
                                                "depth": depth, "result": result})
                                update_entry(campaign_path, campaign, key, "pass",
                                             result=str(result_path), returncode=0)
                            else:
                                failure = {"seed": seed, "interval": interval,
                                           "mode": mode, "chunk": chunk,
                                           "depth": depth, "returncode": 1,
                                           "error": "FULL restore gate failed"}
                                failures.append(failure)
                                update_entry(campaign_path, campaign, key, "fail",
                                             **failure)
                        else:
                            failure = {"seed": seed, "interval": interval,
                                       "mode": mode, "chunk": chunk,
                                       "depth": depth, "returncode": proc.returncode,
                                       "stdout": (proc.stdout or "")[-4000:],
                                       "stderr": (proc.stderr or "")[-4000:]}
                            failures.append(failure)
                            update_entry(campaign_path, campaign, key, "fail", **failure)
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
    atomic_json(root / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "passed": len(records),
                      "failed": len(failures), "output": str(root)}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
