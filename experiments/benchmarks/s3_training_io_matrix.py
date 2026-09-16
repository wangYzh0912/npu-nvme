#!/usr/bin/env python3
"""Stage-3 single-card FULL training/I/O matrix orchestrator."""

from __future__ import annotations

import argparse
import itertools
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
from ppt_evidence import command, environment_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gpt2", "gpt2_xl"), default="gpt2_xl")
    parser.add_argument("--modes", nargs="+", choices=("none", "serial", "queue",
                                                        "async", "frozen_async",
                                                        "live_async"),
                        default=["none", "serial", "queue", "async"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--intervals", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--total-steps", type=int, default=110)
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--chunks", nargs="+", type=int, default=[4 * 1024**2])
    parser.add_argument("--depths", nargs="+", type=int, default=[4])
    parser.add_argument("--snapshot-slots", nargs="+", type=int, default=[1])
    parser.add_argument("--request-slots", nargs="+", type=int, default=[1])
    parser.add_argument("--generation-delays-ms", nargs="+", type=int, default=[0])
    parser.add_argument("--restore-retained", action="store_true")
    parser.add_argument("--npu", type=int, default=2)
    parser.add_argument("--numa-node", type=int, default=4)
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
    unsupported = []
    commit_id = checked_stdout(
        command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]), "git commit")
    campaign_config = {
        "model": args.model, "modes": args.modes, "seeds": args.seeds,
        "intervals": args.intervals, "total_steps": args.total_steps,
        "seq_len": args.seq_len, "chunks": args.chunks,
        "depths": args.depths, "npu": args.npu, "pci": args.pci,
        "snapshot_slots": args.snapshot_slots,
        "request_slots": args.request_slots,
        "generation_delays_ms": args.generation_delays_ms,
        "restore_retained": args.restore_retained,
        "numa_node": args.numa_node,
        "shm_id": args.shm_id, "timeout": args.timeout,
    }
    atomic_json(root / "config.json", {
        **campaign_config, "experiment": "single-card-FULL-training-matrix"})
    atomic_json(root / "environment.json", environment_snapshot(
        pci=args.pci, npu=str(args.npu), repo_root=ROOT,
        npu_info=command(["npu-smi", "info"])))
    campaign_path = root / "campaign.json"
    campaign = open_campaign(campaign_path, commit_id, campaign_config,
                             resume=args.resume)
    configs = itertools.product(
        args.seeds, args.intervals, args.chunks, args.depths, args.modes,
        args.snapshot_slots, args.request_slots, args.generation_delays_ms)
    for config_index, config in enumerate(configs):
        (seed, interval, chunk, depth, mode, snapshot_slots,
         request_slots, delay_ms) = config
        checkpoints = list(range(interval, args.total_steps, interval))
        if not checkpoints:
            parser.error("each interval must leave a checkpoint and continuation step")
        run_dir = root / (
            f"{args.model}_seed{seed}_i{interval}_{mode}_c{chunk}_d{depth}_"
            f"ss{snapshot_slots}_rs{request_slots}_delay{delay_ms}")
        command_line = [
            "numactl", f"--cpunodebind={args.numa_node}",
            f"--membind={args.numa_node}", sys.executable, str(RUNNER),
            "--model", args.model, "--mode", mode, "--checkpoint-steps",
            *[str(step) for step in checkpoints],
            "--total-steps", str(args.total_steps), "--seq-len", str(args.seq_len),
            "--seed", str(seed), "--npu", str(args.npu), "--pci", args.pci,
            "--chunk-size", str(chunk), "--pipeline-depth", str(depth),
            "--checkpoint-slots", str(snapshot_slots),
            "--request-slots", str(request_slots),
            "--generation-delay-ms", str(delay_ms),
            "--shm-id", str(args.shm_id + config_index),
            "--timeout", str(args.timeout), "--run-dir", str(run_dir),
        ]
        if args.restore_retained:
            command_line.append("--restore-retained")
        dimensions = {
            "seed": seed, "interval": interval, "mode": mode, "chunk": chunk,
            "depth": depth, "snapshot_slots": snapshot_slots,
            "request_slots": request_slots, "generation_delay_ms": delay_ms,
        }
        result_path = run_dir / "result.json"
        key = run_dir.name
        prior = completed_result(campaign, key, result_path)
        if prior is not None:
            records.append({**dimensions, "result": prior})
            continue
        update_entry(campaign_path, campaign, key, "running",
                     run_dir=str(run_dir), shm_id=args.shm_id + config_index)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(command_line, text=True, capture_output=True,
                                  timeout=args.timeout * 3)
        except subprocess.TimeoutExpired as error:
            proc = subprocess.CompletedProcess(
                error.cmd, 124, error.stdout or "", error.stderr or "")
        (run_dir / "orchestrator.stdout.log").write_text(
            proc.stdout or "", encoding="utf-8")
        (run_dir / "orchestrator.stderr.log").write_text(
            proc.stderr or "", encoding="utf-8")
        if proc.returncode == 2 and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "unsupported":
                unsupported.append({**dimensions, "result": result})
                update_entry(campaign_path, campaign, key, "unsupported",
                             result=str(result_path), returncode=2)
                continue
        if proc.returncode == 0 and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            gate_ok = (result.get("status") == "pass" and
                       (mode == "none" or
                        (result.get("persisted") is True and
                         result.get("restore_verified") is True and
                         result.get("loss_allclose") is True and
                         result.get("loaded_state_byte_exact") is True)))
            if gate_ok:
                records.append({**dimensions, "result": result})
                update_entry(campaign_path, campaign, key, "pass",
                             result=str(result_path), returncode=0)
                continue
        failure = {**dimensions, "returncode": proc.returncode,
                   "stdout": (proc.stdout or "")[-4000:],
                   "stderr": (proc.stderr or "")[-4000:],
                   "error": "FULL restore gate failed"}
        failures.append(failure)
        update_entry(campaign_path, campaign, key, "fail", **failure)
        if args.fail_fast:
            break
    summary = {"status": "pass" if not failures else "fail",
               "records": records, "unsupported": unsupported,
               "failures": failures,
               "live_async_supported": not any(
                   item["mode"] == "live_async" for item in unsupported)}
    atomic_json(root / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "passed": len(records),
                      "unsupported": len(unsupported), "failed": len(failures),
                      "output": str(root)}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
