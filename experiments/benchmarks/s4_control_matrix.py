#!/usr/bin/env python3
"""Stage-4 FULL backpressure, fault, and lifecycle long-run matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
TRAINING = ROOT / "experiments" / "benchmarks" / "run_single_card_full.py"
FAULTS = ROOT / "tests" / "hardware" / "stage4_fault_lifecycle.py"

from experiments.benchmarks.longrun_utils import (atomic_json, checked_stdout,
                                                   completed_result, open_campaign,
                                                   update_entry)
from ppt_evidence import command


def execute(command_line, run_dir, timeout):
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.run(command_line, capture_output=True, text=True,
                                 timeout=timeout)
    except subprocess.TimeoutExpired as error:
        process = subprocess.CompletedProcess(
            error.cmd, 124, error.stdout or "", error.stderr or "")
    (run_dir / "orchestrator.stdout.log").write_text(
        process.stdout or "", encoding="utf-8")
    (run_dir / "orchestrator.stderr.log").write_text(
        process.stderr or "", encoding="utf-8")
    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dma-slots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--checkpoint-slots", nargs="+", type=int,
                        default=[1, 2, 4])
    parser.add_argument("--delays-ms", nargs="+", type=int,
                        default=[0, 100, 1000, 5000])
    parser.add_argument("--intervals", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=26000)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items()
              if key not in {"resume", "fail_fast", "output_root"}}
    commit_id = checked_stdout(
        command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]), "git commit")
    campaign_path = root / "campaign.json"
    campaign = open_campaign(campaign_path, commit_id, config, resume=args.resume)
    records = []
    failures = []
    ordinal = 0

    for dma_slots in args.dma_slots:
        for checkpoint_slots in args.checkpoint_slots:
            for delay_ms in args.delays_ms:
                for interval in args.intervals:
                    key = (f"pressure_dma{dma_slots}_ckpt{checkpoint_slots}_"
                           f"delay{delay_ms}_i{interval}")
                    run_dir = root / key
                    result_path = run_dir / "result.json"
                    prior = completed_result(campaign, key, result_path)
                    if prior is not None:
                        records.append({"kind": "pressure", "key": key,
                                        "result": prior})
                        ordinal += 1
                        continue
                    steps = [interval * value for value in range(1, 7)]
                    command_line = [
                        sys.executable, str(TRAINING), "--model", "gpt2",
                        "--mode", "async", "--checkpoint-steps",
                        *[str(step) for step in steps], "--total-steps",
                        str(interval * 6 + 2), "--seq-len", str(args.seq_len),
                        "--seed", str(args.seed), "--npu", str(args.npu),
                        "--pci", args.pci, "--chunk-size", str(4 * 1024**2),
                        "--pipeline-depth", str(dma_slots),
                        "--checkpoint-slots", str(checkpoint_slots),
                        "--admission", "try", "--generation-delay-ms",
                        str(delay_ms), "--shm-id", str(args.shm_id + ordinal),
                        "--timeout", str(args.timeout), "--run-dir", str(run_dir),
                    ]
                    update_entry(campaign_path, campaign, key, "running",
                                 kind="pressure", run_dir=str(run_dir),
                                 shm_id=args.shm_id + ordinal)
                    process = execute(command_line, run_dir, args.timeout * 3)
                    result = (json.loads(result_path.read_text(encoding="utf-8"))
                              if result_path.exists() else {})
                    passed = (process.returncode == 0 and result.get("status") == "pass"
                              and result.get("persisted") is True
                              and result.get("restore_verified") is True
                              and result.get("loss_allclose") is True
                              and result.get("loaded_state_byte_exact") is True
                              and int(result.get("accepted_generations", 0)) > 0)
                    if passed:
                        records.append({"kind": "pressure", "key": key,
                                        "result": result})
                        update_entry(campaign_path, campaign, key, "pass",
                                     result=str(result_path), returncode=0)
                    else:
                        failure = {"kind": "pressure", "key": key,
                                   "returncode": process.returncode,
                                   "error": "pressure FULL restore gate failed"}
                        failures.append(failure)
                        update_entry(campaign_path, campaign, key, "fail", **failure)
                    ordinal += 1
                    if failures and args.fail_fast:
                        break
                if failures and args.fail_fast:
                    break
            if failures and args.fail_fast:
                break
        if failures and args.fail_fast:
            break

    if not failures or not args.fail_fast:
        for dma_slots in args.dma_slots:
            for checkpoint_slots in args.checkpoint_slots:
                key = f"fault_dma{dma_slots}_ckpt{checkpoint_slots}"
                run_dir = root / key
                result_path = run_dir / "result.json"
                prior = completed_result(campaign, key, result_path)
                if prior is not None:
                    records.append({"kind": "fault", "key": key,
                                    "result": prior})
                    ordinal += 1
                    continue
                command_line = [
                    sys.executable, str(FAULTS), "--depth", str(dma_slots),
                    "--checkpoint-slots", str(checkpoint_slots), "--npu",
                    str(args.npu), "--pci", args.pci, "--shm-id",
                    str(args.shm_id + ordinal * 400), "--output", str(run_dir),
                ]
                update_entry(campaign_path, campaign, key, "running", kind="fault",
                             run_dir=str(run_dir), shm_id=args.shm_id + ordinal * 400)
                process = execute(command_line, run_dir, args.timeout)
                result = (json.loads(result_path.read_text(encoding="utf-8"))
                          if result_path.exists() else {})
                if process.returncode == 0 and result.get("status") == "pass":
                    records.append({"kind": "fault", "key": key,
                                    "result": result})
                    update_entry(campaign_path, campaign, key, "pass",
                                 result=str(result_path), returncode=0)
                else:
                    failure = {"kind": "fault", "key": key,
                               "returncode": process.returncode,
                               "error": "fault/lifecycle gate failed"}
                    failures.append(failure)
                    update_entry(campaign_path, campaign, key, "fail", **failure)
                ordinal += 1
                if failures and args.fail_fast:
                    break
            if failures and args.fail_fast:
                break

    pressure = [item for item in records if item["kind"] == "pressure"]
    fault_records = [item for item in records if item["kind"] == "fault"]
    accepted = sum(int(item["result"].get("accepted_generations", 0))
                   for item in pressure)
    busy = sum(int(item["result"].get("busy_requests", 0))
               for item in pressure)
    requested = sum(int(item["result"].get("samples", 0))
                    for item in pressure)
    request_ring_busy = any(
        case.get("case") == "request_ring_busy" and
        case.get("status") == "pass" and int(case.get("busy_rc", 0)) != 0
        for item in fault_records for case in item["result"].get("cases", []))
    explicit_busy = busy > 0 or request_ring_busy
    if pressure and not explicit_busy:
        failures.append({"kind": "backpressure_gate",
                         "error": "no explicit FULL checkpoint BUSY was observed"})
    if pressure and accepted + busy != requested:
        failures.append({"kind": "admission_accounting_gate",
                         "requested": requested, "accepted": accepted,
                         "busy": busy})
    summary = {"status": "pass" if not failures else "fail",
               "expected_pressure_configs": (len(args.dma_slots) *
                                             len(args.checkpoint_slots) *
                                             len(args.delays_ms) *
                                             len(args.intervals)),
               "expected_fault_configs": (len(args.dma_slots) *
                                          len(args.checkpoint_slots)),
               "backpressure": {"requested_checkpoints": requested,
                                "accepted_generations": accepted,
                                "busy_requests": busy,
                                "training_admission_busy_observed": busy > 0,
                                "request_ring_busy_observed": request_ring_busy,
                                "explicit_busy_observed": explicit_busy},
               "records": records, "failures": failures}
    atomic_json(root / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "passed": len(records),
                      "failed": len(failures), "output": str(root)}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
