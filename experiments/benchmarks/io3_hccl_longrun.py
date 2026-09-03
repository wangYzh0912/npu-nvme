#!/usr/bin/env python3
"""Resumable IO-3 HCCL FULL long-run campaign.

Every child uses one long-lived HCCL source job, one SPDK owner, repeated
global commits, complete source-process exit, then fresh HCCL restore jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C2 = ROOT / "tests" / "hardware" / "c2_multirank_state.py"
sys.path.insert(0, str(ROOT / "python"))
from ppt_evidence import command, environment_snapshot  # noqa: E402


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(
        ROOT / "results/io-next-20260903/IO3_hccl_longrun"))
    parser.add_argument("--model", choices=("gpt2", "gpt2_xl"), default="gpt2")
    parser.add_argument("--world-sizes", nargs="+", type=int, default=(2, 4))
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--intervals", nargs="+", type=int, default=(10, 50))
    parser.add_argument("--total-steps", type=int, default=500)
    parser.add_argument("--continue-steps", nargs="+", type=int,
                        choices=(10, 100), default=(10, 100))
    parser.add_argument("--keep-last-n", type=int, default=3)
    parser.add_argument("--rank-devices-2", default="2,3")
    parser.add_argument("--rank-devices-4", default="0,1,2,3")
    parser.add_argument("--coordinator-npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=20269400)
    parser.add_argument("--master-port", type=int, default=8300)
    parser.add_argument("--timeout", type=float, default=43200)
    parser.add_argument("--restore-retained", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(world not in (2, 4) for world in args.world_sizes):
        raise ValueError("world size must be 2 or 4")
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items()
              if key not in ("resume", "dry_run")}
    config.update({"output_root": str(root), "scope": "FULL-only"})
    atomic_json(root / "config.json", config)
    atomic_json(root / "environment.json", environment_snapshot(
        pci=args.pci, npu="2,3/0,1,2,3", repo_root=ROOT,
        npu_info=command(["npu-smi", "info"])))
    matrix = list(itertools.product(args.world_sizes, args.seeds, args.intervals,
                                    args.continue_steps))
    campaign_path = root / "campaign.json"
    commit = command(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.resume and campaign_path.exists():
        campaign = json.loads(campaign_path.read_text())
        if campaign.get("config_digest") != digest:
            raise ValueError("resume config differs from existing campaign")
    else:
        campaign = {"status": "planned", "commit": commit,
                    "config_digest": digest, "entries": {}}
    atomic_json(campaign_path, campaign)
    failures = []
    for index, (world, seed, interval, continuation) in enumerate(matrix):
        key = f"{args.model}_w{world}_seed{seed}_i{interval}_c{continuation}"
        run_dir = root / key
        prior = campaign["entries"].get(key, {})
        result_path = run_dir / "result.json"
        if (args.resume and prior.get("status") == "pass" and result_path.exists() and
                json.loads(result_path.read_text()).get("status") == "pass"):
            continue
        devices = args.rank_devices_2 if world == 2 else args.rank_devices_4
        argv = [sys.executable, str(C2), "--run-dir", str(run_dir),
                "--model", args.model, "--world-size", str(world), "--hccl",
                "--rank-devices", devices, "--coordinator-npu", str(args.coordinator_npu),
                "--seed", str(seed), "--total-steps", str(args.total_steps),
                "--checkpoint-interval", str(interval),
                "--continue-steps", str(continuation),
                "--keep-last-n", str(args.keep_last_n), "--pci", args.pci,
                "--shm-id", str(args.shm_id + index),
                "--master-port", str(args.master_port + index * 10)]
        if args.restore_retained:
            argv.append("--restore-retained")
        campaign["entries"][key] = {"status": "planned" if args.dry_run else "running",
                                     "command": argv, "run_dir": str(run_dir)}
        campaign["status"] = "planned" if args.dry_run else "running"
        atomic_json(campaign_path, campaign)
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time_ns()
        try:
            completed = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True,
                                       timeout=args.timeout, check=False)
            rc, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as error:
            rc, stdout, stderr = 124, error.stdout or "", error.stderr or ""
        (run_dir / "stdout.log").write_text(stdout)
        (run_dir / "stderr.log").write_text(stderr)
        status = "pass" if rc == 0 and result_path.exists() and json.loads(
            result_path.read_text()).get("status") == "pass" else "fail"
        campaign["entries"][key].update({"status": status, "returncode": rc,
                                         "started_unix_ns": started,
                                         "finished_unix_ns": time.time_ns()})
        if status != "pass":
            failures.append(key)
        atomic_json(campaign_path, campaign)
    campaign["status"] = ("planned" if args.dry_run else
                          "pass" if not failures else "fail")
    campaign["failures"] = failures
    campaign["finished_unix_ns"] = time.time_ns()
    atomic_json(campaign_path, campaign)
    atomic_json(root / "result.json", campaign)
    print(json.dumps({"status": campaign["status"], "runs": len(matrix),
                      "failures": failures}, sort_keys=True))
    raise SystemExit(0 if campaign["status"] in ("pass", "planned") else 1)


if __name__ == "__main__":
    main()
