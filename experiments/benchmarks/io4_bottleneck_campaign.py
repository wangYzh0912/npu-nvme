#!/usr/bin/env python3
"""Resumable IO-4 B0-B5 bottleneck decomposition and Reactor decision."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "experiments/benchmarks/io4_unix_pipeline.py"
ASYNC = ROOT / "experiments/benchmarks/s2_async_data_plane.py"
sys.path.insert(0, str(ROOT / "python"))
from ppt_evidence import command, environment_snapshot  # noqa: E402


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(fraction * (len(values) - 1)))]


def aggregate(records):
    groups = {}
    for item in records:
        key = tuple(item[name] for name in
                    ("path", "producers", "payload", "chunk", "depth",
                     "numa_node", "slow_delay_ms"))
        groups.setdefault(key, []).append(item["result"])
    output = []
    for key, values in groups.items():
        throughput = [item["throughput_bytes_per_second"] for item in values]
        elapsed = [item["elapsed_ns"] / 1e6 for item in values]
        output.append({
            **dict(zip(("path", "producers", "payload", "chunk", "depth",
                        "numa_node", "slow_delay_ms"), key)),
            "samples": len(values), "throughput_mean": statistics.mean(throughput),
            "throughput_p50": statistics.median(throughput),
            "latency_ms_p99": percentile(elapsed, 0.99),
            "all_byte_exact": all(item.get("byte_exact") is True for item in values),
            "reactor_cpu_fraction_mean": statistics.mean([
                item.get("spdk_stats", {}).get("reactor_cpu_us", 0) /
                max(item["elapsed_ns"] / 1000.0, 1) for item in values]),
            "queue_wait_fraction_mean": statistics.mean([
                item.get("coordinator_queue_wait_ns", 0) /
                max(item["elapsed_ns"] * max(1, item.get("producers", 1)), 1)
                for item in values]),
            "nvme_outstanding_peak": max(
                item.get("spdk_stats", {}).get("nvme_outstanding_peak", 0)
                for item in values),
        })
    return output


def reactor_decision(groups):
    b0 = [item for item in groups if item["path"] == "B0" and
          item["numa_node"] == 4 and item["slow_delay_ms"] == 0]
    b4 = [item for item in groups if item["path"] == "B4" and
          item["numa_node"] == 4 and item["slow_delay_ms"] == 0]
    if not b0 or not b4:
        return {"implement_multi_reactor": False, "status": "insufficient_evidence",
                "missing": [name for name, values in (("B0", b0), ("B4", b4))
                            if not values]}
    calibration = max(item["throughput_mean"] for item in b0)
    candidate = max(b4, key=lambda item: item["throughput_mean"])
    gates = {
        "single_reactor_cpu_saturated": candidate["reactor_cpu_fraction_mean"] >= 0.90,
        "reactor_queue_significant": candidate["queue_wait_fraction_mean"] >= 0.20,
        "throughput_below_calibration": candidate["throughput_mean"] < 0.80 * calibration,
        "qpair_underfed": candidate["nvme_outstanding_peak"] < candidate["depth"],
        # This campaign has one owner by construction.  A separate controlled
        # multi-owner result must be supplied before implementation is allowed.
        "controlled_multi_owner_gain": False,
    }
    return {
        "implement_multi_reactor": all(gates.values()), "status": "decided",
        "gates": gates, "calibration_bytes_per_second": calibration,
        "best_b4": candidate,
        "reason": ("all decision gates passed" if all(gates.values()) else
                   "single-Reactor evidence does not satisfy every implementation gate"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(
        ROOT / "results/io-next-20260903/IO4_bottleneck"))
    parser.add_argument("--paths", nargs="+", choices=("B0", "B1", "B2", "B3", "B4"),
                        default=("B0", "B1", "B2", "B3", "B4"))
    parser.add_argument("--producers", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--payloads", nargs="+", type=int, default=(256 * 1024**2,))
    parser.add_argument("--chunks", nargs="+", type=int,
                        default=(1 * 1024**2, 4 * 1024**2, 16 * 1024**2))
    parser.add_argument("--depths", nargs="+", type=int, default=(2, 4, 8))
    parser.add_argument("--numa-nodes", nargs="+", type=int, default=(4, 0))
    parser.add_argument("--slow-delays-ms", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--rank-devices", default="2,3,0,1")
    parser.add_argument("--coordinator-npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--offset", type=int, default=160 * 1024**3)
    parser.add_argument("--shm-id", type=int, default=20269500)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0:
        raise ValueError("invalid sample counts")
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items()
              if key not in ("resume", "dry_run")}
    config["output_root"] = str(root)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    campaign_path = root / "campaign.json"
    if args.resume and campaign_path.exists():
        campaign = json.loads(campaign_path.read_text())
        if campaign.get("config_digest") != digest:
            raise ValueError("resume config differs from existing campaign")
    else:
        campaign = {"status": "planned", "config_digest": digest,
                    "commit": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                    "entries": {}}
    atomic_json(root / "config.json", config)
    atomic_json(root / "environment.json", environment_snapshot(
        pci=args.pci, npu=args.rank_devices, repo_root=ROOT,
        npu_info=command(["npu-smi", "info"])))
    atomic_json(campaign_path, campaign)
    records, failures = [], []
    index = 0
    for path, producer, payload, chunk, depth, numa, delay in itertools.product(
            args.paths, args.producers, args.payloads, args.chunks, args.depths,
            args.numa_nodes, args.slow_delays_ms):
        if path in ("B0", "B1") and producer != 1:
            continue
        for iteration in range(args.warmups + args.samples):
            warmup = iteration < args.warmups
            sample = iteration if warmup else iteration - args.warmups
            label = f"w{iteration:02d}" if warmup else f"s{sample:02d}"
            key = (f"{path}_p{producer}_bytes{payload}_c{chunk}_d{depth}_"
                   f"numa{numa}_slow{delay:g}_{label}")
            run_dir = root / key
            result_path = run_dir / "result.json"
            prior = campaign["entries"].get(key, {})
            if (args.resume and prior.get("status") == "pass" and result_path.exists()):
                result = json.loads(result_path.read_text())
            else:
                if path == "B1":
                    argv = [sys.executable, str(ASYNC), "--child", "--run-dir", str(run_dir),
                            "--payload", str(payload), "--chunk", str(chunk),
                            "--depth", str(depth), "--mode", "async", "--npu", "2",
                            "--pci", args.pci, "--offset", str(args.offset),
                            "--shm-id", str(args.shm_id + index)]
                else:
                    argv = [sys.executable, str(PIPELINE), "--path", path,
                            "--producers", str(producer), "--payload", str(payload),
                            "--chunk", str(chunk), "--depth", str(depth),
                            "--rank-devices", args.rank_devices,
                            "--coordinator-npu", str(args.coordinator_npu),
                            "--pci", args.pci, "--offset", str(args.offset),
                            "--sink-delay-ms", str(delay), "--run-dir", str(run_dir)]
                argv = ["numactl", f"--cpunodebind={numa}", f"--membind={numa}", *argv]
                campaign["entries"][key] = {
                    "status": "planned" if args.dry_run else "running",
                    "command": argv, "run_dir": str(run_dir)}
                atomic_json(campaign_path, campaign)
                if args.dry_run:
                    index += 1
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                try:
                    completed = subprocess.run(argv, cwd=ROOT, text=True,
                                               capture_output=True, timeout=args.timeout,
                                               check=False)
                    rc = completed.returncode
                    (run_dir / "stdout.log").write_text(completed.stdout)
                    (run_dir / "stderr.log").write_text(completed.stderr)
                except subprocess.TimeoutExpired as error:
                    rc = 124
                    (run_dir / "stdout.log").write_text(error.stdout or "")
                    (run_dir / "stderr.log").write_text(error.stderr or "")
                if path == "B1" and result_path.exists():
                    raw = json.loads(result_path.read_text())
                    result = {
                        **raw, "path": "B1", "source": "hbm", "sink": "spdk",
                        "producers": 1, "payload_per_producer": payload,
                        "pipeline_depth": depth,
                        "throughput_bytes_per_second": payload / raw["elapsed_seconds"],
                        "elapsed_ns": int(raw["elapsed_seconds"] * 1e9),
                        "byte_exact": raw["expected_sha256"] == raw["readback_sha256"],
                        "spdk_stats": raw.get("stats", {}),
                    }
                    atomic_json(result_path, result)
                elif result_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    result = {"status": "fail"}
                status = "pass" if rc == 0 and result.get("status") == "pass" else "fail"
                campaign["entries"][key].update({"status": status, "returncode": rc})
                atomic_json(campaign_path, campaign)
            if not warmup and result.get("status") == "pass":
                records.append({"path": path, "producers": producer,
                                "payload": payload, "chunk": chunk, "depth": depth,
                                "numa_node": numa, "slow_delay_ms": delay,
                                "sample": sample, "result": result})
            elif not args.dry_run and result.get("status") != "pass":
                failures.append(key)
            index += 1
    groups = aggregate(records)
    decision = reactor_decision(groups)
    status = "planned" if args.dry_run else "pass" if not failures else "fail"
    summary = {"status": status, "records": records, "groups": groups,
               "failures": failures, "multi_reactor_decision": decision,
               "b5_source": "IO3_hccl_longrun results and C2 coordinator metrics"}
    campaign.update({"status": status, "failures": failures,
                     "finished_unix_ns": time.time_ns()})
    atomic_json(campaign_path, campaign)
    atomic_json(root / "result.json", summary)
    print(json.dumps({"status": status, "samples": len(records),
                      "failures": failures, "decision": decision}, sort_keys=True))
    raise SystemExit(0 if status in ("pass", "planned") else 1)


if __name__ == "__main__":
    main()
