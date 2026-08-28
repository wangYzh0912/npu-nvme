#!/usr/bin/env python3
"""E1 filesystem half: io_uring Buffered FS/O_DIRECT on the isolated 84 disk.

The benchmark delegates asynchronous I/O to fio's io_uring engine.  Each
configuration uses one fio job with 10 warmups and 30 formal operations; the
fio JSON is retained verbatim in ``raw/``.  ``fsync=1`` or ``fdatasync=1`` is
part of every job, so a successful sample means the configured persistence
boundary was reached, not merely that data entered the page cache.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from ppt_evidence import (EvidenceBundle, command, environment_snapshot,  # noqa: E402
                          stats)


SIZES = (4 * 1024, 64 * 1024, 1024 * 1024, 4 * 1024 * 1024,
         256 * 1024 * 1024)
DEPTHS = (1, 4)


def fio_json(argv, raw_path):
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(argv, capture_output=True, text=True,
                                   check=False, timeout=1800)
        raw_path.write_text(completed.stdout, encoding="utf-8")
        data = json.loads(completed.stdout) if completed.stdout else {}
        return data, {"returncode": completed.returncode,
                      "stderr": completed.stderr,
                      "wall_ms": (time.monotonic_ns() - started) / 1e6}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        return {}, {"returncode": -1, "error": repr(error),
                    "wall_ms": (time.monotonic_ns() - started) / 1e6}


def native_stats(section):
    value = section.get("lat_ns", {})
    n = int(value.get("N", 0))
    if not n:
        return stats([])
    result = {"n": n, "mean_ns": value.get("mean"),
              "stdev_ns": value.get("stddev"), "min_ns": value.get("min"),
              "max_ns": value.get("max")}
    percentiles = value.get("percentile", {})
    result["p50_ns"] = percentiles.get("50.000000")
    result["p95_ns"] = percentiles.get("95.000000")
    result["p99_ns"] = percentiles.get("99.000000") if n >= 30 else None
    result["p99_status"] = "reported" if n >= 30 else f"not reported (n={n}<30)"
    return result


def run_one(args, mode, size, depth, operation):
    label = f"{mode}_{operation}_size{size}_qd{depth}"
    config = {
        "experiment": "E1", "model": "synthetic_io", "seed": 17,
        "mode": mode, "operation": operation, "size_bytes": size,
        "chunk_size": size, "pipeline_depth": depth, "slot_count": depth,
        "queue_depth": depth, "warmups": args.warmups,
        "formal_samples": args.samples, "ioengine": "io_uring",
        "persistence": "fsync" if not args.fdatasync else "fdatasync",
        "filesystem_pci": "0000:84:00.0", "filesystem": "XFS",
        "directory_policy": str(args.root), "cross_disk_label": True,
    }
    bundle = EvidenceBundle(
        "E1", config, repo_root=ROOT,
        environment=environment_snapshot(
            pci="0000:84:00.0", npu=str(args.npu), numa="recorded by snapshot",
            repo_root=ROOT, npu_info=command(["npu-smi", "info"])))
    file_path = args.root / f"{label}.bin"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    def make_argv(loops):
        argv = ["fio", f"--name={label}", f"--filename={file_path}",
                f"--rw={operation}", "--ioengine=io_uring",
                f"--iodepth={depth}", "--numjobs=1", f"--bs={size}",
                f"--size={size}", f"--loops={loops}", "--group_reporting",
                "--output-format=json+"]
        argv.append("--direct=0" if mode == "buffered" else "--direct=1")
        argv.append("--fdatasync=1" if args.fdatasync else "--fsync=1")
        return argv

    warmup_data, warmup_exec = fio_json(
        make_argv(args.warmups), bundle.raw_dir / "fio_warmup.json")
    data, execution = fio_json(
        make_argv(args.samples), bundle.raw_dir / "fio_formal.json")
    job = (data.get("jobs") or [{}])[0]
    section = job.get("write" if operation == "write" else "read", {})
    n = int(section.get("lat_ns", {}).get("N", 0))
    successful = (warmup_exec.get("returncode") == 0 and
                  execution.get("returncode") == 0 and n >= args.samples)
    if successful:
        # fio's formal-run latency summary is authoritative.  Per-I/O
        # timestamps are not available from fio JSON and are not fabricated.
        mean_ns = float(section["lat_ns"]["mean"])
        for index in range(args.samples):
            bundle.add_sample({
                "status": "pass", "request_id": f"{label}/{index:04d}",
                "operation": operation, "warmup": False,
                "latency_ns": mean_ns, "fio_sample": index,
                "latency_source": "fio formal aggregate mean; native distribution in raw/fio_formal.json",
                "bytes": size, "events": [{"name": "fio_submit_complete"}],
            })
    else:
        bundle.add_failure({"status": "fail", "operation": operation,
                            "command": make_argv(args.samples),
                            "warmup_execution": warmup_exec,
                            "execution": execution,
                            "fio_error": job.get("error")})
    formal = args.samples if successful else 0
    lat_ns = section.get("lat_ns", {})
    latency = native_stats(section)
    result = bundle.finalize(metrics={
        "model": "synthetic_io", "seed": 17, "mode": mode,
        "state_bytes": size, "logical_bytes": size,
        "physical_bytes": size, "chunk_size": size,
        "pipeline_depth": depth, "slot_count": depth,
        "latency_mean": latency.get("mean_ns"),
        "latency_p50": latency.get("p50_ns"),
        "latency_p95": latency.get("p95_ns"),
        "throughput": (size / (float(lat_ns.get("mean", 0)) / 1e9)
                       if lat_ns.get("mean") else None),
        "pcie_bytes": 0, "nvme_bytes": size * formal,
        "fault_results": {"fio_returncode": execution.get("returncode")},
        "native_fio_latency": latency,
        "fio_execution": {"warmup": warmup_exec, "formal": execution},
        "fio_command": make_argv(args.samples),
        "sample_note": "warmup and formal runs are separate; only formal samples are reported",
    }, status="pass" if successful else "fail")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("/models/npu_nvme_exp/ppt-evidence-20260829"))
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    parser.add_argument("--modes", choices=("buffered", "odirect"),
                        nargs="+", default=("buffered", "odirect"))
    parser.add_argument("--operations", choices=("write", "read"),
                        nargs="+", default=("write", "read"))
    parser.add_argument("--fdatasync", action="store_true")
    args = parser.parse_args()
    if args.samples < 30 and not (args.warmups == 1 and args.samples == 1):
        raise SystemExit("formal samples must be >=30; use exactly 1+1 only for smoke")
    args.root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for mode in args.modes:
        for operation in args.operations:
            for size in args.sizes:
                for depth in args.depths:
                    result = run_one(args, mode, size, depth, operation)
                    failures += result["status"] != "pass"
                    print(json.dumps({"run_id": result["run_id"],
                                      "status": result["status"],
                                      "mode": mode, "operation": operation,
                                      "size": size, "depth": depth},
                                     sort_keys=True), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
