#!/usr/bin/env python3
"""P2 software-stack instrumentation for representative P1 workloads.

The collector never fabricates unavailable layer times.  It records perf,
strace and tracefs capabilities, computes only disjoint intervals that can be
observed, and marks the result ``degraded`` when the <=10% closure target is
not demonstrable.
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
from ppt_evidence import EvidenceBundle, command, environment_snapshot


def run(argv, timeout, output):
    started = time.perf_counter_ns()
    try:
        proc = subprocess.run(argv, text=True, capture_output=True,
                              check=False, timeout=timeout)
        output.write_text(proc.stdout + ("\n[stderr]\n" + proc.stderr
                                         if proc.stderr else ""), encoding="utf-8")
        return {"argv": argv, "returncode": proc.returncode,
                "wall_ms": (time.perf_counter_ns() - started) / 1e6,
                "stdout": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        output.write_text(repr(error), encoding="utf-8")
        return {"argv": argv, "returncode": -1,
                "wall_ms": (time.perf_counter_ns() - started) / 1e6,
                "error": repr(error)}


def tracefs_capabilities():
    root = Path("/sys/kernel/tracing")
    wanted = ("block_rq_insert", "block_rq_issue", "block_rq_complete",
              "sys_enter_write", "sys_exit_write", "writeback_dirty_page")
    available = []
    for event in wanted:
        path = root / "events" / ("block" if event.startswith("block_") else
                                  "syscalls" if event.startswith("sys_") else
                                  "writeback") / event / "enable"
        if path.exists():
            available.append(event)
    return {"root": str(root), "available": available,
            "enabled": False, "reason": "collector does not mutate global tracefs"}


def trace_start(enabled):
    info = tracefs_capabilities()
    if not enabled:
        info["reason"] = "disabled by command line"
        return info
    root = Path(info["root"])
    try:
        (root / "tracing_on").write_text("0")
        (root / "trace").write_text("")
        selected = []
        for event in info["available"]:
            group = ("block" if event.startswith("block_") else
                     "syscalls" if event.startswith("sys_") else "writeback")
            (root / "events" / group / event / "enable").write_text("1")
            selected.append(f"{group}:{event}")
        (root / "tracing_on").write_text("1")
        return {**info, "enabled": True, "selected": selected,
                "reason": None}
    except OSError as error:
        return {**info, "enabled": False, "reason": repr(error)}


def trace_stop(info, output):
    if not info.get("enabled"):
        return {"events": {}, "lines": 0}
    root = Path(info["root"])
    try:
        (root / "tracing_on").write_text("0")
        text = (root / "trace").read_text(errors="replace")
        output.write_text(text, encoding="utf-8")
        counts = {event: text.count(event) for event in info["available"]}
        return {"events": counts, "lines": len(text.splitlines())}
    finally:
        for event in info["available"]:
            group = ("block" if event.startswith("block_") else
                     "syscalls" if event.startswith("sys_") else "writeback")
            try:
                (root / "events" / group / event / "enable").write_text("0")
            except OSError:
                pass


def one(args, mode, size):
    exp = f"P2_{mode}_{size}"
    bundle = EvidenceBundle("P2", {
        "mode": mode, "operation": "write", "size_bytes": size,
        "total_bytes": args.total_bytes, "queue_depth": args.depth,
        "persistence": "fsync/fdatasync" if mode != "spdk" else "flush+metadata",
        "representative_run": True,
    }, repo_root=ROOT, environment=environment_snapshot(
        pci=args.pci if mode == "spdk" else "0000:84:00.0",
        npu=str(args.npu), repo_root=ROOT,
        npu_info=command(["npu-smi", "info"])))
    perf = shutil.which("perf")
    strace = shutil.which("strace")
    raw = bundle.raw_dir
    if mode in ("buffered", "odirect"):
        target = args.fs_root / f"p2_{size}.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        workload = ["fio", f"--name={exp}", f"--filename={target}", "--rw=write",
                    "--ioengine=io_uring", f"--iodepth={args.depth}",
                    f"--bs={size}", f"--size={args.total_bytes}",
                    "--direct=0" if mode == "buffered" else "--direct=1",
                    "--end_fsync=1", "--output-format=json"]
    else:
        workload = [sys.executable, str(ROOT / "experiments/benchmarks/p1_fair_io.py"),
                    "--path", "spdk", "--operations", "write", "--sizes", str(size),
                    "--depths", str(args.depth), "--total-bytes", str(args.total_bytes),
                    "--warmups", "1", "--samples", "30", "--pci", args.pci,
                    "--npu", str(args.npu), "--offset", str(args.offset),
                    "--timeout", str(args.timeout)]
    if perf:
        perf_argv = [perf, "stat", "-x,", "-o", str(raw / "perf_stat.csv"), "--"] + workload
    else:
        perf_argv = workload
    trace = trace_start(not args.no_tracefs)
    perf_result = run(perf_argv, args.timeout, raw / "workload.txt")
    trace_counts = trace_stop(trace, raw / "trace.txt")
    if strace:
        strace_result = run([strace, "-f", "-c", "-o", str(raw / "strace.txt"), "--"] + workload,
                            args.timeout, raw / "strace_stdout.txt")
    else:
        strace_result = {"returncode": -1, "reason": "strace unavailable"}
    trace["counts"] = trace_counts
    observed = {"application_wall_ms": perf_result.get("wall_ms"),
                "device_service_ms": None, "queue_ms": None,
                "filesystem_block_ms": None, "page_cache_ms": None,
                "flush_ms": None, "residual_ms": None}
    status = "pass" if perf_result.get("returncode") == 0 else "fail"
    if status == "pass":
        status = "degraded"
        bundle.add_failure({"type": "time_closure", "reason":
                            "exclusive block/device intervals are unavailable",
                            "tracefs": trace})
    bundle.add_sample({"status": status, "mode": mode, "size_bytes": size,
                       "perf_returncode": perf_result.get("returncode"),
                       "strace_returncode": strace_result.get("returncode"),
                       "tracefs": trace, "observed": observed}, events=[])
    result = bundle.finalize(metrics={"status": status, "mode": mode,
        "logical_bytes": args.total_bytes, "physical_bytes": args.total_bytes,
        "layer_times": observed, "perf": perf_result,
        "strace": strace_result, "closure": {"target_residual_fraction": 0.10,
                                                "achieved": False},
        "instrumentation_note": "No unavailable layer time is represented as zero."}, status=status)
    print(json.dumps({"run_id": result["run_id"], "status": status,
                      "mode": mode, "size": size}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=("buffered", "odirect", "spdk"), default=("buffered", "odirect", "spdk"))
    parser.add_argument("--sizes", nargs="+", type=int, default=(4 * 1024 * 1024, 256 * 1024 * 1024))
    parser.add_argument("--total-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--fs-root", type=Path, default=Path("/models/npu_nvme_exp/ppt-evidence-20260829"))
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--offset", type=int, default=64 * 1024**3)
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-tracefs", action="store_true")
    args = parser.parse_args()
    for mode in args.modes:
        for size in args.sizes:
            one(args, mode, size)


if __name__ == "__main__":
    main()
