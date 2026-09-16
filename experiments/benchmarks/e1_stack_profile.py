#!/usr/bin/env python3
"""E1 representative software-stack profile.

The existing E1 matrix supplies latency distributions.  This companion run
adds kernel/user accounting for one representative size and queue depth per
path.  It never converts counters into invented absolute layer times:
unavailable tracepoints are recorded as unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def command_for(args, run_dir):
    if args.mode == "fs":
        target = Path(args.fs_root) / f"e1_profile_{args.size}_{args.depth}.bin"
        return ["fio", f"--name=e1_fs_{args.size}_{args.depth}",
                f"--filename={target}", "--rw=write", "--ioengine=io_uring",
                f"--iodepth={args.depth}", "--numjobs=1", f"--bs={args.size}",
                f"--size={args.size}", "--loops=30", "--direct=0",
                "--fsync=1", "--group_reporting", "--output-format=json+"]
    return [os.environ.get("PYTHON", "/root/miniconda3/envs/ms_2.5/bin/python"),
            str(ROOT / "experiments/benchmarks/io_matrix.py"),
            "--experiment", "E1", "--item-bytes", str(args.size),
            "--depths", str(args.depth), "--warmups", "10",
            "--repetitions", "30", "--npu", str(args.npu),
            "--pci", "0000:83:00.0", "--shm-id", str(args.shm_id),
            "--offset", str(args.offset), "--output-root", str(run_dir / "spdk")]


def trace_setup(trace_dir):
    tracefs = Path("/sys/kernel/tracing")
    events = {"syscalls:sys_enter_write", "syscalls:sys_exit_write",
              "writeback:writeback_start", "writeback:writeback_written",
              "block:block_rq_issue", "block:block_rq_complete"}
    available = set()
    try:
        available = set((tracefs / "available_events").read_text().splitlines())
        selected = sorted(events & available)
        if not selected:
            return {"status": "unavailable", "reason": "no requested tracepoints"}
        (tracefs / "tracing_on").write_text("0")
        (tracefs / "trace").write_text("")
        (tracefs / "set_event").write_text("\n".join(selected) + "\n")
        (tracefs / "tracing_on").write_text("1")
        return {"status": "enabled", "events": selected}
    except OSError as error:
        return {"status": "unavailable", "reason": repr(error),
                "available_checked": sorted(available)}


def trace_stop(trace_info, trace_dir):
    if trace_info.get("status") != "enabled":
        return
    tracefs = Path("/sys/kernel/tracing")
    try:
        (tracefs / "tracing_on").write_text("0")
        trace_dir.write_text((tracefs / "trace").read_text(errors="replace"))
    except OSError as error:
        trace_dir.write_text(json.dumps({"error": repr(error)}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fs", "spdk"), required=True)
    parser.add_argument("--size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=8800)
    parser.add_argument("--offset", type=int, default=256 * 1024 * 1024 * 1024)
    parser.add_argument("--fs-root", default="/models/npu_nvme_exp/e1_stack_profile")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    run_dir = Path(args.output_root or ROOT / "results/ppt-evidence-20260829/E1/stack")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"experiment": "E1", "mode": args.mode, "size": args.size,
              "depth": args.depth, "persistence": "fsync" if args.mode == "fs"
              else "nvme_flush+metadata_commit", "ssd_pci": "0000:84:00.0"
              if args.mode == "fs" else "0000:83:00.0",
              "command": command_for(args, run_dir),
              "trace_scope": "representative run; counters not layer-time estimates"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.mode == "fs":
        Path(args.fs_root).mkdir(parents=True, exist_ok=True)
    trace_info = trace_setup(run_dir / "trace.txt")
    perf_stat = run_dir / "perf.stat"
    perf_data = run_dir / "perf.data"
    perf_cmd = ["perf", "stat", "-x,", "-o", str(perf_stat), "-e",
                "task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions",
                "--"] + config["command"]
    start = time.monotonic_ns()
    completed = subprocess.run(perf_cmd, cwd=ROOT, text=True,
                               stdout=(run_dir / "stdout.log").open("w"),
                               stderr=subprocess.STDOUT, check=False,
                               timeout=3600)
    trace_stop(trace_info, run_dir / "trace.txt")
    # A short call-graph profile is retained separately from perf stat.  If
    # kernel restrictions reject it, the failure is evidence, not fabricated.
    record_cmd = ["perf", "record", "-o", str(perf_data), "-F", "99", "-g",
                  "--"] + config["command"]
    record = subprocess.run(record_cmd, cwd=ROOT,
                            stdout=(run_dir / "record.stdout.log").open("w"),
                            stderr=subprocess.STDOUT, check=False,
                            timeout=3600)
    result = {"status": "pass" if completed.returncode == 0 else "fail",
              "experiment": "E1", "mode": args.mode, "size": args.size,
              "depth": args.depth, "persistence": config["persistence"],
              "command": config["command"], "perf_stat_returncode": completed.returncode,
              "perf_record_returncode": record.returncode,
              "trace": trace_info,
              "wall_ms": (time.monotonic_ns() - start) / 1e6,
              "layer_time_policy": "only end-to-end is measured; per-layer absolute times require trace closure"}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
