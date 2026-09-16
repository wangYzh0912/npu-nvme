#!/usr/bin/env python3
"""Resumable minimal IO-1/IO-2 acceptance campaign."""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "experiments" / "benchmarks" / "s3_training_io_matrix.py"
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from experiments.benchmarks.longrun_utils import atomic_json  # noqa: E402
from ppt_evidence import command, environment_snapshot  # noqa: E402

MIB = 1024 * 1024


def matrix_command(args, output, model="gpt2", modes=("frozen_async",),
                   seeds=(41,), intervals=(10,), chunks=(4 * MIB,), depths=(4,),
                   snapshot_slots=(2,), request_slots=(2,), delays=(0,),
                   total_steps=110, restore_retained=False):
    return [
        sys.executable, str(MATRIX), "--model", model,
        "--modes", *modes, "--seeds", *map(str, seeds),
        "--intervals", *map(str, intervals), "--chunks", *map(str, chunks),
        "--depths", *map(str, depths),
        "--snapshot-slots", *map(str, snapshot_slots),
        "--request-slots", *map(str, request_slots),
        "--generation-delays-ms", *map(str, delays),
        "--total-steps", str(total_steps), "--seq-len", str(args.seq_len),
        "--npu", str(args.npu), "--numa-node", str(args.numa_node),
        "--pci", args.pci, "--shm-id", str(args.shm_id),
        "--timeout", str(args.timeout), "--output-root", str(output),
        *( ["--resume"] if args.resume else []),
        *( ["--restore-retained"] if restore_retained else []),
    ]


def score_record(record):
    values = record.get("result", {}).get("checkpoint_latency_seconds", [])
    return statistics.median(values) if values else float("inf")


def candidates_from_summary(path, limit=3):
    summary = json.loads(path.read_text(encoding="utf-8"))
    records = sorted(summary.get("records", []), key=score_record)
    output = []
    seen = set()
    for record in records:
        key = (record["chunk"], record["depth"])
        if key in seen:
            continue
        seen.add(key)
        output.append({"chunk": key[0], "depth": key[1],
                       "median_persist_seconds": score_record(record)})
        if len(output) == limit:
            break
    return output


def run_command(command_line, log, dry_run):
    if dry_run:
        return {"status": "planned", "command": command_line}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command_line, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            returncode = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    if returncode:
        raise RuntimeError(f"campaign command failed rc={returncode}: {command_line}")
    return {"status": "pass", "command": command_line}


def io1_mechanism_specs(args, root):
    base = root / "IO1_single_card_async"
    return [
        ("io1_mechanism", matrix_command(
            args, base / "mechanism", modes=("none", "serial", "frozen_async",
                                              "live_async"), seeds=(41,),
            intervals=(10,), chunks=(4 * MIB,), depths=(4,),
            snapshot_slots=(2,), request_slots=(2,), delays=(0,),
            total_steps=101, restore_retained=True)),
    ]


def io1_formal_specs(args, root, candidates=None):
    base = root / "IO1_single_card_async"
    return [
        ("io1_formal", matrix_command(
            args, base / "formal", modes=("none", "frozen_async", "live_async"),
            seeds=(41, 42, 43), intervals=(10,), chunks=(4 * MIB,), depths=(4,),
            snapshot_slots=(2,), request_slots=(2,), delays=(0,),
            total_steps=201, restore_retained=True)),
        ("io1_live_stress", matrix_command(
            args, base / "live_stress", modes=("live_async",), seeds=(41,),
            intervals=(1,), chunks=(4 * MIB,), depths=(4,), snapshot_slots=(2,),
            request_slots=(2,), delays=(1000,), total_steps=21,
            restore_retained=True)),
    ]


def io2_specs(args, root, candidates=None):
    base = root / "IO2_gpt2xl"
    return [("io2_formal", matrix_command(
        args, base / "formal", model="gpt2_xl",
        modes=("none", "frozen_async", "live_async"), seeds=(41, 42, 43),
        intervals=(10,), chunks=(4 * MIB,), depths=(4,), snapshot_slots=(2,),
        request_slots=(2,), delays=(0,), total_steps=121,
        restore_retained=True))]


def io2_formal_specs(args, root, candidates=None):
    base = root / "IO2_gpt2xl"
    return [("io2_live_stress", matrix_command(
        args, base / "live_stress", model="gpt2_xl", modes=("live_async",),
        seeds=(41,), intervals=(1,), chunks=(4 * MIB,), depths=(4,),
        snapshot_slots=(2,), request_slots=(2,), delays=(0,), total_steps=13,
        restore_retained=True))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+",
                        choices=("io1_mechanism", "io1_formal", "io2_formal",
                                 "io2_stress"),
                        default=["io1_mechanism", "io1_formal", "io2_formal",
                                 "io2_stress"])
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results" / "io-next-20260903")
    parser.add_argument("--npu", type=int, default=2)
    parser.add_argument("--numa-node", type=int, default=4)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--shm-id", type=int, default=20269300)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "config.json", vars(args))
    atomic_json(root / "environment.json", environment_snapshot(
        pci=args.pci, npu=str(args.npu), repo_root=ROOT,
        npu_info=command(["npu-smi", "info"])))
    records = []
    candidates = [{"chunk": 4 * MIB, "depth": 4}]
    if "io1_mechanism" in args.phases:
        for name, command_line in io1_mechanism_specs(args, root):
            records.append({"phase": name, **run_command(
                command_line, root / f"{name}.log", args.dry_run)})
    if "io1_formal" in args.phases:
        for name, command_line in io1_formal_specs(args, root, candidates):
            records.append({"phase": name, **run_command(
                command_line, root / f"{name}.log", args.dry_run)})
    if "io2_formal" in args.phases:
        for name, command_line in io2_specs(args, root, candidates):
            records.append({"phase": f"io2_{name}", **run_command(
                command_line, root / f"io2_{name}.log", args.dry_run)})
    if "io2_stress" in args.phases:
        for name, command_line in io2_formal_specs(args, root, candidates):
            records.append({"phase": f"io2_{name}", **run_command(
                command_line, root / f"io2_{name}.log", args.dry_run)})
    result = {"status": "planned" if args.dry_run else "pass",
              "records": records, "candidates": candidates}
    atomic_json(root / "result.json", result)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
