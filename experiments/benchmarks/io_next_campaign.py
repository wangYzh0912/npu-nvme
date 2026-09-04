#!/usr/bin/env python3
"""Resumable IO-1/IO-2 campaign with staged candidate promotion."""

from __future__ import annotations

import argparse
import json
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
        completed = subprocess.run(command_line, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"campaign command failed rc={completed.returncode}: {command_line}")
    return {"status": "pass", "command": command_line}


def io1_specs(args, root):
    base = root / "IO1_single_card_async"
    return [
        ("capability", matrix_command(
            args, base / "capability", modes=("live_async",), intervals=(10,),
            snapshot_slots=(2,), request_slots=(2,), total_steps=100)),
        ("layer1_none", matrix_command(
            args, base / "layer1_none", modes=("none",), delays=(0,))),
        ("layer1_modes_delays", matrix_command(
            args, base / "layer1_modes_delays",
            modes=("serial", "queue", "frozen_async"),
            delays=(0, 100, 1000, 5000))),
        ("layer2_chunk_depth", matrix_command(
            args, base / "layer2_chunk_depth", modes=("frozen_async",),
            chunks=(1 * MIB, 4 * MIB, 16 * MIB), depths=(1, 2, 4, 8))),
    ]


def io1_formal_specs(args, root, candidates):
    base = root / "IO1_single_card_async"
    selected = candidates[:3] or [{"chunk": 4 * MIB, "depth": 4}]
    specs = []
    for index, candidate in enumerate(selected):
        specs.append((f"candidate{index}_gate100", matrix_command(
            args, base / f"candidate{index}_gate100", modes=("frozen_async",),
            seeds=(41,), intervals=(10,), chunks=(candidate["chunk"],),
            depths=(candidate["depth"],), total_steps=110)))
    for interval in (1, 10, 50):
        total_steps = interval * 30 + 1
        specs.append((f"formal_none_i{interval}", matrix_command(
            args, base / f"formal_none_i{interval}", modes=("none",),
            seeds=(41, 42, 43), intervals=(interval,), delays=(0,),
            total_steps=total_steps)))
        specs.append((f"formal_comparison_i{interval}", matrix_command(
            args, base / f"formal_comparison_i{interval}",
            modes=("serial", "queue"), seeds=(41, 42, 43),
            intervals=(interval,), chunks=(4 * MIB,), depths=(4,), delays=(0,),
            total_steps=total_steps)))
        for index, candidate in enumerate(selected):
            specs.append((f"formal_frozen_candidate{index}_i{interval}", matrix_command(
                args, base / f"formal_frozen_candidate{index}_i{interval}",
                modes=("frozen_async",), seeds=(41, 42, 43),
                intervals=(interval,), chunks=(candidate["chunk"],),
                depths=(candidate["depth"],), delays=(0,),
                total_steps=total_steps)))
    best = selected[0]
    specs.append(("formal_slow_i10", matrix_command(
        args, base / "formal_slow_i10", modes=("frozen_async",),
        seeds=(41,), intervals=(10,), chunks=(best["chunk"],),
        depths=(best["depth"],), delays=(5000,), total_steps=301)))
    return specs


def io2_specs(args, root, candidates):
    base = root / "IO2_gpt2xl"
    selected = candidates[0] if candidates else {"chunk": 4 * MIB, "depth": 4}
    chunk, depth = selected["chunk"], selected["depth"]
    return [
        ("screen_i10", matrix_command(
            args, base / "screen", model="gpt2_xl",
            modes=("none", "serial", "queue", "frozen_async", "live_async"),
            intervals=(10,), chunks=tuple(sorted({chunk, 4 * MIB})),
            depths=tuple(sorted({depth, 4})), total_steps=31)),
        ("screen_i50", matrix_command(
            args, base / "screen_i50", model="gpt2_xl",
            modes=("none", "serial", "queue", "frozen_async", "live_async"),
            intervals=(50,), chunks=tuple(sorted({chunk, 4 * MIB})),
            depths=tuple(sorted({depth, 4})), total_steps=151)),
        ("gate100", matrix_command(
            args, base / "gate100", model="gpt2_xl",
            modes=("none", "serial", "frozen_async", "live_async"),
            seeds=(41, 42, 43), intervals=(10, 50), chunks=(chunk,),
            depths=(depth,), delays=(0, 5000), total_steps=110)),
    ]


def io2_formal_specs(args, root, candidates):
    base = root / "IO2_gpt2xl"
    selected = candidates[0] if candidates else {"chunk": 4 * MIB, "depth": 4}
    chunk, depth = selected["chunk"], selected["depth"]
    specs = []
    for interval in (10, 50):
        specs.append((f"formal_i{interval}", matrix_command(
            args, base / f"formal_i{interval}", model="gpt2_xl",
            modes=("none", "serial", "frozen_async", "live_async"),
            seeds=(41, 42, 43), intervals=(interval,), chunks=(chunk,),
            depths=(depth,), delays=(0,), total_steps=interval * 30 + 1)))
    for interval in (1, 5):
        specs.append((f"stress_i{interval}", matrix_command(
            args, base / f"stress_i{interval}", model="gpt2_xl",
            modes=("serial", "frozen_async"), seeds=(41,),
            intervals=(interval,), chunks=(chunk,), depths=(depth,), delays=(0,),
            total_steps=interval * 10 + 1)))
    specs.append(("formal_slow_i10", matrix_command(
        args, base / "formal_slow_i10", model="gpt2_xl",
        modes=("frozen_async",), seeds=(41,), intervals=(10,),
        chunks=(chunk,), depths=(depth,), delays=(5000,), total_steps=301)))
    specs.append(("retained_restore_i10", matrix_command(
        args, base / "retained_restore_i10", model="gpt2_xl",
        modes=("frozen_async",), seeds=(41,), intervals=(10,),
        chunks=(chunk,), depths=(depth,), delays=(0,), total_steps=31,
        restore_retained=True)))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+",
                        choices=("io1", "io1_formal", "io2", "io2_formal"),
                        default=["io1", "io2"])
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
    if "io1" in args.phases:
        for name, command_line in io1_specs(args, root):
            records.append({"phase": name, **run_command(
                command_line, root / f"{name}.log", args.dry_run)})
        layer2 = root / "IO1_single_card_async" / "layer2_chunk_depth" / "summary.json"
        candidates = [] if args.dry_run else candidates_from_summary(layer2)
        atomic_json(root / "IO1_single_card_async" / "candidates.json",
                    {"status": "planned" if args.dry_run else "pass",
                     "candidates": candidates})
    else:
        candidate_path = root / "IO1_single_card_async" / "candidates.json"
        candidates = (json.loads(candidate_path.read_text()).get("candidates", [])
                      if candidate_path.exists() else [])
    if "io1_formal" in args.phases:
        if not candidates:
            raise RuntimeError("IO-1 formal requires completed screening candidates")
        for name, command_line in io1_formal_specs(args, root, candidates):
            records.append({"phase": name, **run_command(
                command_line, root / f"{name}.log", args.dry_run)})
    if "io2" in args.phases:
        for name, command_line in io2_specs(args, root, candidates):
            records.append({"phase": f"io2_{name}", **run_command(
                command_line, root / f"io2_{name}.log", args.dry_run)})
    if "io2_formal" in args.phases:
        if not candidates:
            raise RuntimeError("IO-2 formal requires completed IO-1 candidates")
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
