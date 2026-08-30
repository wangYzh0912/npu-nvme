#!/usr/bin/env python3
"""P3 real DMA--NVMe pipeline matrix on GPT-2 XL training state."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
from experiments.common import init_env, make_causal_lm_training, warmup_model
from direct_checkpoint import DirectCheckpoint
from ppt_evidence import EvidenceBundle, environment_snapshot, stats, command


def read_timeline(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            def value(key):
                try:
                    return float(row.get(key, 0) or 0)
                except (TypeError, ValueError):
                    return 0.0
            dma = max(0.0, value("ts_dma_done_us") - value("ts_dma_submit_us"))
            nvme = max(0.0, value("ts_nvme_done_us") - value("ts_nvme_submit_us"))
            total = max(0.0, value("ts_nvme_done_us") - value("ts_dma_submit_us"))
            rows.append({"item": int(float(row.get("item", 0))),
                         "dma_us": dma, "nvme_us": nvme,
                         "total_us": total,
                         "dma_start_us": value("ts_dma_submit_us"),
                         "dma_end_us": value("ts_dma_done_us"),
                         "nvme_start_us": value("ts_nvme_submit_us"),
                         "nvme_end_us": value("ts_nvme_done_us"),
                         "slot_wait_us": value("slot_wait_us"),
                         "queue_depth": int(float(row.get("queue_depth", 0) or 0))})
    return rows


def batch_overlap(rows):
    """Return overlap using aggregate DMA/NVMe service and batch makespan."""
    valid = [row for row in rows if row["dma_start_us"] and row["nvme_end_us"]]
    if not valid:
        return 0.0
    def union_duration(intervals):
        total = 0.0
        end = None
        for start, stop in sorted(intervals):
            if stop <= start:
                continue
            if end is None or start > end:
                total += stop - start
                end = stop
            elif stop > end:
                total += stop - end
                end = stop
        return total
    dma = union_duration((row["dma_start_us"], row["dma_end_us"])
                         for row in valid)
    nvme = union_duration((row["nvme_start_us"], row["nvme_end_us"])
                          for row in valid)
    total = max(row["nvme_end_us"] for row in valid) - min(
        row["dma_start_us"] for row in valid)
    return max(0.0, min(1.0, (dma + nvme - total) /
                        max(min(dma, nvme), 1e-9)))


def run_one(args, mode, chunk, depth, delay_ms, seed):
    init_env(device_id=args.npu, seed=seed)
    model, dataset, optimizer = make_causal_lm_training(
        "gpt2_xl", total_steps=1, device_id=args.npu,
        seq_len=args.seq_len, dropout_rate=0.0)
    warmup_model(model, optimizer, dataset)
    run_root = Path(args.output_root or ROOT / "results/ppt-evidence-20260829")
    bundle = EvidenceBundle("P3", {
        "model": "gpt2_xl", "seed": seed, "mode": mode,
        "chunk_size": chunk, "pipeline_depth": depth,
        "delay_ms": delay_ms,
        "state": "model+optimizer+control",
        "warmups": args.warmups, "formal_samples": args.samples,
        "persistence": "metadata_commit+flush",
    }, root=run_root, repo_root=ROOT,
    environment=environment_snapshot(pci=args.pci, npu=str(args.npu),
                                      repo_root=ROOT,
                                      npu_info=command(["npu-smi", "info"])))
    old_delay = os.environ.get("NPU_NVME_TEST_GENERATION_DELAY_MS")
    if delay_ms:
        os.environ["NPU_NVME_TEST_GENERATION_DELAY_MS"] = str(delay_ms)
    ckpt = None
    elapsed_values, foreground_values, overlaps = [], [], []
    try:
        ckpt = DirectCheckpoint(
            nvme_addr=args.pci, npu_device_id=args.npu,
            pipeline_depth=depth, requested_chunk_size=chunk,
            enable_profiling=True, profiling_dir=str(bundle.raw_dir),
            spdk_shm_id=args.shm_id + seed + depth,
            keep_last_n=3, slot_size_gb=args.slot_size_gb)
        for index in range(args.warmups + args.samples):
            timeline_path = bundle.raw_dir / "time_write.csv"
            timeline_path.unlink(missing_ok=True)
            started = time.perf_counter_ns()
            handle = ckpt.save_state(
                {"model": model, "optimizer": optimizer},
                {"global_step": index}, step=index + 1,
                meta_path=str(bundle.raw_dir / f"meta_{index:04d}.pkl"),
                io_mode=mode)
            dispatch_ms = (time.perf_counter_ns() - started) / 1e6
            wait_started = time.perf_counter_ns()
            handle.wait()
            wait_ms = (time.perf_counter_ns() - wait_started) / 1e6
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            timeline = read_timeline(timeline_path)
            if index < args.warmups:
                continue
            overlap = batch_overlap(timeline)
            overlaps.append(overlap)
            elapsed_values.append(elapsed_ms)
            foreground_values.append(wait_ms)
            bundle.add_sample({"status": "pass", "seed": seed,
                               "sample": index - args.warmups,
                               "dispatch_ms": dispatch_ms,
                               "elapsed_ms": elapsed_ms,
                               "foreground_wait_ms": wait_ms,
                               "chunks": len(timeline),
                               "overlap_rate": overlap,
                               "timeline": timeline,
                               "events": [{"name": "save_dispatched"},
                                          {"name": "persisted"}]})
    except BaseException as error:
        bundle.add_failure({"error": repr(error), "mode": mode,
                            "chunk_size": chunk, "depth": depth,
                            "delay_ms": delay_ms})
    finally:
        if ckpt is not None:
            ckpt.cleanup()
        if old_delay is None:
            os.environ.pop("NPU_NVME_TEST_GENERATION_DELAY_MS", None)
        else:
            os.environ["NPU_NVME_TEST_GENERATION_DELAY_MS"] = old_delay
    elapsed = stats(elapsed_values)
    result = bundle.finalize(metrics={
        "model": "gpt2_xl", "seed": seed, "mode": mode,
        "chunk_size": chunk, "pipeline_depth": depth,
        "delay_ms": delay_ms,
        "latency_mean": elapsed.get("mean"),
        "latency_p50": elapsed.get("median"), "latency_p95": elapsed.get("p95"),
        "foreground_wait": stats(foreground_values),
        "overlap_rate": stats(overlaps),
        "state_scope": "complete training state; metadata/control included",
        "gate": {"overlap_median_min": 0.30,
                  "overlap_ci_lower_positive": bool(overlaps and min(overlaps) > 0),
                  "serial_speedup_required": 0.10},
    }, status="pass" if len(elapsed_values) == args.samples and not bundle.failures else "fail")
    print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                      "mode": mode, "chunk": chunk, "depth": depth,
                      "delay_ms": delay_ms, "seed": seed}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=("serial", "queue", "async"),
                        default=("serial", "queue", "async"))
    parser.add_argument("--chunks", nargs="+", type=int,
                        default=(1 * 1024**2, 4 * 1024**2, 16 * 1024**2))
    parser.add_argument("--depths", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--delays", nargs="+", type=int, default=(0, 100, 1000, 5000))
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=9200)
    parser.add_argument("--slot-size-gb", type=int, default=10,
                        help="must match the formatted FULL-slot layout")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.samples < 30:
        raise SystemExit("formal samples must be >= 30")
    for seed in args.seeds:
        for mode in args.modes:
            for chunk in args.chunks:
                for depth in args.depths:
                    for delay in args.delays:
                        run_one(args, mode, chunk, depth, delay, seed)


if __name__ == "__main__":
    main()
