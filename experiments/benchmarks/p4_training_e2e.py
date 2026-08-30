#!/usr/bin/env python3
"""P4 end-to-end training impact of full checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
from experiments.common import init_env, make_causal_lm_training
from direct_checkpoint import DirectCheckpoint, ProbeTrainOneStepCell
from ppt_evidence import EvidenceBundle, environment_snapshot, stats, command


def run_one(args, mode, interval, seed):
    init_env(device_id=args.npu, seed=seed)
    formal_steps = (args.total_formal_steps if args.total_formal_steps is not None
                    else interval * args.checkpoints + 2)
    total_steps = formal_steps + args.warmup_steps
    model, dataset, optimizer = make_causal_lm_training(
        args.model, total_steps=total_steps, device_id=args.npu,
        seq_len=args.seq_len, dropout_rate=0.0)
    cell = ProbeTrainOneStepCell(model, optimizer, enable_probe=False,
                                 ckpt_interval=999999)
    iterator = dataset.create_tuple_iterator()
    root = Path(args.output_root or ROOT / "results/ppt-evidence-20260829")
    bundle = EvidenceBundle("P4", {
        "model": args.model, "seed": seed, "mode": mode,
        "checkpoint_interval": interval, "formal_checkpoints": args.checkpoints,
        "state": "model+optimizer+control", "warmup_steps": args.warmup_steps,
    }, root=root, repo_root=ROOT,
    environment=environment_snapshot(pci=args.pci, npu=str(args.npu),
                                      repo_root=ROOT,
                                      npu_info=command(["npu-smi", "info"])))
    ckpt = None
    ordinary, checkpoint, waits, backlog, losses = [], [], [], [], []
    active_handle = None
    wall_start = time.perf_counter_ns()
    final_drain_ms = 0.0
    restore_verified = mode == "none"
    try:
        if mode != "none":
            ckpt = DirectCheckpoint(
                nvme_addr=args.pci, npu_device_id=args.npu,
                pipeline_depth=args.pipeline_depth,
                requested_chunk_size=args.chunk_size,
                enable_profiling=True, profiling_dir=str(bundle.raw_dir),
                spdk_shm_id=args.shm_id + seed + interval,
                keep_last_n=3, slot_size_gb=args.slot_size_gb)
        for step in range(1, total_steps + 1):
            batch = next(iterator)
            start = time.perf_counter_ns()
            output = cell(*batch)
            if hasattr(output, "asnumpy"):
                loss = float(np.asarray(output.asnumpy()).reshape(-1)[0])
            else:
                loss = float(output)
            if hasattr(__import__("mindspore"), "hal"):
                __import__("mindspore").hal.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if step <= args.warmup_steps:
                continue
            is_ckpt = mode != "none" and step % interval == 0
            wait_ms = 0.0
            if is_ckpt:
                ckpt_start = time.perf_counter_ns()
                pending_before = int(active_handle is not None and
                                     not active_handle.done())
                prior_wait_ms = 0.0
                if pending_before:
                    prior_wait_start = time.perf_counter_ns()
                    active_handle.wait()
                    prior_wait_ms = ((time.perf_counter_ns() - prior_wait_start)
                                     / 1e6)
                    active_handle = None
                handle = ckpt.save_state(
                    {"model": model, "optimizer": optimizer},
                    {"global_step": step}, step=step,
                    meta_path=str(bundle.raw_dir / f"meta_{step:06d}.pkl"),
                    io_mode="serial" if mode == "sync" else mode)
                dispatch_ms = (time.perf_counter_ns() - ckpt_start) / 1e6
                if mode == "sync":
                    handle.wait()
                    active_handle = None
                    wait_ms = (time.perf_counter_ns() - ckpt_start) / 1e6
                else:
                    # save_state waits for the prior generation before it
                    # snapshots this one.  Its dispatch duration is therefore
                    # exactly the foreground stall caused by a full queue.
                    active_handle = handle
                    wait_ms = prior_wait_ms + dispatch_ms
                checkpoint.append(elapsed + wait_ms)
                waits.append(wait_ms)
                backlog.append(pending_before)
            else:
                ordinary.append(elapsed)
            losses.append(loss)
            bundle.add_sample({"status": "pass", "step": step,
                               "checkpoint": is_ckpt, "step_ms": elapsed,
                               "foreground_wait_ms": wait_ms, "loss": loss,
                               "backlog": pending_before if is_ckpt else
                               int(active_handle is not None and not active_handle.done()),
                               "generation": step if is_ckpt else None,
                               "events": ([{"name": "train_step_end"}] +
                                          ([{"name": "checkpoint_dispatch"}]
                                           if is_ckpt else []))})
        if active_handle is not None:
            drain_start = time.perf_counter_ns()
            active_handle.wait()
            final_drain_ms = (time.perf_counter_ns() - drain_start) / 1e6
            bundle.add_sample({"status": "pass", "event": "final_drain",
                               "foreground_wait_ms": final_drain_ms})
            active_handle = None
        if ckpt is not None:
            ckpt.load_state({"model": model, "optimizer": optimizer},
                            verify_checksums=True)
            restore_verified = True
    except BaseException as error:
        bundle.add_failure({"mode": mode, "interval": interval,
                            "seed": seed, "error": repr(error)})
    finally:
        if ckpt is not None:
            ckpt.cleanup()
    step_overhead_ratio = (
        ((sum(ordinary) + sum(checkpoint)) /
         max(len(ordinary) + len(checkpoint), 1)) /
        max(stats(ordinary).get("mean") or 1.0, 1e-9) - 1.0
        if ordinary else None
    )
    accepted = bool(restore_verified and step_overhead_ratio is not None and
                    step_overhead_ratio <= 0.05)
    result = bundle.finalize(metrics={
        "model": args.model, "seed": seed, "mode": mode,
        "step_ms": stats(ordinary), "checkpoint_step_ms": stats(checkpoint),
        "foreground_wait": stats(waits), "backlog": stats(backlog),
        "loss": stats(losses),
        "training_throughput_steps_s": (len(losses) /
            max((time.perf_counter_ns() - wall_start) / 1e9, 1e-9)),
        "final_drain_ms": final_drain_ms,
        "restore_verified": restore_verified,
        "step_overhead": step_overhead_ratio,
        "step_overhead_percent": (step_overhead_ratio * 100.0
                                  if step_overhead_ratio is not None else None),
        "acceptance_status": "pass" if accepted else "fail",
        "gate": {"mean_step_overhead_max": 0.05,
                  "backlog_monotonic": False,
                  "restore_required": mode != "none"},
    }, status="pass" if not bundle.failures and len(losses) >= args.checkpoints
        and restore_verified else "fail")
    print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                      "mode": mode, "interval": interval, "seed": seed}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=("none", "sync", "queue", "async"),
                        default=("none", "sync", "queue", "async"))
    parser.add_argument("--intervals", nargs="+", type=int, default=(1, 5, 10, 20, 50))
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--checkpoints", type=int, default=30)
    parser.add_argument("--total-formal-steps", type=int, default=None,
                        help="use the same formal step count for every mode")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=9400)
    parser.add_argument("--slot-size-gb", type=int, default=10,
                        help="must match the formatted FULL-slot layout")
    parser.add_argument("--model", choices=("gpt2", "gpt2_xl"), default="gpt2_xl")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    for seed in args.seeds:
        for mode in args.modes:
            intervals = (1,) if mode == "none" else args.intervals
            for interval in intervals:
                run_one(args, mode, interval, seed)


if __name__ == "__main__":
    main()
