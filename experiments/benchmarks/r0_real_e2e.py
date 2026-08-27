#!/usr/bin/env python3
"""R0 real-training correctness and step-latency gate.

The driver deliberately keeps the three measurements separate:

* ``baseline``: real MindFormers training with no R0 work;
* ``capture_only``: R0 HBM compare + ACK reference update, without NVMe I/O;
* ``r0``: complete FULL + native-dtype replacement Delta persistence.

The process is single-rank and uses the already formatted 83.0.0 layout.  A
fresh process is used by the caller for replay; this file focuses on producing
durable per-step timing and phase evidence for the source process.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "python")]

from direct_checkpoint import DirectCheckpoint  # noqa: E402
from c_bindings import lib  # noqa: E402
from experiments.common import (  # noqa: E402
    init_env, make_causal_lm_training,
)
from r0_pipeline import R0NpuWriter  # noqa: E402
from s2_r0_cell import R0NpuState  # noqa: E402
from training_cell import ProbeTrainOneStepCell  # noqa: E402


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def control_state(optimizer, step):
    global_step = np.asarray(optimizer.global_step.asnumpy()).copy()
    return {
        "global_step": global_step,
        "loss_scale": np.float32(1.0),
        "data_cursor": {"epoch": 0, "sample": int(step)},
    }


def batch_for_step(ms, step, seq_len, vocab_size=50257):
    start = (int(step) * 104729) % vocab_size
    ids = (np.arange(seq_len, dtype=np.int32) + start) % vocab_size
    mask = np.ones(seq_len, dtype=np.int32)
    return ms.Tensor(ids[None, :]), ms.Tensor(mask[None, :])


def train_one(cell, ms, step, seq_len):
    start = time.perf_counter_ns()
    loss = cell(*batch_for_step(ms, step, seq_len))
    ms.hal.synchronize()
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    value = float(np.asarray(loss.asnumpy()).reshape(()))
    if not np.isfinite(value):
        raise FloatingPointError(f"non-finite loss at step {step}: {value}")
    return value, elapsed_ms


def build(args):
    import mindspore as ms

    init_env(device_id=args.npu, seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model, _dataset, optimizer = make_causal_lm_training(
        args.model, total_steps=1, device_id=args.npu, seq_len=args.seq_len,
        dropout_rate=0.0)
    cell = ProbeTrainOneStepCell(
        model, optimizer, enable_probe=False, ckpt_interval=999999)
    # Excluded compile/materialisation step.  Formal timing starts afterward.
    _, warmup_ms = train_one(cell, ms, 0, args.seq_len)
    print(f"[R0] excluded warmup={warmup_ms:.3f} ms", flush=True)
    return ms, model, optimizer, cell


def run(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    events = []
    ms, model, optimizer, cell = build(args)
    ckpt = None
    state = None
    writer = None
    steps = []
    try:
        if args.mode == "r0":
            ckpt = DirectCheckpoint(
                nvme_addr=args.pci, npu_device_id=args.npu,
                pipeline_depth=args.pipeline_depth,
                requested_chunk_size=4 * 1024 * 1024,
                spdk_shm_id=args.shm_id,
                slot_size_gb=args.slot_size_gb,
                profiling_dir=str(output.parent / "profiling"))
            rc = lib.npu_nvme_set_io_timeout_ms(
                ckpt.ctx, int(args.io_timeout * 1000))
            if rc != 0:
                raise RuntimeError(f"failed to set FULL I/O timeout: {rc}")
            full_start = time.perf_counter_ns()
            full_handle = ckpt.save_state(
                {"model": model, "optimizer": optimizer},
                control_state(optimizer, 0), step=0,
                meta_path=str(output.parent / "checkpoint_meta.pkl"))
            full_handle.wait(timeout=args.io_timeout)
            full_ms = (time.perf_counter_ns() - full_start) / 1e6
            events.append({"event": "full_persisted", "elapsed_ms": full_ms,
                           "generation": int(full_handle.generation)})
            state = R0NpuState(
                {"model": model, "optimizer": optimizer},
                block_size=524288, shard_fields=args.commit_fields,
                capture_blocks=args.capture_blocks)
            init_start = time.perf_counter_ns()
            state.initialize()
            ms.hal.synchronize()
            events.append({"event": "r0_reference_initialized",
                           "elapsed_ms": (time.perf_counter_ns() - init_start) / 1e6,
                           "fields": len(state.fields),
                           "blocks": sum(len(f.blocks) for f in state.fields)})
            writer = R0NpuWriter(
                ckpt, state, full_generation=full_handle.generation,
                batch_blocks=args.io_batch_blocks, event_sink=events.append)
            if hasattr(ckpt.ctx, "contents"):
                # A 128-block batch is expected to finish well below one
                # minute; this avoids the historical 30-second XL timeout.
                lib.npu_nvme_set_io_timeout_ms(ckpt.ctx, args.delta_timeout_ms)
        elif args.mode == "capture_only":
            state = R0NpuState(
                {"model": model, "optimizer": optimizer},
                block_size=524288, shard_fields=args.commit_fields,
                capture_blocks=args.capture_blocks)
            init_start = time.perf_counter_ns()
            state.initialize()
            ms.hal.synchronize()
            events.append({"event": "r0_reference_initialized",
                           "elapsed_ms": (time.perf_counter_ns() - init_start) / 1e6,
                           "fields": len(state.fields),
                           "blocks": sum(len(f.blocks) for f in state.fields)})

        for step in range(1, args.steps + 1):
            loss, train_ms = train_one(cell, ms, step, args.seq_len)
            row = {"step": step, "loss": loss,
                   "training_ms_before_r0": train_ms}
            if args.mode == "capture_only" and step % args.ckpt_every == 0:
                start = time.perf_counter_ns()
                flags = state.capture()
                changed = state.changed_buffers(flags)
                state.commit_ack()
                ms.hal.synchronize()
                row["r0_capture_ms"] = (time.perf_counter_ns() - start) / 1e6
                row["changed_blocks"] = len(changed)
            elif args.mode == "r0" and step % args.ckpt_every == 0:
                start = time.perf_counter_ns()
                record = writer.capture_and_commit(
                    step, control_state(optimizer, step))
                row["r0_total_ms"] = (time.perf_counter_ns() - start) / 1e6
                row["changed_blocks"] = int(record["n_blocks"])
                row["frame_bytes"] = int(record["frame_size"])
            row["step_wall_ms"] = train_ms + row.get("r0_total_ms", 0.0) + \
                row.get("r0_capture_ms", 0.0)
            steps.append(row)
            if step <= 2 or step % 5 == 0:
                print(f"[R0] step={step} loss={loss:.8g} "
                      f"train={train_ms:.3f}ms wall={row['step_wall_ms']:.3f}ms",
                      flush=True)
        result = {
            "status": "PASS", "mode": args.mode, "model": args.model,
            "seed": args.seed, "npu": args.npu, "pci": args.pci,
            "steps": steps, "events": events,
            "config": vars(args),
        }
        write_json(output, result)
        return result
    except BaseException as error:
        write_json(output, {"status": "FAIL", "mode": args.mode,
                             "model": args.model, "seed": args.seed,
                             "steps": steps, "events": events,
                             "error": repr(error), "config": vars(args)})
        raise
    finally:
        if ckpt is not None:
            ckpt.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "capture_only", "r0"),
                        default="baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--npu", type=int, default=0)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=2301)
    parser.add_argument("--pipeline-depth", type=int, default=8)
    parser.add_argument("--slot-size-gb", type=int, default=10)
    parser.add_argument("--capture-blocks", type=int, default=128)
    parser.add_argument("--commit-fields", type=int, default=16)
    parser.add_argument("--io-batch-blocks", type=int, default=128)
    parser.add_argument("--delta-timeout-ms", type=int, default=60000)
    parser.add_argument("--io-timeout", type=float, default=900.0)
    args = parser.parse_args()
    if args.steps <= 0 or args.ckpt_every <= 0:
        parser.error("steps and ckpt-every must be positive")
    if args.mode != "r0" and args.ckpt_every > args.steps:
        parser.error("ckpt-every cannot exceed steps for non-R0 mode")
    run(args)


if __name__ == "__main__":
    main()
