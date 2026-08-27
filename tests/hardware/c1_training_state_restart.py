#!/usr/bin/env python3
"""C1: complete MindSpore training-state restart and continuation gate.

The orchestrator runs three independent Python processes:

1. an uninterrupted reference trajectory;
2. a trajectory stopped after a complete training-state checkpoint;
3. a fresh process that restores the checkpoint and continues.

The dataset is generated deterministically from the logical step so a saved
data cursor has an observable effect without depending on MindDataset's
internal prefetch state.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "python")]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def build_training(args):
    import mindspore as ms
    from experiments.common import init_env, make_causal_lm_training
    from direct_checkpoint import ProbeTrainOneStepCell

    init_env(device_id=args.npu, seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model, _dataset, optimizer = make_causal_lm_training(
        args.model, total_steps=1, device_id=args.npu, seq_len=args.seq_len,
        dropout_rate=args.dropout_rate)
    cell = ProbeTrainOneStepCell(
        model, optimizer, enable_probe=False, ckpt_interval=999999)
    # A fresh GRAPH_MODE model still owns lazy initializers.  Loading before
    # the first compiled step appears to work byte-for-byte, but graph
    # materialisation can subsequently overwrite that state.  Compile and
    # allocate with the same excluded step in every process; the restore
    # process loads the checkpoint only after this barrier.
    warmup_start = time.perf_counter()
    warmup_loss = cell(*batch_for_step(ms, 0, args.seq_len))
    ms.hal.synchronize()
    warmup_value = float(np.asarray(warmup_loss.asnumpy()).reshape(()))
    if not np.isfinite(warmup_value):
        raise FloatingPointError(f"non-finite excluded warmup: {warmup_value}")
    print(f"[C1] excluded_step=0 loss={warmup_value:.9g} "
          f"time={time.perf_counter() - warmup_start:.3f}s", flush=True)
    return ms, model, optimizer, cell


def batch_for_step(ms, step, seq_len, vocab_size=50257):
    # GPT-2 consumes seq_len tokens and internally shifts to seq_len - 1.
    start = (step * 104729) % vocab_size
    ids = (np.arange(seq_len, dtype=np.int32) + start) % vocab_size
    mask = np.ones(seq_len, dtype=np.int32)
    return ms.Tensor(ids[None, :]), ms.Tensor(mask[None, :])


def train_range(ms, cell, begin, end, seq_len):
    losses = []
    times = []
    for step in range(begin, end + 1):
        batch = batch_for_step(ms, step, seq_len)
        start = time.perf_counter()
        loss = cell(*batch)
        ms.hal.synchronize()
        times.append(time.perf_counter() - start)
        value = float(np.asarray(loss.asnumpy()).reshape(()))
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite loss at step {step}: {value}")
        losses.append(value)
        print(f"[C1] step={step} loss={value:.9g} time={times[-1]:.3f}s",
              flush=True)
    return losses, times


def iter_unique_parameters(model, optimizer):
    seen = set()
    for component, obj in (("model", model), ("optimizer", optimizer)):
        for name, parameter in obj.parameters_and_names():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield f"{component}/{name}", parameter


def state_digest(model, optimizer):
    digest = hashlib.sha256()
    fields = 0
    total_bytes = 0
    for name, parameter in iter_unique_parameters(model, optimizer):
        array = np.ascontiguousarray(parameter.value().asnumpy())
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        fields += 1
        total_bytes += int(array.nbytes)
    return {"sha256": digest.hexdigest(), "fields": fields,
            "bytes": total_bytes}


def write_state_oracle(root, model, optimizer):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (name, parameter) in enumerate(
            iter_unique_parameters(model, optimizer)):
        array = np.ascontiguousarray(parameter.value().asnumpy())
        filename = f"{index:04d}.npy"
        np.save(root / filename, array, allow_pickle=False)
        records.append({"name": name, "file": filename,
                        "shape": list(array.shape), "dtype": array.dtype.str,
                        "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
    write_json(root / "manifest.json", {"fields": records})


def compare_state_oracle(root, model, optimizer, rtol, atol):
    root = Path(root)
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8"))["fields"]
    actual_fields = list(iter_unique_parameters(model, optimizer))
    if len(actual_fields) != len(manifest):
        raise AssertionError("final state field count changed")
    byte_exact = 0
    max_abs = 0.0
    max_rel = 0.0
    mismatches = []
    for record, (name, parameter) in zip(manifest, actual_fields):
        if name != record["name"]:
            raise AssertionError(
                f"final state order/name changed: {name} != {record['name']}")
        expected = np.load(root / record["file"], mmap_mode="r",
                           allow_pickle=False)
        actual = np.ascontiguousarray(parameter.value().asnumpy())
        if list(actual.shape) != record["shape"] or actual.dtype.str != record["dtype"]:
            mismatches.append({"name": name, "reason": "shape_or_dtype"})
            continue
        if hashlib.sha256(actual.tobytes()).hexdigest() == record["sha256"]:
            byte_exact += 1
            continue
        if np.issubdtype(actual.dtype, np.inexact):
            expected_float = np.asarray(expected, dtype=np.float64)
            actual_float = actual.astype(np.float64)
            difference = np.abs(actual_float - expected_float)
            field_max_abs = float(difference.max(initial=0.0))
            denominator = np.maximum(np.abs(expected_float), 1e-12)
            field_max_rel = float((difference / denominator).max(initial=0.0))
            max_abs = max(max_abs, field_max_abs)
            max_rel = max(max_rel, field_max_rel)
            if not np.allclose(actual, expected, rtol=rtol, atol=atol,
                               equal_nan=False):
                mismatches.append({"name": name, "reason": "not_allclose",
                                   "max_abs": field_max_abs,
                                   "max_rel": field_max_rel})
        elif not np.array_equal(actual, expected):
            mismatches.append({"name": name, "reason": "integer_mismatch"})
    return {"fields": len(manifest), "byte_exact_fields": byte_exact,
            "allclose": not mismatches, "max_abs": max_abs,
            "max_rel": max_rel, "mismatches": mismatches[:20],
            "rtol": rtol, "atol": atol}


def control_state(ms, optimizer, cursor, args):
    global_step = np.asarray(optimizer.global_step.asnumpy()).copy()
    return {
        "global_step": global_step,
        "loss_scale": np.float32(args.loss_scale),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "mindspore_seed": int(args.seed),
        "mindspore_rng": np.asarray(ms.get_rng_state().asnumpy()).copy(),
        "data_cursor": {"epoch": 0, "sample": int(cursor)},
    }


def apply_control_state(ms, controls, expected_cursor, args):
    required = {"global_step", "loss_scale", "python_rng", "numpy_rng",
                "mindspore_seed", "mindspore_rng", "data_cursor"}
    if set(controls) != required:
        raise AssertionError(
            f"control fields differ: expected={sorted(required)} "
            f"actual={sorted(controls)}")
    if controls["data_cursor"] != {"epoch": 0, "sample": expected_cursor}:
        raise AssertionError("restored data cursor is incorrect")
    if int(controls["mindspore_seed"]) != args.seed:
        raise AssertionError("restored MindSpore seed is incorrect")
    if float(controls["loss_scale"]) != float(np.float32(args.loss_scale)):
        raise AssertionError("restored loss scale is incorrect")
    random.setstate(controls["python_rng"])
    np.random.set_state(controls["numpy_rng"])
    ms.common.set_seed(int(controls["mindspore_seed"]))
    ms.set_rng_state(ms.Tensor(controls["mindspore_rng"]))


def baseline_phase(args):
    ms, model, optimizer, cell = build_training(args)
    losses, times = train_range(
        ms, cell, 1, args.save_step + args.continue_steps, args.seq_len)
    final_state = state_digest(model, optimizer)
    write_state_oracle(Path(args.run_dir) / "baseline_state", model, optimizer)
    write_json(Path(args.run_dir) / "baseline.json", {
        "model": args.model,
        "seed": args.seed,
        "save_step": args.save_step,
        "continue_steps": args.continue_steps,
        "continuation_losses": losses[args.save_step:],
        "all_step_seconds": times,
        "final_state": final_state,
    })


def save_phase(args):
    from direct_checkpoint import DirectCheckpoint

    ms, model, optimizer, cell = build_training(args)
    losses, times = train_range(ms, cell, 1, args.save_step, args.seq_len)
    before = state_digest(model, optimizer)
    controls = control_state(ms, optimizer, args.save_step, args)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu,
        pipeline_depth=args.pipeline_depth,
        requested_chunk_size=4 * 1024 * 1024,
        rank_id=0, world_size=1, keep_last_n=3,
        slot_size_gb=args.slot_size_gb, spdk_shm_id=args.shm_id,
        profiling_dir=str(Path(args.run_dir) / "profiling_save"))
    try:
        started = time.perf_counter()
        handle = ckpt.save_state(
            {"model": model, "optimizer": optimizer}, controls,
            step=args.save_step,
            meta_path=str(Path(args.run_dir) / "checkpoint_meta.pkl"))
        handle.wait(timeout=args.io_timeout)
        persist_seconds = time.perf_counter() - started
        record = ckpt.meta_dict["checkpoints"][f"step_{args.save_step}"]
        # Continue from the exact in-memory state that was frozen for the
        # checkpoint.  Ascend GRAPH_MODE may produce slightly different
        # floating-point reductions after an independent process rebuilds a
        # graph, so this is the semantic continuation oracle for C1.  The
        # files intentionally reuse baseline_state's directory to keep the
        # large oracle from being duplicated on the nearly-full home volume.
        continuation_losses, continuation_times = train_range(
            ms, cell, args.save_step + 1,
            args.save_step + args.continue_steps, args.seq_len)
        continuation_state = state_digest(model, optimizer)
        write_state_oracle(Path(args.run_dir) / "baseline_state",
                           model, optimizer)
        write_json(Path(args.run_dir) / "save.json", {
            "status": handle.status,
            "metadata_generation": record["generation"],
            "snapshot_generation": handle.generation,
            "state": before,
            "losses": losses,
            "step_seconds": times,
            "persist_seconds": persist_seconds,
            "control_names": record["control_names"],
            "components": record["components"],
            "continuation_losses": continuation_losses,
            "continuation_step_seconds": continuation_times,
            "continuation_state": continuation_state,
        })
    finally:
        ckpt.cleanup()


def restore_phase(args):
    from direct_checkpoint import DirectCheckpoint

    saved = json.loads(
        (Path(args.run_dir) / "save.json").read_text(encoding="utf-8"))
    ms, model, optimizer, cell = build_training(args)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu,
        pipeline_depth=args.pipeline_depth,
        requested_chunk_size=4 * 1024 * 1024,
        rank_id=0, world_size=1, keep_last_n=3,
        slot_size_gb=args.slot_size_gb, spdk_shm_id=args.shm_id,
        profiling_dir=str(Path(args.run_dir) / "profiling_load"))
    try:
        started = time.perf_counter()
        controls = ckpt.load_state(
            {"model": model, "optimizer": optimizer}, step=args.save_step)
        restore_seconds = time.perf_counter() - started
        restored_digest = state_digest(model, optimizer)
        if restored_digest != saved["state"]:
            raise AssertionError(
                f"loaded state differs from saved state: "
                f"{restored_digest} != {saved['state']}")
        actual_global_step = np.asarray(optimizer.global_step.asnumpy())
        if not np.array_equal(actual_global_step, controls["global_step"]):
            raise AssertionError("optimizer and control global_step differ")
        apply_control_state(ms, controls, args.save_step, args)
        losses, times = train_range(
            ms, cell, args.save_step + 1,
            args.save_step + args.continue_steps, args.seq_len)
        expected_losses = np.asarray(saved["continuation_losses"],
                                     dtype=np.float64)
        actual_losses = np.asarray(losses, dtype=np.float64)
        if not np.allclose(actual_losses, expected_losses,
                           rtol=args.loss_rtol, atol=args.loss_atol):
            raise AssertionError(
                f"continuation losses differ: expected={expected_losses.tolist()} "
                f"actual={actual_losses.tolist()}")
        final = state_digest(model, optimizer)
        final_comparison = compare_state_oracle(
            Path(args.run_dir) / "baseline_state", model, optimizer,
            args.state_rtol, args.state_atol)
        if not final_comparison["allclose"]:
            raise AssertionError(
                f"final training state exceeds tolerance: "
                f"{final_comparison['mismatches'][:3]}")
        write_json(Path(args.run_dir) / "result.json", {
            "status": "pass",
            "gate": "C1",
            "model": args.model,
            "npu": args.npu,
            "pci": args.pci,
            "save_step": args.save_step,
            "continuation_steps": args.continue_steps,
            "loaded_state_byte_exact": True,
            "final_state_byte_exact": final == saved["continuation_state"],
            "final_state_comparison": final_comparison,
            "loss_allclose": True,
            "loss_rtol": args.loss_rtol,
            "loss_atol": args.loss_atol,
            "restore_seconds": restore_seconds,
            "continuation_step_seconds": times,
            "state": final,
        })
        print("[C1] PASS complete state restart and continuation", flush=True)
    finally:
        ckpt.cleanup()


def child_command(args, phase):
    return [
        sys.executable, str(Path(__file__).resolve()), "--phase", phase,
        "--run-dir", str(Path(args.run_dir).resolve()),
        "--pci", args.pci, "--npu", str(args.npu),
        "--model", args.model, "--seq-len", str(args.seq_len),
        "--seed", str(args.seed), "--save-step", str(args.save_step),
        "--continue-steps", str(args.continue_steps),
        "--loss-scale", str(args.loss_scale),
        "--dropout-rate", str(args.dropout_rate),
        "--loss-rtol", str(args.loss_rtol), "--loss-atol", str(args.loss_atol),
        "--state-rtol", str(args.state_rtol),
        "--state-atol", str(args.state_atol),
        "--pipeline-depth", str(args.pipeline_depth),
        "--slot-size-gb", str(args.slot_size_gb),
        "--shm-id", str(args.shm_id), "--io-timeout", str(args.io_timeout),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("orchestrate", "baseline", "save", "restore"),
                        default="orchestrate")
    parser.add_argument("--run-dir", default=str(
        REPO_ROOT / "results" / "next-correctness" /
        time.strftime("c1_%Y%m%d_%H%M%S")))
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=0)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-step", type=int, default=2)
    parser.add_argument("--continue-steps", type=int, default=2)
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument(
        "--dropout-rate", type=float, default=0.0,
        help="Ascend dropout is not reproducible across restart; C1 defaults to 0")
    parser.add_argument("--loss-rtol", type=float, default=1e-5)
    parser.add_argument("--loss-atol", type=float, default=1e-6)
    parser.add_argument("--state-rtol", type=float, default=1e-5)
    parser.add_argument("--state-atol", type=float, default=1e-6)
    parser.add_argument("--pipeline-depth", type=int, default=8)
    parser.add_argument("--slot-size-gb", type=int, default=10)
    parser.add_argument("--shm-id", type=int, default=91)
    parser.add_argument("--io-timeout", type=float, default=900.0)
    args = parser.parse_args()
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)

    if args.phase == "baseline":
        baseline_phase(args)
    elif args.phase == "save":
        save_phase(args)
    elif args.phase == "restore":
        restore_phase(args)
    else:
        for phase in ("baseline", "save", "restore"):
            subprocess.run(child_command(args, phase), cwd=args.run_dir, check=True)
        print((Path(args.run_dir) / "result.json").read_text(encoding="utf-8"),
              flush=True)


if __name__ == "__main__":
    main()
