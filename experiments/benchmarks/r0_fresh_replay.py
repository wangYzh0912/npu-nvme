#!/usr/bin/env python3
"""Fresh-process FULL + one S2-R0 frame replay gate.

The source and restore phases intentionally run in different processes.  The
source records per-field hashes at the committed Delta generation, then keeps
training only to produce a two-step continuation oracle.  The restore phase
loads the FULL, applies the R0 frame from raw NVMe, restores all control state,
and verifies the committed state before continuing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "python")]

from c_bindings import lib  # noqa: E402
from direct_checkpoint import DirectCheckpoint  # noqa: E402
from experiments.benchmarks.r0_real_e2e import (  # noqa: E402
    build, control_state, train_one,
)
from r0_pipeline import R0NpuReader, R0NpuWriter  # noqa: E402
from s2_r0_cell import R0NpuState  # noqa: E402
from training_state import (encode_control_value,
                            restore_training_controls)  # noqa: E402


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def field_manifest(state):
    records = []
    digest = hashlib.sha256()
    total = 0
    for field in state.fields:
        name = field.canonical_name
        array = np.ascontiguousarray(state.current[name].value().asnumpy())
        raw = array.tobytes()
        sha = hashlib.sha256(raw).hexdigest()
        records.append({
            "name": name, "state_index": int(field.state_index),
            "shape": list(array.shape), "dtype": array.dtype.str,
            "bytes": len(raw), "sha256": sha,
        })
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(raw)
        total += len(raw)
    return {"fields": records, "field_count": len(records),
            "bytes": total, "sha256": digest.hexdigest()}


def control_manifest(controls):
    records = []
    for name in sorted(controls):
        payload, metadata = encode_control_value(controls[name])
        records.append({"name": name, "bytes": int(payload.nbytes),
                        "sha256": metadata["sha256"],
                        "codec": metadata["codec"]})
    return {"fields": records,
            "data_cursor": controls.get("data_cursor")}


def checkpoint(args, profiling):
    return DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu,
        pipeline_depth=args.pipeline_depth,
        requested_chunk_size=4 * 1024 * 1024,
        spdk_shm_id=args.shm_id, slot_size_gb=args.slot_size_gb,
        profiling_dir=str(profiling))


def source_phase(args):
    ms, model, optimizer, cell = build(args)
    ckpt = checkpoint(args, Path(args.run_dir) / "profiling_source")
    events = []
    try:
        rc = lib.npu_nvme_set_io_timeout_ms(
            ckpt.ctx, int(args.io_timeout * 1000))
        if rc != 0:
            raise RuntimeError(f"failed to set FULL I/O timeout: {rc}")
        started = time.perf_counter_ns()
        full = ckpt.save_state(
            {"model": model, "optimizer": optimizer},
            control_state(ms, optimizer, 0, args.seed), step=0,
            meta_path=str(Path(args.run_dir) / "checkpoint_meta.pkl"))
        full.wait(timeout=args.io_timeout)
        events.append({"event": "full_persisted",
                       "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                       "generation": int(full.generation)})

        state = R0NpuState(
            {"model": model, "optimizer": optimizer},
            block_size=args.block_size, shard_fields=args.commit_fields,
            capture_blocks=args.capture_blocks)
        started = time.perf_counter_ns()
        state.initialize()
        ms.hal.synchronize()
        events.append({"event": "reference_initialized",
                       "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                       "fields": len(state.fields),
                       "blocks": sum(len(field.blocks) for field in state.fields)})
        writer = R0NpuWriter(
            ckpt, state, full_generation=full.generation,
            batch_blocks=args.io_batch_blocks, event_sink=events.append)
        lib.npu_nvme_set_io_timeout_ms(ckpt.ctx, args.delta_timeout_ms)

        loss, training_ms = train_one(cell, ms, 1, args.seq_len)
        controls = control_state(ms, optimizer, 1, args.seed)
        started = time.perf_counter_ns()
        delta = writer.capture_and_commit(1, controls)
        r0_ms = (time.perf_counter_ns() - started) / 1e6
        committed = field_manifest(state)
        committed_controls = control_manifest(controls)

        continuation = []
        for step in range(2, 2 + args.continue_steps):
            next_loss, step_ms = train_one(cell, ms, step, args.seq_len)
            continuation.append({"step": step, "loss": next_loss,
                                 "training_ms": step_ms})
        result = {
            "status": "PASS", "phase": "source", "model": args.model,
            "seed": args.seed, "full_generation": int(full.generation),
            "delta": delta, "checkpoint_loss": loss,
            "checkpoint_training_ms": training_ms, "r0_total_ms": r0_ms,
            "committed_state": committed,
            "committed_controls": committed_controls,
            "continuation": continuation, "events": events,
            "config": vars(args),
        }
        write_json(args.source_result, result)
        return result
    finally:
        ckpt.cleanup()


def compare_field_manifests(expected, actual):
    expected_by_name = {item["name"]: item for item in expected["fields"]}
    actual_by_name = {item["name"]: item for item in actual["fields"]}
    mismatches = []
    if set(expected_by_name) != set(actual_by_name):
        mismatches.append({"reason": "field_set"})
    for name in sorted(set(expected_by_name) & set(actual_by_name)):
        left = expected_by_name[name]
        right = actual_by_name[name]
        for key in ("state_index", "shape", "dtype", "bytes", "sha256"):
            if left[key] != right[key]:
                mismatches.append({"name": name, "reason": key,
                                   "expected": left[key], "actual": right[key]})
                break
    return {"byte_exact": not mismatches,
            "expected_fields": len(expected_by_name),
            "actual_fields": len(actual_by_name),
            "mismatches": mismatches[:20]}


def restore_phase(args):
    source = json.loads(Path(args.source_result).read_text(encoding="utf-8"))
    ms, model, optimizer, cell = build(args)
    ckpt = checkpoint(args, Path(args.run_dir) / "profiling_restore")
    try:
        lib.npu_nvme_set_io_timeout_ms(ckpt.ctx, int(args.io_timeout * 1000))
        started = time.perf_counter_ns()
        full_controls = ckpt.load_state(
            {"model": model, "optimizer": optimizer}, step=0)
        full_load_ms = (time.perf_counter_ns() - started) / 1e6
        if control_manifest(full_controls)["data_cursor"] != {
                "epoch": 0, "sample": 0}:
            raise AssertionError("FULL control cursor mismatch")

        state = R0NpuState(
            {"model": model, "optimizer": optimizer},
            block_size=args.block_size, shard_fields=args.commit_fields,
            capture_blocks=args.capture_blocks)
        state.initialize()
        record = source["delta"]
        started = time.perf_counter_ns()
        controls, info = R0NpuReader(ckpt, state).apply(record)
        replay_ms = (time.perf_counter_ns() - started) / 1e6
        actual_state = field_manifest(state)
        state_comparison = compare_field_manifests(
            source["committed_state"], actual_state)
        actual_controls = control_manifest(controls)
        controls_exact = actual_controls == source["committed_controls"]
        if not state_comparison["byte_exact"]:
            raise AssertionError(
                f"R0 replay state mismatch: {state_comparison['mismatches'][:3]}")
        if not controls_exact:
            raise AssertionError("R0 replay control-state mismatch")
        restore_training_controls(ms, optimizer, controls)

        continuation = []
        expected = source["continuation"]
        for row in expected:
            loss, step_ms = train_one(cell, ms, int(row["step"]), args.seq_len)
            continuation.append({"step": int(row["step"]), "loss": loss,
                                 "expected_loss": float(row["loss"]),
                                 "training_ms": step_ms})
        expected_losses = np.asarray([row["loss"] for row in expected])
        actual_losses = np.asarray([row["loss"] for row in continuation])
        loss_match = bool(np.allclose(
            actual_losses, expected_losses, rtol=args.loss_rtol,
            atol=args.loss_atol))
        if not loss_match:
            raise AssertionError(
                f"continuation losses differ: expected={expected_losses.tolist()} "
                f"actual={actual_losses.tolist()}")
        result = {
            "status": "PASS", "phase": "restore", "model": args.model,
            "seed": args.seed, "full_load_ms": full_load_ms,
            "replay_ms": replay_ms, "frame_version": info["version"],
            "generation": info["generation"],
            "state_comparison": state_comparison,
            "controls_exact": controls_exact, "loss_match": loss_match,
            "continuation": continuation, "config": vars(args),
        }
        write_json(args.restore_result, result)
        return result
    finally:
        ckpt.cleanup()


def child_command(args, phase, shm_id):
    command = [
        sys.executable, str(Path(__file__).resolve()), "--phase", phase,
        "--run-dir", args.run_dir, "--source-result", args.source_result,
        "--restore-result", args.restore_result, "--model", args.model,
        "--seq-len", str(args.seq_len), "--seed", str(args.seed),
        "--npu", str(args.npu), "--pci", args.pci,
        "--shm-id", str(shm_id), "--pipeline-depth", str(args.pipeline_depth),
        "--slot-size-gb", str(args.slot_size_gb),
        "--block-size", str(args.block_size),
        "--capture-blocks", str(args.capture_blocks),
        "--commit-fields", str(args.commit_fields),
        "--io-batch-blocks", str(args.io_batch_blocks),
        "--delta-timeout-ms", str(args.delta_timeout_ms),
        "--io-timeout", str(args.io_timeout),
        "--continue-steps", str(args.continue_steps),
        "--loss-rtol", str(args.loss_rtol), "--loss-atol", str(args.loss_atol),
    ]
    return command


def orchestrate(args):
    for phase, shm_id in (("source", args.shm_id),
                          ("restore", args.shm_id + 1)):
        completed = subprocess.run(
            child_command(args, phase, shm_id), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (Path(args.run_dir) / f"{phase}.log").write_text(
            completed.stdout, encoding="utf-8")
        if completed.returncode:
            result = {"status": "FAIL", "phase": phase,
                      "returncode": completed.returncode}
            write_json(args.output, result)
            raise RuntimeError(
                f"{phase} phase failed with rc={completed.returncode}; "
                f"see {Path(args.run_dir) / (phase + '.log')}")
    source = json.loads(Path(args.source_result).read_text(encoding="utf-8"))
    restore = json.loads(Path(args.restore_result).read_text(encoding="utf-8"))
    result = {"status": "PASS", "gate": "R0_XL_FRESH_REPLAY",
              "source": source, "restore": restore}
    write_json(args.output, result)
    print(json.dumps({"status": "PASS", "gate": result["gate"],
                      "fields": restore["state_comparison"]["actual_fields"],
                      "generation": restore["generation"]}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("orchestrate", "source", "restore"),
                        default="orchestrate")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--source-result")
    parser.add_argument("--restore-result")
    parser.add_argument("--model", default="gpt2_xl")
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--npu", type=int, default=0)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=2401)
    parser.add_argument("--pipeline-depth", type=int, default=8)
    parser.add_argument("--slot-size-gb", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=524288)
    parser.add_argument("--capture-blocks", type=int, default=128)
    parser.add_argument("--commit-fields", type=int, default=16)
    parser.add_argument("--io-batch-blocks", type=int, default=128)
    parser.add_argument("--delta-timeout-ms", type=int, default=60000)
    parser.add_argument("--io-timeout", type=float, default=900.0)
    parser.add_argument("--continue-steps", type=int, default=2)
    parser.add_argument("--loss-rtol", type=float, default=1e-5)
    parser.add_argument("--loss-atol", type=float, default=1e-6)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    args.output = args.output or str(run_dir / "result.json")
    args.source_result = args.source_result or str(run_dir / "source.json")
    args.restore_result = args.restore_result or str(run_dir / "restore.json")
    if args.phase == "source":
        source_phase(args)
    elif args.phase == "restore":
        restore_phase(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
