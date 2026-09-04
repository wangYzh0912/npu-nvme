#!/usr/bin/env python3
"""Reusable single-card/single-disk FULL training baseline.

The orchestrator deliberately uses separate processes for baseline, source
save, and restore.  A result is a pass only when the source process has exited
after PERSISTED and a fresh process has restored and continued training.
"""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from ppt_evidence import environment_snapshot, command  # noqa: E402
from full_checkpoint_protocol import validate_result_gate  # noqa: E402


def batch_for_step(ms, step, seq_len, vocab_size=50257):
    start = (int(step) * 104729) % vocab_size
    ids = (np.arange(seq_len, dtype=np.int32) + start) % vocab_size
    mask = np.ones(seq_len, dtype=np.int32)
    return ms.Tensor(ids[None, :]), ms.Tensor(mask[None, :])


def digest_state(model, optimizer):
    digest = hashlib.sha256()
    fields = 0
    total = 0
    seen = set()
    items = []
    for component, obj in (("model", model), ("optimizer", optimizer)):
        for name, parameter in obj.parameters_and_names():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            array = np.ascontiguousarray(parameter.value().asnumpy())
            items.append((f"{component}/{name}", array))
    for name, array in sorted(items, key=lambda item: item[0]):
        digest.update(name.encode())
        digest.update(array.tobytes())
        fields += 1
        total += int(array.nbytes)
    return {"sha256": digest.hexdigest(), "fields": fields, "bytes": total}


def train_steps(ms, cell, begin, end, seq_len):
    losses = []
    elapsed = []
    for step in range(int(begin), int(end) + 1):
        started = time.perf_counter()
        value = cell(*batch_for_step(ms, step, seq_len))
        ms.hal.synchronize()
        loss = float(np.asarray(value.asnumpy()).reshape(()))
        if not np.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        losses.append(loss)
        elapsed.append(time.perf_counter() - started)
    return losses, elapsed


def stream_pointer(stream):
    """Extract the ACL stream from MindSpore's documented device capsule."""
    import ctypes
    capsule = stream.device_stream()
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    pointer = get_pointer(capsule, None)
    if not pointer:
        raise RuntimeError("MindSpore current stream has no device pointer")
    return int(pointer)


def train_live_steps(ms, cells, begin, end, seq_len, pending_handle=None):
    """Overlap one generation's D2H with next-step forward/backward."""
    forward_backward, optimizer_cell = cells
    losses, elapsed = [], []
    handle = pending_handle
    for step in range(int(begin), int(end) + 1):
        started = time.perf_counter()
        fb_begin = time.monotonic_ns()
        loss, grads = forward_backward(*batch_for_step(ms, step, seq_len))
        fb_submitted = time.monotonic_ns()
        if handle is not None:
            handle.install_update_fence(
                stream_pointer(ms.runtime.current_stream()))
        update_submitted = time.monotonic_ns()
        optimizer_cell(*grads)
        step_done = ms.runtime.Event()
        step_done.record(ms.runtime.current_stream())
        step_done.synchronize()
        update_complete = time.monotonic_ns()
        if handle is not None:
            handle.collect_update_wait()
            handle.training_dependency = {
                "forward_backward_begin_ns": fb_begin,
                "forward_backward_submitted_ns": fb_submitted,
                "optimizer_submitted_ns": update_submitted,
                "optimizer_complete_ns": update_complete,
            }
            handle = None
        value = float(np.asarray(loss.asnumpy()).reshape(()))
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite loss at step {step}")
        losses.append(value)
        elapsed.append(time.perf_counter() - started)
    return losses, elapsed


def continuation_oracle(source, baseline, target):
    """Return the continuous source-process trajectory after ``target``.

    Ascend GRAPH_MODE may choose slightly different floating-point reduction
    orders when the same graph is compiled in an independent process.  The
    source process continuing from the frozen state is therefore the semantic
    no-restart oracle; the separate baseline remains the performance control.
    """
    source_losses = source.get("continuous_losses")
    if source_losses is None:
        return baseline["losses"][target:], "independent_process_baseline"
    if len(source_losses) < target:
        raise ValueError(
            f"source loss trajectory has {len(source_losses)} steps, target={target}")
    return source_losses[target:], "source_process_continuation"


def request_timing(request):
    """Derive request latency from the request's monotonic state transitions."""
    event_times = {
        event["state"]: int(event["monotonic_ns"])
        for event in request.get("events", [])
    }
    persisted_ns = event_times.get("PERSISTED")
    created_ns = event_times.get("CREATED")
    api_enter_ns = request.get("api_enter_ns")
    if persisted_ns is None or created_ns is None or api_enter_ns is None:
        raise ValueError("request is missing API, CREATED, or PERSISTED timestamp")
    if persisted_ns < created_ns or persisted_ns < int(api_enter_ns):
        raise ValueError("request timestamps are not monotonic")
    return {
        "persist_seconds": (persisted_ns - int(api_enter_ns)) / 1e9,
        "state_machine_seconds": (persisted_ns - created_ns) / 1e9,
    }


def build(args):
    import mindspore as ms
    from experiments.common import init_env, make_causal_lm_training
    from direct_checkpoint import ProbeTrainOneStepCell
    from training_cell import LiveForwardBackwardCell, LiveOptimizerCell
    init_env(device_id=args.npu, seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model, _dataset, optimizer = make_causal_lm_training(
        args.model, total_steps=1, device_id=args.npu,
        seq_len=args.seq_len, dropout_rate=0.0, require_dataset=False)
    if args.mode == "live_async":
        cell = (LiveForwardBackwardCell(model, optimizer),
                LiveOptimizerCell(optimizer))
        _loss, _grads = cell[0](*batch_for_step(ms, 0, args.seq_len))
        cell[1](*_grads)
    else:
        cell = ProbeTrainOneStepCell(model, optimizer, enable_probe=False,
                                     ckpt_interval=999999)
        _ = cell(*batch_for_step(ms, 0, args.seq_len))
    # Allocate lazy parameters consistently in every process.
    ms.hal.synchronize()
    # The excluded compilation step updates optimizer state.  Every process
    # sees the same warmup state; reset only the logical counter so checkpoint
    # step and optimizer global_step remain the same contract.
    global_step = np.asarray(optimizer.global_step.asnumpy())
    optimizer.global_step.set_data(ms.Tensor(np.zeros_like(global_step)))
    return ms, model, optimizer, cell


def phase_baseline(args, run_dir):
    ms, model, optimizer, cell = build(args)
    trainer = train_live_steps if args.mode == "live_async" else train_steps
    losses, elapsed = trainer(ms, cell, 1, args.total_steps, args.seq_len)
    payload = {"losses": losses, "step_seconds": elapsed,
               "state": digest_state(model, optimizer)}
    (run_dir / "baseline.json").write_text(json.dumps(payload, indent=2),
                                            encoding="utf-8")


def phase_source(args, run_dir):
    from direct_checkpoint import (CheckpointBusyError, DirectCheckpoint)
    ms, model, optimizer, cell = build(args)
    steps = [int(item) for item in args.checkpoint_steps]
    previous = 0
    records = []
    continuous_losses = []
    continuous_step_seconds = []
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu,
        pipeline_depth=args.pipeline_depth, requested_chunk_size=args.chunk_size,
        enable_profiling=True, profiling_dir=str(run_dir / "raw"),
        keep_last_n=args.keep_last_n, slot_size_gb=args.slot_size_gb,
        spdk_shm_id=args.shm_id, checkpoint_slots=args.checkpoint_slots,
        request_slots=args.request_slots, admission=args.admission)
    try:
        pending = []
        admission_events = []

        def finish_one():
            if not pending:
                return
            handle, pending_record = pending.pop(0)
            wait_started = time.perf_counter()
            handle.wait(timeout=args.timeout)
            pending_record["foreground_wait_seconds"] = (
                time.perf_counter() - wait_started)
            if handle.state.value != "PERSISTED":
                raise RuntimeError(
                    f"checkpoint did not persist: {handle.as_dict()}")
            pending_record["request"] = handle.as_dict()
            if pending_record.get("state") is None:
                pending_record["state"] = handle.snapshot_state_digest
            if pending_record["state"] is None:
                raise RuntimeError("live checkpoint has no stable state digest")
            timing = request_timing(pending_record["request"])
            interval = pending_record.get("overlapped_training_interval")
            if interval:
                event_times = {event["state"]: event["monotonic_ns"]
                               for event in handle.events}
                io_start = event_times.get("DMA_COPYING")
                io_end = event_times.get("PERSISTED")
                if io_start is not None and io_end is not None:
                    pending_record["training_io_overlap_ns"] = max(
                        0, min(interval["end_monotonic_ns"], io_end) -
                        max(interval["start_monotonic_ns"], io_start))
            pending_record.pop("started")
            pending_record.update(timing)
            pending_record["runtime_stats"] = ckpt.get_runtime_stats()
            records.append(pending_record)

        for step in steps:
            train_started = time.monotonic_ns()
            if args.mode == "live_async":
                live_handle = pending[-1][0] if pending else None
                interval_losses, interval_elapsed = train_live_steps(
                    ms, cell, previous + 1, step, args.seq_len,
                    pending_handle=live_handle)
            else:
                interval_losses, interval_elapsed = train_steps(
                    ms, cell, previous + 1, step, args.seq_len)
            continuous_losses.extend(interval_losses)
            continuous_step_seconds.extend(interval_elapsed)
            train_ended = time.monotonic_ns()
            for _handle, record in pending:
                record.setdefault("overlapped_training_interval", {
                    "start_monotonic_ns": train_started,
                    "end_monotonic_ns": train_ended,
                })
            if args.mode == "serial":
                while pending:
                    finish_one()
            controls = {
                "global_step": np.asarray(optimizer.global_step.asnumpy()).copy(),
                "loss_scale": np.float32(1.0),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "mindspore_seed": int(args.seed),
                "mindspore_rng": np.asarray(ms.get_rng_state().asnumpy()).copy(),
                "data_cursor": {"epoch": 0, "sample": int(step)},
            }
            checkpoint_state = (None if args.mode == "live_async" else
                                digest_state(model, optimizer))
            started = time.perf_counter()
            try:
                handle = ckpt.save_state(
                    {"model": model, "optimizer": optimizer}, controls,
                    step=step, meta_path=str(run_dir / f"meta_{step:06d}.pkl"),
                    io_mode=args.mode, admission=args.admission,
                    timeout=args.timeout)
            except CheckpointBusyError as error:
                admission_events.append({"step": step, "status": "BUSY",
                                         "generation_created": False,
                                         "error": str(error)})
                previous = step
                continue
            pending_record = {
                "step": step, "started": started,
                "dispatch_seconds": time.perf_counter() - started,
                "state": checkpoint_state,
                "preceding_training_interval": {
                    "start_monotonic_ns": train_started,
                    "end_monotonic_ns": train_ended,
                },
            }
            pending.append((handle, pending_record))
            admission_events.append({"step": step, "status": "ACCEPTED",
                                     "request_id": handle.request_id,
                                     "generation": handle.generation})
            previous = step
        if previous < args.total_steps:
            trailing_started = time.monotonic_ns()
            if args.mode == "live_async":
                trailing_losses, trailing_elapsed = train_live_steps(
                    ms, cell, previous + 1, args.total_steps, args.seq_len,
                    pending_handle=pending[-1][0] if pending else None)
            else:
                trailing_losses, trailing_elapsed = train_steps(
                    ms, cell, previous + 1, args.total_steps, args.seq_len)
            continuous_losses.extend(trailing_losses)
            continuous_step_seconds.extend(trailing_elapsed)
            trailing_ended = time.monotonic_ns()
            for _handle, record in pending:
                record.setdefault("overlapped_training_interval", {
                    "start_monotonic_ns": trailing_started,
                    "end_monotonic_ns": trailing_ended,
                })
        while pending:
            finish_one()
        if not records:
            raise RuntimeError("no checkpoint generation was accepted")
        if len(continuous_losses) != args.total_steps:
            raise RuntimeError(
                f"continuous loss trajectory has {len(continuous_losses)} steps, "
                f"expected {args.total_steps}")
        (run_dir / "source.json").write_text(
            json.dumps({"status": "pass", "checkpoints": records,
                        "admission_events": admission_events,
                        "continuous_losses": continuous_losses,
                        "continuous_step_seconds": continuous_step_seconds,
                        "continuous_final_state": digest_state(model, optimizer)}, indent=2,
                       default=str), encoding="utf-8")
    finally:
        ckpt.cleanup()


def phase_restore(args, run_dir):
    import mindspore as ms
    from direct_checkpoint import DirectCheckpoint
    from training_state import restore_training_controls
    baseline = json.loads((run_dir / "baseline.json").read_text())
    source = json.loads((run_dir / "source.json").read_text())
    target = int(args.restore_step or source["checkpoints"][-1]["step"])
    ms, model, optimizer, cell = build(args)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu,
        pipeline_depth=args.pipeline_depth, requested_chunk_size=args.chunk_size,
        keep_last_n=args.keep_last_n, slot_size_gb=args.slot_size_gb,
        spdk_shm_id=args.shm_id)
    try:
        controls = ckpt.load_state({"model": model, "optimizer": optimizer},
                                   step=target, verify_checksums=True)
        target_record = next(record for record in source["checkpoints"]
                             if int(record["step"]) == target)
        loaded_state = digest_state(model, optimizer)
        if loaded_state != target_record["state"]:
            raise AssertionError("fresh restore state digest mismatch")
        expected_step = int(np.asarray(controls["global_step"]).reshape(-1)[0])
        if expected_step != target:
            raise AssertionError(f"global_step={expected_step} target={target}")
        restored = restore_training_controls(ms, optimizer, controls)
        if restored["data_cursor"] != {"epoch": 0, "sample": target}:
            raise AssertionError("data cursor mismatch")
        if args.mode == "live_async":
            losses, elapsed = train_live_steps(
                ms, cell, target + 1, args.total_steps, args.seq_len)
        else:
            losses, elapsed = train_steps(ms, cell, target + 1,
                                          args.total_steps, args.seq_len)
        expected_losses, oracle = continuation_oracle(source, baseline, target)
        expected = np.asarray(expected_losses, dtype=np.float64)
        actual = np.asarray(losses, dtype=np.float64)
        if expected.shape != actual.shape or not np.allclose(
                expected, actual, rtol=args.loss_rtol, atol=args.loss_atol):
            max_abs = (float(np.max(np.abs(expected - actual)))
                       if expected.shape == actual.shape and expected.size else None)
            raise AssertionError(
                f"fresh continuation loss mismatch oracle={oracle} "
                f"expected={expected.tolist()} actual={actual.tolist()} "
                f"max_abs={max_abs}")
        result = {"status": "pass", "model": args.model, "mode": args.mode,
                  "seed": args.seed, "pci": args.pci, "npu": args.npu,
                  "request_id": target_record["request"]["request_id"],
                  "generation": target_record["request"]["metadata_generation"],
                  "checkpoint_step": target, "persisted": True,
                  "restore_verified": True, "loss_allclose": True,
                  "continuation_oracle": oracle,
                  "loaded_state_byte_exact": True,
                  "state_after_restore": loaded_state,
                  "state_after_continuation": digest_state(model, optimizer),
                  "continuation_losses": losses, "step_seconds": elapsed}
        (run_dir / "restore.json").write_text(json.dumps(result, indent=2),
                                               encoding="utf-8")
    finally:
        ckpt.cleanup()


def phase_capability(args, run_dir):
    from direct_checkpoint import DirectCheckpoint
    capability = DirectCheckpoint.live_async_capability()
    result = {
        "status": "pass" if capability["supported"] else "unsupported",
        "mode": "live_async", "run_id": run_dir.name,
        "persisted": False, "restore_verified": False,
        "capability": capability,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def run_orchestrated(args, run_dir):
    config = vars(args).copy()
    config.update({"experiment_id": "SINGLE_CARD_FULL",
                   "run_id": run_dir.name,
                   "state": "model+optimizer+control",
                   "persistence": "data_complete+flush+metadata_commit"})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2,
                                                     sort_keys=True),
                                          encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(environment_snapshot(pci=args.pci, npu=str(args.npu),
                                        repo_root=ROOT,
                                        npu_info=command(["npu-smi", "info"])),
                   indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "commit.json").write_text(json.dumps({
        "repo": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "branch": command(["git", "-C", str(ROOT), "branch", "--show-current"]),
        "status": command(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "spdk": command(["git", "-C", str(ROOT / "third_party" / "spdk"),
                          "rev-parse", "HEAD"]),
    }, indent=2, sort_keys=True), encoding="utf-8")
    for filename in ("samples.jsonl", "timeline.jsonl", "events.jsonl",
                     "failures.jsonl"):
        (run_dir / filename).touch()
    if args.mode == "live_async":
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--phase", "capability", "--run-dir", str(run_dir),
             "--model", args.model, "--mode", args.mode,
             "--npu", str(args.npu), "--pci", args.pci],
            cwd=run_dir, check=False)
        if completed.returncode != 0:
            raise RuntimeError("live_async capability probe failed")
        capability_result = json.loads((run_dir / "result.json").read_text())
        if capability_result["status"] != "pass":
            raise SystemExit(2)
    if args.mode == "none":
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--phase", "baseline", "--run-dir", str(run_dir),
                        "--model", args.model, "--mode", "none",
                        "--checkpoint-steps", *[str(x) for x in args.checkpoint_steps],
                        "--total-steps", str(args.total_steps),
                        "--seq-len", str(args.seq_len), "--seed", str(args.seed),
                        "--npu", str(args.npu), "--pci", args.pci],
                       cwd=run_dir, check=True)
        baseline = json.loads((run_dir / "baseline.json").read_text())
        result = {"status": "pass", "model": args.model, "mode": "none",
                  "seed": args.seed, "pci": args.pci, "npu": args.npu,
                  "persisted": None, "restore_verified": None,
                  "state": baseline["state"], "run_id": run_dir.name,
                  "samples": args.total_steps, "failed_samples": 0,
                  "performance": summarize_performance(
                      baseline["step_seconds"], [], [])}
        validate_result_gate(result)
        (run_dir / "result.json").write_text(json.dumps(result, indent=2),
                                             encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return
    phases = ("baseline", "source")
    try:
        for phase in phases:
            command_line = [sys.executable, str(Path(__file__).resolve()),
                        "--phase", phase, "--run-dir", str(run_dir),
                        "--model", args.model, "--mode", args.mode,
                        "--checkpoint-steps", *[str(x) for x in args.checkpoint_steps],
                        "--total-steps", str(args.total_steps),
                        "--seq-len", str(args.seq_len), "--seed", str(args.seed),
                        "--npu", str(args.npu), "--pci", args.pci,
                        "--chunk-size", str(args.chunk_size),
                        "--pipeline-depth", str(args.pipeline_depth),
                        "--keep-last-n", str(args.keep_last_n),
                        "--slot-size-gb", str(args.slot_size_gb),
                        "--checkpoint-slots", str(args.checkpoint_slots),
                        "--request-slots", str(args.request_slots),
                        "--admission", args.admission,
                        "--generation-delay-ms", str(args.generation_delay_ms),
                        "--shm-id", str(args.shm_id), "--timeout", str(args.timeout),
                        "--loss-rtol", str(args.loss_rtol),
                        "--loss-atol", str(args.loss_atol)]
            subprocess.run(command_line, cwd=run_dir, check=True)
        source = json.loads((run_dir / "source.json").read_text())
        if args.restore_step is not None:
            restore_steps = [args.restore_step]
        elif args.restore_retained:
            restore_steps = [int(record["step"])
                             for record in source["checkpoints"][-args.keep_last_n:]]
        else:
            restore_steps = [int(source["checkpoints"][-1]["step"])]
        restore_results = []
        for restore_step in restore_steps:
            phase = "restore"
            command_line = [sys.executable, str(Path(__file__).resolve()),
                        "--phase", phase, "--run-dir", str(run_dir),
                        "--model", args.model, "--mode", args.mode,
                        "--checkpoint-steps", *[str(x) for x in args.checkpoint_steps],
                        "--total-steps", str(args.total_steps),
                        "--seq-len", str(args.seq_len), "--seed", str(args.seed),
                        "--npu", str(args.npu), "--pci", args.pci,
                        "--chunk-size", str(args.chunk_size),
                        "--pipeline-depth", str(args.pipeline_depth),
                        "--keep-last-n", str(args.keep_last_n),
                        "--slot-size-gb", str(args.slot_size_gb),
                        "--checkpoint-slots", str(args.checkpoint_slots),
                        "--request-slots", str(args.request_slots),
                        "--admission", args.admission,
                        "--generation-delay-ms", str(args.generation_delay_ms),
                        "--shm-id", str(args.shm_id), "--timeout", str(args.timeout),
                        "--loss-rtol", str(args.loss_rtol),
                        "--loss-atol", str(args.loss_atol),
                        "--restore-step", str(restore_step)]
            subprocess.run(command_line, cwd=run_dir, check=True)
            restored = json.loads((run_dir / "restore.json").read_text())
            (run_dir / f"restore_step_{restore_step:06d}.json").write_text(
                json.dumps(restored, indent=2), encoding="utf-8")
            restore_results.append(restored)
    except BaseException as error:
        failure = {"status": "fail", "phase": phase, "error": repr(error)}
        with (run_dir / "failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(failure, sort_keys=True) + "\n")
        source_path = run_dir / "source.json"
        source_persisted = False
        if source_path.exists():
            source_result = json.loads(source_path.read_text(encoding="utf-8"))
            source_persisted = bool(source_result.get("checkpoints")) and all(
                record.get("request", {}).get("state") == "PERSISTED"
                for record in source_result.get("checkpoints", []))
        (run_dir / "result.json").write_text(
            json.dumps({**failure, "run_id": run_dir.name,
                        "persisted": source_persisted,
                        "restore_verified": False}, indent=2),
            encoding="utf-8")
        raise
    result = restore_results[-1]
    result["restored_steps"] = restore_steps
    result["all_retained_restores_verified"] = all(
        item.get("status") == "pass" and item.get("loaded_state_byte_exact") is True and
        item.get("loss_allclose") is True for item in restore_results)
    source = json.loads((run_dir / "source.json").read_text())
    checkpoint_steps = {int(record["step"]) for record in source["checkpoints"]}
    checkpoint_step_seconds = [
        value for index, value in enumerate(source["continuous_step_seconds"], 1)
        if index in checkpoint_steps]
    non_checkpoint_step_seconds = [
        value for index, value in enumerate(source["continuous_step_seconds"], 1)
        if index not in checkpoint_steps]
    result["performance"] = summarize_performance(
        source["continuous_step_seconds"], checkpoint_step_seconds,
        non_checkpoint_step_seconds)
    result["checkpoint_latency_seconds"] = [
        record["persist_seconds"] for record in source["checkpoints"]]
    result["checkpoint_state_machine_seconds"] = [
        record["state_machine_seconds"] for record in source["checkpoints"]]
    result["latency_semantics"] = {
        "checkpoint_latency_seconds": "api_enter_to_persisted_event",
        "checkpoint_state_machine_seconds": "created_to_persisted_event",
        "foreground_wait_seconds": "host_wait_inside_finish_one",
    }
    result["api_return_seconds"] = [
        max(0, record["request"]["api_return_ns"] -
            record["request"]["api_enter_ns"]) / 1e9
        for record in source["checkpoints"]
        if record["request"].get("api_enter_ns") is not None and
        record["request"].get("api_return_ns") is not None]
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as samples:
        for record in source["checkpoints"]:
            samples.write(json.dumps({
                "run_id": run_dir.name, "status": "pass",
                "step": record["step"],
                "request_id": record["request"]["request_id"],
                "generation": record["request"]["metadata_generation"],
                "persist_seconds": record["persist_seconds"],
                "state_machine_seconds": record["state_machine_seconds"],
                "foreground_wait_seconds": record.get(
                    "foreground_wait_seconds", 0.0),
                "checksum": record["request"]["checksum"],
                "runtime_stats": record.get("runtime_stats", {}),
            }, sort_keys=True) + "\n")
    with (run_dir / "timeline.jsonl").open("w", encoding="utf-8") as timeline, \
            (run_dir / "events.jsonl").open("w", encoding="utf-8") as events:
        for record in source["checkpoints"]:
            event_record = {
                "run_id": run_dir.name,
                "rank": 0,
                "request_id": record["request"]["request_id"],
                "checkpoint_step": record["step"],
                "generation": record["request"]["metadata_generation"],
                "api_enter_ns": record["request"].get("api_enter_ns"),
                "api_return_ns": record["request"].get("api_return_ns"),
                "freeze_wait_ns": record["request"].get("freeze_wait_ns", 0),
                "update_wait_ns": record["request"].get("update_wait_ns", 0),
                "update_deadline_missed": record["request"].get(
                    "update_deadline_missed", False),
                "dma_submit_ns": record["request"].get("dma_submit_ns"),
                "dma_complete_ns": record["request"].get("dma_complete_ns"),
                "dma_chunks": record["request"].get("dma_chunks", []),
                "training_dependency": record["request"].get(
                    "training_dependency"),
                "events": record["request"]["events"],
            }
            encoded = json.dumps(event_record, sort_keys=True) + "\n"
            timeline.write(encoded)
            events.write(encoded)
        result.update({"run_id": run_dir.name, "samples": len(args.checkpoint_steps),
                   "failed_samples": 0,
                   "accepted_generations": len(source["checkpoints"]),
                   "busy_requests": sum(
                       event.get("status") == "BUSY"
                       for event in source.get("admission_events", [])),
                   "paths": {
                       "config": "config.json", "environment": "environment.json",
                       "commit": "commit.json", "samples": "samples.jsonl",
                       "timeline": "timeline.jsonl", "events": "events.jsonl",
                       "failures": "failures.jsonl",
                       "source": "source.json", "restore": "restore.json",
                       "raw": "raw/", "result": "result.json"}})
    validate_result_gate(result)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2),
                                         encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def summarize_performance(all_steps, checkpoint_steps, non_checkpoint_steps):
    def distribution(values):
        array = np.asarray(values, dtype=np.float64)
        if not array.size:
            return {"count": 0, "mean_seconds": None,
                    "p95_seconds": None, "p99_seconds": None}
        return {"count": int(array.size), "mean_seconds": float(array.mean()),
                "p95_seconds": float(np.percentile(array, 95)),
                "p99_seconds": float(np.percentile(array, 99))}
    return {"all": distribution(all_steps),
            "checkpoint": distribution(checkpoint_steps),
            "non_checkpoint": distribution(non_checkpoint_steps)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("orchestrate", "baseline", "source",
                                             "restore", "capability"),
                        default="orchestrate")
    parser.add_argument("--run-dir", default=str(ROOT / "results" / "single-card-full" /
                                                  time.strftime("run_%Y%m%d_%H%M%S")))
    parser.add_argument("--model", choices=("gpt2", "gpt2_xl"), default="gpt2")
    parser.add_argument("--mode", choices=("none", "serial", "queue", "async",
                                            "frozen_async", "live_async"),
                        default="serial")
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=None)
    parser.add_argument("--restore-step", type=int, default=None)
    parser.add_argument("--restore-retained", action="store_true")
    parser.add_argument("--total-steps", type=int, default=110)
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--keep-last-n", type=int, default=3)
    parser.add_argument("--slot-size-gb", type=int, default=10)
    parser.add_argument("--checkpoint-slots", type=int, default=1)
    parser.add_argument("--request-slots", type=int, default=None)
    parser.add_argument("--admission", choices=("block", "try"), default="block")
    parser.add_argument("--generation-delay-ms", type=int, default=0)
    parser.add_argument("--shm-id", type=int, default=17041)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--loss-rtol", type=float, default=1e-5)
    parser.add_argument("--loss-atol", type=float, default=1e-6)
    parser.add_argument("--smoke", action="store_true",
                        help="use a short, explicitly non-formal run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.request_slots is None:
        args.request_slots = args.checkpoint_slots
    if args.checkpoint_steps is None:
        args.checkpoint_steps = [2, 5] if args.smoke else [10, 50, 100]
    if args.smoke and args.total_steps == 110:
        args.total_steps = 10
    if args.restore_step is not None and args.restore_step >= args.total_steps:
        parser.error("restore step must leave at least one continuation step")
    if args.total_steps <= 0 or any(step <= 0 or step > args.total_steps
                                    for step in args.checkpoint_steps):
        parser.error("checkpoint steps must be positive and <= total steps")
    if sorted(args.checkpoint_steps) != list(args.checkpoint_steps):
        parser.error("checkpoint steps must be sorted")
    if len(set(args.checkpoint_steps)) != len(args.checkpoint_steps):
        parser.error("checkpoint steps must be unique")
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return 0
    # This branch is an acceptance harness for complete training state only.
    # Child baseline/source/restore processes inherit the explicit freeze.
    os.environ["NPU_NVME_FULL_ONLY"] = "1"
    if args.generation_delay_ms:
        os.environ["NPU_NVME_TEST_GENERATION_DELAY_MS"] = str(
            args.generation_delay_ms)
    else:
        os.environ.pop("NPU_NVME_TEST_GENERATION_DELAY_MS", None)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "orchestrate":
        run_orchestrated(args, run_dir)
    elif args.phase == "baseline":
        phase_baseline(args, run_dir)
    elif args.phase == "source":
        phase_source(args, run_dir)
    elif args.phase == "restore":
        phase_restore(args, run_dir)
    else:
        phase_capability(args, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
