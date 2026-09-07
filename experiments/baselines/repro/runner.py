"""Single MindSpore GPT-2 runner shared by all baseline adapters."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np

from .adapters import ADAPTERS
from .protocol import AdapterError, DependencyBlocked
from .state_bridge import (apply_snapshot, capture_snapshot, create_batches,
                           load_raw_snapshot, save_raw_snapshot, write_json)
from python.training_state import capture_training_controls, restore_training_controls


def _model(config):
    import mindspore as ms
    from experiments.common import init_env, make_causal_lm_training
    from direct_checkpoint import ProbeTrainOneStepCell

    init_env(device_id=int(config["npu_device"]), seed=int(config["seed"]))
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    model, _dataset, optimizer = make_causal_lm_training(
        config["model"], total_steps=1, device_id=int(config["npu_device"]),
        seq_len=int(config["input_tokens"]), dropout_rate=float(config["dropout"]),
        require_dataset=False)
    cell = ProbeTrainOneStepCell(model, optimizer, enable_probe=False,
                                 ckpt_interval=10**9)
    return ms, model, optimizer, cell


def _step(ms, cell, batch_path, step):
    from .state_bridge import load_batch
    started = time.monotonic_ns()
    loss = cell(*load_batch(ms, batch_path, step))
    ms.hal.synchronize()
    value = float(np.asarray(loss.asnumpy()).reshape(()))
    if not np.isfinite(value):
        raise FloatingPointError(f"non-finite loss at step {step}: {value}")
    return {"step": int(step), "loss": value,
            "step_begin_ns": started, "step_end_ns": time.monotonic_ns()}


def prepare(config, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    Path(config["fs_test_dir"]).mkdir(parents=True, exist_ok=True)
    total_steps = (int(config["warmup_steps"]) + int(config["formal_steps"])
                   + int(config["continue_steps"]))
    batches = create_batches(root / "batches.npz", total_steps,
                             int(config["input_tokens"]))
    ms, model, optimizer, cell = _model(config)
    warmups = []
    for step in range(1, int(config["warmup_steps"]) + 1):
        warmups.append(_step(ms, cell, batches["path"], step))
    snapshot = capture_snapshot(ms, {"model": model, "optimizer": optimizer},
                                optimizer, int(config["warmup_steps"]),
                                int(config["seed"]))
    fixture_dir = root / "initial_state"
    metadata = save_raw_snapshot(snapshot, fixture_dir)
    write_json(root / "state_schema.json", snapshot.schema)
    write_json(root / "fixture.json", {"sha256": metadata["sha256"],
                                       "bytes": snapshot.total_bytes,
                                       "warmups": warmups, "batches": batches})
    return {"fixture": str(fixture_dir), "batches": batches,
            "state_schema": snapshot.schema, "warmups": warmups,
            "sha256": metadata["sha256"], "bytes": snapshot.total_bytes}


def run(config, adapter_name, run_dir, fixture_dir=None, batches_path=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    AdapterClass = ADAPTERS[adapter_name]
    adapter = AdapterClass(config, run_dir)
    if fixture_dir is None:
        fixture_dir = Path(config["results_root"]) / "prepare" / "initial_state"
    if batches_path is None:
        batches_path = Path(config["results_root"]) / "prepare" / "batches.npz"
    ms, model, optimizer, cell = _model(config)
    fixture = load_raw_snapshot(fixture_dir)
    apply_snapshot(ms, fixture, {"model": model, "optimizer": optimizer}, optimizer)
    records = []
    checkpoints = []
    start = time.monotonic_ns()
    warmup_steps = int(config["warmup_steps"])
    for formal_step in range(1, int(config["formal_steps"]) + 1):
        logical_step = warmup_steps + formal_step
        row = _step(ms, cell, batches_path, logical_step)
        row["formal_step"] = formal_step
        records.append(row)
        if adapter_name != "none" and formal_step % int(config["checkpoint_every"]) == 0:
            generation = len(checkpoints) + 1
            request_started = time.monotonic_ns()
            control_state = capture_training_controls(
                ms, optimizer,
                {"epoch": 0, "sample": int(logical_step),
                 "next_step": int(logical_step) + 1},
                np.float32(1.0), int(config["seed"]))
            handle = adapter.submit(
                generation,
                {"snapshot": lambda: capture_snapshot(
                    ms, {"model": model, "optimizer": optimizer}, optimizer,
                    logical_step, int(config["seed"])),
                 "components": {"model": model, "optimizer": optimizer},
                 "control_state": control_state, "step": logical_step},
                control_state)
            adapter.wait_source_release(handle, config["timeout_seconds"])
            checkpoints.append({"handle": handle, "step": logical_step,
                                "submit_ns": request_started})
    if adapter_name != "none":
        adapter.drain(config["timeout_seconds"])
        adapter.close()
        materialized = []
        for item in checkpoints:
            checkpoint = item["handle"].as_dict()
            checkpoint.update({"step": item["step"],
                               "submit_ns": item["submit_ns"]})
            checkpoint["persisted_ns"] = checkpoint["timestamps_ns"].get(
                "PERSISTED", checkpoint["timestamps_ns"].get("persisted"))
            materialized.append(checkpoint)
        checkpoints = materialized
    # Continue from the last committed state in the source process for oracle.
    oracle = []
    final_step = warmup_steps + int(config["formal_steps"])
    for step in range(final_step + 1, final_step + int(config["continue_steps"]) + 1):
        # The fixed batch fixture is extended deterministically for the oracle.
        start_token = (step * 104729) % 50257
        ids = (np.arange(int(config["input_tokens"]), dtype=np.int32) + start_token) % 50257
        mask = np.ones_like(ids)
        started = time.monotonic_ns()
        loss = cell(ms.Tensor(ids[None, :]), ms.Tensor(mask[None, :]))
        ms.hal.synchronize()
        value = float(np.asarray(loss.asnumpy()).reshape(()))
        oracle.append({"step": step, "loss": value,
                      "step_begin_ns": started, "step_end_ns": time.monotonic_ns()})
    native_save_metrics = None
    if adapter_name == "mindspore_native_save":
        def transition_detail(checkpoint, event):
            for transition in checkpoint.get("transitions", ()):
                if transition.get("event") == event:
                    return transition.get("detail", {})
            return {}

        def summary(values):
            values = [int(value) for value in values if value is not None]
            if not values:
                return {"count": 0, "values_ns": []}
            ordered = sorted(values)
            return {
                "count": len(values),
                "values_ns": values,
                "mean_ns": sum(values) // len(values),
                "min_ns": min(values),
                "max_ns": max(values),
                "p95_ns": ordered[min(len(ordered) - 1,
                                      int(len(ordered) * 0.95))],
            }

        native_save_metrics = {
            "native_api": summary([
                transition_detail(checkpoint, "PERSISTED").get("native_api_ns")
                for checkpoint in checkpoints]),
            "pre_save_capture_oracle": summary([
                transition_detail(checkpoint, "SNAPSHOT_READY").get("capture_ns")
                for checkpoint in checkpoints]),
            "post_write_file_flush": summary([
                transition_detail(checkpoint, "PERSISTED").get("flush_ns")
                for checkpoint in checkpoints]),
            "submit_to_persist": summary([
                checkpoint.get("timestamps_ns", {}).get("PERSISTED", 0) -
                checkpoint.get("submit_ns", 0)
                for checkpoint in checkpoints]),
            "storage_backend": "filesystem",
            "api": "mindspore.save_checkpoint(async_save=False)",
        }
    result = {
        "status": "trend_measured", "adapter": adapter_name,
        "kind": AdapterClass.kind, "model": config["model"],
        "port_class": getattr(AdapterClass, "kind", None),
        "upstream_core_invoked": getattr(AdapterClass,
                                           "upstream_core_invoked", None),
        "mechanisms_preserved": list(getattr(
            AdapterClass, "mechanisms_preserved", ())),
        "platform_substitutions": list(getattr(
            AdapterClass, "platform_substitutions", ())),
        "seed": int(config["seed"]), "formal_steps": int(config["formal_steps"]),
        "checkpoint_count": len(checkpoints), "checkpoints": checkpoints,
        "steps": records, "source_oracle": oracle,
        "storage_backend": ("raw_spdk" if adapter_name == "ours" else
                            "none" if adapter_name == "none" else "filesystem"),
        "fresh_restore_required": adapter_name != "none",
        "total_wall_seconds": (time.monotonic_ns() - start) / 1e9,
        "state_bytes": fixture.total_bytes,
    }
    if native_save_metrics is not None:
        result["native_save_metrics"] = native_save_metrics
    write_json(run_dir / "source.json", result)
    write_json(run_dir / "result.json", result)
    return result


def restore(config, adapter_name, run_dir, generation="latest-committed",
            continue_steps=None, fixture_dir=None):
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "source.json").read_text())
    checkpoints = result.get("checkpoints", [])
    if not checkpoints:
        raise ValueError("source run has no committed checkpoints")
    selected = checkpoints[-1] if generation == "latest-committed" else next(
        item for item in checkpoints if int(item["generation"]) == int(generation))
    ms, model, optimizer, cell = _model(config)
    adapter = ADAPTERS[adapter_name](config, run_dir)
    destination = {"model": model, "optimizer": optimizer,
                   "_step": int(selected["step"])}
    restored = adapter.restore(int(selected["generation"]), destination)
    if hasattr(restored, "arrays"):
        snapshot = restored
        applied = apply_snapshot(ms, snapshot,
                                  {"model": model, "optimizer": optimizer}, optimizer)
        restored_digest = snapshot.digest()
    else:
        snapshot = None
        applied = restore_training_controls(ms, optimizer, restored)
        restored_digest = selected.get("sha256")
    steps = int(config["continue_steps"] if continue_steps is None else continue_steps)
    losses = []
    final_step = int(selected["step"])
    for step in range(final_step + 1, final_step + steps + 1):
        start_token = (step * 104729) % 50257
        ids = (np.arange(int(config["input_tokens"]), dtype=np.int32) + start_token) % 50257
        mask = np.ones_like(ids)
        loss = cell(ms.Tensor(ids[None, :]), ms.Tensor(mask[None, :]))
        ms.hal.synchronize()
        losses.append(float(np.asarray(loss.asnumpy()).reshape(())))
    oracle = [row["loss"] for row in result.get("source_oracle", [])[:steps]]
    deviations = [abs(float(a) - float(b)) for a, b in zip(losses, oracle)]
    restore_result = {
        "status": "pass" if restored_digest == selected.get("sha256") and
                  (not deviations or max(deviations) <= float(config["loss_atol"]) +
                   float(config["loss_rtol"]) * max(1.0, max(map(abs, oracle)))) else "restore_failed",
        "adapter": adapter_name, "generation": int(selected["generation"]),
        "checkpoint_step": int(selected["step"]),
        "byte_exact": restored_digest == selected.get("sha256"),
        "controls": {key: str(value) for key, value in applied.items()},
        "restored_losses": losses, "source_oracle_losses": oracle,
        "loss_deviation": deviations,
    }
    write_json(run_dir / "restore.json", restore_result)
    return restore_result
