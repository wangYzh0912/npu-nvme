"""Single MindSpore GPT-2 runner shared by all baseline adapters."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np

from .adapters import ADAPTERS
from .protocol import AdapterError, DependencyBlocked, EventLog
from .state_bridge import (apply_snapshot, capture_snapshot, create_batches,
                           load_raw_snapshot, save_raw_snapshot, write_json)
from npu_nvme.framework.training_state import capture_training_controls, restore_training_controls


def _model(config, initialized=False):
    from types import SimpleNamespace
    from experiments.training.full_fixture import build_training
    args = SimpleNamespace(model=config['model'], npu=int(config['npu_device']),
        seed=int(config['seed']), seq_len=int(config['input_tokens']),
        dropout_rate=float(config['dropout']), deterministic='ON')
    return build_training(args, initialized=initialized)


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
    if fixture_dir is None:
        fixture_dir = Path(config["results_root"]) / "prepare" / "initial_state"
    if batches_path is None:
        batches_path = Path(config["results_root"]) / "prepare" / "batches.npz"
    ms, model, optimizer, cell = _model(config)
    adapter = AdapterClass(config, run_dir)
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
            persisted_ns = checkpoint['persisted_ns']
            if not isinstance(persisted_ns, int) or persisted_ns < item['submit_ns']:
                raise ValueError('checkpoint lacks a valid persistence timestamp')
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
                checkpoint["persisted_ns"] - checkpoint["submit_ns"]
                for checkpoint in checkpoints]),
            "storage_backend": "filesystem",
            "api": "mindspore.save_checkpoint(async_save=False)",
        }
    result = {
        "schema_version": 2, "status": "trend_measured", "adapter": adapter_name,
        "project_commit": config["project_commit"],
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
            continue_steps=None, fixture_dir=None, output_path=None,
            mode="verify", repeat_index=None):
    """Restore in a fresh process and optionally emit phase timing evidence.

    ``output_path`` is deliberately independent from ``restore.json`` so
    repeated timing processes cannot overwrite correctness evidence.
    """
    run_dir = Path(run_dir)
    process_start = time.monotonic_ns()
    events = []
    def mark(event, **detail):
        row = {"event": event, "monotonic_ns": time.monotonic_ns()}
        row.update(detail)
        events.append(row)
        return row["monotonic_ns"]
    mark("process_start", pid=os.getpid(), mode=mode,
         repeat_index=repeat_index)
    result = json.loads((run_dir / "source.json").read_text())
    checkpoints = result.get("checkpoints", [])
    if not checkpoints:
        raise ValueError("source run has no committed checkpoints")
    selected = checkpoints[-1] if generation == "latest-committed" else next(
        item for item in checkpoints if int(item["generation"]) == int(generation))
    adapter = None
    try:
        ms, model, optimizer, cell = _model(config)
        if adapter_name == "ours":
            import gc
            del model, optimizer, cell
            gc.collect()
        mark("model_constructed")
        ms.hal.synchronize()
        mark("restore_begin", selected_generation=int(selected["generation"]),
             selected_step=int(selected["step"]))
        adapter = ADAPTERS[adapter_name](config, run_dir)
        if output_path:
            adapter.events = EventLog(Path(output_path).with_suffix(".events.jsonl"))
        mark("metadata_ready")
        if adapter_name == 'ours':
            from npu_nvme.framework.full_state import MindSporeRestoreTarget
            from npu_nvme.storage.bindings import load_backend
            from .adapters.ours import OursAdapter
            def factory(spec):
                framework, new_model, new_optimizer, new_cell = _model(config, initialized=True)
                return MindSporeRestoreTarget(framework=framework, acl=load_backend().acl_lib,
                    npu=int(config['npu_device']), model=new_model, optimizer=new_optimizer, cell=new_cell,
                    identity=OursAdapter.training_identity(config, framework))
            destination = dict(target_factory=factory, expected_spec=selected['expected_spec'], _step=int(selected['step']))
            ready, receipt = adapter.restore(int(selected['generation']), destination)
            model, optimizer, cell, restored = ready.model, ready.optimizer, ready.cell, ready.controls
        else:
            destination = {'model':model, 'optimizer':optimizer, '_step':int(selected['step']),
                           '_verify_checksums':mode=='verify'}
            restored = adapter.restore(int(selected['generation']), destination)
        mark("read_deserialize_done",
             combined=adapter_name in {"mindspore_native_save", "bytecheckpoint_host"})
        if hasattr(restored, "arrays"):
            snapshot = restored
            applied = apply_snapshot(ms, snapshot,
                                      {"model": model, "optimizer": optimizer}, optimizer)
        else:
            snapshot = None
            applied = restored if adapter_name == "ours" else restore_training_controls(ms, optimizer, restored)
        ms.hal.synchronize()
        mark("state_ready")

        # The performance metric ends at state_ready.  Full-state hashing is
        # deliberately restricted to the independent verification process so
        # a 1.5 GiB oracle scan is never reported as restore latency.
        persisted_digest = None
        applied_digest = None
        byte_exact = None
        if mode == "verify":
            mark("verify_begin")
            if snapshot is not None:
                persisted_digest = snapshot.digest()
            applied_snapshot = capture_snapshot(
                ms, {"model": model, "optimizer": optimizer}, optimizer,
                int(selected["step"]), int(config["seed"]))
            applied_digest = applied_snapshot.digest()
            byte_exact = (
                (persisted_digest is None or
                 persisted_digest == selected.get("sha256")) and
                applied_digest == selected.get("sha256"))
            mark("verify_end", persisted_state_sha256=persisted_digest,
                 applied_state_sha256=applied_digest,
                 expected_state_sha256=selected.get("sha256"),
                 byte_exact=byte_exact)
        steps = int(config["continue_steps"] if continue_steps is None else continue_steps)
        losses = []
        final_step = int(selected["step"])
        for step in range(final_step + 1, final_step + steps + 1):
            if step == final_step + 1:
                mark("first_step_begin", step=step)
            start_token = (step * 104729) % 50257
            ids = (np.arange(int(config["input_tokens"]), dtype=np.int32) + start_token) % 50257
            mask = np.ones_like(ids)
            loss = cell(ms.Tensor(ids[None, :]), ms.Tensor(mask[None, :]))
            ms.hal.synchronize()
            value = float(np.asarray(loss.asnumpy()).reshape(()))
            losses.append(value)
            if step == final_step + 1:
                mark("first_step_end", step=step, loss=value,
                     includes_graph_compile=True)
        reference = {row['step']: row['loss'] for row in result['steps'] + result['source_oracle']}
        wanted = list(range(final_step + 1, final_step + steps + 1))
        if steps <= 0 or any(step not in reference for step in wanted):
            raise ValueError('source trajectory does not cover requested continuation')
        oracle = [reference[step] for step in wanted]
        deviations = [abs(float(a) - float(b)) for a, b in zip(losses, oracle)]
        loss_ok = (len(losses) == len(oracle) == steps and
                   bool(np.all(np.isfinite(losses))) and
                   bool(np.allclose(losses, oracle, rtol=float(config['loss_rtol']), atol=float(config['loss_atol']))))
        restore_result = {
            "status": ("timing_measured" if mode == "timing" else "pass") if (mode == "timing" or byte_exact is True) and
                      loss_ok else "restore_failed",
            "adapter": adapter_name, "generation": int(selected["generation"]),
            "checkpoint_step": int(selected["step"]),
            "verification_performed": mode == "verify",
            "byte_exact": byte_exact,
            "persisted_state_sha256": persisted_digest,
            "applied_state_sha256": applied_digest,
            "expected_state_sha256": selected.get("sha256"),
            "controls": {key: str(value) for key, value in applied.items()},
            "restored_losses": losses, "source_oracle_losses": oracle,
            "loss_deviation": deviations,
            "timing": {"events": events,
                       "state_ready_ns": next((e["monotonic_ns"] for e in events if e["event"] == "state_ready"), None),
                       "restore_begin_ns": next((e["monotonic_ns"] for e in events if e["event"] == "restore_begin"), None),
                       "first_step_begin_ns": next((e["monotonic_ns"] for e in events if e["event"] == "first_step_begin"), None),
                       "first_step_end_ns": next((e["monotonic_ns"] for e in events if e["event"] == "first_step_end"), None),
                       "process_start_ns": process_start},
            "mode": mode, "repeat_index": repeat_index,
        }
    except Exception as error:
        restore_result = {"status": "restore_failed", "adapter": adapter_name,
                          "error": repr(error), "timing": {"events": events},
                          "mode": mode, "repeat_index": repeat_index}
        raise
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                restore_result.setdefault("cleanup_error", repr(error))
    destination_path = Path(output_path) if output_path else run_dir / "restore.json"
    if destination_path.exists() and output_path:
        raise FileExistsError(f"refusing to overwrite restore output: {destination_path}")
    write_json(destination_path, restore_result)
    return restore_result
