#!/usr/bin/env python3
"""I1 real MindSpore trajectory collector for the S2 oracle.

The collector streams one snapshot at a time and records block statistics
instead of retaining every training state.  GPT-2 124M is the exact
weight+Adam state lane; larger models can use the same collector with
``--no-optimizer`` for a scale-only observation lane.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from experiments.benchmarks.io_matrix import (  # noqa: E402
    ResultWriter, check_npu_free, environment_snapshot,
)
from experiments.common import (  # noqa: E402
    init_env, make_causal_lm_training, warmup_model,
)
from direct_checkpoint import ProbeTrainOneStepCell  # noqa: E402
from s2_delta import S2DeltaOracle, score_manifest_blocks  # noqa: E402


def snapshot_state(model, optimizer, include_optimizer):
    state = {}
    for parameter in model.get_parameters():
        state[f"model/{parameter.name}"] = parameter.asnumpy()
    if include_optimizer:
        for prefix, parameters in (
                ("optimizer/m", optimizer.moments1),
                ("optimizer/v", optimizer.moments2)):
            for parameter in parameters:
                state[f"{prefix}/{parameter.name}"] = parameter.asnumpy()
        state["optimizer/global_step"] = optimizer.global_step.asnumpy()
    return state


def finite_diagnostics(state):
    """Return a compact numeric-health report without retaining tensors."""
    bad = []
    finite_arrays = 0
    for name, value in state.items():
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.inexact):
            continue
        finite_arrays += 1
        nonfinite = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        if nonfinite:
            bad.append({
                "name": name,
                "dtype": array.dtype.name,
                "elements": int(array.size),
                "nonfinite": nonfinite,
                "nan": int(np.isnan(array).sum()),
                "inf": int(np.isinf(array).sum()),
            })
    return {"arrays": len(state), "floating_arrays": finite_arrays,
            "nonfinite_arrays": len(bad), "nonfinite": bad}


def block_scores(current, reference, manifest):
    return score_manifest_blocks(current, reference, manifest)


def coverage(scores, budgets):
    total = sum(item["score"] ** 2 for item in scores)
    output = {}
    for budget in budgets:
        count = max(1, int(np.ceil(len(scores) * budget))) if scores else 0
        selected = scores[:count]
        energy = sum(item["score"] ** 2 for item in selected)
        output[str(int(budget * 100))] = {
            "count": len(selected),
            "energy_fraction": (energy / total) if total else 1.0,
            "block_ids": [item["block_id"] for item in selected],
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gpt2", "gpt2_xl"), default="gpt2")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=1025)
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--small-threshold", type=int, default=10000)
    parser.add_argument("--top-k-percent", type=float, default=10.0)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--no-optimizer", action="store_true")
    parser.add_argument("--numeric-only", action="store_true",
                        help="check loss/state health without S2 scoring/frames")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.steps <= 0 or args.sample_every <= 0 or not 0 < args.top_k_percent <= 100:
        raise ValueError("invalid trajectory dimensions")

    writer = ResultWriter("I1_REAL", args)
    writer.config.update({
        "model": args.model,
        "state": "weights_only" if args.no_optimizer else "weights+adam_m_v+global_step",
        "oracle": "S2 persisted-reference replacement",
    })
    writer.write_json("config.json", writer.config)
    npu_info = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, npu_info))

    init_env(device_id=args.npu, seed=args.seed)
    model, dataset, optimizer = make_causal_lm_training(
        args.model, total_steps=args.steps + 2, device_id=args.npu,
        seq_len=args.seq_len)
    warmup_model(model, optimizer, dataset)
    cell = ProbeTrainOneStepCell(model, optimizer, enable_probe=False,
                                 ckpt_interval=9999)
    iterator = dataset.create_tuple_iterator()

    initial = snapshot_state(model, optimizer, not args.no_optimizer)
    initial_health = finite_diagnostics(initial)
    writer.write_json("numeric_gate.json", {
        "stage": "post_warmup_initial_state",
        **initial_health,
    })
    if initial_health["nonfinite_arrays"]:
        failure = {
            "error": "post-warmup model state contains non-finite values",
            "stage": "post_warmup_initial_state",
            "numeric_gate": initial_health,
        }
        writer.add_failure(failure)
        result = writer.finalize({"numeric_gate": initial_health}, status="fail")
        print(json.dumps({"status": result["status"],
                          "run_id": writer.run_id,
                          "summary": result["summary"]}, indent=2), flush=True)
        return
    if args.numeric_only:
        samples = []
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = dataset.create_tuple_iterator()
                batch = next(iterator)
            loss = cell(*batch)
            import mindspore as ms
            ms.hal.synchronize()
            loss_value = float(np.asarray(loss.asnumpy()).reshape(()))
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite training loss at step {step}: {loss_value}")
            if step % args.sample_every:
                continue
            health = finite_diagnostics(snapshot_state(
                model, optimizer, not args.no_optimizer))
            if health["nonfinite_arrays"]:
                raise FloatingPointError(
                    f"non-finite training state at step {step}: "
                    f"{health['nonfinite'][:3]}")
            sample = {
                "run_id": writer.run_id,
                "request_id": f"{writer.run_id}/step_{step:04d}",
                "checkpoint_id": f"numeric_{step:04d}",
                "step": step,
                "status": "pass",
                "loss": loss_value,
                "numeric_health": health,
                "events": [{"name": "numeric_health_pass",
                            "monotonic_ns": time.monotonic_ns()}],
                "timeline_us": {"end_to_end": 0},
            }
            writer.add_sample(sample)
            samples.append(sample)
        result = writer.finalize({
            "steps": args.steps,
            "samples": len(samples),
            "seed": args.seed,
            "numeric_health": "all sampled weights/optimizer states finite",
            "losses": [sample["loss"] for sample in samples],
        }, status="pass")
        print(json.dumps({"status": result["status"],
                          "run_id": writer.run_id,
                          "summary": result["summary"]}, indent=2), flush=True)
        return
    oracle = S2DeltaOracle(initial, block_size=args.block_size,
                           small_threshold=args.small_threshold,
                           top_k=None)
    previous = {name: np.array(value, copy=True)
                for name, value in initial.items()}
    persisted = {name: np.array(value, copy=True)
                 for name, value in initial.items()}
    policy_oracle = S2DeltaOracle(initial, block_size=args.block_size,
                                  small_threshold=args.small_threshold,
                                  top_k=max(1, int(len(oracle.manifest["blocks"])
                                                 * args.top_k_percent / 100)))
    manifest = oracle.manifest
    budgets = (0.01, 0.05, 0.10, 0.20, 0.50, 1.0)
    last_selected = set()
    last_acked = {int(item["block_id"]): 0 for item in manifest["blocks"]}
    trajectory = []
    try:
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = dataset.create_tuple_iterator()
                batch = next(iterator)
            loss = cell(*batch)
            import mindspore as ms
            ms.hal.synchronize()
            loss_value = float(np.asarray(loss.asnumpy()).reshape(()))
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite training loss at step {step}: {loss_value}")
            if step % args.sample_every:
                continue
            current = snapshot_state(model, optimizer, not args.no_optimizer)
            health = finite_diagnostics(current)
            if health["nonfinite_arrays"]:
                raise FloatingPointError(
                    f"non-finite training state at step {step}: "
                    f"{health['nonfinite'][:3]}")
            oracle.set_current(current)
            policy_oracle.set_current(current)
            adjacent = block_scores(current, previous, manifest)
            persisted_scores = block_scores(current, persisted, manifest)
            selected = {item["block_id"] for item in persisted_scores[:max(
                1, int(len(persisted_scores) * args.top_k_percent / 100))]}
            for block_id in selected:
                last_acked[block_id] = step
            ages = [step - last_acked[item["block_id"]]
                    for item in manifest["blocks"]]
            frame = policy_oracle.observe(step)
            policy_oracle.ack(frame)
            sample = {
                "run_id": writer.run_id,
                "request_id": f"{writer.run_id}/step_{step:04d}",
                "checkpoint_id": f"trajectory_{step:04d}",
                "step": step,
                "status": "pass",
                "loss": loss_value,
                "numeric_health": health,
                "state_arrays": len(current),
                "block_count": len(adjacent),
                "frame_bytes_top_k": len(frame),
                "adjacent_l2": float(np.sqrt(sum(item["score"] ** 2
                                                   for item in adjacent))),
                "persisted_l2": float(np.sqrt(sum(item["score"] ** 2
                                                    for item in persisted_scores))),
                "selected_count": len(selected),
                "selected_jaccard": (len(selected & last_selected) /
                                      len(selected | last_selected)
                                      if selected | last_selected else 1.0),
                "age": {"max": max(ages, default=0),
                        "mean": float(np.mean(ages)) if ages else 0.0},
                "coverage": coverage(adjacent, budgets),
                "timeline_us": {"end_to_end": 0},
                "events": [{"name": "trajectory_sample",
                            "monotonic_ns": time.monotonic_ns()}],
            }
            writer.add_sample(sample)
            trajectory.append(sample)
            last_selected = selected
            previous = {name: np.array(value, copy=True)
                        for name, value in current.items()}
            persisted = {name: np.array(value, copy=True)
                         for name, value in policy_oracle.persisted_reference.items()}
            del current
        result = writer.finalize({
            "steps": args.steps,
            "samples": len(trajectory),
            "manifest_digest": oracle.manifest_digest,
            "final_generation": policy_oracle.persisted_generation,
            "final_frame_bytes": trajectory[-1]["frame_bytes_top_k"] if trajectory else 0,
        }, status="pass")
        print(json.dumps({"status": result["status"], "run_id": writer.run_id,
                          "summary": result["summary"]}, indent=2), flush=True)
    except BaseException as error:
        writer.add_failure({"error": repr(error)})
        writer.finalize({}, status="fail")
        raise


if __name__ == "__main__":
    main()
