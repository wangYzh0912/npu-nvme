#!/usr/bin/env python3
"""Online CPU R1/R2 scan over a deterministic real MindSpore trajectory.

Each invocation owns only one candidate batch.  ``--preset grid`` is split by
``--batch-index/--batch-count`` so the full Cartesian scan never needs every
decoded persisted reference resident at once.  Repeated batches use the same
seed and logical-step data and record losses for trajectory identity checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "python")]

from experiments.benchmarks.io_matrix import (  # noqa: E402
    check_npu_free, environment_snapshot,
)
from experiments.benchmarks.r0_real_e2e import (  # noqa: E402
    build, control_state, train_one,
)
from experiments.benchmarks.s2_real_trajectory import (  # noqa: E402
    snapshot_state, state_category,
)
from s2_delta import build_block_manifest, score_manifest_blocks  # noqa: E402
from s2_policy import S2SelectivePolicy  # noqa: E402
from training_state import encode_control_value  # noqa: E402


def align(value, alignment=4096):
    return ((int(value) + alignment - 1) // alignment) * alignment


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def threshold_calibration(initial, current, block_sizes, fractions):
    thresholds = {}
    for block_size in block_sizes:
        manifest = build_block_manifest(initial, block_size, small_threshold=0)
        scores = score_manifest_blocks(current, initial, manifest)
        values = np.asarray([item["score"] for item in scores], dtype=np.float64)
        thresholds[str(block_size)] = {}
        for fraction in fractions:
            count = max(1, int(math.ceil(values.size * fraction)))
            thresholds[str(block_size)][str(fraction)] = float(
                values[min(count - 1, values.size - 1)]) if values.size else 0.0
    return thresholds


def candidate_grid(thresholds):
    candidates = []
    block_sizes = (65536, 262144, 524288)
    topk = (("topk", value) for value in (0.01, 0.05, 0.10, 0.20))
    energy = (("error_budget", value) for value in (0.80, 0.90, 0.95, 0.99))
    modes = list(topk) + list(energy) + [
        ("threshold", value) for value in (0.01, 0.05, 0.10, 0.20)]
    for block_size in block_sizes:
        for selection_mode, budget in modes:
            threshold = (thresholds[str(block_size)][str(budget)]
                         if selection_mode == "threshold" else 0.0)
            for encoding in ("fp16", "int8"):
                for full_interval in (20, 50, 100, 200):
                    candidates.append({
                        "strategy": "r1", "block_size": block_size,
                        "selection_mode": selection_mode, "budget": budget,
                        "score_threshold": threshold, "encoding": encoding,
                        "max_age": 0, "full_interval": full_interval,
                    })
                    for max_age in (4, 8, 16):
                        candidates.append({
                            "strategy": "r2", "block_size": block_size,
                            "selection_mode": selection_mode, "budget": budget,
                            "score_threshold": threshold, "encoding": encoding,
                            "max_age": max_age,
                            "full_interval": full_interval,
                        })
    return candidates


def quick_candidates():
    return [
        {"strategy": "r2", "block_size": 262144,
         "selection_mode": "topk", "budget": 0.05,
         "score_threshold": 0.0, "encoding": "int8", "max_age": 8,
         "full_interval": 100},
        {"strategy": "r2", "block_size": 65536,
         "selection_mode": "topk", "budget": 0.20,
         "score_threshold": 0.0, "encoding": "fp16", "max_age": 4,
         "full_interval": 20},
        {"strategy": "r2", "block_size": 262144,
         "selection_mode": "error_budget", "budget": 0.90,
         "score_threshold": 0.0, "encoding": "int8", "max_age": 8,
         "full_interval": 50},
    ]


def candidate_id(config):
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def control_bytes(ms, optimizer, step, seed):
    controls = control_state(ms, optimizer, step, seed)
    payload = 0
    for value in controls.values():
        encoded, _metadata = encode_control_value(value)
        payload += int(encoded.nbytes)
    return payload, len(controls)


def assign_state(ms, model, optimizer, state):
    """Assign one canonical host state to the compiled training cell."""
    parameters = {}
    for parameter in model.get_parameters():
        parameters[f"model/{parameter.name}"] = parameter
    for prefix, values in (("optimizer/m", optimizer.moments1),
                           ("optimizer/v", optimizer.moments2)):
        for parameter in values:
            parameters[f"{prefix}/{parameter.name}"] = parameter
    parameters["optimizer/global_step"] = optimizer.global_step
    if set(parameters) != set(state):
        missing = sorted(set(parameters) - set(state))
        extra = sorted(set(state) - set(parameters))
        raise ValueError(f"state assignment field mismatch: missing={missing[:3]} "
                         f"extra={extra[:3]}")
    for name, parameter in parameters.items():
        parameter.set_data(ms.Tensor(np.asarray(state[name])))


def relative_l2_by_category(current, reference):
    totals = {}
    for name, value in current.items():
        actual = np.asarray(value).astype(np.float64)
        restored = np.asarray(reference[name]).astype(np.float64)
        category = state_category(name)
        record = totals.setdefault(category, [0.0, 0.0])
        difference = actual - restored
        record[0] += float(np.vdot(difference, difference))
        record[1] += float(np.vdot(actual, actual))
    return {category: float(np.sqrt(error / max(energy, 1e-30)))
            for category, (error, energy) in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gpt2", "gpt2_xl"), default="gpt2")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--npu", type=int, default=0)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--preset", choices=("quick", "grid"), default="quick")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_count <= 0 or not 0 <= args.batch_index < args.batch_count:
        raise ValueError("invalid scan dimensions")
    npu_info = check_npu_free(args.npu)
    started = time.perf_counter()
    ms, model, optimizer, cell = build(args)
    initial = snapshot_state(model, optimizer, True)
    state_bytes = sum(int(value.nbytes) for value in initial.values())

    first_loss, first_ms = train_one(cell, ms, 1, args.seq_len)
    current = snapshot_state(model, optimizer, True)
    thresholds = threshold_calibration(
        initial, current, (65536, 262144, 524288),
        (0.01, 0.05, 0.10, 0.20))
    all_candidates = (quick_candidates() if args.preset == "quick"
                      else candidate_grid(thresholds))
    candidates = [config for index, config in enumerate(all_candidates)
                  if index % args.batch_count == args.batch_index]
    policies = {}
    metrics = {}
    for config in candidates:
        key = candidate_id(config)
        policies[key] = S2SelectivePolicy(
            initial, config["block_size"], config["budget"],
            encoding=config["encoding"], max_age=config["max_age"],
            small_threshold=10000,
            selection_mode=config["selection_mode"],
            score_threshold=config["score_threshold"])
        metrics[key] = {"candidate_id": key, "config": config,
                        "physical_bytes": 0, "delta_bytes": 0,
                        "full_bytes": 0, "errors": [], "max_ages": [],
                        "selected_blocks": [], "payload_bytes": 0,
                        "raw_payload_bytes": 0,
                        "encoded_payload_bytes": 0,
                        "descriptor_bytes": 0, "control_bytes": 0,
                        "category_errors": []}
    estimated_reference_bytes = state_bytes * len(candidates)
    losses = [first_loss]
    training_ms = [first_ms]
    full_only_bytes = 0

    for step in range(1, args.steps + 1):
        if step > 1:
            loss, elapsed = train_one(cell, ms, step, args.seq_len)
            losses.append(loss)
            training_ms.append(elapsed)
            current = snapshot_state(model, optimizer, True)
        controls_payload, controls_count = control_bytes(
            ms, optimizer, step, args.seed)
        full_only_bytes += align(
            state_bytes + controls_payload + controls_count * 96 + 4096)
        for key, policy in policies.items():
            row = metrics[key]
            interval = row["config"]["full_interval"]
            if interval and step % interval == 0:
                physical = align(state_bytes + controls_payload +
                                 controls_count * 96 + 4096)
                row["physical_bytes"] += physical
                row["full_bytes"] += physical
                row["control_bytes"] += controls_payload
                row["selected_blocks"].append(0)
                policy.reset_full(current)
            else:
                frame = policy.observe(current, step)
                policy.ack(frame["generation"])
                physical = align(4096 + frame["descriptor_bytes"] +
                                 frame["payload_bytes"] + controls_payload +
                                 controls_count * 96)
                row["physical_bytes"] += physical
                row["delta_bytes"] += physical
                row["payload_bytes"] += frame["payload_bytes"]
                row["raw_payload_bytes"] += frame["raw_payload_bytes"]
                row["encoded_payload_bytes"] += frame[
                    "encoded_payload_bytes"]
                row["descriptor_bytes"] += frame["descriptor_bytes"]
                row["control_bytes"] += controls_payload
                row["selected_blocks"].append(len(frame["selected"]))
            row["errors"].append(policy.relative_l2(current))
            row["category_errors"].append(relative_l2_by_category(
                current, policy.reference))
            row["max_ages"].append(int(policy.age.max(initial=0)))
        del current

    committed_state = snapshot_state(model, optimizer, True)
    assign_state(ms, model, optimizer, committed_state)
    oracle_recovery_loss, _oracle_ms = train_one(
        cell, ms, args.steps + 1, args.seq_len)
    for key, policy in policies.items():
        assign_state(ms, model, optimizer, policy.reference)
        recovered_loss, _recovered_ms = train_one(
            cell, ms, args.steps + 1, args.seq_len)
        row = metrics[key]
        row["oracle_recovery_loss"] = oracle_recovery_loss
        row["recovered_loss"] = recovered_loss
        row["recovery_loss_relative_error"] = abs(
            recovered_loss - oracle_recovery_loss) / max(
                abs(oracle_recovery_loss), 1e-30)
    rows = []
    for row in metrics.values():
        row["write_ratio"] = row["physical_bytes"] / full_only_bytes
        row["final_relative_l2_error"] = row["errors"][-1]
        row["max_relative_l2_error"] = max(row["errors"])
        categories = sorted({category for sample in row["category_errors"]
                             for category in sample})
        row["max_category_relative_l2_error"] = {
            category: max(sample.get(category, 0.0)
                          for sample in row["category_errors"])
            for category in categories}
        row["final_category_relative_l2_error"] = (
            row["category_errors"][-1] if row["category_errors"] else {})
        row["max_block_age"] = max(row["max_ages"], default=0)
        row["eligible_go"] = (
            row["write_ratio"] < 0.20 and
            (row["config"]["strategy"] == "r2" and
             row["max_block_age"] < row["config"]["max_age"]) and
            row["final_relative_l2_error"] <= 1e-2 and
            row["recovery_loss_relative_error"] <= 0.01)
        rows.append(row)
    rows.sort(key=lambda row: (row["write_ratio"],
                               row["final_relative_l2_error"]))
    result = {
        "status": "PASS", "experiment": "S2_REAL_POLICY_SCAN",
        "semantic": "S2 implicit current-minus-decoded-reference feedback",
        "model": args.model, "seed": args.seed, "steps": args.steps,
        "preset": args.preset, "batch_index": args.batch_index,
        "batch_count": args.batch_count,
        "all_candidates": len(all_candidates),
        "batch_candidates": len(candidates),
        "state_bytes": state_bytes,
        "estimated_reference_bytes": estimated_reference_bytes,
        "full_only_bytes": full_only_bytes,
        "thresholds": thresholds, "losses": losses,
        "training_ms": training_ms, "rows": rows,
        "environment": environment_snapshot(args, npu_info),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "model": args.model,
                      "steps": args.steps, "candidates": len(rows),
                      "best_write_ratio": rows[0]["write_ratio"] if rows else None},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
