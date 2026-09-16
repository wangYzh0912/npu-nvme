#!/usr/bin/env python3
"""Deterministic CPU Pareto scan for the R1/R2 candidate policies.

This is a policy/oracle experiment, not a MindSpore training benchmark.  It
uses a three-phase parameter trajectory and measures the things that decide
whether a policy is worth promoting to an NPU run: physical bytes, final
replacement error, and maximum block age.

R1 stores selected current blocks with a per-block symmetric INT8 scale.
R2 adds residual feedback and a maximum-age rule: selection is driven by the
unpersisted residual and an old block is forced into the frame.  Both models
use ACK-only reference advancement and replacement semantics.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from s2_policy import S2SelectivePolicy  # noqa: E402


def trajectory(steps, seed=42):
    rng = np.random.default_rng(seed)
    sizes = (4096, 2048, 1024, 511)
    state = {f"p{i}": rng.normal(0, 0.05, n).astype(np.float32)
             for i, n in enumerate(sizes)}
    initial = {name: value.copy() for name, value in state.items()}
    states = []
    for step in range(steps):
        current = {name: value.copy() for name, value in state.items()}
        if step < steps // 3:
            current["p0"][step % 4096] += 0.8
            current["p1"][:8] += 0.03
        elif step < 2 * steps // 3:
            for name, value in current.items():
                value += rng.normal(0, 0.002, value.size).astype(np.float32)
        else:
            current["p0"][:64] += 0.012
            current["p2"][step % 1024] += 0.15
            current["p3"] += 0.001
        states.append(current)
        state = current
    return initial, states


def scan_policy(initial, states, block_size, fraction, policy, max_age=0,
                encoding="int8", full_interval=0):
    oracle = S2SelectivePolicy(
        initial, block_size, fraction, encoding=encoding,
        max_age=max_age if policy == "r2_implicit" else 0,
        small_threshold=0)
    delta_bytes = 0
    periodic_full_bytes = 0
    selected_counts = []
    errors = []
    max_ages = []
    state_bytes = sum(value.nbytes for value in initial.values())
    for step, state in enumerate(states, 1):
        if full_interval and step % full_interval == 0:
            periodic_full_bytes += ((state_bytes + 4095) // 4096) * 4096
            oracle.reset_full(state)
            selected_counts.append(0)
        else:
            frame = oracle.observe(state, step)
            oracle.ack(frame["generation"])
            delta_bytes += frame["physical_bytes"]
            selected_counts.append(len(frame["selected"]))
        max_ages.append(int(oracle.age.max(initial=0)))
        errors.append(oracle.relative_l2(state))
    full_bytes = state_bytes * len(states)
    return {
        "policy": policy,
        "block_size": block_size,
        "selection_fraction": fraction,
        "max_age": max_age if policy == "r2_implicit" else None,
        "encoding": encoding, "full_interval": full_interval,
        "steps": len(states),
        "blocks": len(oracle.blocks),
        "mean_selected_blocks": float(np.mean(selected_counts)),
        "delta_physical_bytes": int(delta_bytes),
        "periodic_full_bytes": int(periodic_full_bytes),
        "total_physical_bytes": int(delta_bytes + periodic_full_bytes),
        "full_value_bytes": int(full_bytes),
        "write_ratio": float((delta_bytes + periodic_full_bytes) /
                             max(full_bytes, 1)),
        "final_relative_l2_error": errors[-1],
        "max_relative_l2_error": max(errors),
        "max_block_age": max(max_ages),
        "selected_blocks": selected_counts,
        "error_by_step": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    started = time.perf_counter()
    initial, states = trajectory(args.steps, args.seed)
    rows = []
    for block_size in (64, 256, 1024):
        for fraction in (0.01, 0.05, 0.10, 0.20):
            for encoding in ("fp16", "int8"):
                for full_interval in (20, 50, 100, 200):
                    rows.append(scan_policy(
                        initial, states, block_size, fraction, "r1",
                        encoding=encoding, full_interval=full_interval))
                    for max_age in (4, 8, 16):
                        rows.append(scan_policy(
                            initial, states, block_size, fraction,
                            "r2_implicit", max_age=max_age,
                            encoding=encoding, full_interval=full_interval))
    result = {
        "status": "pass",
        "experiment": "R1_R2_CPU_POLICY_SCAN",
        "semantic_scope": "S2 lossless replacement oracle with INT8 candidate encoding",
        "note": ("synthetic deterministic trajectory; S2-R2 uses implicit "
                 "current-minus-decoded-reference error feedback"),
        "seed": args.seed,
        "steps": args.steps,
        "rows": rows,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "experiment", "steps", "elapsed_ms")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
