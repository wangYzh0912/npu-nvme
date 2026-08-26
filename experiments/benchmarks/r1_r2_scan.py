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


def trajectory(steps, seed=42):
    rng = np.random.default_rng(seed)
    sizes = (4096, 2048, 1024, 511)
    state = {f"p{i}": rng.normal(0, 0.05, n).astype(np.float32)
             for i, n in enumerate(sizes)}
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
    return states


def blocks(states, block_size):
    names = list(states[0])
    layout = []
    for name in names:
        n = states[0][name].size
        for start in range(0, n, block_size):
            layout.append((name, start, min(block_size, n - start)))
    return layout


def quantize(values):
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    scale = peak / 127.0 if peak > 0 else 1.0
    q = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
    return q.astype(np.float32) * scale, scale


def scan_policy(states, block_size, fraction, policy, max_age=0):
    initial = {name: value.copy() for name, value in states[0].items()}
    layout = blocks(states, block_size)
    reference = {name: value.copy() for name, value in initial.items()}
    residual = {name: np.zeros_like(value) for name, value in initial.items()}
    age = np.zeros(len(layout), dtype=np.int64)
    total_bytes = 0
    selected_counts = []
    errors = []
    max_ages = []
    for state in states:
        scores = []
        for index, (name, start, count) in enumerate(layout):
            end = start + count
            delta = state[name][start:end] - reference[name][start:end]
            if policy == "r2_residual":
                delta = residual[name][start:end] + delta
            scores.append(float(np.linalg.norm(delta.astype(np.float64))))
        target = max(1, int(np.ceil(len(layout) * fraction)))
        selected = set(np.argsort(np.asarray(scores))[::-1][:target].tolist())
        if policy == "r2_residual" and max_age > 0:
            selected.update(np.nonzero(age >= max_age - 1)[0].tolist())
        for index, (name, start, count) in enumerate(layout):
            end = start + count
            if index in selected:
                stored, scale = quantize(state[name][start:end])
                reference[name][start:end] = stored
                total_bytes += count + 4 + 24  # INT8 data + scale + frame header
                residual[name][start:end] = state[name][start:end] - stored
                age[index] = 0
            else:
                residual[name][start:end] = state[name][start:end] - reference[name][start:end]
                age[index] += 1
        selected_counts.append(len(selected))
        max_ages.append(int(age.max(initial=0)))
        sq = 0.0
        denom = 0.0
        for name in state:
            diff = (state[name] - reference[name]).astype(np.float64)
            sq += float(np.dot(diff, diff))
            denom += float(np.dot(state[name].astype(np.float64), state[name]))
        errors.append(float(np.sqrt(sq / max(denom, 1e-30))))
    full_bytes = sum(value.nbytes for value in states[-1].values()) * len(states)
    return {
        "policy": policy,
        "block_size": block_size,
        "selection_fraction": fraction,
        "max_age": max_age if policy == "r2_residual" else None,
        "steps": len(states),
        "blocks": len(layout),
        "mean_selected_blocks": float(np.mean(selected_counts)),
        "total_int8_frame_bytes": int(total_bytes),
        "full_value_bytes": int(full_bytes),
        "write_ratio": float(total_bytes / max(full_bytes, 1)),
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
    states = trajectory(args.steps, args.seed)
    rows = []
    for block_size in (64, 256, 1024):
        for fraction in (0.01, 0.05, 0.10, 0.20):
            rows.append(scan_policy(states, block_size, fraction, "r1_int8"))
            rows.append(scan_policy(states, block_size, fraction, "r2_residual", max_age=8))
    result = {
        "status": "pass",
        "experiment": "R1_R2_CPU_POLICY_SCAN",
        "semantic_scope": "S2 lossless replacement oracle with INT8 candidate encoding",
        "note": "synthetic deterministic trajectory; not a real MindSpore training result",
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
