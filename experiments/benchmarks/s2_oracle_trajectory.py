#!/usr/bin/env python3
"""CPU S2/R0 trajectory replay for I1 and O1/O2 measurements.

The default trajectory is deterministic and intentionally has early sparse,
middle dense, and late hot-block phases.  A caller can replace it with real
captured states through ``run_trajectory``; the CLI is a reproducible smoke
trajectory and is not presented as a MindSpore training-quality result.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from s2_delta import S2DeltaOracle, apply_s2_replacements  # noqa: E402


def _state_equal(left, right):
    return all(np.array_equal(left[name], right[name], equal_nan=True)
               for name in left)


def synthetic_trajectory(steps, seed=42):
    rng = np.random.default_rng(seed)
    initial = {
        "backbone.blocks.0.weight": np.zeros(96, dtype=np.float32),
        "backbone.blocks.1.weight": np.zeros(65, dtype=np.float16),
        "backbone.layernorm.bias": np.zeros(7, dtype=np.float32),
    }
    states = []
    state = {name: value.copy() for name, value in initial.items()}
    for step in range(steps):
        update = {name: value.copy() for name, value in state.items()}
        if step < max(1, steps // 3):
            update["backbone.blocks.0.weight"][step % 96] += 1.0
        elif step < max(2, 2 * steps // 3):
            update["backbone.blocks.0.weight"] += rng.normal(0, 0.01, 96)
            update["backbone.blocks.1.weight"] += rng.normal(0, 0.02, 65).astype(np.float16)
        else:
            update["backbone.blocks.0.weight"][0:8] += 0.25
            update["backbone.layernorm.bias"][step % 7] += 0.125
        states.append(update)
        state = update
    return initial, states


def run_trajectory(states, initial, block_size=16, small_threshold=8,
                   top_k=None, epsilon=0.0):
    oracle = S2DeltaOracle(initial, block_size=block_size,
                           small_threshold=small_threshold,
                           top_k=top_k, change_epsilon=epsilon)
    frames = []
    rows = []
    previous_ids = set()
    started = time.perf_counter()
    for step, state in enumerate(states):
        oracle.set_current(state)
        frame = oracle.observe(step)
        decoded_step, blocks, smalls, info = __import__(
            "delta_protocol").unpack_s2_replacement_frame(frame)
        ids = {int(block["block_id"]) for block in blocks}
        union = ids | previous_ids
        jaccard = len(ids & previous_ids) / len(union) if union else 1.0
        rows.append({
            "step": decoded_step,
            "generation": info["generation"],
            "blocks": len(blocks),
            "small_params": len(smalls),
            "frame_bytes": len(frame),
            "jaccard_previous": jaccard,
            "selected_block_ids": sorted(ids),
        })
        frames.append(frame)
        oracle.ack(frame)
        previous_ids = ids

    recovered = oracle.recover(initial, frames)
    if not states:
        expected = initial
    else:
        expected = states[-1]
    if not _state_equal(recovered["state"], expected):
        raise AssertionError("S2 trajectory recovery mismatch")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "status": "pass",
        "steps": len(states),
        "block_size": block_size,
        "small_threshold": small_threshold,
        "top_k": top_k,
        "manifest_digest": oracle.manifest_digest,
        "total_frame_bytes": sum(len(frame) for frame in frames),
        "oracle_ms": elapsed_ms,
        "recovery_generation": recovered["generation"],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--small-threshold", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    initial, states = synthetic_trajectory(args.steps)
    result = run_trajectory(states, initial, args.block_size,
                            args.small_threshold, args.top_k, args.epsilon)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
