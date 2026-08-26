#!/usr/bin/env python3
"""I4 ordinary-file cross-process FULL + S2 frame replay gate."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from experiments.benchmarks.io_matrix import (  # noqa: E402
    ResultWriter, environment_snapshot, run_command,
)
from s2_delta import FileS2Ring, S2DeltaOracle  # noqa: E402
from delta_protocol import unpack_s2_replacement_frame  # noqa: E402


def digest(state):
    sha = hashlib.sha256()
    for name in sorted(state):
        value = np.asarray(state[name])
        sha.update(name.encode())
        sha.update(value.dtype.str.encode())
        sha.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        sha.update(value.tobytes(order="C"))
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.steps <= 0 or args.slots < args.steps:
        raise ValueError("I4 requires positive steps and enough ring slots")

    writer = ResultWriter("I4_CROSS_PROCESS", args)
    writer.config.update({"scope": "ordinary-file FULL + S2 replay",
                          "full_format": "numpy savez",
                          "frame_format": "S2 replacement v3"})
    writer.write_json("config.json", writer.config)
    writer.write_json("environment.json", environment_snapshot(
        args, run_command(["npu-smi", "info"])))
    run_dir = Path(writer.run_dir)
    ring_dir = run_dir / "ring"
    ring = FileS2Ring(ring_dir, slot_count=args.slots, slot_size=1024 * 1024)
    initial = {
        "backbone.blocks.0.weight": np.zeros(33, dtype=np.float32),
        "backbone.blocks.1.weight": np.ones(19, dtype=np.float16),
        "backbone.layernorm.bias": np.zeros(5, dtype=np.float32),
    }
    current = {name: value.copy() for name, value in initial.items()}
    oracle = S2DeltaOracle(initial, block_size=8, small_threshold=4)
    full_path = run_dir / "full_state.npz"
    np.savez(full_path, **initial)
    with full_path.open("rb") as stream:
        os.fsync(stream.fileno())
    expected = None
    for step in range(1, args.steps + 1):
        current = {name: value.copy() for name, value in current.items()}
        current["backbone.blocks.0.weight"][step % 33] += np.float32(step * 0.25)
        current["backbone.blocks.1.weight"][step % 19] += np.float16(0.5)
        if step % 2:
            current["backbone.layernorm.bias"][step % 5] += np.float32(0.125)
        oracle.set_current(current)
        frame = oracle.observe(step)
        slot = ring.write(frame)
        oracle.ack(frame)
        expected = {name: value.copy() for name, value in current.items()}

    child = r'''import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, sys.argv[3])
from s2_delta import FileS2Ring, S2DeltaOracle
from delta_protocol import unpack_s2_replacement_frame
def digest(state):
    sha = hashlib.sha256()
    for name in sorted(state):
        value = np.asarray(state[name])
        sha.update(name.encode()); sha.update(value.dtype.str.encode())
        sha.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        sha.update(value.tobytes(order="C"))
    return sha.hexdigest()
full = np.load(sys.argv[2], allow_pickle=False)
initial = {name: full[name] for name in full.files}
ring = FileS2Ring(sys.argv[1], int(sys.argv[4]), 1048576)
frames = [ring.read(slot) for slot in range(ring.slot_count)
          if os.path.exists(ring._path(slot))]
frames.sort(key=lambda frame: unpack_s2_replacement_frame(frame)[3]["generation"])
oracle = S2DeltaOracle(initial, block_size=8, small_threshold=4)
recovered = oracle.recover(initial, frames)
print(json.dumps({"generation": recovered["generation"],
                  "last_step": recovered["last_step"],
                  "digest": digest(recovered["state"])}, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-c", child, str(ring_dir), str(full_path),
         str(REPO_ROOT / "python"), str(args.slots)],
        capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"child replay failed: {completed.stderr}")
    child_result = json.loads(completed.stdout)
    passed = (child_result["generation"] == args.steps and
              child_result["last_step"] == args.steps and
              child_result["digest"] == digest(expected))
    sample = {
        "run_id": writer.run_id,
        "request_id": writer.run_id + "/cross_process_0001",
        "checkpoint_id": "full_plus_delta_chain",
        "status": "pass" if passed else "fail",
        "steps": args.steps,
        "ring_slots": args.slots,
        "child": child_result,
        "expected_digest": digest(expected),
        "events": [{"name": "child_replay_complete", "monotonic_ns": time.monotonic_ns()}],
        "timeline_us": {"cross_process": 0},
    }
    writer.add_sample(sample)
    result = writer.finalize({"steps": args.steps, "ring_slots": args.slots,
                              "child_generation": child_result["generation"],
                              "byte_exact": passed},
                             status="pass" if passed else "fail")
    print(json.dumps({"status": result["status"], "run_id": writer.run_id,
                      "summary": result["summary"]}, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
