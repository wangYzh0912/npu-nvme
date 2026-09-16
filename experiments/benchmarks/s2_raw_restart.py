#!/usr/bin/env python3
"""I6 first gate: raw 83.0.0 frame-chain restart and corruption rejection."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from direct_checkpoint import DirectCheckpoint  # noqa: E402
from experiments.benchmarks.io_matrix import (  # noqa: E402
    ResultWriter, SAFE_OFFSET, check_npu_free, environment_snapshot,
)
from s2_delta import S2DeltaOracle  # noqa: E402
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


def chain_states(steps):
    initial = {
        "backbone.blocks.0.weight": np.zeros(33, dtype=np.float32),
        "backbone.blocks.1.weight": np.ones(19, dtype=np.float16),
        "backbone.layernorm.bias": np.zeros(5, dtype=np.float32),
    }
    current = {name: value.copy() for name, value in initial.items()}
    oracle = S2DeltaOracle(initial, block_size=8, small_threshold=4)
    frames = []
    for step in range(1, steps + 1):
        current = {name: value.copy() for name, value in current.items()}
        current["backbone.blocks.0.weight"][step % 33] += np.float32(step * 0.25)
        current["backbone.blocks.1.weight"][step % 19] += np.float16(0.5)
        if step % 2:
            current["backbone.layernorm.bias"][step % 5] += np.float32(0.125)
        oracle.set_current(current)
        frame = oracle.observe(step)
        oracle.ack(frame)
        frames.append(frame)
    return initial, current, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=4)
    parser.add_argument("--shm-id", type=int, default=1406)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--offset", type=int, default=SAFE_OFFSET + 512 * 1024**2)
    parser.add_argument("--stride", type=int, default=2 * 1024**2)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.steps <= 0 or args.offset % 4096 or args.stride % 4096:
        raise ValueError("I6 offsets must be positive 4 KiB aligned values")

    writer = ResultWriter("I6_RAW_RESTART", args)
    writer.config.update({"scope": "83.0.0 raw safe-region restart",
                          "metadata_touched": False,
                          "formatting": False,
                          "corruption": "payload byte flip; CRC must reject"})
    writer.write_json("config.json", writer.config)
    npu_info = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, npu_info))
    initial, expected, frames = chain_states(args.steps)
    run_dir = Path(writer.run_dir)
    manifest = {
        "pci": args.pci, "npu": args.npu, "shm_id": args.shm_id,
        "steps": args.steps, "offset": args.offset, "stride": args.stride,
        "frames": [{"step": step, "offset": args.offset + (step - 1) * args.stride,
                    "size": len(frame)}
                   for step, frame in enumerate(frames, 1)],
        "corrupt_offset": args.offset + args.steps * args.stride,
        "expected_digest": digest(expected),
    }
    writer.write_json("chain_manifest.json", manifest)

    ckpt = DirectCheckpoint(nvme_addr=args.pci, npu_device_id=args.npu,
                            pipeline_depth=4, requested_chunk_size=4 * 1024**2,
                            spdk_shm_id=args.shm_id)
    try:
        for item, frame in zip(manifest["frames"], frames):
            ckpt.write_host_frame(frame, item["offset"])
        corrupted = bytearray(frames[-1])
        corrupted[-1] ^= 0x01
        ckpt.write_host_frame(bytes(corrupted), manifest["corrupt_offset"])
    finally:
        ckpt.cleanup()

    child = r'''import json, os, sys
import numpy as np
sys.path.insert(0, sys.argv[2]); sys.path.insert(0, sys.argv[3])
from direct_checkpoint import DirectCheckpoint
from experiments.benchmarks.io_matrix import check_npu_free
from s2_delta import S2DeltaOracle
from delta_protocol import unpack_s2_replacement_frame
def digest(state):
 import hashlib
 sha = hashlib.sha256()
 for name in sorted(state):
  value = np.asarray(state[name]); sha.update(name.encode()); sha.update(value.dtype.str.encode())
  sha.update(np.asarray(value.shape, dtype=np.int64).tobytes()); sha.update(value.tobytes(order="C"))
 return sha.hexdigest()
manifest = json.load(open(sys.argv[1]))
check_npu_free(manifest["npu"])
os.environ["SPDK_SHM_ID"] = str(manifest["shm_id"] + 1)
ckpt = DirectCheckpoint(nvme_addr=manifest["pci"], npu_device_id=manifest["npu"],
                        pipeline_depth=4, requested_chunk_size=4 * 1024**2,
                        spdk_shm_id=manifest["shm_id"] + 1)
try:
 frames = [ckpt.read_host_frame(item["offset"], item["size"])
           for item in manifest["frames"]]
 for frame in frames: unpack_s2_replacement_frame(frame)
 full = {"backbone.blocks.0.weight": np.zeros(33, dtype=np.float32),
         "backbone.blocks.1.weight": np.ones(19, dtype=np.float16),
         "backbone.layernorm.bias": np.zeros(5, dtype=np.float32)}
 oracle = S2DeltaOracle(full, block_size=8, small_threshold=4)
 recovered = oracle.recover(full, frames)
 corrupt_error = None
 try:
  unpack_s2_replacement_frame(ckpt.read_host_frame(manifest["corrupt_offset"], manifest["frames"][-1]["size"]))
 except ValueError as error:
  corrupt_error = str(error)
 print(json.dumps({"generation": recovered["generation"],
                   "last_step": recovered["last_step"],
                   "digest": digest(recovered["state"]),
                   "corruption_rejected": corrupt_error is not None,
                   "corruption_error": corrupt_error}, sort_keys=True))
finally:
 ckpt.cleanup()
'''
    completed = subprocess.run(
        [sys.executable, "-c", child, str(run_dir / "chain_manifest.json"),
         str(REPO_ROOT), str(REPO_ROOT / "python")],
        capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"restart child failed: {completed.stderr}")
    child_lines = [line for line in completed.stdout.splitlines()
                   if line.strip().startswith("{")]
    if not child_lines:
        raise RuntimeError("restart child emitted no JSON: "
                           f"stdout={completed.stdout!r} stderr={completed.stderr!r}")
    child_result = json.loads(child_lines[-1])
    passed = (child_result["generation"] == args.steps and
              child_result["last_step"] == args.steps and
              child_result["digest"] == manifest["expected_digest"] and
              child_result["corruption_rejected"])
    sample = {"run_id": writer.run_id,
              "request_id": writer.run_id + "/restart_0001",
              "checkpoint_id": "raw_s2_chain_restart",
              "status": "pass" if passed else "fail",
              "child": child_result,
              "events": [{"name": "restart_replay_complete", "monotonic_ns": __import__("time").monotonic_ns()}],
              "timeline_us": {"restart_replay": 0}}
    writer.add_sample(sample)
    result = writer.finalize({"steps": args.steps, "generation": child_result["generation"],
                              "byte_exact": child_result["digest"] == manifest["expected_digest"],
                              "corruption_rejected": child_result["corruption_rejected"]},
                             status="pass" if passed else "fail")
    print(json.dumps({"status": result["status"], "run_id": writer.run_id,
                      "summary": result["summary"]}, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
