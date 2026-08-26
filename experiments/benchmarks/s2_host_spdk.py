#!/usr/bin/env python3
"""I5 Host-SPDK byte-preserving S2 frame loopback."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from direct_checkpoint import DirectCheckpoint  # noqa: E402
from s2_delta import S2DeltaOracle  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=950)
    parser.add_argument("--offset", type=int, default=64 * 1024**3)
    args = parser.parse_args()

    initial = {
        "backbone.blocks.0.weight": np.arange(33, dtype=np.float32),
        "backbone.layernorm.bias": np.array([1, 2, 3], dtype=np.float16),
    }
    current = {name: value.copy() for name, value in initial.items()}
    current["backbone.blocks.0.weight"][32] = -17.25
    oracle = S2DeltaOracle(initial, block_size=8, small_threshold=4)
    oracle.set_current(current)
    frame = oracle.observe(1)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu, pipeline_depth=4,
        requested_chunk_size=4 * 1024 * 1024, spdk_shm_id=args.shm_id)
    try:
        write_stats = ckpt.write_host_frame(frame, args.offset)
        actual = ckpt.read_host_frame(args.offset, len(frame))
        if actual != frame:
            raise AssertionError("Host-SPDK frame bytes changed")
        decoded = oracle.ack(actual)
        recovered = oracle.recover(initial, [actual])
        if any(not np.array_equal(recovered["state"][name], current[name])
               for name in current):
            raise AssertionError("Host-SPDK S2 recovery mismatch")
        result = {"status": "pass", "frame_bytes": len(frame),
                  "write": write_stats, "ack": decoded,
                  "recovery_generation": recovered["generation"],
                  "offset": args.offset}
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        ckpt.cleanup()


if __name__ == "__main__":
    main()
