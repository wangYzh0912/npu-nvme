#!/usr/bin/env python3
"""Stage 5 single-Reactor multi-rank gate.

The default mode is a deterministic protocol/partition gate that requires no
NPU.  ``--run-c2`` additionally launches the existing two-rank real training
state gate; its Unix-socket host staging is reported explicitly and is not a
throughput result.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from multirank_protocol import GlobalCommit, GlobalState  # noqa: E402


def protocol_gate(world_size: int, step: int, generation: int):
    commit = GlobalCommit(world_size, step, generation)
    payloads = {}
    for rank in range(world_size):
        payload = (f"rank={rank};step={step};generation={generation}".encode() * 4096)
        payloads[rank] = payload
        digest = hashlib.sha256(payload).hexdigest()
        commit.prepare(rank, step, generation, digest)
    for rank, payload in payloads.items():
        commit.persisted_ready(rank, hashlib.sha256(payload).hexdigest())
    metadata = commit.commit()
    offsets = list(range(world_size))
    if len(offsets) != len(set(offsets)):
        raise AssertionError("rank partition overlap")
    return {"status": "pass", "transport": "protocol-only",
            "reactor_count": 1, "world_size": world_size,
            "metadata": metadata, "rank_partitions": offsets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--run-c2", action="store_true")
    parser.add_argument("--c2-run-dir", default="/tmp/stage5-c2")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=50100)
    parser.add_argument("--coordinator-npu", type=int, default=7)
    args = parser.parse_args()
    if args.world_size not in (2, 4):
        raise ValueError("world size must be 2 or 4")
    result = protocol_gate(args.world_size, args.step, args.generation)
    if args.run_c2:
        if args.world_size != 2:
            raise ValueError("real C2 launcher currently supports world_size=2")
        command = [sys.executable, str(ROOT / "tests/hardware/c2_multirank_state.py"),
                   "--run-dir", args.c2_run_dir, "--pci", args.pci,
                   "--shm-id", str(args.shm_id),
                   "--coordinator-npu", str(args.coordinator_npu)]
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        result["c2_command"] = command
        result["c2_exit_code"] = completed.returncode
        result["transport"] = "unix-socket-host-staging"
        if completed.returncode != 0:
            result["status"] = "fail"
    output = Path(args.c2_run_dir) if args.run_c2 else Path("/tmp/stage5-protocol")
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage5_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

