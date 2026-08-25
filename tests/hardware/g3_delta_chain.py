#!/usr/bin/env python3
"""G3 R0 Delta-chain gate on the formatted 83.0.0 Delta ring.

The gate writes 100 logical steps with three self-described FP16 frames per
step (300 frames total), which wraps the 128-slot hardware ring more than
twice.  The restart phase reads the retained 128-frame window and compares it
to a deterministic CPU oracle.  The frame parser is also tested with a
corrupted payload; corruption must fail rather than silently apply.
"""

import argparse
import json
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, REPO_ROOT)

from delta_protocol import (apply_delta_patches, pack_lossless_delta_frame,
                            unpack_delta_frame)
from direct_checkpoint import DirectCheckpoint


LOGICAL_STEPS = 100
FRAMES_PER_STEP = 3
FRAME_COUNT = LOGICAL_STEPS * FRAMES_PER_STEP
FIRST_FRAME_STEP = 2
LAST_FRAME_STEP = FIRST_FRAME_STEP + FRAME_COUNT - 1
RING_SLOTS = 128
LATEST_FIRST_STEP = LAST_FRAME_STEP - RING_SLOTS + 1
PARAM_NAME = "g3.synthetic.weight"


def delta_for_frame(frame_index):
    return np.array([(frame_index % 7 - 3) * 0.125], dtype=np.float16)


def oracle_states():
    base = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    states = {1: {PARAM_NAME: base.copy()}}
    current = base.copy()
    for frame_index in range(FRAME_COUNT):
        current = current.copy()
        current[0] += np.float32(delta_for_frame(frame_index)[0])
        states[FIRST_FRAME_STEP + frame_index] = {PARAM_NAME: current.copy()}
    return states


def make_ckpt(args, run_dir):
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu, pipeline_depth=4,
        requested_chunk_size=4 * 1024 * 1024, rank_id=0, world_size=1,
        keep_last_n=3, slot_size_gb=10, spdk_shm_id=args.shm_id,
        profiling_dir=os.path.join(run_dir, "profiling"))
    ckpt._meta_pkl = os.path.join(run_dir, "checkpoint_meta.pkl")
    ckpt.delta_init(slot_size_mb=256, slot_count=128)
    return ckpt


def write_phase(args, run_dir):
    ckpt = make_ckpt(args, run_dir)
    try:
        full = ckpt.meta_dict.get("checkpoints", {}).get("step_1")
        if not full or full.get("type", "FULL") != "FULL":
            raise RuntimeError("G3 requires the G1 FULL checkpoint step_1")
        base_generation = int(full["generation"])

        # Start a clean logical ring while preserving the G1 FULL base.
        ckpt.meta_dict["delta_chain"] = {}
        ckpt.meta_dict["delta_head"] = 0
        ckpt.meta_dict["delta_tail"] = 0
        ckpt._delta_next_slot = 0
        ckpt._persist_metadata(ckpt.metadata_generation + 1)

        for frame_index in range(FRAME_COUNT):
            step = FIRST_FRAME_STEP + frame_index
            patch = [{
                "name": PARAM_NAME,
                "layer_id": 0,
                "block_idx": 0,
                "element_offset": 0,
                "fp16_data": delta_for_frame(frame_index),
            }]
            ckpt.delta_save_lossless(step, patch, [], base_generation=base_generation)
            if (frame_index + 1) % 50 == 0:
                print(f"[G3/write] frames={frame_index + 1}/{FRAME_COUNT}", flush=True)

        manifest = {
            "gate": "G3",
            "logical_steps": LOGICAL_STEPS,
            "frames_per_step": FRAMES_PER_STEP,
            "frame_count": FRAME_COUNT,
            "first_frame_step": FIRST_FRAME_STEP,
            "last_frame_step": LAST_FRAME_STEP,
            "latest_first_step": LATEST_FIRST_STEP,
            "ring_slots": RING_SLOTS,
            "base_generation": base_generation,
        }
        with open(os.path.join(run_dir, "g3_manifest.json"), "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        print(f"[G3/write] PASS wrote {FRAME_COUNT} frames; "
              f"head={ckpt.meta_dict['delta_head']} "
              f"tail={ckpt.meta_dict['delta_tail']} "
              f"retained={len(ckpt.meta_dict['delta_chain'])}", flush=True)
    finally:
        ckpt.cleanup()


def verify_phase(args, run_dir):
    with open(os.path.join(run_dir, "g3_manifest.json"), "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    states = oracle_states()
    ckpt = make_ckpt(args, run_dir)
    try:
        chain_records = ckpt.meta_dict.get("delta_chain", {})
        expected_keys = {
            f"step_{step}"
            for step in range(manifest["latest_first_step"], manifest["last_frame_step"] + 1)
        }
        if set(chain_records) != expected_keys:
            raise AssertionError(
                f"ring metadata window mismatch: {len(chain_records)} records")
        chain = ckpt.delta_load_chain(
            manifest["latest_first_step"] - 1, manifest["last_frame_step"])
        recovered = states[manifest["latest_first_step"] - 1]
        for step, blocks, smalls in chain:
            recovered = apply_delta_patches(recovered, blocks, smalls, block_size=16)
            if step not in states:
                raise AssertionError(f"unexpected step {step}")
        np.testing.assert_array_equal(
            recovered[PARAM_NAME], states[manifest["last_frame_step"]][PARAM_NAME])

        # CPU parser negative test: valid frame plus one byte mutation.
        good = pack_lossless_delta_frame(
            999, [{"name": PARAM_NAME, "fp16_data": np.array([1], dtype=np.float16)}],
            [], base_generation=manifest["base_generation"], generation=77)
        bad = bytearray(good)
        bad[-1] ^= 1
        try:
            unpack_delta_frame(bad)
        except ValueError as error:
            if "CRC" not in str(error):
                raise AssertionError(f"unexpected corruption error: {error}")
        else:
            raise AssertionError("corrupted Delta frame was accepted")

        print(f"[G3/verify] PASS restart window {len(chain)} frames; "
              "CPU oracle exact; corruption rejected", flush=True)
    finally:
        ckpt.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("orchestrate", "write", "verify"),
                        default="orchestrate")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=1)
    parser.add_argument("--shm-id", type=int, default=83)
    parser.add_argument("--run-dir", default=os.path.join(
        REPO_ROOT, "experiments", "output", "gates", "g3"))
    args = parser.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    if args.phase == "write":
        write_phase(args, run_dir)
        return
    if args.phase == "verify":
        verify_phase(args, run_dir)
        return

    write_cmd = [sys.executable, os.path.abspath(__file__), "--phase", "write",
                 "--pci", args.pci, "--npu", str(args.npu), "--shm-id", str(args.shm_id),
                 "--run-dir", run_dir]
    verify_cmd = [sys.executable, os.path.abspath(__file__), "--phase", "verify",
                  "--pci", args.pci, "--npu", str(args.npu), "--shm-id", str(args.shm_id),
                  "--run-dir", run_dir]
    import subprocess
    subprocess.run(write_cmd, cwd=run_dir, check=True)
    subprocess.run(verify_cmd, cwd=run_dir, check=True)
    print("[G3] PASS", flush=True)


if __name__ == "__main__":
    main()
