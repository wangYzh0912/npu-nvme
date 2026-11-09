#!/usr/bin/env python3
"""G2 metadata durability/fallback gate on 83.0.0.

Corrupts one metadata replica and the superblock in turn, verifies deterministic
failure/fallback, then restores the exact original bytes and remounts.  The
test never touches 84.0.0 and leaves 83.0.0 in its original state.
"""

import argparse
import ctypes
import json
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from c_bindings import NPUNVMEContext, lib  # noqa: E402
from direct_checkpoint import (  # noqa: E402
    META_SLOT_A_OFFSET,
    META_SLOT_B_OFFSET,
    META_SLOT_BYTES,
    SUPERBLOCK_OFFSET,
    DirectCheckpoint,
)


def read_region(ckpt, offset, size):
    buffer = ctypes.create_string_buffer(size)
    rc = lib.npu_nvme_sync_meta_io(
        ckpt.ctx, offset, size, 1, ctypes.byref(buffer))
    if rc != 0:
        raise RuntimeError(f"metadata read failed at {offset}: {rc}")
    return buffer.raw[:size]


def write_region(ckpt, offset, payload):
    buffer = ctypes.create_string_buffer(bytes(payload), len(payload))
    rc = lib.npu_nvme_sync_meta_io(
        ckpt.ctx, offset, len(payload), 0, ctypes.byref(buffer))
    if rc != 0:
        raise RuntimeError(f"metadata write failed at {offset}: {rc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=1)
    parser.add_argument("--shm-id", type=int, default=83)
    parser.add_argument("--output", default=os.path.join(
        REPO_ROOT, "experiments", "output", "gates", "g2_metadata.json"))
    args = parser.parse_args()

    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu, pipeline_depth=4,
        requested_chunk_size=4 * 1024 * 1024, rank_id=0, world_size=1,
        keep_last_n=3, slot_size_gb=10, spdk_shm_id=args.shm_id)
    try:
        original_sb = read_region(ckpt, SUPERBLOCK_OFFSET, 4096)
        original_a = read_region(ckpt, META_SLOT_A_OFFSET, META_SLOT_BYTES)
        original_b = read_region(ckpt, META_SLOT_B_OFFSET, META_SLOT_BYTES)
        before_generation = ckpt.metadata_generation
        active_slot = ckpt.active_meta_slot
        active_offset = META_SLOT_A_OFFSET if active_slot == 0 else META_SLOT_B_OFFSET
        inactive_offset = META_SLOT_B_OFFSET if active_slot == 0 else META_SLOT_A_OFFSET
        active_original = original_a if active_slot == 0 else original_b
        inactive_original = original_b if active_slot == 0 else original_a

        corrupted_active = bytearray(active_original)
        corrupted_active[40] ^= 0x01  # first metadata JSON byte, CRC must fail
        write_region(ckpt, active_offset, corrupted_active)
        ckpt._mount_filesystem()
        fallback_generation = ckpt.metadata_generation
        if fallback_generation >= before_generation:
            raise AssertionError(
                f"corrupted active replica was not rejected: {fallback_generation}")
        if "step_1" not in ckpt.meta_dict.get("checkpoints", {}):
            raise AssertionError("fallback replica lost the FULL checkpoint index")
        write_region(ckpt, active_offset, active_original)
        ckpt._mount_filesystem()
        if ckpt.metadata_generation != before_generation:
            raise AssertionError("restored metadata did not remount at latest generation")

        corrupted_sb = bytearray(original_sb)
        corrupted_sb[32] ^= 0x01
        write_region(ckpt, SUPERBLOCK_OFFSET, corrupted_sb)
        try:
            ckpt._mount_filesystem()
        except RuntimeError as error:
            if "Superblock verification failed" not in str(error):
                raise AssertionError(f"unexpected superblock error: {error}")
        else:
            raise AssertionError("corrupted superblock was accepted")
        write_region(ckpt, SUPERBLOCK_OFFSET, original_sb)
        ckpt._mount_filesystem()

        result = {
            "gate": "G2",
            "pci": args.pci,
            "active_slot": active_slot,
            "latest_generation": before_generation,
            "fallback_generation": fallback_generation,
            "metadata_repair": True,
            "superblock_corruption_rejected": True,
            "status": "pass",
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print(f"[G2] PASS A/B fallback generation={fallback_generation}; "
              "superblock corruption rejected and repaired", flush=True)
    finally:
        ckpt.cleanup()


if __name__ == "__main__":
    main()
