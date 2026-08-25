"""Inspect NVMe metadata slots and list saved checkpoints.

Usage:
- python python/inspect_npu_disk.py --pci_addr <PCI> --npu_id <ID>

Inputs:
- PCI address and NPU device id.
Outputs:
- Prints parsed metadata and checkpoint summary.
"""
import ctypes
import json
import argparse
import sys

from disk_layout import (SUPERBLOCK_OFFSET, META_SLOT_A_OFFSET, META_SLOT_B_OFFSET,
                          META_SLOT_BYTES, CHUNK_SIZE, unpack_metadata,
                          unpack_superblock)
from c_bindings import lib, NPUNVMEContext


def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0


def parse_metadata_slot(ctx, slot_name, offset_bytes):
    meta_buf = ctypes.create_string_buffer(META_SLOT_BYTES)
    rc = lib.npu_nvme_sync_meta_io(
        ctx, offset_bytes, META_SLOT_BYTES, 1,
        ctypes.c_void_p(ctypes.addressof(meta_buf)))

    if rc != 0:
        return {"status": "I/O Error", "checkpoints": {}}

    try:
        generation, data = unpack_metadata(meta_buf.raw)
        data["generation"] = generation
        data["status"] = "Valid JSON"
        return data
    except (ValueError, json.JSONDecodeError):
        return {"status": "Corrupted metadata envelope", "checkpoints": {}}


def inspect_disk(pci_addr, npu_id=0):
    print(f"\n{'='*70}")
    print(f"NPUNVME Disk Inspector")
    print(f"{'='*70}")
    print(f"Target Device : {pci_addr}")

    ctx = ctypes.POINTER(NPUNVMEContext)()
    ret = lib.npu_nvme_init(ctypes.byref(ctx), pci_addr.encode('utf-8'),
                             npu_id, 1, CHUNK_SIZE, False, b".")
    if ret != 0:
        print("[Error] SPDK initialization failed.")
        sys.exit(1)

    try:
        # -- Superblock --
        sb_buf = ctypes.create_string_buffer(4096)
        rc = lib.npu_nvme_sync_meta_io(
            ctx, SUPERBLOCK_OFFSET, 4096, 1,
            ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc != 0:
            raise RuntimeError("Failed to read Superblock.")

        try:
            layout = unpack_superblock(sb_buf.raw)
            is_formatted = True
        except ValueError:
            layout = None
            is_formatted = False

        print("\n[1] SUPERBLOCK (Offset 0)")
        print("-" * 50)
        print(f"  Magic Number    : {sb_buf.raw[:8]} "
              f"{'(OK)' if is_formatted else '(INVALID)'}")
        if not is_formatted:
            return

        active_slot = layout.active_meta_slot
        total_bytes = layout.total_bytes

        print(f"  Active Meta Slot: "
              f"{'Slot A' if active_slot == 0 else 'Slot B'} "
              f"(Pointer: {active_slot})")
        print(f"  Total Capacity  : {format_size(total_bytes)} ({total_bytes} Bytes)")
        print(f"  FULL Base       : {format_size(layout.full_base)}")
        print(f"  Delta Base      : {format_size(layout.delta_base)}")

        # -- Disk layout --
        meta_end_bytes = META_SLOT_B_OFFSET + META_SLOT_BYTES
        heap_size = layout.full_bytes
        stack_size = layout.delta_bytes

        print("\n[2] DISK LAYOUT (Macro View)")
        print("-" * 50)
        print(f"  [ 0        ~ 804 KB   ] Metadata Area "
              f"({format_size(meta_end_bytes)})")
        print(f"  FULL Area      : {format_size(heap_size)}")
        print(f"  Delta Area     : {format_size(stack_size)}")

        # -- Metadata slots --
        print("\n[3] METADATA SLOTS")
        print("-" * 50)

        slot_a_data = parse_metadata_slot(ctx, "A", META_SLOT_A_OFFSET)
        slot_b_data = parse_metadata_slot(ctx, "B", META_SLOT_B_OFFSET)

        def extract_step_num(key_str):
            try:
                return int(key_str.split('_')[-1])
            except ValueError:
                return 0

        for name, data, is_active in [
            ("Slot A", slot_a_data, active_slot == 0),
            ("Slot B", slot_b_data, active_slot == 1),
        ]:
            active_marker = "<-- [ACTIVE]" if is_active else ""
            print(f"\n* {name} {active_marker}")
            print(f"  Status: {data['status']}")

            ckpts = data.get("checkpoints", {})
            if not ckpts:
                print("  Checkpoints: None")
            else:
                print(f"  Checkpoints Found: {len(ckpts)}")
                sorted_steps = sorted(ckpts.items(),
                                      key=lambda x: extract_step_num(x[0]))

                for ckpt_name, info in sorted_steps:
                    c_type = info.get("type", "UNKNOWN")
                    ranks = info.get("world_size", 1)
                    tensors_count = len(info.get("params", {}))
                    ckpt_bytes = sum(
                        p.get("size", 0)
                        for p in info.get("params", {}).values())

                    if c_type == "COMPLETE":
                        print(f"    * [{ckpt_name}] Type: {c_type}, "
                              f"Tensors: {tensors_count}, "
                              f"Size: {format_size(ckpt_bytes)}")
                    else:
                        print(f"    - [{ckpt_name}] Type: {c_type}, "
                              f"Ranks: {ranks}, Tensors: {tensors_count}, "
                              f"Size: {format_size(ckpt_bytes)}")

        print(f"\n{'='*70}\n")

    except Exception as e:
        print(f"\n[Fatal Error] Inspection failed: {e}")
    finally:
        lib.npu_nvme_cleanup(ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPUNVME Disk Inspector Tool")
    parser.add_argument("--pci_addr", type=str, default="0000:83:00.0")
    parser.add_argument("--npu_id", type=int, default=0)
    args = parser.parse_args()
    inspect_disk(args.pci_addr, args.npu_id)
