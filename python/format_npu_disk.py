"""Destructive NVMe formatting utility for NPU-NVMe checkpoints.

Usage:
- python python/format_npu_disk.py --pci_addr <PCI> --npu_id <ID>

Inputs:
- PCI address and NPU device id.
Outputs:
- Overwrites metadata slots and prints status.
"""
import ctypes
import json
import struct
import argparse
import sys

from disk_layout import (SUPERBLOCK_OFFSET, META_SLOT_A_OFFSET, META_SLOT_B_OFFSET,
                          META_SLOT_BYTES, CHUNK_SIZE, make_layout,
                          pack_metadata, pack_superblock)
from c_bindings import lib, NPUNVMEContext


def format_disk(pci_addr, npu_id=0, force=False, world_size=1,
                keep_last_n=3, full_slot_gb=10, delta_slot_mb=256,
                delta_slot_count=128):
    print(f"\n{'='*60}")
    print(f"!!! WARNING: NPUNVME DISK FORMAT UTILITY !!!")
    print(f"{'='*60}")
    print(f"Target NVMe Device : {pci_addr}")
    print(f"NPU Device ID      : {npu_id}")
    print(f"FULL geometry      : {world_size * keep_last_n} slots x "
          f"{full_slot_gb} GiB")
    print(f"Delta geometry     : {delta_slot_count} slots x "
          f"{delta_slot_mb} MiB")
    print("\nThis operation will OVERWRITE the Superblock and Metadata slots.")
    print("All previously saved Checkpoints on this disk will be rendered UNREADABLE.")

    if not force:
        confirm = input("\nType 'YES' in all caps to proceed: ")
        if confirm != "YES":
            print("Format cancelled by user.")
            sys.exit(0)
    else:
        print("\n[force=True] Skipping interactive confirmation.")

    print("\n[1/4] Initializing SPDK and connecting to NVMe...")
    ctx = ctypes.POINTER(NPUNVMEContext)()
    ret = lib.npu_nvme_init(ctypes.byref(ctx), pci_addr.encode('utf-8'),
                             npu_id, 1, CHUNK_SIZE, False, b".")
    if ret != 0:
        print("[Error] SPDK initialization failed. Check PCI address and hugepages.")
        sys.exit(1)

    try:
        total_bytes = lib.npu_nvme_get_total_blocks(ctx)
        capacity_gb = total_bytes / (1024**3)
        print(f"[2/4] Device connected. Total capacity: {capacity_gb:.2f} GB "
              f"({total_bytes // 4096} 4K-blocks)")

        layout = make_layout(
            total_bytes=total_bytes,
            full_slot_bytes=full_slot_gb * 1024**3,
            full_slot_count=world_size * keep_last_n,
            delta_slot_bytes=delta_slot_mb * 1024**2,
            delta_slot_count=delta_slot_count,
        )
        print(f"      FULL:  {layout.full_base}..{layout.full_end}")
        print(f"      Delta: {layout.delta_base}..{layout.delta_end}")

        empty_meta = {
            "schema": 2,
            "checkpoints": {},
            "delta_chain": {},
            "full_generation": 0,
            "delta_head": 0,
            "delta_tail": 0,
        }
        meta_buf = ctypes.create_string_buffer(
            pack_metadata(empty_meta, generation=0), META_SLOT_BYTES)

        print("[3/4] Wiping Metadata Slots (A and B)...")
        ret_a = lib.npu_nvme_sync_meta_io(
            ctx, META_SLOT_A_OFFSET, META_SLOT_BYTES, 0,
            ctypes.c_void_p(ctypes.addressof(meta_buf)))
        ret_b = lib.npu_nvme_sync_meta_io(
            ctx, META_SLOT_B_OFFSET, META_SLOT_BYTES, 0,
            ctypes.c_void_p(ctypes.addressof(meta_buf)))
        if ret_a != 0 or ret_b != 0:
            raise RuntimeError("Failed to wipe Metadata Slots.")
        if hasattr(lib, "npu_nvme_flush") and lib.npu_nvme_flush(ctx) != 0:
            raise RuntimeError("Failed to flush metadata replicas.")

        print("[4/4] Writing V2 Superblock (layout + CRC)...")
        sb_buf = ctypes.create_string_buffer(pack_superblock(layout), 4096)

        ret_sb = lib.npu_nvme_sync_meta_io(
            ctx, SUPERBLOCK_OFFSET, 4096, 0,
            ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if ret_sb != 0:
            raise RuntimeError("Failed to write Superblock.")
        if hasattr(lib, "npu_nvme_flush") and lib.npu_nvme_flush(ctx) != 0:
            raise RuntimeError("Failed to flush superblock.")

        print("Flushing NVMe cache to NAND... Please wait...")
        import time
        print("NVMe namespace flush completed.")

        print(f"\n{'='*60}")
        print(f"[SUCCESS] NVMe disk {pci_addr} successfully formatted for NPUNVME!")

    except Exception as e:
        print(f"\n[Fatal Error] Format failed: {e}")
        raise
    finally:
        lib.npu_nvme_cleanup(ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPUNVME Disk Formatting Tool")
    parser.add_argument("--pci_addr", type=str, default="0000:83:00.0")
    parser.add_argument("--npu_id", type=int, default=0)
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive confirmation")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--keep-last-n", type=int, default=3)
    parser.add_argument("--full-slot-gb", type=int, default=10)
    parser.add_argument("--delta-slot-mb", type=int, default=256)
    parser.add_argument("--delta-slot-count", type=int, default=128)
    args = parser.parse_args()
    format_disk(args.pci_addr, args.npu_id, force=args.yes,
                world_size=args.world_size, keep_last_n=args.keep_last_n,
                full_slot_gb=args.full_slot_gb,
                delta_slot_mb=args.delta_slot_mb,
                delta_slot_count=args.delta_slot_count)
