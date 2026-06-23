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
import os

# -- Disk layout constants (byte-addressed, must match direct_checkpoint.py) --
BLOCK_SIZE = 4096
SUPERBLOCK_OFFSET  = 0
META_SLOT_A_OFFSET = 4096
META_SLOT_B_OFFSET = 4096 + 400 * 1024
META_SLOT_BYTES    = 400 * 1024
MAGIC_NUMBER       = b"NPUNVME1"

# -- C interface (reuse direct_checkpoint bindings) --
from direct_checkpoint import lib

# -- Core formatting logic --
def format_disk(pci_addr, npu_id=0):
    print(f"\n{'='*60}")
    print(f"!!! WARNING: NPUNVME DISK FORMAT UTILITY !!!")
    print(f"{'='*60}")
    print(f"Target NVMe Device : {pci_addr}")
    print(f"NPU Device ID      : {npu_id}")
    print("\nThis operation will OVERWRITE the Superblock and Metadata slots.")
    print("All previously saved Checkpoints on this disk will be rendered UNREADABLE.")

    confirm = input("\nType 'YES' in all caps to proceed: ")
    if confirm != "YES":
        print("Format cancelled by user.")
        sys.exit(0)

    print("\n[1/4] Initializing SPDK and connecting to NVMe...")
    ctx = ctypes.c_void_p()
    # Pipeline depth = 1 is sufficient for formatting (no bulk transfer needed).
    ret = lib.npu_nvme_init(ctypes.byref(ctx), pci_addr.encode('utf-8'), npu_id,
                             1, BLOCK_SIZE, False, b".")
    if ret != 0:
        print("[Error] SPDK initialization failed. Check PCI address and hugepages.")
        sys.exit(1)

    try:
        total_bytes = lib.npu_nvme_get_total_blocks(ctx)
        capacity_gb = total_bytes / (1024**3)
        print(f"[2/4] Device connected. Total capacity: {capacity_gb:.2f} GB "
              f"({total_bytes // BLOCK_SIZE} 4K-blocks)")

        # Write empty metadata JSON to both slots
        empty_meta = {"checkpoints": {}}
        meta_json = json.dumps(empty_meta).encode('utf-8')
        meta_buf = ctypes.create_string_buffer(meta_json, META_SLOT_BYTES)

        print("[3/4] Wiping Metadata Slots (A and B)...")
        ret_a = lib.npu_nvme_sync_meta_io(ctx, META_SLOT_A_OFFSET, META_SLOT_BYTES,
                                           0, ctypes.c_void_p(ctypes.addressof(meta_buf)))
        ret_b = lib.npu_nvme_sync_meta_io(ctx, META_SLOT_B_OFFSET, META_SLOT_BYTES,
                                           0, ctypes.c_void_p(ctypes.addressof(meta_buf)))
        if ret_a != 0 or ret_b != 0:
            raise RuntimeError("Failed to wipe Metadata Slots.")

        print("[4/4] Writing Superblock (Magic Number)...")
        sb_buf = ctypes.create_string_buffer(BLOCK_SIZE)
        active_slot = 0
        # stack_start_bytes = 0 signals runtime-calculation in _mount_filesystem()
        stack_start_bytes = 0

        # Superblock: magic(8s) + slot(I) + capacity_bytes(Q) + stack_start_bytes(Q)
        struct.pack_into("<8s I Q Q", sb_buf, 0,
                         MAGIC_NUMBER, active_slot, total_bytes, stack_start_bytes)

        ret_sb = lib.npu_nvme_sync_meta_io(ctx, SUPERBLOCK_OFFSET, BLOCK_SIZE,
                                            0, ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if ret_sb != 0:
            raise RuntimeError("Failed to write Superblock.")

        print("Flushing NVMe cache to NAND... Please wait...")
        import time
        time.sleep(2)

        print(f"\n{'='*60}")
        print(f"[SUCCESS] NVMe disk {pci_addr} successfully formatted for NPUNVME!")

    except Exception as e:
        print(f"\n[Fatal Error] Format failed: {e}")
    finally:
        lib.npu_nvme_cleanup(ctx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPUNVME Disk Formatting Tool")
    parser.add_argument("--pci_addr", type=str, default="0000:83:00.0", help="PCI address of the NVMe SSD")
    parser.add_argument("--npu_id", type=int, default=0, help="NPU Device ID for SPDK init")
    
    args = parser.parse_args()
    format_disk(args.pci_addr, args.npu_id)