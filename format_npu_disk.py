import ctypes
import json
import struct
import argparse
import sys
import os

# ============================================================
# 裸盘物理布局常量 (必须与 direct_checkpoint.py 绝对对齐)
# ============================================================
BLOCK_SIZE = 4096
SUPERBLOCK_LBA = 0
META_SLOT_A_LBA = 1
META_SLOT_B_LBA = 101
META_SLOT_BLOCKS = 100       # 每个槽位占用约 400KB
MAGIC_NUMBER = b"NPUNVME1"

# ============================================================
# 绑定 C 接口
# ============================================================
LIB_PATH = os.environ.get("NPU_NVME_LIB", "./out/lib/libnpu_nvme.so")

try:
    lib = ctypes.CDLL(LIB_PATH)
    lib.npu_nvme_init.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_int, 
        ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_char_p
    ]
    lib.npu_nvme_init.restype = ctypes.c_int

    lib.npu_nvme_cleanup.argtypes = [ctypes.c_void_p]
    
    lib.npu_nvme_get_total_blocks.argtypes = [ctypes.c_void_p]
    lib.npu_nvme_get_total_blocks.restype = ctypes.c_uint64

    lib.npu_nvme_sync_meta_io.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p
    ]
    lib.npu_nvme_sync_meta_io.restype = ctypes.c_int
except OSError as e:
    print(f"[Error] Failed to load C library at {LIB_PATH}: {e}")
    sys.exit(1)

# ============================================================
# 核心格式化逻辑
# ============================================================
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
    # Pipeline depth 设为 1 即可，格式化不需要极速并发
    ret = lib.npu_nvme_init(ctypes.byref(ctx), pci_addr.encode('utf-8'), npu_id, 1, 1, False, b".")
    if ret != 0:
        print("[Error] SPDK initialization failed. Check PCI address and hugepages.")
        sys.exit(1)

    try:
        total_bytes = lib.npu_nvme_get_total_blocks(ctx)
        total_4k_blocks = total_bytes // BLOCK_SIZE
        capacity_gb = total_bytes / (1024**3)
        print(f"[2/4] Device connected. Total capacity: {capacity_gb:.2f} GB ({total_4k_blocks} 4K-blocks)")

        # 准备空的元数据 JSON
        empty_meta = {"checkpoints": {}}
        meta_json = json.dumps(empty_meta).encode('utf-8')
        meta_buf = ctypes.create_string_buffer(meta_json, META_SLOT_BLOCKS * BLOCK_SIZE)

        print("[3/4] Wiping Metadata Slots (A and B)...")
        ret_a = lib.npu_nvme_sync_meta_io(ctx, META_SLOT_A_LBA, META_SLOT_BLOCKS, 0, ctypes.c_void_p(ctypes.addressof(meta_buf)))
        ret_b = lib.npu_nvme_sync_meta_io(ctx, META_SLOT_B_LBA, META_SLOT_BLOCKS, 0, ctypes.c_void_p(ctypes.addressof(meta_buf)))
        if ret_a != 0 or ret_b != 0:
            raise RuntimeError("Failed to wipe Metadata Slots.")

        print("[4/4] Writing Superblock (Magic Number)...")
        sb_buf = ctypes.create_string_buffer(BLOCK_SIZE)
        active_slot = 0       
        stack_start_lba = 0   
        
        # 写入的是 total_4k_blocks，而不是原始的硬件扇区数！
        struct.pack_into("<8s I Q Q", sb_buf, 0, MAGIC_NUMBER, active_slot, total_4k_blocks, stack_start_lba)
        
        ret_sb = lib.npu_nvme_sync_meta_io(ctx, SUPERBLOCK_LBA, 1, 0, ctypes.c_void_p(ctypes.addressof(sb_buf)))
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