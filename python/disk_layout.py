"""NPU-NVMe raw block-device layout constants.

All byte offsets are absolute (addressed from sector 0 of the NVMe device).
Shared by direct_checkpoint.py, format_npu_disk.py, inspect_npu_disk.py,
and export_model.py.
"""

# -- Superblock & metadata area --
SUPERBLOCK_OFFSET      = 0
SUPERBLOCK_HEADER_BYTES = 28  # "<8s I Q Q" = magic(8) + slot(4) + capacity(8) + stack(8)
META_SLOT_A_OFFSET     = 4096
META_SLOT_B_OFFSET     = 4096 + 400 * 1024
META_SLOT_BYTES        = 400 * 1024
MAGIC_NUMBER           = b"NPUNVME1"

# -- Miscellaneous --
UINT32_BYTES           = 4
BLOCK_SIZE             = 4096   # NVMe block size (4 KiB)

# -- Delta frame binary protocol --
DELTA_MAGIC      = 0x414C5444   # "DLTA"
FRAME_HEADER_SIZE = 4096

# -- Default chunk size for bulk data transfer --
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB

# -- Derived constants --
HEAP_START_OFFSET = META_SLOT_B_OFFSET + META_SLOT_BYTES
