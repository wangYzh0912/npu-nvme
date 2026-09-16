"""NPU-NVMe raw layout and checksummed V2 metadata protocol.

All offsets are absolute byte offsets from sector zero.  FULL and Delta
allocators must use :func:`make_layout`; keeping the calculation here avoids
the historical double-allocation from two independent tail allocators.
"""

from dataclasses import dataclass


# -- Superblock and metadata area -------------------------------------------
SUPERBLOCK_OFFSET = 0
SUPERBLOCK_BYTES = 4096
SUPERBLOCK_HEADER_BYTES = 28  # legacy header size, retained for readers
META_SLOT_A_OFFSET = 4096
META_SLOT_B_OFFSET = 4096 + 400 * 1024
META_SLOT_BYTES = 400 * 1024
MAGIC_NUMBER = b"NPUNVME1"
FORMAT_VERSION = 2
METADATA_MAGIC = b"NVMETA02"
METADATA_VERSION = 2
METADATA_VERSION_LEGACY = 1
METADATA_FLAG_ZLIB = 1

# -- Miscellaneous ----------------------------------------------------------
UINT32_BYTES = 4
BLOCK_SIZE = 4096
DATA_START_OFFSET = 1 * 1024 * 1024

# -- Delta frame binary protocol --------------------------------------------
DELTA_MAGIC = 0x414C5444   # "DLTA"
FRAME_HEADER_SIZE = 4096

# -- Default transfer and layout values -------------------------------------
CHUNK_SIZE = 4 * 1024 * 1024
HEAP_START_OFFSET = META_SLOT_B_OFFSET + META_SLOT_BYTES

def align_up(value: int, alignment: int = BLOCK_SIZE) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("invalid alignment arguments")
    return (value + alignment - 1) & ~(alignment - 1)


def align_down(value: int, alignment: int = BLOCK_SIZE) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("invalid alignment arguments")
    return value & ~(alignment - 1)


@dataclass(frozen=True)
class DiskLayout:
    """Validated V2 partition table."""

    total_bytes: int
    full_base: int
    full_slot_bytes: int
    full_slot_count: int
    delta_base: int
    delta_slot_bytes: int
    delta_slot_count: int
    generation: int = 0
    active_meta_slot: int = 0

    @property
    def full_bytes(self) -> int:
        return self.full_slot_bytes * self.full_slot_count

    @property
    def delta_bytes(self) -> int:
        return self.delta_slot_bytes * self.delta_slot_count

    @property
    def full_end(self) -> int:
        return self.full_base + self.full_bytes

    @property
    def delta_end(self) -> int:
        return self.delta_base + self.delta_bytes

    def validate(self) -> None:
        values = (self.total_bytes, self.full_base, self.full_slot_bytes,
                  self.full_slot_count, self.delta_base,
                  self.delta_slot_bytes, self.delta_slot_count)
        if any(value <= 0 for value in values):
            raise ValueError("layout values must be positive")
        if self.active_meta_slot not in (0, 1):
            raise ValueError("active metadata slot must be 0 or 1")
        for value in (self.full_base, self.full_slot_bytes,
                      self.delta_base, self.delta_slot_bytes):
            if value % BLOCK_SIZE:
                raise ValueError("layout values must be 4 KiB aligned")
        if self.full_base < DATA_START_OFFSET:
            raise ValueError("FULL region overlaps metadata area")
        if self.full_end > self.delta_base:
            raise ValueError("FULL and Delta regions overlap")
        if self.delta_end > self.total_bytes:
            raise ValueError("Delta region exceeds device capacity")

    def full_slot_offset(self, rank_id: int, step: int,
                         keep_last_n: int) -> int:
        if rank_id < 0 or keep_last_n <= 0:
            raise ValueError("invalid rank or keep_last_n")
        slot = rank_id * keep_last_n + (step % keep_last_n)
        if slot >= self.full_slot_count:
            raise ValueError("rank/slot selection exceeds FULL partition")
        return self.full_base + slot * self.full_slot_bytes

    def delta_slot_offset(self, slot_idx: int) -> int:
        if slot_idx < 0 or slot_idx >= self.delta_slot_count:
            raise ValueError("Delta slot index out of range")
        return self.delta_base + slot_idx * self.delta_slot_bytes


def make_layout(total_bytes: int, full_slot_bytes: int,
                full_slot_count: int, delta_slot_bytes: int,
                delta_slot_count: int, generation: int = 0,
                active_meta_slot: int = 0) -> DiskLayout:
    """Construct and validate the one true FULL/Delta partition table."""
    layout = DiskLayout(
        total_bytes=total_bytes,
        full_base=align_up(DATA_START_OFFSET),
        full_slot_bytes=align_up(full_slot_bytes),
        full_slot_count=full_slot_count,
        delta_base=align_down(total_bytes - align_up(delta_slot_bytes)
                              * delta_slot_count),
        delta_slot_bytes=align_up(delta_slot_bytes),
        delta_slot_count=delta_slot_count,
        generation=generation,
        active_meta_slot=active_meta_slot,
    )
    layout.validate()
    return layout
