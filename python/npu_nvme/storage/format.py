"""Checksummed V2 records; encoding semantics preserved from disk_layout."""
import binascii
import json
import struct
import zlib
from .layout import (DiskLayout, MAGIC_NUMBER, FORMAT_VERSION, SUPERBLOCK_BYTES,
    META_SLOT_A_OFFSET, META_SLOT_B_OFFSET, META_SLOT_BYTES, METADATA_MAGIC,
    METADATA_VERSION, METADATA_VERSION_LEGACY, METADATA_FLAG_ZLIB)

_SUPERBLOCK_FMT = "<8sIIQQQQQQQQQQQQI"
_SUPERBLOCK_SIZE = struct.calcsize(_SUPERBLOCK_FMT)
_METADATA_HEADER_FMT = "<8sIIQQII"
_METADATA_HEADER_SIZE = struct.calcsize(_METADATA_HEADER_FMT)


def pack_superblock(layout: DiskLayout) -> bytes:
    """Serialize a V2 superblock with CRC over its fixed header."""
    layout.validate()
    raw = struct.pack(
        _SUPERBLOCK_FMT,
        MAGIC_NUMBER, FORMAT_VERSION, 0,
        layout.generation, layout.active_meta_slot,
        layout.total_bytes, layout.full_base, layout.full_slot_bytes,
        layout.full_slot_count, layout.delta_base, layout.delta_slot_bytes,
        layout.delta_slot_count, META_SLOT_A_OFFSET, META_SLOT_B_OFFSET,
        META_SLOT_BYTES, 0,
    )
    crc = binascii.crc32(raw[:-4]) & 0xFFFFFFFF
    return (raw[:-4] + struct.pack("<I", crc)).ljust(SUPERBLOCK_BYTES, b"\0")


def unpack_superblock(raw: bytes) -> DiskLayout:
    if len(raw) < _SUPERBLOCK_SIZE:
        raise ValueError("superblock is truncated")
    fields = struct.unpack(_SUPERBLOCK_FMT, raw[:_SUPERBLOCK_SIZE])
    magic, version, _reserved = fields[:3]
    if magic != MAGIC_NUMBER:
        raise ValueError("invalid superblock magic")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported disk format version: {version}")
    stored_crc = fields[-1]
    actual_crc = binascii.crc32(raw[:_SUPERBLOCK_SIZE - 4]) & 0xFFFFFFFF
    if stored_crc != actual_crc:
        raise ValueError("superblock CRC mismatch")
    if fields[12] != META_SLOT_A_OFFSET or fields[13] != META_SLOT_B_OFFSET:
        raise ValueError("metadata slot geometry mismatch")
    if fields[14] != META_SLOT_BYTES:
        raise ValueError("metadata slot size mismatch")
    layout = DiskLayout(
        total_bytes=fields[5], full_base=fields[6],
        full_slot_bytes=fields[7], full_slot_count=fields[8],
        delta_base=fields[9], delta_slot_bytes=fields[10],
        delta_slot_count=fields[11], generation=fields[3],
        active_meta_slot=fields[4],
    )
    layout.validate()
    return layout


def pack_metadata(payload: dict, generation: int) -> bytes:
    """Wrap compressed JSON metadata in a generation-tagged CRC envelope.

    Version 1 metadata remains readable. Compression is required for large
    real-model manifests, whose parameter names otherwise exceed the fixed
    400 KiB A/B metadata slots.
    """
    if generation < 0:
        raise ValueError("metadata generation must be non-negative")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    body_bytes = zlib.compress(body.encode("utf-8"), level=6)
    if _METADATA_HEADER_SIZE + len(body_bytes) > META_SLOT_BYTES:
        raise ValueError("metadata payload exceeds metadata slot")
    crc = binascii.crc32(body_bytes) & 0xFFFFFFFF
    header = struct.pack(_METADATA_HEADER_FMT, METADATA_MAGIC,
                         METADATA_VERSION, 0, generation,
                         len(body_bytes),
                         crc, METADATA_FLAG_ZLIB)
    return (header + body_bytes).ljust(META_SLOT_BYTES, b"\0")


def unpack_metadata(raw: bytes):
    if len(raw) < _METADATA_HEADER_SIZE:
        raise ValueError("metadata slot is truncated")
    magic, version, _reserved, generation, length, stored_crc, flags = \
        struct.unpack(_METADATA_HEADER_FMT, raw[:_METADATA_HEADER_SIZE])
    if magic != METADATA_MAGIC or version not in (METADATA_VERSION_LEGACY,
                                                  METADATA_VERSION):
        raise ValueError("unsupported metadata envelope")
    end = _METADATA_HEADER_SIZE + length
    if end > len(raw):
        raise ValueError("metadata payload exceeds slot")
    body = raw[_METADATA_HEADER_SIZE:end]
    if binascii.crc32(body) & 0xFFFFFFFF != stored_crc:
        raise ValueError("metadata CRC mismatch")
    if version == METADATA_VERSION and (flags & METADATA_FLAG_ZLIB):
        try:
            body = zlib.decompress(body)
        except zlib.error as error:
            raise ValueError("metadata zlib decompression failed") from error
    return generation, json.loads(body.decode("utf-8"))
