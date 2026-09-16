"""Experiment-only append layout for retained incremental frames.

This is deliberately not a recovery format. It gives the phase-one campaign a
bounded, checksummed definition of "saved" without changing the production D2
catalog or advertising crash recovery.
"""
from __future__ import annotations

import hashlib
import json
import struct


ALIGN = 4096
SUPER_BYTES = 4 << 20
SUPER_MAGIC = b"NPUICF1\0"
FRAME_MAGIC = b"ICFRAME1"
COMMIT_MAGIC = b"ICCOMMIT"
_SUPER = struct.Struct("<8sIQQ32sI")
_FRAME = struct.Struct("<8sIQQQQ32sI")
_COMMIT = struct.Struct("<8sIQQ32sI")


def align(value, boundary=ALIGN):
    if type(value) is not int or value < 0 or boundary <= 0 or boundary & (boundary - 1):
        raise ValueError("invalid alignment")
    return (value + boundary - 1) // boundary * boundary


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _page(header, document):
    raw = canonical(document)
    if len(header) + len(raw) > ALIGN:
        raise ValueError("metadata page exceeds 4 KiB")
    return header + raw + bytes(ALIGN - len(header) - len(raw))


def pack_superblock(*, namespace_bytes, config_sha256, campaign_id):
    if namespace_bytes <= SUPER_BYTES or len(config_sha256) != 64 or not campaign_id:
        raise ValueError("invalid superblock identity")
    document = {"campaign_id": campaign_id, "config_sha256": config_sha256,
                "namespace_bytes": namespace_bytes, "format": "incremental-feasibility-v1"}
    raw = canonical(document)
    header = _SUPER.pack(SUPER_MAGIC, 1, namespace_bytes, SUPER_BYTES,
                         hashlib.sha256(raw).digest(), len(raw))
    return _page(header, document) + bytes(SUPER_BYTES - ALIGN)


def unpack_superblock(raw):
    raw = bytes(raw)
    if len(raw) != SUPER_BYTES:
        raise ValueError("superblock size differs")
    magic, version, namespace, data_start, digest, size = _SUPER.unpack_from(raw)
    if magic != SUPER_MAGIC or version != 1 or data_start != SUPER_BYTES or size <= 0:
        raise ValueError("invalid superblock")
    document = json.loads(raw[_SUPER.size:_SUPER.size + size])
    if hashlib.sha256(canonical(document)).digest() != digest:
        raise ValueError("superblock digest differs")
    if document.get("namespace_bytes") != namespace or document.get("format") != "incremental-feasibility-v1":
        raise ValueError("superblock document differs")
    return document


def pack_frame_prefix(*, generation, step, descriptor, payload_bytes, payload_sha256):
    if generation <= 0 or step <= 0 or payload_bytes <= 0 or len(payload_sha256) != 64:
        raise ValueError("invalid frame identity")
    raw = canonical(descriptor)
    descriptor_bytes = align(len(raw) + _FRAME.size)
    header = _FRAME.pack(FRAME_MAGIC, 1, generation, step, descriptor_bytes,
                         payload_bytes, bytes.fromhex(payload_sha256), len(raw))
    return header + raw + bytes(descriptor_bytes - len(header) - len(raw))


def unpack_frame_prefix(raw):
    raw = bytes(raw)
    if len(raw) < ALIGN:
        raise ValueError("frame prefix is short")
    magic, version, generation, step, descriptor_bytes, payload_bytes, digest, size = _FRAME.unpack_from(raw)
    if (magic != FRAME_MAGIC or version != 1 or generation <= 0 or step <= 0 or
            descriptor_bytes != len(raw) or descriptor_bytes % ALIGN or size <= 0):
        raise ValueError("invalid frame prefix")
    descriptor = json.loads(raw[_FRAME.size:_FRAME.size + size])
    return {"generation": generation, "step": step, "descriptor_bytes": descriptor_bytes,
            "payload_bytes": payload_bytes, "payload_sha256": digest.hex(), "descriptor": descriptor}


def pack_commit(*, generation, frame_offset, frame_bytes, frame_sha256):
    if (generation <= 0 or frame_offset < SUPER_BYTES or frame_offset % ALIGN or
            frame_bytes <= 0 or len(frame_sha256) != 64):
        raise ValueError("invalid commit geometry")
    document = {"generation": generation, "frame_offset": frame_offset,
                "frame_bytes": frame_bytes, "frame_sha256": frame_sha256}
    raw = canonical(document)
    header = _COMMIT.pack(COMMIT_MAGIC, 1, generation, frame_offset,
                          bytes.fromhex(frame_sha256), len(raw))
    return _page(header, document)


def unpack_commit(raw):
    raw = bytes(raw)
    if len(raw) != ALIGN:
        raise ValueError("commit page size differs")
    magic, version, generation, offset, digest, size = _COMMIT.unpack_from(raw)
    if (magic != COMMIT_MAGIC or version != 1 or generation <= 0 or offset < SUPER_BYTES or
            offset % ALIGN or size <= 0):
        raise ValueError("invalid commit page")
    document = json.loads(raw[_COMMIT.size:_COMMIT.size + size])
    if (document.get("generation") != generation or document.get("frame_offset") != offset or
            document.get("frame_sha256") != digest.hex()):
        raise ValueError("commit document differs")
    return document


class Layout:
    def __init__(self, namespace_bytes, full_weight_bytes):
        if namespace_bytes <= SUPER_BYTES or full_weight_bytes <= 0:
            raise ValueError("invalid namespace budget")
        scratch = align(2 * full_weight_bytes + (1 << 30), 1 << 30)
        if SUPER_BYTES + scratch >= namespace_bytes:
            raise MemoryError("raw namespace cannot admit archive and scratch")
        self.namespace_bytes = namespace_bytes
        self.archive_start = SUPER_BYTES
        self.scratch_start = namespace_bytes - scratch
        self.scratch_bytes = scratch
        self.cursor = self.archive_start

    def reserve_archive(self, byte_count):
        byte_count = align(byte_count)
        if byte_count <= 0 or self.cursor > self.scratch_start - byte_count:
            raise MemoryError("canonical archive exceeds raw namespace")
        offset = self.cursor
        self.cursor += byte_count
        return offset

    def validate_frame(self, frame_offset, prefix_bytes, payload_bytes):
        """Check a complete frame before any media write is admitted."""
        if (type(frame_offset) is not int or type(prefix_bytes) is not int or
                type(payload_bytes) is not int or frame_offset % ALIGN or
                prefix_bytes < ALIGN or prefix_bytes % ALIGN or payload_bytes <= 0):
            raise ValueError("invalid frame geometry")
        end = frame_offset + prefix_bytes + align(payload_bytes)
        if frame_offset < self.archive_start or end > self.scratch_start:
            raise MemoryError("frame exceeds archive boundary")
        return end

    def scratch_slot(self, index):
        if index not in (0, 1):
            raise ValueError("scratch slot index")
        slot = self.scratch_bytes // 2
        return self.scratch_start + index * slot, slot

    def scratch_cursor(self, index, byte_count):
        start, size = self.scratch_slot(index)
        byte_count = align(byte_count)
        if byte_count <= 0 or byte_count > size:
            raise MemoryError('scratch run exceeds slot')
        return start, start + byte_count
