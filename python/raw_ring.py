"""Crash-detectable envelope and A/B metadata for raw checkpoint rings."""

from __future__ import annotations

import struct
import zlib


RING_MAGIC = 0x524E4733  # RNG3
META_MAGIC = 0x524D5433  # RMT3
RING_VERSION = 1
STATE_COMMITTED = 1
KIND_FULL = 1
KIND_DELTA = 2
ENVELOPE_SIZE = 64
METADATA_SIZE = 64
_ENVELOPE = struct.Struct("<IHHQQQIII20s")
_METADATA = struct.Struct("<IHHQQQQI20s")


def _crc32(value):
    return zlib.crc32(value) & 0xFFFFFFFF


def pack_ring_slot(frame, slot_generation, step_id, kind, slot_size=None):
    frame = bytes(frame)
    if not frame or slot_generation <= 0 or step_id < 0:
        raise ValueError("invalid ring frame identity")
    if kind not in (KIND_FULL, KIND_DELTA):
        raise ValueError("invalid ring frame kind")
    header = _ENVELOPE.pack(
        RING_MAGIC, RING_VERSION, STATE_COMMITTED, int(slot_generation),
        int(step_id), len(frame), _crc32(frame), int(kind), 0, bytes(20))
    header_crc = _crc32(header)
    header = _ENVELOPE.pack(
        RING_MAGIC, RING_VERSION, STATE_COMMITTED, int(slot_generation),
        int(step_id), len(frame), _crc32(frame), int(kind), header_crc,
        bytes(20))
    packed = header + frame
    if slot_size is not None:
        if len(packed) > slot_size:
            raise ValueError("ring frame exceeds slot capacity")
        packed += bytes(slot_size - len(packed))
    return packed


def unpack_ring_slot(value):
    value = bytes(value)
    if len(value) < ENVELOPE_SIZE:
        raise ValueError("torn ring envelope")
    fields = list(_ENVELOPE.unpack_from(value))
    magic, version, state = fields[:3]
    generation, step_id, frame_length, frame_crc, kind, header_crc = fields[3:9]
    if magic != RING_MAGIC or version != RING_VERSION:
        raise ValueError("invalid ring envelope magic/version")
    if state != STATE_COMMITTED:
        raise ValueError("ring slot is not committed")
    fields[8] = 0
    if _crc32(_ENVELOPE.pack(*fields)) != header_crc:
        raise ValueError("ring envelope CRC mismatch")
    if kind not in (KIND_FULL, KIND_DELTA):
        raise ValueError("invalid ring frame kind")
    end = ENVELOPE_SIZE + frame_length
    if frame_length <= 0 or end > len(value):
        raise ValueError("torn ring payload")
    frame = value[ENVELOPE_SIZE:end]
    if _crc32(frame) != frame_crc:
        raise ValueError("ring whole-frame CRC mismatch")
    return {"slot_generation": generation, "step_id": step_id,
            "kind": kind, "frame_length": frame_length,
            "frame_crc": frame_crc, "frame": frame}


def pack_ring_metadata(generation, head, tail, latest_full):
    if generation <= 0 or min(head, tail, latest_full) < 0:
        raise ValueError("invalid ring metadata")
    raw = _METADATA.pack(META_MAGIC, RING_VERSION, STATE_COMMITTED,
                         int(generation), int(head), int(tail),
                         int(latest_full), 0, bytes(20))
    crc = _crc32(raw)
    return _METADATA.pack(META_MAGIC, RING_VERSION, STATE_COMMITTED,
                          int(generation), int(head), int(tail),
                          int(latest_full), crc, bytes(20))


def unpack_ring_metadata(value):
    if len(value) < METADATA_SIZE:
        raise ValueError("torn ring metadata")
    fields = list(_METADATA.unpack_from(bytes(value)))
    magic, version, state = fields[:3]
    crc = fields[7]
    if magic != META_MAGIC or version != RING_VERSION or state != STATE_COMMITTED:
        raise ValueError("invalid ring metadata header")
    fields[7] = 0
    if _crc32(_METADATA.pack(*fields)) != crc:
        raise ValueError("ring metadata CRC mismatch")
    return {"generation": fields[3], "head": fields[4], "tail": fields[5],
            "latest_full": fields[6]}


def select_ab_metadata(copy_a, copy_b):
    valid = []
    for name, raw in (("A", copy_a), ("B", copy_b)):
        try:
            valid.append((name, unpack_ring_metadata(raw)))
        except ValueError:
            pass
    if not valid:
        raise ValueError("no valid ring metadata copy")
    valid.sort(key=lambda item: item[1]["generation"], reverse=True)
    if (len(valid) == 2 and
            valid[0][1]["generation"] == valid[1][1]["generation"] and
            valid[0][1] != valid[1][1]):
        raise ValueError("conflicting equal-generation metadata copies")
    return valid[0]


def select_recovery_chain(raw_slots):
    """Select latest FULL plus a complete, monotonic committed suffix."""
    records = []
    for raw in raw_slots:
        try:
            records.append(unpack_ring_slot(raw))
        except ValueError:
            continue
    if not records:
        raise ValueError("ring contains no valid committed slots")
    by_generation = {}
    for record in records:
        generation = record["slot_generation"]
        if generation in by_generation:
            raise ValueError("duplicate committed ring generation")
        by_generation[generation] = record
    latest = max(by_generation)
    full_generations = [generation for generation, record in by_generation.items()
                        if generation <= latest and record["kind"] == KIND_FULL]
    if not full_generations:
        raise ValueError("ring contains no retained FULL frame")
    first = max(full_generations)
    expected = list(range(first, latest + 1))
    if any(generation not in by_generation for generation in expected):
        raise ValueError("missing generation after latest FULL")
    chain = [by_generation[generation] for generation in expected]
    first_step = chain[0]["step_id"]
    if any(record["step_id"] != first_step + index
           for index, record in enumerate(chain)):
        raise ValueError("non-contiguous step sequence after latest FULL")
    if any(record["kind"] != KIND_DELTA for record in chain[1:]):
        raise ValueError("unexpected FULL inside selected suffix")
    return chain


__all__ = ["ENVELOPE_SIZE", "KIND_DELTA", "KIND_FULL", "METADATA_SIZE",
           "pack_ring_metadata", "pack_ring_slot", "select_ab_metadata",
           "select_recovery_chain", "unpack_ring_metadata",
           "unpack_ring_slot"]
