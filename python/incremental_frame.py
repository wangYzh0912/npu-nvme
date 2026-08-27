"""Version 4 self-describing replacement frame for S2-R0.

The frame keeps the fixed header small and puts canonical JSON descriptors in
the reserved header area, followed by raw payload bytes.  Names and dtype
strings are carried in descriptors while the manifest digest makes the
descriptor interpretable only with the matching training-state manifest.
Every record has its own CRC32 and the whole descriptor/payload area has a
CRC32.  This is a CPU-testable contract; the NPU writer may fill the same
descriptor after collecting per-chunk CRCs from DMA buffers.
"""

from __future__ import annotations

import binascii
import json
import struct
from typing import Iterable, Mapping

import numpy as np

from disk_layout import DELTA_MAGIC, FRAME_HEADER_SIZE


FRAME_VERSION = 4
FRAME_FLAGS_REPLACEMENT = 1
FRAME_FLAGS_CONTROLS = 2
_HEADER = struct.Struct("<IHHIIIIQQQQ32sIIIIII")
_HEADER_CRC_OFFSET = _HEADER.size - 4
_MAX_DESCRIPTOR_BYTES = FRAME_HEADER_SIZE - _HEADER.size


def _crc32(value: bytes) -> int:
    return binascii.crc32(value) & 0xFFFFFFFF


def _digest(value):
    if not value:
        return bytes(32)
    raw = bytes.fromhex(str(value))
    if len(raw) != 32:
        raise ValueError("manifest_digest must be a SHA-256 hex digest")
    return raw


def _raw_value(value, dtype):
    np_dtype = np.dtype(dtype).newbyteorder("<")
    array = np.ascontiguousarray(np.asarray(value, dtype=np_dtype).reshape(-1))
    return array, array.tobytes()


def pack_r0_frame(step: int, generation: int, base_full_generation: int,
                  base_delta_generation: int, manifest_digest: str,
                  block_records: Iterable[Mapping],
                  control_records: Iterable[Mapping] = (), world_size: int = 1,
                  rank_id: int = 0) -> bytes:
    """Pack native-dtype replacement blocks and control payloads."""
    if step < 0 or generation <= 0 or base_full_generation < 0:
        raise ValueError("invalid frame identity")
    if base_delta_generation < 0 or generation <= base_delta_generation:
        raise ValueError("generation must follow base_delta_generation")
    if world_size <= 0 or rank_id < 0 or rank_id >= world_size:
        raise ValueError("invalid rank identity")

    payload = bytearray()
    blocks = []
    controls = []
    for source in block_records:
        name = str(source["name"])
        dtype = np.dtype(source["dtype"]).newbyteorder("<")
        array, raw = _raw_value(source["value"], dtype)
        expected = int(source["element_count"])
        if array.size != expected:
            raise ValueError(f"block element count mismatch for {name}")
        offset = len(payload)
        payload.extend(raw)
        blocks.append({
            "kind": "block", "name": name,
            "state_index": int(source["state_index"]),
            "block_index": int(source["block_index"]),
            "element_offset": int(source["element_offset"]),
            "element_count": int(array.size), "dtype": dtype.name,
            "encoding": "raw", "payload_offset": offset,
            "payload_bytes": len(raw), "crc32": _crc32(raw),
        })
    for source in control_records:
        name = str(source["name"])
        raw = bytes(source["payload"])
        if not raw:
            raise ValueError(f"empty control payload: {name}")
        offset = len(payload)
        payload.extend(raw)
        controls.append({
            "kind": "control", "name": name, "codec": str(source["codec"]),
            "payload_offset": offset, "payload_bytes": len(raw),
            "crc32": _crc32(raw),
        })

    descriptor = json.dumps({"blocks": blocks, "controls": controls},
                            sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True).encode("utf-8")
    if len(descriptor) > _MAX_DESCRIPTOR_BYTES:
        raise ValueError("frame descriptors exceed fixed header area")
    logical_bytes = len(payload)
    descriptor_crc = _crc32(descriptor)
    payload_crc = _crc32(payload)
    flags = FRAME_FLAGS_REPLACEMENT | (FRAME_FLAGS_CONTROLS if controls else 0)
    values = [DELTA_MAGIC, FRAME_VERSION, flags, 1, int(world_size), int(rank_id),
              0, int(base_full_generation), int(base_delta_generation),
              int(generation), int(step), _digest(manifest_digest),
              len(descriptor), len(payload), logical_bytes, payload_crc,
              descriptor_crc, 0]
    header = bytearray(_HEADER.pack(*values))
    header_crc = _crc32(bytes(header))
    values[-1] = header_crc
    header = _HEADER.pack(*values)
    return header + bytes(FRAME_HEADER_SIZE - len(header)) + descriptor + bytes(payload)


def unpack_r0_frame(frame: bytes) -> dict:
    """Validate and unpack an R0 frame into descriptors and owned arrays."""
    frame = bytes(frame)
    if len(frame) < FRAME_HEADER_SIZE or len(frame) < _HEADER.size:
        raise ValueError("frame is shorter than the v4 header")
    values = list(_HEADER.unpack_from(frame))
    (magic, version, flags, schema, world_size, rank_id, strategy,
     base_full, base_delta, generation, step, digest, descriptor_bytes,
     payload_bytes, logical_bytes, payload_crc, descriptor_crc,
     header_crc) = values
    if magic != DELTA_MAGIC or version != FRAME_VERSION:
        raise ValueError("invalid v4 frame magic/version")
    if schema != 1 or strategy != 0:
        raise ValueError("unsupported v4 schema or strategy")
    if world_size <= 0 or rank_id >= world_size:
        raise ValueError("invalid v4 rank identity")
    if generation <= base_delta or generation <= 0:
        raise ValueError("invalid v4 generation lineage")
    if descriptor_bytes > _MAX_DESCRIPTOR_BYTES:
        raise ValueError("v4 descriptor area exceeds fixed header")
    end = FRAME_HEADER_SIZE + descriptor_bytes + payload_bytes
    if end != len(frame) or logical_bytes != payload_bytes:
        raise ValueError("v4 frame length mismatch")
    values[-1] = 0
    if _crc32(_HEADER.pack(*values)) != header_crc:
        raise ValueError("v4 header CRC mismatch")
    descriptor = frame[FRAME_HEADER_SIZE:FRAME_HEADER_SIZE + descriptor_bytes]
    payload = frame[FRAME_HEADER_SIZE + descriptor_bytes:end]
    if _crc32(descriptor) != descriptor_crc:
        raise ValueError("v4 descriptor CRC mismatch")
    if _crc32(payload) != payload_crc:
        raise ValueError("v4 payload CRC mismatch")
    try:
        document = json.loads(descriptor.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid v4 descriptor JSON") from error
    blocks = []
    occupied = []
    for record in document.get("blocks", []):
        if record.get("kind") != "block" or record.get("encoding") != "raw":
            raise ValueError("unsupported v4 block record")
        dtype = np.dtype(record["dtype"]).newbyteorder("<")
        offset = int(record["payload_offset"])
        size = int(record["payload_bytes"])
        count = int(record["element_count"])
        if count < 0 or size != count * dtype.itemsize or offset < 0 or offset + size > len(payload):
            raise ValueError("invalid v4 block bounds")
        raw = payload[offset:offset + size]
        if _crc32(raw) != int(record["crc32"]):
            raise ValueError("v4 block CRC mismatch")
        if any(offset < end and start < offset + size
               for start, end in occupied):
            raise ValueError("overlapping v4 payload records")
        occupied.append((offset, offset + size))
        blocks.append({**record, "dtype": dtype.name,
                       "value": np.frombuffer(raw, dtype=dtype).copy()})
    controls = []
    for record in document.get("controls", []):
        offset = int(record["payload_offset"])
        size = int(record["payload_bytes"])
        if not record.get("name") or not record.get("codec") or size <= 0 or offset < 0 or offset + size > len(payload):
            raise ValueError("invalid v4 control bounds")
        if any(offset < end and start < offset + size
               for start, end in occupied):
            raise ValueError("overlapping v4 payload records")
        raw = payload[offset:offset + size]
        if _crc32(raw) != int(record["crc32"]):
            raise ValueError("v4 control CRC mismatch")
        occupied.append((offset, offset + size))
        controls.append({**record, "payload": bytes(raw)})
    return {
        "version": version, "flags": flags, "schema_version": schema,
        "world_size": world_size, "rank_id": rank_id,
        "base_full_generation": base_full, "base_delta_generation": base_delta,
        "generation": generation, "step": step,
        "manifest_digest": digest.hex(), "blocks": blocks, "controls": controls,
        "logical_payload_bytes": logical_bytes,
    }


__all__ = ["FRAME_VERSION", "pack_r0_frame", "unpack_r0_frame"]
