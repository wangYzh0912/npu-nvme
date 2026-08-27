"""Direct HBM-to-NVMe S2-R0 writer and fresh-process replay helper.

This module is intentionally synchronous.  R0 first proves the complete
replacement/ACK/restart contract while keeping the training parameters stable
between capture and ACK.  It uses the native-dtype HBM DMA API for tensor
blocks; only the descriptor and small control payloads use host buffers.
"""

from __future__ import annotations

import binascii
import ctypes
import json
import math
from typing import Mapping

import numpy as np

from c_bindings import lib
from chunk_helpers import build_chunks, build_chunks_host, build_ctypes_arrays
from direct_checkpoint import get_dev_ptr
from incremental_frame import (pack_r0_frame_prefix, unpack_r0_frame_prefix)
from training_state import decode_control_value, encode_control_value


BLOCK_SIZE = 4096


def _align(value, alignment=BLOCK_SIZE):
    return (int(value) + alignment - 1) // alignment * alignment


def _gf2_matrix_times(matrix, vector):
    result = 0
    index = 0
    while vector:
        if vector & 1:
            result ^= matrix[index]
        vector >>= 1
        index += 1
    return result


def _gf2_matrix_square(square, matrix):
    for index in range(32):
        square[index] = _gf2_matrix_times(matrix, matrix[index])


def crc32_combine(crc1, crc2, length2):
    """Return CRC32(A+B) from CRC32(A), CRC32(B), and len(B)."""
    if length2 < 0:
        raise ValueError("CRC suffix length must be non-negative")
    if length2 == 0:
        return (int(crc1) ^ int(crc2)) & 0xFFFFFFFF
    odd = [0] * 32
    even = [0] * 32
    odd[0] = 0xEDB88320
    row = 1
    for index in range(1, 32):
        odd[index] = row
        row <<= 1
    _gf2_matrix_square(even, odd)
    _gf2_matrix_square(odd, even)
    length = int(length2)
    first = int(crc1) & 0xFFFFFFFF
    while length:
        _gf2_matrix_square(even, odd)
        if length & 1:
            first = _gf2_matrix_times(even, first)
        length >>= 1
        if not length:
            break
        _gf2_matrix_square(odd, even)
        if length & 1:
            first = _gf2_matrix_times(odd, first)
        length >>= 1
    return (first ^ (int(crc2) & 0xFFFFFFFF)) & 0xFFFFFFFF


def _combine_records(records, payload_bytes):
    """Combine CRCs for aligned records and their zero padding."""
    crc = 0
    cursor = 0
    zero_crc = {}
    for record in sorted(records, key=lambda item: item["payload_offset"]):
        offset = int(record["payload_offset"])
        if offset < cursor:
            raise ValueError("overlapping payload records")
        gap = offset - cursor
        if gap:
            if gap not in zero_crc:
                zero_crc[gap] = binascii.crc32(bytes(gap)) & 0xFFFFFFFF
            crc = crc32_combine(crc, zero_crc[gap], gap)
        size = int(record["payload_bytes"])
        crc = crc32_combine(crc, int(record["crc32"]), size)
        cursor = offset + size
        padded = _align(cursor)
        if padded != cursor:
            pad = padded - cursor
            if pad not in zero_crc:
                zero_crc[pad] = binascii.crc32(bytes(pad)) & 0xFFFFFFFF
            crc = crc32_combine(crc, zero_crc[pad], pad)
            cursor = padded
    if cursor != int(payload_bytes):
        raise ValueError("payload CRC records do not cover payload")
    return crc


class R0NpuWriter:
    """Capture changed HBM blocks, persist a frame, metadata, then ACK."""

    def __init__(self, checkpoint, state, full_generation: int,
                 slot_count: int = None, slot_size_mb: int = None):
        self.checkpoint = checkpoint
        self.state = state
        self.full_generation = int(full_generation)
        if slot_count is None:
            slot_count = checkpoint.layout.delta_slot_count
        if slot_size_mb is None:
            slot_size_mb = checkpoint.layout.delta_slot_bytes // 1024**2
        checkpoint.delta_init(slot_size_mb=slot_size_mb, slot_count=slot_count)

    @property
    def manifest_digest(self):
        return self.state.manifest.digest

    def _write_device_records(self, records, slot_offset, descriptor_bytes):
        if not records:
            return
        params = [{
            "ptr": int(record["pointer"]),
            "size": int(record["payload_bytes"]),
            "offset": int(slot_offset + 4096 + descriptor_bytes +
                           record["payload_offset"]),
            "name": record["name"],
        } for record in records]
        chunks, _ = build_chunks(params, self.checkpoint.chunk_size)
        ptrs, offsets, sizes = build_ctypes_arrays(chunks)
        if not hasattr(lib, "npu_nvme_write_batch_crc"):
            raise RuntimeError("C library lacks npu_nvme_write_batch_crc")
        crc_values = (ctypes.c_uint32 * len(chunks))()
        rc = lib.npu_nvme_write_batch_crc(
            self.checkpoint.ctx, ptrs, offsets, sizes, crc_values, len(chunks))
        if rc != 0:
            raise RuntimeError(f"R0 HBM write failed (rc={rc})")
        # R0's fixed block size is <= the 4 MiB transfer limit, so one record
        # maps to one chunk.  Retain the assertion to prevent silent CRC
        # ambiguity if a future configuration violates that contract.
        if len(chunks) != len(records):
            raise ValueError("R0 block exceeds one DMA transfer")
        for record, crc in zip(records, crc_values):
            record["crc32"] = int(crc)

    def _write_host(self, raw, offset):
        raw = bytes(raw)
        aligned = _align(len(raw))
        buffer = ctypes.create_string_buffer(aligned)
        ctypes.memmove(buffer, raw, len(raw))
        chunks, _ = build_chunks_host(ctypes.addressof(buffer), offset,
                                       aligned, self.checkpoint.chunk_size)
        ptrs, offsets, sizes = build_ctypes_arrays(chunks)
        rc = lib.npu_nvme_write_batch_host(
            self.checkpoint.ctx, ptrs, offsets, sizes, len(chunks))
        if rc != 0:
            raise RuntimeError(f"R0 host write failed (rc={rc})")

    def capture_and_commit(self, step: int, control_state: Mapping,
                           base_delta_generation: int = 0):
        if not self.state.initialized:
            raise RuntimeError("R0 state has not been initialized from FULL")
        flags = self.state.capture()
        changed = self.state.changed_buffers(flags)
        slot_idx = self.checkpoint._delta_next_slot % self.checkpoint._delta_slot_count
        slot_offset = self.checkpoint.layout.delta_slot_offset(slot_idx)
        delta_generation = max(
            [int(record.get("generation", 0))
             for record in self.checkpoint.meta_dict.get("delta_chain", {}).values()
             if record.get("type") == "S2_R0"] + [0]) + 1

        payload_cursor = 0
        block_records = []
        for buffer in changed:
            size = int(buffer.element_count) * np.dtype(buffer.dtype).itemsize
            payload_cursor = _align(payload_cursor)
            block_records.append({
                "kind": "block", "name": buffer.name,
                "state_index": buffer.state_index,
                "block_index": buffer.block_index,
                "element_offset": buffer.element_offset,
                "element_count": buffer.element_count,
                "dtype": np.dtype(buffer.dtype).newbyteorder("<").name,
                "encoding": "raw", "payload_offset": payload_cursor,
                "payload_bytes": size, "crc32": 0,
                "pointer": buffer.pointer,
            })
            payload_cursor += size

        control_payloads = []
        for name in sorted(control_state):
            payload, metadata = encode_control_value(control_state[name])
            payload_cursor = _align(payload_cursor)
            raw = payload.tobytes()
            control_payloads.append({
                "kind": "control", "name": name,
                "codec": metadata["codec"], "sha256": metadata["sha256"],
                "payload_offset": payload_cursor,
                "payload_bytes": len(raw), "crc32": binascii.crc32(raw) & 0xFFFFFFFF,
                "payload": raw,
            })
            payload_cursor += len(raw)
        payload_bytes = _align(payload_cursor)

        # CRCs are obtained by the HBM DMA write.  A provisional descriptor
        # determines the aligned descriptor area; CRC digits do not normally
        # cross its alignment boundary, but the retry handles that edge.
        descriptors_blocks = [dict(record) for record in block_records]
        for record in descriptors_blocks:
            record.pop("pointer", None)
        descriptors_controls = [{key: value for key, value in record.items()
                                 if key != "payload"}
                                for record in control_payloads]
        self._write_device_records(block_records, slot_offset, _align(1))
        descriptors_blocks = [dict(record) for record in block_records]
        for record in descriptors_blocks:
            record.pop("pointer", None)
        prefix_probe = pack_r0_frame_prefix(
            step, delta_generation, self.full_generation,
            int(base_delta_generation), self.manifest_digest,
            descriptors_blocks, descriptors_controls, payload_bytes, 0)
        descriptor_bytes = len(prefix_probe) - 4096
        # If the descriptor size changed after real CRC values were known,
        # rewrite the HBM payload at its final address.
        self._write_device_records(block_records, slot_offset, descriptor_bytes)
        descriptors_blocks = [dict(record) for record in block_records]
        for record in descriptors_blocks:
            record.pop("pointer", None)
        payload_records = descriptors_blocks + descriptors_controls
        payload_crc = _combine_records(payload_records, payload_bytes)
        prefix = pack_r0_frame_prefix(
            step, delta_generation, self.full_generation,
            int(base_delta_generation), self.manifest_digest,
            descriptors_blocks, descriptors_controls, payload_bytes, payload_crc)
        final_descriptor_bytes = len(prefix) - 4096
        if final_descriptor_bytes != descriptor_bytes:
            self._write_device_records(block_records, slot_offset,
                                       final_descriptor_bytes)
            descriptors_blocks = [dict(record) for record in block_records]
            for record in descriptors_blocks:
                record.pop("pointer", None)
            payload_crc = _combine_records(
                descriptors_blocks + descriptors_controls, payload_bytes)
            prefix = pack_r0_frame_prefix(
                step, delta_generation, self.full_generation,
                int(base_delta_generation), self.manifest_digest,
                descriptors_blocks, descriptors_controls, payload_bytes, payload_crc)

        for record in control_payloads:
            self._write_host(record["payload"], slot_offset + 4096 +
                             len(prefix) - 4096 + record["payload_offset"])
        self._write_host(prefix, slot_offset)
        self.checkpoint.flush_nvme()

        key = f"step_{int(step)}"
        for old_key, old_record in list(
                self.checkpoint.meta_dict.get("delta_chain", {}).items()):
            if old_record.get("slot") == slot_idx:
                del self.checkpoint.meta_dict["delta_chain"][old_key]
        self.checkpoint.meta_dict.setdefault("delta_chain", {})[key] = {
            "type": "S2_R0", "generation": delta_generation,
            "slot": slot_idx, "frame_size": len(prefix) + payload_bytes,
            "frame_crc32": payload_crc, "step": int(step),
            "base_full_generation": self.full_generation,
            "base_delta_generation": int(base_delta_generation),
            "manifest_digest": self.manifest_digest,
            "n_blocks": len(descriptors_blocks),
            "n_controls": len(descriptors_controls),
            "payload_bytes": payload_bytes,
        }
        self.checkpoint.meta_dict["delta_head"] = int(self.checkpoint._delta_next_slot) + 1
        self.checkpoint.meta_dict["delta_tail"] = max(
            0, self.checkpoint.meta_dict["delta_head"] -
            self.checkpoint._delta_slot_count)
        self.checkpoint._persist_metadata(self.checkpoint.metadata_generation + 1)
        self.checkpoint._delta_next_slot += 1
        self.state.commit_ack()
        return self.checkpoint.meta_dict["delta_chain"][key]


class R0NpuReader:
    """Apply a persisted R0 frame directly into live NPU parameters."""

    def __init__(self, checkpoint, state):
        self.checkpoint = checkpoint
        self.state = state
        self.by_name = {field.canonical_name: field for field in state.fields}
        self.current = state.current

    def _read_host(self, offset, size):
        return self.checkpoint.read_host_frame(int(offset), int(size))

    def apply(self, record):
        slot = int(record["slot"])
        slot_offset = self.checkpoint.layout.delta_slot_offset(slot)
        header = self._read_host(slot_offset, 4096)
        # v4 header layout places descriptor_bytes after the 32-byte digest:
        # offset 88, followed by payload_bytes at 92.
        descriptor_size = int.from_bytes(header[88:92], "little")
        prefix = header + self._read_host(slot_offset + 4096, descriptor_size)
        info = unpack_r0_frame_prefix(prefix)
        if info["step"] != int(record["step"]):
            raise ValueError("R0 frame step does not match metadata")
        if info["generation"] != int(record["generation"]):
            raise ValueError("R0 frame generation does not match metadata")
        if info["manifest_digest"] != self.state.manifest.digest:
            raise ValueError("R0 frame manifest mismatch")
        if info["payload_bytes"] + len(prefix) != int(record["frame_size"]):
            raise ValueError("R0 frame size does not match metadata")

        dev_params = []
        for item in info["blocks"]:
            name = item["name"]
            field = self.by_name.get(name)
            if field is None or field.state_index != int(item["state_index"]):
                raise ValueError(f"unknown R0 field: {name}")
            block = field.blocks[int(item["block_index"])]
            if (block.element_offset != int(item["element_offset"]) or
                    block.element_count != int(item["element_count"]) or
                    np.dtype(field.dtype).newbyteorder("<").name != item["dtype"]):
                raise ValueError(f"R0 block descriptor mismatch: {name}")
            pointer = get_dev_ptr(self.current[name])
            if not pointer:
                raise RuntimeError(f"no HBM pointer for {name}")
            itemsize = np.dtype(field.dtype).itemsize
            dev_params.append({
                "ptr": pointer + block.element_offset * itemsize,
                "size": int(item["payload_bytes"]),
                "offset": slot_offset + len(prefix) + int(item["payload_offset"]),
                "name": name,
            })
        if dev_params:
            chunks, _ = build_chunks(dev_params, self.checkpoint.chunk_size)
            ptrs, offsets, sizes = build_ctypes_arrays(chunks)
            rc = lib.npu_nvme_read_batch(
                self.checkpoint.ctx, ptrs, offsets, sizes, len(chunks))
            if rc != 0:
                raise RuntimeError(f"R0 HBM replay failed (rc={rc})")
        controls = {}
        for item in info["controls"]:
            raw = self._read_host(slot_offset + len(prefix) +
                                   int(item["payload_offset"]),
                                   int(item["payload_bytes"]))
            if binascii.crc32(raw) & 0xFFFFFFFF != int(item["crc32"]):
                raise ValueError(f"R0 control CRC mismatch: {item['name']}")
            controls[item["name"]] = decode_control_value(
                np.frombuffer(raw, dtype=np.uint8), item)
        if hasattr(__import__("mindspore"), "hal"):
            import mindspore as ms
            if hasattr(ms.hal, "synchronize"):
                ms.hal.synchronize()
        return controls, info


__all__ = ["R0NpuReader", "R0NpuWriter", "crc32_combine"]
