"""Category-aware, self-describing selective checkpoint frames.

The codec is intentionally storage-path agnostic: P8 writes the returned
aligned bytes through SPDK and P9 applies exactly those bytes during recovery.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from s2_policy import S2SelectivePolicy


MAGIC = b"NVDLT01\0"
VERSION = 1
ALIGNMENT = 4096
HEADER = struct.Struct("<8sIIQQQII")


def align(value, alignment=ALIGNMENT):
    return ((int(value) + alignment - 1) // alignment) * alignment


def state_category(name):
    if name.startswith("model/"):
        return "model"
    if name.startswith("optimizer/m/"):
        return "adam_m"
    if name.startswith("optimizer/v/"):
        return "adam_v"
    return "other"


@dataclass(frozen=True)
class CategoryConfig:
    fraction: float = 1.0
    encoding: str = "raw"
    max_age: int = 0
    refresh_interval: int = 1


class CategoryAwarePolicy:
    """Compose independent persisted-reference policies by tensor category."""

    def __init__(self, initial: Mapping[str, np.ndarray], block_size: int,
                 configs: Mapping[str, CategoryConfig]):
        self.configs = {name: (value if isinstance(value, CategoryConfig)
                               else CategoryConfig(**value))
                        for name, value in configs.items()}
        grouped = {}
        for name, value in initial.items():
            grouped.setdefault(state_category(name), {})[name] = value
        self.policies = {}
        for category, state in grouped.items():
            cfg = self.configs.get(category, CategoryConfig())
            self.policies[category] = S2SelectivePolicy(
                state, block_size=block_size,
                selection_fraction=cfg.fraction, encoding=cfg.encoding,
                max_age=cfg.max_age, small_threshold=0)
        self.generation = 0
        self.pending = None

    def observe(self, current: Mapping[str, np.ndarray], step: int):
        if self.pending is not None:
            raise RuntimeError("one category generation is already awaiting ACK")
        records = {}
        for category, policy in self.policies.items():
            cfg = self.configs.get(category, CategoryConfig())
            subset = {name: current[name] for name in policy.reference}
            # A reduced-refresh category still calls observe so age advances.
            # Between refresh points only max-age-forced blocks are selected.
            original_mode = policy.selection_mode
            original_threshold = policy.score_threshold
            if cfg.refresh_interval > 1 and step % cfg.refresh_interval:
                policy.selection_mode = "threshold"
                policy.score_threshold = float("inf")
            try:
                records[category] = policy.observe(subset, step)
            finally:
                policy.selection_mode = original_mode
                policy.score_threshold = original_threshold
        self.pending = {"generation": self.generation + 1,
                        "step": int(step), "categories": records}
        return self.pending

    def pack(self, current: Mapping[str, np.ndarray]):
        if self.pending is None:
            raise RuntimeError("observe must precede pack")
        descriptors = []
        payload = bytearray()
        for category, policy in sorted(self.policies.items()):
            pending = self.pending["categories"][category]
            decoded_by_index = {index: (scale, encoding)
                                for index, _value, scale, encoding
                                in pending["decoded"]}
            for index in pending["selected"]:
                block = policy.blocks[index]
                source = np.asarray(current[block.name]).reshape(-1)[
                    block.offset:block.offset + block.count]
                scale, encoding = decoded_by_index[index]
                if encoding == "raw":
                    encoded = np.ascontiguousarray(source).tobytes()
                elif encoding == "fp16":
                    encoded = np.ascontiguousarray(
                        source.astype(np.float16)).tobytes()
                else:
                    scale = float(scale)
                    encoded = np.clip(np.rint(source / scale), -127, 127).astype(
                        np.int8).tobytes()
                descriptors.append({
                    "category": category, "name": block.name,
                    "element_offset": block.offset, "element_count": block.count,
                    "dtype": np.asarray(current[block.name]).dtype.str,
                    "encoding": encoding, "scale": scale,
                    "payload_offset": len(payload), "payload_bytes": len(encoded),
                    "age_forced": bool(policy.max_age and
                                       policy.age[index] >= policy.max_age - 1),
                })
                payload.extend(encoded)
        descriptor_bytes = json.dumps(
            descriptors, separators=(",", ":"), sort_keys=True).encode("utf-8")
        header_bytes = align(HEADER.size + len(descriptor_bytes))
        payload_crc = zlib.crc32(payload) & 0xFFFFFFFF
        prefix = HEADER.pack(MAGIC, VERSION, header_bytes,
                             self.pending["generation"], self.pending["step"],
                             len(payload), len(descriptor_bytes), payload_crc)
        frame = bytearray(prefix)
        frame.extend(descriptor_bytes)
        frame.extend(b"\0" * (header_bytes - len(frame)))
        frame.extend(payload)
        frame.extend(b"\0" * (align(len(frame)) - len(frame)))
        return bytes(frame), {
            "generation": self.pending["generation"],
            "step": self.pending["step"], "blocks": len(descriptors),
            "payload_bytes": len(payload),
            "metadata_bytes": header_bytes,
            "alignment_bytes": len(frame) - header_bytes - len(payload),
            "physical_bytes": len(frame), "payload_crc32": payload_crc,
        }

    def ack(self, generation):
        if self.pending is None or generation != self.pending["generation"]:
            raise ValueError("stale or unknown category ACK")
        for policy in self.policies.values():
            policy.ack(generation)
        self.generation = generation
        self.pending = None

    def reference(self):
        output = {}
        for policy in self.policies.values():
            output.update({name: np.array(value, copy=True)
                           for name, value in policy.reference.items()})
        return output


def unpack_frame(frame: bytes):
    if len(frame) < HEADER.size:
        raise ValueError("truncated delta frame")
    magic, version, header_bytes, generation, step, payload_bytes, desc_bytes, crc = (
        HEADER.unpack_from(frame))
    if magic != MAGIC or version != VERSION or header_bytes % ALIGNMENT:
        raise ValueError("invalid delta frame header")
    if header_bytes + payload_bytes > len(frame):
        raise ValueError("truncated delta payload")
    descriptors = json.loads(frame[HEADER.size:HEADER.size + desc_bytes])
    payload = memoryview(frame)[header_bytes:header_bytes + payload_bytes]
    if zlib.crc32(payload) & 0xFFFFFFFF != crc:
        raise ValueError("delta frame CRC mismatch")
    return {"generation": generation, "step": step,
            "header_bytes": header_bytes, "payload_bytes": payload_bytes,
            "physical_bytes": len(frame), "descriptors": descriptors,
            "payload": payload}


def apply_frame(state: Mapping[str, np.ndarray], frame: bytes):
    parsed = unpack_frame(frame)
    output = {name: np.array(value, copy=True) for name, value in state.items()}
    for item in parsed["descriptors"]:
        if item["name"] not in output:
            raise KeyError(f"missing recovery tensor: {item['name']}")
        target = output[item["name"]].reshape(-1)
        dtype = np.dtype(item["dtype"])
        begin = int(item["payload_offset"])
        end = begin + int(item["payload_bytes"])
        raw = parsed["payload"][begin:end]
        encoding = item["encoding"]
        if encoding == "raw":
            values = np.frombuffer(raw, dtype=dtype)
        elif encoding == "fp16":
            values = np.frombuffer(raw, dtype=np.float16).astype(dtype)
        elif encoding == "int8":
            values = (np.frombuffer(raw, dtype=np.int8).astype(np.float32) *
                      float(item["scale"])).astype(dtype)
        else:
            raise ValueError(f"unsupported frame encoding: {encoding}")
        count = int(item["element_count"])
        if values.size != count:
            raise ValueError("delta descriptor length mismatch")
        offset = int(item["element_offset"])
        target[offset:offset + count] = values
    return output, parsed


__all__ = ["CategoryAwarePolicy", "CategoryConfig", "apply_frame",
           "unpack_frame", "state_category", "align"]
