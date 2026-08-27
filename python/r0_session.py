"""Correctness-first S2-R0 reference ledger and replay session.

This module is intentionally framework-free.  The hardware session will use
the same lineage and ACK rules, while this implementation provides the oracle
for unit tests and for validating frames produced by an NPU capture path.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

import numpy as np

from incremental_frame import pack_r0_frame, unpack_r0_frame


class R0Session:
    """Lossless replacement session with at most one unacknowledged frame."""

    def __init__(self, initial: Mapping[str, np.ndarray], manifest,
                 base_full_generation: int = 1):
        if set(initial) != {field.canonical_name for field in manifest.fields}:
            raise ValueError("initial state does not match manifest")
        self.manifest = manifest
        self.base_full_generation = int(base_full_generation)
        self.base_state = {name: np.ascontiguousarray(value).copy()
                           for name, value in initial.items()}
        self.persisted = {name: value.copy()
                          for name, value in self.base_state.items()}
        self.current = {name: value.copy() for name, value in self.persisted.items()}
        self.persisted_generation = 0
        self._in_flight = None

    @property
    def in_flight_generation(self):
        return None if self._in_flight is None else self._in_flight["generation"]

    def set_current(self, state: Mapping[str, np.ndarray]):
        if set(state) != set(self.current):
            raise ValueError("current state field set changed")
        for name, old in self.current.items():
            value = np.ascontiguousarray(state[name])
            if value.shape != old.shape or value.dtype != old.dtype:
                raise ValueError(f"state shape/dtype changed: {name}")
            if np.issubdtype(value.dtype, np.inexact) and not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite state: {name}")
        self.current = {name: np.ascontiguousarray(value).copy()
                        for name, value in state.items()}

    def _changed_records(self):
        records = []
        for field in self.manifest.fields:
            current = self.current[field.canonical_name].reshape(-1)
            persisted = self.persisted[field.canonical_name].reshape(-1)
            for block in field.blocks:
                start = block.element_offset
                end = start + block.element_count
                if not np.array_equal(current[start:end], persisted[start:end]):
                    records.append({
                        "name": field.canonical_name,
                        "state_index": field.state_index,
                        "block_index": block.block_index,
                        "element_offset": start,
                        "element_count": block.element_count,
                        "dtype": field.dtype,
                        "value": current[start:end].copy(),
                    })
        return records

    def observe(self, step: int, generation: int, controls=()) -> bytes:
        if self._in_flight is not None:
            raise RuntimeError("an R0 generation is awaiting ACK")
        if generation != self.persisted_generation + 1:
            raise ValueError("generation is not the next persisted generation")
        control_records = [{"name": name, "codec": codec, "payload": payload}
                           for name, codec, payload in controls]
        frame = pack_r0_frame(
            step=step, generation=generation,
            base_full_generation=self.base_full_generation,
            base_delta_generation=self.persisted_generation,
            manifest_digest=self.manifest.digest,
            block_records=self._changed_records(),
            control_records=control_records)
        self._in_flight = {"generation": generation,
                           "checksum": hashlib.sha256(frame).hexdigest()}
        return frame

    def ack(self, frame: bytes):
        info = unpack_r0_frame(frame)
        if info["manifest_digest"] != self.manifest.digest:
            raise ValueError("R0 manifest digest mismatch")
        if self._in_flight is None:
            raise ValueError("R0 ACK has no in-flight generation")
        if info["generation"] != self._in_flight["generation"]:
            raise ValueError("R0 ACK generation mismatch")
        if info["base_delta_generation"] != self.persisted_generation:
            raise ValueError("R0 ACK base generation mismatch")
        by_index = {field.state_index: field for field in self.manifest.fields}
        for record in info["blocks"]:
            field = by_index.get(int(record["state_index"]))
            if field is None or field.canonical_name != record["name"]:
                raise ValueError("R0 ACK block manifest mismatch")
            block = field.blocks[int(record["block_index"])]
            if (block.element_offset != int(record["element_offset"]) or
                    block.element_count != int(record["element_count"])):
                raise ValueError("R0 ACK block location mismatch")
            target = self.persisted[field.canonical_name].reshape(-1)
            start = block.element_offset
            target[start:start + block.element_count] = record["value"]
        self.persisted_generation = info["generation"]
        self._in_flight = None
        return {"generation": info["generation"], "step": info["step"],
                "blocks": len(info["blocks"]),
                "controls": len(info["controls"])}

    def recover(self, frames: Iterable[bytes]):
        state = {name: value.copy() for name, value in self.base_state.items()}
        generation = 0
        last_step = None
        for frame in frames:
            info = unpack_r0_frame(frame)
            if info["manifest_digest"] != self.manifest.digest:
                raise ValueError("R0 recovery manifest mismatch")
            if info["base_delta_generation"] != generation:
                raise ValueError("R0 recovery generation gap or reordering")
            by_index = {field.state_index: field for field in self.manifest.fields}
            for record in info["blocks"]:
                field = by_index.get(int(record["state_index"]))
                if field is None or field.canonical_name != record["name"]:
                    raise ValueError("R0 recovery block mismatch")
                target = state[field.canonical_name].reshape(-1)
                start = int(record["element_offset"])
                target[start:start + int(record["element_count"])] = record["value"]
            generation = info["generation"]
            last_step = info["step"]
        return {"state": state, "generation": generation, "last_step": last_step}


def state_digest(state: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = np.ascontiguousarray(state[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


__all__ = ["R0Session", "state_digest"]
