"""Correctness-first S2-R0 reference ledger and replay session.

This module is intentionally framework-free.  The hardware session will use
the same lineage and ACK rules, while this implementation provides the oracle
for unit tests and for validating frames produced by an NPU capture path.
"""

from __future__ import annotations

import hashlib
import copy
from typing import Iterable, Mapping

import numpy as np

from incremental_frame import pack_r0_frame, unpack_r0_frame


class R0Session:
    """Lossless replacement session with at most one unacknowledged frame."""

    def __init__(self, initial: Mapping[str, np.ndarray], manifest,
                 base_full_generation: int = 1):
        if set(initial) != {field.canonical_name for field in manifest.fields}:
            raise ValueError("initial state does not match manifest")
        if type(base_full_generation) is not int or base_full_generation<=0:
            raise ValueError('FULL root must be a positive integer')
        for field in manifest.fields:
            value=np.asarray(initial[field.canonical_name])
            if tuple(value.shape)!=tuple(field.shape) or value.dtype!=np.dtype(field.dtype):
                raise ValueError('initial state geometry differs from manifest')
        self.manifest = copy.deepcopy(manifest)
        self.base_full_generation = int(base_full_generation)
        self.base_state = {name: np.ascontiguousarray(value).copy()
                           for name, value in initial.items()}
        self.persisted = {name: value.copy()
                          for name, value in self.base_state.items()}
        self.current = {name: value.copy() for name, value in self.persisted.items()}
        self.persisted_generation = 0
        self._in_flight = None
        self._last_ack = None

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

    def _changed_records(self, changed_blocks=None):
        records = []
        block_count=sum(len(field.blocks) for field in self.manifest.fields)
        if changed_blocks is not None:
            changed_blocks=np.asarray(changed_blocks)
            if changed_blocks.dtype!=np.bool_ or changed_blocks.shape!=(block_count,):
                raise ValueError('R0 NPU changed bitmap geometry')
        cursor=0
        for field in self.manifest.fields:
            current = self.current[field.canonical_name].reshape(-1)
            persisted = self.persisted[field.canonical_name].reshape(-1)
            for block in field.blocks:
                start = block.element_offset
                end = start + block.element_count
                changed=current[start:end].tobytes() != persisted[start:end].tobytes()
                if changed_blocks is not None and bool(changed_blocks[cursor])!=changed:
                    raise ValueError('R0 NPU bitmap differs from exact byte oracle')
                cursor+=1
                if changed:
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

    def observe(self, step: int, generation: int, controls=(), changed_blocks=None) -> bytes:
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
            block_records=self._changed_records(changed_blocks),
            control_records=control_records)
        self._in_flight = {"generation": generation,
                           "checksum": hashlib.sha256(frame).hexdigest()}
        return frame

    def ack(self, frame: bytes):
        checksum = hashlib.sha256(frame).hexdigest()
        if self._last_ack is not None and checksum == self._last_ack[0]:
            return dict(self._last_ack[1])
        info = unpack_r0_frame(frame)
        if info["manifest_digest"] != self.manifest.digest:
            raise ValueError("R0 manifest digest mismatch")
        if self._in_flight is None:
            raise ValueError("R0 ACK has no in-flight generation")
        if checksum != self._in_flight['checksum']:
            raise ValueError("R0 ACK differs from pending frame")
        if info["generation"] != self._in_flight["generation"]:
            raise ValueError("R0 ACK generation mismatch")
        if info["base_delta_generation"] != self.persisted_generation:
            raise ValueError("R0 ACK base generation mismatch")
        updated = self._apply_validated(info, self.persisted)
        self.persisted = updated
        self.persisted_generation = info["generation"]
        self._in_flight = None
        result = {"generation": info["generation"], "step": info["step"],
                "blocks": len(info["blocks"]),
                "controls": len(info["controls"])}
        self._last_ack = (checksum, dict(result))
        return result

    def _apply_validated(self, info, state):
        if info['base_full_generation'] != self.base_full_generation:
            raise ValueError('R0 FULL root differs')
        if info['world_size'] != 1 or info['rank_id'] != 0:
            raise ValueError('R0 session requires single rank')
        by_index = {field.state_index: field for field in self.manifest.fields}
        validated = []; seen = set()
        for record in info['blocks']:
            field = by_index.get(record['state_index'])
            index = record['block_index']
            if field is None or field.canonical_name != record['name'] or type(index) is not int or not 0 <= index < len(field.blocks):
                raise ValueError('R0 block manifest mismatch')
            block = field.blocks[index]
            key = (field.state_index, index)
            if key in seen or record['element_offset'] != block.element_offset or record['element_count'] != block.element_count or np.dtype(record['dtype']) != np.dtype(field.dtype):
                raise ValueError('R0 block geometry/dtype mismatch')
            seen.add(key); validated.append((field, block, record['value']))
        updated = {name: value.copy() for name, value in state.items()}
        for field, block, value in validated:
            updated[field.canonical_name].reshape(-1)[block.element_offset:block.element_offset+block.element_count] = value
        return updated

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
            state = self._apply_validated(info, state)
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
