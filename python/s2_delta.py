"""S2/R0 replacement semantics and a small CPU correctness oracle.

S2 compares the current model against the last *acknowledged* persisted
reference.  A frame contains replacement values (not deltas), and the
reference advances only after the frame is acknowledged.  Blocks are local
to a parameter; no block may straddle two parameters.

This module is intentionally independent of MindSpore and SPDK.  It is the
executable oracle for I0/I1 and the reference implementation used by later
frame and device tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from delta_protocol import (pack_s2_replacement_frame,
                            unpack_s2_replacement_frame)


def _as_array(value) -> np.ndarray:
    return np.asarray(value)


def _dtype_name(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype.fields is not None:
        raise ValueError(f"structured dtype is not supported: {dtype}")
    return dtype.name


@dataclass(frozen=True)
class Block:
    """Stable, parameter-local block descriptor."""

    block_id: int
    name: str
    layer_id: int
    block_idx: int
    element_offset: int
    element_count: int
    dtype: str

    def as_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "name": self.name,
            "layer_id": self.layer_id,
            "block_idx": self.block_idx,
            "element_offset": self.element_offset,
            "element_count": self.element_count,
            "dtype": self.dtype,
        }


def _layer_id(name: str) -> int:
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "blocks" and parts[index + 1].isdigit():
            return int(parts[index + 1])
    if "embedding" in name:
        return -2
    if "layernorm" in name or "layer_norm" in name:
        return -1
    return -3


def build_block_manifest(params: Mapping[str, np.ndarray],
                         block_size: int,
                         small_threshold: int = 10000) -> dict:
    """Build a deterministic manifest with parameter-local block offsets."""
    if block_size <= 0 or small_threshold < 0:
        raise ValueError("block_size must be positive and threshold non-negative")

    blocks: List[Block] = []
    small: List[dict] = []
    block_id = 0
    # Sorting names makes the manifest stable across dict construction order.
    for name in sorted(params):
        value = _as_array(params[name])
        count = int(value.size)
        dtype = _dtype_name(value.dtype)
        if count < small_threshold:
            small.append({"name": name, "layer_id": _layer_id(name),
                          "element_count": count, "dtype": dtype})
            continue
        for block_idx, start in enumerate(range(0, count, block_size)):
            element_count = min(block_size, count - start)
            blocks.append(Block(block_id, name, _layer_id(name), block_idx,
                                start, element_count, dtype))
            block_id += 1

    payload = {"block_size": int(block_size),
               "small_threshold": int(small_threshold),
               "blocks": [block.as_dict() for block in blocks],
               "small": small}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def score_manifest_blocks(current: Mapping[str, np.ndarray],
                          reference: Mapping[str, np.ndarray],
                          manifest: Mapping) -> List[dict]:
    """Score manifest blocks with one FP64 conversion per parameter.

    The original real-trajectory collector converted the same parameter
    slice twice for every block.  At GPT-2 XL scale that made a two-step
    correctness smoke spend minutes in Python and grow to ~80 GiB RSS.  This
    implementation retains FP64 norm accumulation while grouping contiguous
    parameter-local blocks into vectorized reductions.
    """
    grouped: Dict[str, List[dict]] = {}
    for item in manifest["blocks"]:
        grouped.setdefault(item["name"], []).append(item)
    output = []
    for name, items in grouped.items():
        if name not in current or name not in reference:
            raise ValueError(f"manifest parameter missing from state: {name}")
        a = np.asarray(current[name]).reshape(-1)
        b = np.asarray(reference[name]).reshape(-1)
        if a.shape != b.shape:
            raise ValueError(f"state shape mismatch: {name}")
        diff = np.subtract(a, b, dtype=np.float64)
        block_size = int(manifest["block_size"])
        full_count = diff.size // block_size
        full_scores = np.empty(0, dtype=np.float64)
        full_nonzero = np.empty(0, dtype=np.int64)
        if full_count:
            full = diff[:full_count * block_size].reshape(full_count,
                                                          block_size)
            full_scores = np.sqrt(np.einsum("ij,ij->i", full, full))
            full_nonzero = np.count_nonzero(full, axis=1)
        for item in items:
            block_idx = int(item["block_idx"])
            count = int(item["element_count"])
            if count == block_size and block_idx < full_count:
                score = float(full_scores[block_idx])
                nonzero = int(full_nonzero[block_idx])
            else:
                start = int(item["element_offset"])
                tail = diff[start:start + count]
                score = float(np.linalg.norm(tail))
                nonzero = int(np.count_nonzero(tail))
            output.append({**item, "score": score, "nonzero": nonzero})
        del diff
    output.sort(key=lambda item: (-item["score"], int(item["block_id"])))
    return output


def _copy_state(state: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True) for name, value in state.items()}


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(a, b, equal_nan=True)


def apply_s2_replacements(state: Mapping[str, np.ndarray],
                          block_patches: Sequence[Mapping],
                          small_patches: Sequence[Mapping],
                          manifest: Mapping) -> Dict[str, np.ndarray]:
    """Apply replacement records to a copied state with strict bounds."""
    result = _copy_state(state)
    manifest_blocks = {b.block_id: b for b in (
        Block(**item) for item in manifest["blocks"])}
    manifest_small = {item["name"]: item for item in manifest["small"]}
    seen = set()
    for patch in block_patches:
        block_id = int(patch["block_id"])
        if block_id in seen or block_id not in manifest_blocks:
            raise ValueError("unknown or duplicate S2 block")
        seen.add(block_id)
        block = manifest_blocks[block_id]
        if patch["name"] != block.name or int(patch["block_idx"]) != block.block_idx:
            raise ValueError("S2 block does not match manifest")
        value = np.asarray(patch["value"], dtype=np.dtype(block.dtype)).reshape(-1)
        if value.size != block.element_count:
            raise ValueError("S2 block element count mismatch")
        flat = result[block.name].reshape(-1)
        end = block.element_offset + block.element_count
        if end > flat.size:
            raise ValueError("S2 block exceeds parameter")
        flat[block.element_offset:end] = value

    seen_small = set()
    for patch in small_patches:
        name = patch["name"]
        if name not in manifest_small or name in seen_small:
            raise ValueError("unknown or duplicate S2 small parameter")
        seen_small.add(name)
        value = np.asarray(patch["value"], dtype=np.dtype(manifest_small[name]["dtype"]))
        if value.size != int(manifest_small[name]["element_count"]):
            raise ValueError("S2 small parameter element count mismatch")
        result[name] = value.reshape(result[name].shape).copy()
    return result


class S2DeltaOracle:
    """Reference-state oracle for replacement frames.

    ``observe`` never mutates ``persisted_reference``.  ``ack`` is the only
    operation that advances it, which makes an unacknowledged frame safe to
    retry and makes stale/out-of-order acknowledgements explicit failures.
    """

    def __init__(self, initial: Mapping[str, np.ndarray], block_size: int,
                 small_threshold: int = 10000, change_epsilon: float = 0.0,
                 top_k: Optional[int] = None):
        self.current = _copy_state(initial)
        self.persisted_reference = _copy_state(initial)
        self.manifest = build_block_manifest(self.current, block_size,
                                             small_threshold)
        self.change_epsilon = float(change_epsilon)
        self.top_k = top_k
        self.persisted_generation = 0

    @property
    def manifest_digest(self) -> str:
        return self.manifest["digest"]

    def set_current(self, state: Mapping[str, np.ndarray]) -> None:
        if set(state) != set(self.current):
            raise ValueError("parameter set changed after manifest creation")
        for name in self.current:
            value = _as_array(state[name])
            if value.shape != self.current[name].shape or value.dtype != self.current[name].dtype:
                raise ValueError(f"parameter shape/dtype changed: {name}")
        self.current = _copy_state(state)

    def _changed_blocks(self) -> List[Tuple[float, dict]]:
        changed = [(item["score"], item)
                   for item in score_manifest_blocks(
                       self.current, self.persisted_reference, self.manifest)
                   if item["score"] > self.change_epsilon]
        if self.top_k is not None:
            if self.top_k < 0:
                raise ValueError("top_k must be non-negative")
            changed = changed[:self.top_k]
        return changed

    def observe(self, step_id: int, generation: Optional[int] = None) -> bytes:
        """Create an S2 replacement frame relative to the last ACK."""
        if step_id < 0:
            raise ValueError("step_id must be non-negative")
        for name, value in self.current.items():
            if np.issubdtype(value.dtype, np.inexact) and not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite parameter value: {name}")
        if generation is None:
            generation = self.persisted_generation + 1
        if generation <= self.persisted_generation:
            raise ValueError("generation must advance")

        block_patches = []
        for _score, item in self._changed_blocks():
            start = int(item["element_offset"])
            count = int(item["element_count"])
            value = self.current[item["name"]].reshape(-1)[start:start + count]
            block_patches.append({**item, "value": np.array(value, copy=True)})

        small_patches = []
        for item in self.manifest["small"]:
            name = item["name"]
            if not _same(self.current[name], self.persisted_reference[name]):
                small_patches.append({**item, "value": np.array(self.current[name], copy=True)})

        return pack_s2_replacement_frame(
            step_id, block_patches, small_patches,
            base_generation=self.persisted_generation,
            generation=generation,
            manifest_digest=self.manifest_digest)

    def ack(self, frame: bytes) -> dict:
        """Commit a frame to the reference after validating its lineage."""
        step, blocks, smalls, info = unpack_s2_replacement_frame(frame)
        if info["manifest_digest"] != self.manifest_digest:
            raise ValueError("S2 manifest digest mismatch")
        if info["base_generation"] != self.persisted_generation:
            raise ValueError("stale or out-of-order S2 acknowledgement")
        self.persisted_reference = apply_s2_replacements(
            self.persisted_reference, blocks, smalls, self.manifest)
        self.persisted_generation = info["generation"]
        return {"step_id": step, "generation": self.persisted_generation,
                "blocks": len(blocks), "small": len(smalls)}

    def recover(self, initial: Mapping[str, np.ndarray],
                frames: Iterable[bytes], base_generation: int = 0) -> dict:
        """Replay a complete ordered chain into an independent state."""
        state = _copy_state(initial)
        generation = int(base_generation)
        last_step = None
        for frame in frames:
            step, blocks, smalls, info = unpack_s2_replacement_frame(frame)
            if info["manifest_digest"] != self.manifest_digest:
                raise ValueError("S2 manifest digest mismatch")
            if info["base_generation"] != generation:
                raise ValueError("S2 recovery generation gap or reordering")
            state = apply_s2_replacements(state, blocks, smalls, self.manifest)
            generation = info["generation"]
            last_step = step
        return {"state": state, "generation": generation, "last_step": last_step}


class FileS2Ring:
    """Atomic ordinary-file ring used by the I4 cross-process gate.

    The frame itself is the source of truth.  A slot is replaced using
    ``os.replace`` only after the complete frame has been flushed and fsync'd,
    so a reader sees either the previous complete frame or the new one.
    """

    def __init__(self, directory, slot_count=128, slot_size=256 * 1024 * 1024):
        if slot_count <= 0 or slot_size < 4096:
            raise ValueError("invalid S2 ring dimensions")
        self.directory = os.fspath(directory)
        self.slot_count = int(slot_count)
        self.slot_size = int(slot_size)
        os.makedirs(self.directory, exist_ok=True)
        self.next_slot = 0

    def _path(self, slot):
        if slot < 0 or slot >= self.slot_count:
            raise ValueError("S2 ring slot out of range")
        return os.path.join(self.directory, f"s2_slot_{slot:04d}.bin")

    def write(self, frame):
        """Atomically write a complete frame and return its slot index."""
        _step, _blocks, _smalls, _info = unpack_s2_replacement_frame(frame)
        if len(frame) > self.slot_size:
            raise ValueError("S2 frame exceeds ring slot")
        slot = self.next_slot % self.slot_count
        path = self._path(slot)
        temporary = f"{path}.tmp.{os.getpid()}"
        with open(temporary, "wb") as stream:
            stream.write(frame)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        self.next_slot += 1
        return slot

    def read(self, slot):
        """Read and validate one complete frame from a slot."""
        with open(self._path(int(slot)), "rb") as stream:
            frame = stream.read()
        unpack_s2_replacement_frame(frame)
        return frame


__all__ = ["Block", "FileS2Ring", "S2DeltaOracle",
           "apply_s2_replacements", "build_block_manifest",
           "score_manifest_blocks"]
