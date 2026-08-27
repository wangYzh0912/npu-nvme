"""CPU policy oracle for selective S2 replacement checkpoints.

R2 uses implicit error feedback: the acknowledged reference stores the value
that a recovery process would actually decode.  Unselected updates and
quantization error therefore remain in ``current - decoded_persisted`` and
must not be added through a second residual vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _align(value, alignment=4096):
    return ((int(value) + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class PolicyBlock:
    block_id: int
    name: str
    offset: int
    count: int
    forced: bool = False


class S2SelectivePolicy:
    """Stateful R1/R2 oracle with max-one-unacknowledged semantics."""

    def __init__(self, initial: Mapping[str, np.ndarray], block_size: int,
                 selection_fraction: float, encoding: str = "int8",
                 max_age: int = 0, small_threshold: int = 10000,
                 selection_mode: str = "topk", score_threshold: float = 0.0):
        if block_size <= 0 or not 0 < selection_fraction <= 1:
            raise ValueError("invalid S2 policy dimensions")
        if encoding not in {"fp16", "int8", "raw"}:
            raise ValueError("unsupported S2 policy encoding")
        if max_age < 0 or small_threshold < 0:
            raise ValueError("age and small threshold must be non-negative")
        if selection_mode not in {"topk", "error_budget", "threshold"}:
            raise ValueError("unsupported S2 selection mode")
        if score_threshold < 0:
            raise ValueError("score threshold must be non-negative")
        self.block_size = int(block_size)
        self.selection_fraction = float(selection_fraction)
        self.encoding = encoding
        self.max_age = int(max_age)
        self.selection_mode = selection_mode
        self.score_threshold = float(score_threshold)
        self.reference = {name: np.array(value, copy=True)
                          for name, value in initial.items()}
        self.blocks = []
        block_id = 0
        for name in sorted(initial):
            value = np.asarray(initial[name])
            forced = value.size < small_threshold
            width = value.size if forced and value.size else self.block_size
            for offset in range(0, value.size, width):
                self.blocks.append(PolicyBlock(
                    block_id, name, offset,
                    min(width, value.size - offset), forced))
                block_id += 1
        self.age = np.zeros(len(self.blocks), dtype=np.int64)
        self.generation = 0
        self.pending = None

    @staticmethod
    def _encode(values, encoding):
        values = np.asarray(values)
        if encoding == "raw":
            return values.copy(), int(values.nbytes), None
        if encoding == "fp16":
            encoded = values.astype(np.float16)
            return encoded.astype(values.dtype), int(encoded.nbytes), None
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        scale = peak / 127.0 if peak else 1.0
        encoded = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
        decoded = (encoded.astype(np.float32) * scale).astype(values.dtype)
        return decoded, int(encoded.nbytes) + 4, scale

    def observe(self, current: Mapping[str, np.ndarray], step: int):
        if self.pending is not None:
            raise RuntimeError("one S2 generation is already awaiting ACK")
        if set(current) != set(self.reference):
            raise ValueError("S2 policy state field set changed")
        scores = np.empty(len(self.blocks), dtype=np.float64)
        for index, block in enumerate(self.blocks):
            now = np.asarray(current[block.name]).reshape(-1)
            old = self.reference[block.name].reshape(-1)
            if now.shape != old.shape or now.dtype != old.dtype:
                raise ValueError(f"S2 policy shape/dtype changed: {block.name}")
            delta = now[block.offset:block.offset + block.count].astype(
                np.float64) - old[block.offset:block.offset + block.count]
            scores[index] = np.linalg.norm(delta)

        selectable = [index for index, block in enumerate(self.blocks)
                      if not block.forced]
        ranked = sorted((index for index in selectable if scores[index] > 0),
                        key=lambda index: (-scores[index], index))
        if self.selection_mode == "topk":
            target = int(np.ceil(len(selectable) * self.selection_fraction))
            selected = set(ranked[:target])
        elif self.selection_mode == "threshold":
            selected = {index for index in selectable
                        if scores[index] >= self.score_threshold}
        else:
            target_energy = float(np.dot(scores[selectable], scores[selectable]))
            target_energy *= self.selection_fraction
            selected = set()
            accumulated = 0.0
            for index in ranked:
                if accumulated >= target_energy and selected:
                    break
                selected.add(index)
                accumulated += scores[index] ** 2
        selected.update(index for index, block in enumerate(self.blocks)
                        if block.forced)
        if self.max_age:
            selected.update(np.nonzero(self.age >= self.max_age - 1)[0].tolist())

        decoded = []
        payload_bytes = 0
        for index in sorted(selected):
            block = self.blocks[index]
            values = np.asarray(current[block.name]).reshape(-1)[
                block.offset:block.offset + block.count]
            restored, encoded_bytes, scale = self._encode(values, self.encoding)
            decoded.append((index, np.array(restored, copy=True), scale))
            payload_bytes += encoded_bytes
        descriptor_bytes = _align(max(1, len(decoded)) * 96)
        physical_bytes = _align(4096 + descriptor_bytes + payload_bytes)
        self.pending = {
            "step": int(step), "generation": self.generation + 1,
            "selected": sorted(selected), "decoded": decoded,
            "payload_bytes": int(payload_bytes),
            "descriptor_bytes": int(descriptor_bytes),
            "physical_bytes": int(physical_bytes),
            "scores": scores,
        }
        return self.pending

    def ack(self, generation: int):
        if self.pending is None or int(generation) != self.pending["generation"]:
            raise ValueError("stale or unknown S2 policy ACK")
        selected = set(self.pending["selected"])
        for index, values, _scale in self.pending["decoded"]:
            block = self.blocks[index]
            target = self.reference[block.name].reshape(-1)
            target[block.offset:block.offset + block.count] = values
        self.age += 1
        if selected:
            self.age[np.asarray(sorted(selected), dtype=np.int64)] = 0
        self.generation = self.pending["generation"]
        result = self.pending
        self.pending = None
        return result

    def fail(self, generation: int):
        if self.pending is None or int(generation) != self.pending["generation"]:
            raise ValueError("stale or unknown S2 policy failure")
        self.pending = None

    def reset_full(self, current: Mapping[str, np.ndarray]):
        if self.pending is not None:
            raise RuntimeError("cannot reset FULL while a Delta awaits ACK")
        self.reference = {name: np.array(value, copy=True)
                          for name, value in current.items()}
        self.age.fill(0)

    def relative_l2(self, current: Mapping[str, np.ndarray]):
        numerator = 0.0
        denominator = 0.0
        for name, value in current.items():
            actual = np.asarray(value).astype(np.float64)
            restored = self.reference[name].astype(np.float64)
            diff = actual - restored
            numerator += float(np.vdot(diff, diff))
            denominator += float(np.vdot(actual, actual))
        return float(np.sqrt(numerator / max(denominator, 1e-30)))


__all__ = ["PolicyBlock", "S2SelectivePolicy"]
