"""Generation-tagged frame buffers with explicit persistence ownership.

The pool is intentionally independent of SPDK and MindSpore.  It is the
control-plane contract used by I3: a producer owns a FILLING buffer, the
writer owns a READY/WRITING buffer, and reuse is legal only after ACK or an
explicit failure transition.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from enum import Enum


class FrameState(str, Enum):
    FREE = "FREE"
    FILLING = "FILLING"
    READY = "READY"
    WRITING = "WRITING"
    PERSISTED = "PERSISTED"
    FAILED = "FAILED"


@dataclass
class FrameRecord:
    slot_id: int
    state: FrameState = FrameState.FREE
    generation: int = 0
    step_id: int = -1
    payload: bytes = b""
    checksum: str = ""
    error: str = ""
    source_kind: str = ""
    device_ptr: int = 0
    device_segments: tuple = ()
    capacity: int = 0
    valid_bytes: int = 0
    event_token: str = ""


class FrameBufferPool:
    """Bounded immutable-payload pool for one producer/writer contract."""

    def __init__(self, slot_count=2):
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        self._records = [FrameRecord(index) for index in range(slot_count)]
        self._condition = threading.Condition()
        self._last_generation = 0

    @property
    def records(self):
        with self._condition:
            return [FrameRecord(**record.__dict__) for record in self._records]

    def acquire(self, generation, step_id, timeout=None):
        generation = int(generation)
        if generation <= self._last_generation:
            raise ValueError("generation must increase")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                record = next((item for item in self._records
                               if item.state == FrameState.FREE), None)
                if record is not None:
                    record.state = FrameState.FILLING
                    record.generation = generation
                    record.step_id = int(step_id)
                    record.payload = b""
                    record.checksum = ""
                    record.error = ""
                    record.source_kind = ""
                    record.device_ptr = 0
                    record.device_segments = ()
                    record.capacity = 0
                    record.valid_bytes = 0
                    record.event_token = ""
                    self._last_generation = generation
                    return record.slot_id
                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("no reusable frame slot before deadline")
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    def publish(self, slot_id, payload):
        with self._condition:
            record = self._record(slot_id)
            self._require(record, FrameState.FILLING)
            # bytes makes the writer's view immutable even if the producer's
            # source was a mutable bytearray or a recycled NPU staging buffer.
            record.payload = bytes(payload)
            record.checksum = hashlib.sha256(record.payload).hexdigest()
            record.source_kind = "HOST_BYTES"
            record.capacity = len(record.payload)
            record.valid_bytes = len(record.payload)
            record.state = FrameState.READY
            self._condition.notify_all()

    def publish_hbm(self, slot_id, segments, valid_bytes, checksum,
                    event_token=""):
        """Publish an immutable, generation-owned HBM segment descriptor."""
        normalized = tuple((int(pointer), int(size))
                           for pointer, size in segments)
        if not normalized or any(pointer <= 0 or size <= 0
                                 for pointer, size in normalized):
            raise ValueError("HBM publication requires positive segments")
        capacity = sum(size for _pointer, size in normalized)
        if valid_bytes <= 0 or valid_bytes > capacity:
            raise ValueError("invalid HBM valid-byte count")
        if not checksum:
            raise ValueError("HBM publication requires a checksum")
        with self._condition:
            record = self._record(slot_id)
            self._require(record, FrameState.FILLING)
            record.payload = b""
            record.checksum = str(checksum)
            record.source_kind = "HBM_SEGMENTS"
            record.device_ptr = normalized[0][0]
            record.device_segments = normalized
            record.capacity = capacity
            record.valid_bytes = int(valid_bytes)
            record.event_token = str(event_token)
            record.state = FrameState.READY
            self._condition.notify_all()

    def begin_write(self, slot_id):
        with self._condition:
            record = self._record(slot_id)
            self._require(record, FrameState.READY)
            if record.source_kind != "HOST_BYTES":
                raise RuntimeError("HBM descriptor requires begin_hbm_write")
            record.state = FrameState.WRITING
            return record.payload, record.checksum

    def begin_hbm_write(self, slot_id):
        with self._condition:
            record = self._record(slot_id)
            self._require(record, FrameState.READY)
            if record.source_kind != "HBM_SEGMENTS":
                raise RuntimeError("host payload requires begin_write")
            record.state = FrameState.WRITING
            return FrameRecord(**record.__dict__)

    def ack(self, slot_id, checksum):
        with self._condition:
            record = self._record(slot_id)
            self._require(record, FrameState.WRITING)
            if checksum != record.checksum:
                raise ValueError("frame checksum does not match ACK")
            record.state = FrameState.PERSISTED
            record.state = FrameState.FREE
            self._condition.notify_all()

    def fail(self, slot_id, error):
        with self._condition:
            record = self._record(slot_id)
            if record.state not in (FrameState.FILLING, FrameState.READY,
                                    FrameState.WRITING):
                raise RuntimeError(f"cannot fail frame in {record.state}")
            record.error = repr(error)
            record.state = FrameState.FAILED
            record.state = FrameState.FREE
            self._condition.notify_all()

    def _record(self, slot_id):
        if not 0 <= int(slot_id) < len(self._records):
            raise IndexError("frame slot out of range")
        return self._records[int(slot_id)]

    @staticmethod
    def _require(record, expected):
        if record.state != expected:
            raise RuntimeError(
                f"slot {record.slot_id} is {record.state}, expected {expected}")


__all__ = ["FrameBufferPool", "FrameRecord", "FrameState"]
