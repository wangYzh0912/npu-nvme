"""Small, dependency-free lifecycle ledger for HBM snapshot slots.

The ledger is intentionally separate from MindSpore and SPDK.  It makes the
ownership rule executable and testable: a slot can only be reused after an
explicit persisted or failed transition, and every transition carries the
request generation that owns the frozen buffer.
"""

from dataclasses import dataclass
from enum import Enum


class SlotState(str, Enum):
    FREE = "FREE"
    SNAPSHOT = "SNAPSHOT"
    READY = "READY"
    IO = "IO"
    PERSISTED = "PERSISTED"
    FAILED = "FAILED"


@dataclass
class SlotRecord:
    slot_id: int
    state: SlotState = SlotState.FREE
    generation: int = 0
    step_id: int = -1
    request_id: str = ""
    error: str = ""


class SlotLedger:
    """Finite-state ledger used by the HBM snapshot runner."""

    _ALLOWED = {
        SlotState.FREE: {SlotState.SNAPSHOT},
        SlotState.SNAPSHOT: {SlotState.READY, SlotState.FAILED},
        SlotState.READY: {SlotState.IO, SlotState.FAILED},
        SlotState.IO: {SlotState.PERSISTED, SlotState.FAILED},
        SlotState.PERSISTED: {SlotState.FREE},
        SlotState.FAILED: {SlotState.FREE},
    }

    def __init__(self, count: int):
        if count <= 0:
            raise ValueError("slot count must be positive")
        self.slots = [SlotRecord(slot_id=i) for i in range(count)]

    def transition(self, slot_id: int, target: SlotState, *, generation=None,
                   step_id=None, request_id=None, error=None) -> SlotRecord:
        record = self.slots[int(slot_id)]
        if target not in self._ALLOWED[record.state]:
            raise RuntimeError(
                f"slot {record.slot_id}: {record.state.value} -> {target.value} "
                "is not allowed")
        if target is SlotState.SNAPSHOT:
            if generation is None or step_id is None or not request_id:
                raise ValueError("snapshot requires generation, step and request")
            if generation <= record.generation:
                raise RuntimeError("slot generation must advance before reuse")
            record.generation = int(generation)
            record.step_id = int(step_id)
            record.request_id = str(request_id)
            record.error = ""
        elif target is SlotState.FAILED:
            record.error = str(error or "unspecified slot failure")
        record.state = target
        if target is SlotState.FREE:
            record.step_id = -1
            record.request_id = ""
            record.error = ""
        return record

    def free_ids(self):
        return [record.slot_id for record in self.slots
                if record.state is SlotState.FREE]

    def assert_no_reuse_before_ack(self):
        for record in self.slots:
            if record.state in (SlotState.SNAPSHOT, SlotState.READY, SlotState.IO):
                if record.generation <= 0 or record.step_id < 0 or not record.request_id:
                    raise AssertionError(f"incomplete ownership metadata: {record}")


__all__ = ["SlotLedger", "SlotRecord", "SlotState"]
