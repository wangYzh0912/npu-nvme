"""Shared FULL-checkpoint request vocabulary and result validation helpers."""

from __future__ import annotations

from enum import Enum
from typing import Mapping


class CheckpointState(str, Enum):
    CREATED = "CREATED"
    SNAPSHOTTING = "SNAPSHOTTING"
    SNAPSHOT_READY = "SNAPSHOT_READY"
    QUEUED = "QUEUED"
    DMA_COPYING = "DMA_COPYING"
    NVME_WRITING = "NVME_WRITING"
    FLUSHING = "FLUSHING"
    METADATA_COMMITTING = "METADATA_COMMITTING"
    PERSISTED = "PERSISTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


TERMINAL_STATES = frozenset({
    CheckpointState.PERSISTED, CheckpointState.FAILED,
    CheckpointState.CANCELLED, CheckpointState.TIMED_OUT,
})

_TRANSITIONS = {
    CheckpointState.CREATED: {CheckpointState.SNAPSHOTTING,
                              CheckpointState.FAILED,
                              CheckpointState.CANCELLED,
                              CheckpointState.TIMED_OUT},
    CheckpointState.SNAPSHOTTING: {CheckpointState.SNAPSHOT_READY,
                                   CheckpointState.FAILED,
                                   CheckpointState.CANCELLED,
                                   CheckpointState.TIMED_OUT},
    CheckpointState.SNAPSHOT_READY: {CheckpointState.QUEUED,
                                     CheckpointState.FAILED,
                                     CheckpointState.CANCELLED,
                                     CheckpointState.TIMED_OUT},
    CheckpointState.QUEUED: {CheckpointState.DMA_COPYING,
                             CheckpointState.FAILED,
                             CheckpointState.CANCELLED,
                             CheckpointState.TIMED_OUT},
    CheckpointState.DMA_COPYING: {CheckpointState.NVME_WRITING,
                                  CheckpointState.FAILED,
                                  CheckpointState.CANCELLED,
                                  CheckpointState.TIMED_OUT},
    CheckpointState.NVME_WRITING: {CheckpointState.FLUSHING,
                                   CheckpointState.FAILED,
                                   CheckpointState.CANCELLED,
                                   CheckpointState.TIMED_OUT},
    CheckpointState.FLUSHING: {CheckpointState.METADATA_COMMITTING,
                               CheckpointState.FAILED,
                               CheckpointState.CANCELLED,
                               CheckpointState.TIMED_OUT},
    CheckpointState.METADATA_COMMITTING: {CheckpointState.PERSISTED,
                                          CheckpointState.FAILED,
                                          CheckpointState.CANCELLED,
                                          CheckpointState.TIMED_OUT},
}


def can_transition(current: CheckpointState, target: CheckpointState) -> bool:
    return target in _TRANSITIONS.get(CheckpointState(current), set())


def require_transition(current: CheckpointState, target: CheckpointState) -> None:
    if not can_transition(current, target):
        raise ValueError(f"invalid checkpoint transition {current}->{target}")


def validate_result_gate(result: Mapping) -> None:
    """Reject a result that claims success without durable restore evidence."""
    if result.get("mode") == "none":
        if result.get("restore_verified") is True:
            raise ValueError("none baseline cannot claim restore_verified")
        return
    required = ("request_id", "generation", "persisted", "restore_verified")
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(f"result missing FULL gate fields: {missing}")
    if result.get("status") == "pass" and not all(
            bool(result.get(name)) for name in ("persisted", "restore_verified")):
        raise ValueError("successful FULL result lacks persistence/restore proof")


__all__ = ["CheckpointState", "TERMINAL_STATES", "can_transition",
           "require_transition", "validate_result_gate"]
