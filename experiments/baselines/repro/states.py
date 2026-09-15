"""Lifecycle of the retained file/worker baseline adapters."""
from enum import Enum

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


