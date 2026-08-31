import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from direct_checkpoint import (CheckpointBusyError,
                               CheckpointHandle,
                               CheckpointQueuePoisonedError,
                               DirectCheckpoint)
from full_checkpoint_protocol import CheckpointState


def manager(admission="try"):
    value = object.__new__(DirectCheckpoint)
    value.admission = admission
    value._queue_poisoned = False
    value._slot_sem = threading.BoundedSemaphore(1)
    value._admission_lock = threading.Lock()
    value._handles_lock = threading.Lock()
    value._active_handles = set()
    value._handle_threads = {}
    value._io_order = threading.Condition()
    value._next_io_sequence = 1
    value._request_counter = 0
    value.metadata_generation = 0
    value._accepted_generation = 0
    value._io_error = None
    value.wait_for_io_completion = lambda timeout=None: None
    return value


def test_try_admission_returns_explicit_busy():
    value = manager()
    value._admit_checkpoint()
    with pytest.raises(CheckpointBusyError):
        value._admit_checkpoint()
    value._release_checkpoint_slot()
    value._admit_checkpoint()


def test_poison_rejects_until_explicit_reset():
    value = manager()
    value._queue_poisoned = True
    with pytest.raises(CheckpointQueuePoisonedError):
        value._admit_checkpoint()
    value.reset_checkpoint_queue()
    value._admit_checkpoint()


def test_first_failure_cancels_later_accepted_generation():
    value = manager()
    failed = CheckpointHandle(value, "failed", 1, 10)
    pending = CheckpointHandle(value, "pending", 2, 20)
    pending.transition(CheckpointState.SNAPSHOTTING)
    pending.transition(CheckpointState.SNAPSHOT_READY)
    pending.transition(CheckpointState.QUEUED)
    value._active_handles = {failed, pending}
    value._poison_checkpoint_queue(failed)
    assert pending.state == CheckpointState.CANCELLED
    assert pending.status == CheckpointHandle.CANCELLED
    with pytest.raises(CheckpointQueuePoisonedError):
        value._admit_checkpoint()
