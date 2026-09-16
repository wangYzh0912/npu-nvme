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
from npu_nvme.runtime.scheduler import CheckpointScheduler
from npu_nvme.runtime.commit import LegacyCommitCoordinator


def manager(admission="try"):
    value = object.__new__(DirectCheckpoint)
    value._commit = LegacyCommitCoordinator(None)
    value.admission = admission
    value._scheduler = CheckpointScheduler(1, 1)
    value.checkpoint_slots = value.request_slots = 1
    value._closed = False
    value.metadata_generation = 0
    value._io_error = None
    value.wait_for_io_completion = lambda timeout=None: None
    return value


def test_try_admission_returns_explicit_busy():
    value = manager()
    value._admit_checkpoint()


def test_snapshot_admission_returns_explicit_busy_and_releases_request():
    value = manager()
    assert value._scheduler.snapshot_budget.acquire(blocking=False)
    with pytest.raises(CheckpointBusyError, match="snapshot"):
        value._admit_checkpoint()
    assert value._scheduler.request_budget.acquire(blocking=False)
    with pytest.raises(CheckpointBusyError):
        value._admit_checkpoint()
    value._scheduler.request_budget.release()
    value._scheduler.snapshot_budget.release()
    value._admit_checkpoint()


def test_poison_rejects_until_explicit_reset():
    value = manager()
    value._scheduler.poisoned = True
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
    value._scheduler.active_handles = {failed, pending}
    value._poison_checkpoint_queue(failed)
    assert pending.state == CheckpointState.CANCELLED
    assert pending.status == CheckpointHandle.CANCELLED
    with pytest.raises(CheckpointQueuePoisonedError):
        value._admit_checkpoint()
