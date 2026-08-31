import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from full_checkpoint_protocol import (CheckpointState, can_transition,
                                      require_transition, validate_result_gate)


class FullCheckpointProtocolTests(unittest.TestCase):
    def test_happy_path_transitions(self):
        state = CheckpointState.CREATED
        for target in (CheckpointState.SNAPSHOTTING,
                       CheckpointState.SNAPSHOT_READY,
                       CheckpointState.QUEUED, CheckpointState.DMA_COPYING,
                       CheckpointState.NVME_WRITING, CheckpointState.FLUSHING,
                       CheckpointState.METADATA_COMMITTING,
                       CheckpointState.PERSISTED):
            self.assertTrue(can_transition(state, target))
            require_transition(state, target)
            state = target

    def test_metadata_cannot_precede_flush(self):
        with self.assertRaises(ValueError):
            require_transition(CheckpointState.NVME_WRITING,
                               CheckpointState.METADATA_COMMITTING)

    def test_none_baseline_is_not_restore_success(self):
        with self.assertRaises(ValueError):
            validate_result_gate({"mode": "none", "restore_verified": True})
        validate_result_gate({"mode": "none", "restore_verified": None})

    def test_full_success_requires_persistence_and_restore(self):
        with self.assertRaises(ValueError):
            validate_result_gate({"mode": "serial", "status": "pass",
                                  "request_id": 1, "generation": 1,
                                  "persisted": True, "restore_verified": False})
        validate_result_gate({"mode": "serial", "status": "pass",
                              "request_id": 1, "generation": 1,
                              "persisted": True, "restore_verified": True})


if __name__ == "__main__":
    unittest.main()
