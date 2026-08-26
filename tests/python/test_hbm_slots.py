import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from hbm_slots import SlotLedger, SlotState


class HbmSlotLedgerTests(unittest.TestCase):
    def test_normal_lifecycle_and_reuse_after_persist(self):
        ledger = SlotLedger(1)
        ledger.transition(0, SlotState.SNAPSHOT, generation=1, step_id=5,
                         request_id="r/5")
        ledger.transition(0, SlotState.READY)
        ledger.transition(0, SlotState.IO)
        with self.assertRaisesRegex(RuntimeError, "not allowed"):
            ledger.transition(0, SlotState.SNAPSHOT, generation=2, step_id=10,
                              request_id="r/10")
        ledger.transition(0, SlotState.PERSISTED)
        ledger.transition(0, SlotState.FREE)
        ledger.transition(0, SlotState.SNAPSHOT, generation=2, step_id=10,
                         request_id="r/10")

    def test_failed_slot_is_reusable_but_unacknowledged_slot_is_not(self):
        ledger = SlotLedger(2)
        ledger.transition(0, SlotState.SNAPSHOT, generation=1, step_id=1,
                         request_id="r/1")
        ledger.transition(0, SlotState.READY)
        ledger.transition(0, SlotState.FAILED, error="timeout")
        self.assertEqual(ledger.slots[0].error, "timeout")
        ledger.transition(0, SlotState.FREE)
        self.assertEqual(ledger.free_ids(), [0, 1])
        ledger.transition(1, SlotState.SNAPSHOT, generation=1, step_id=1,
                         request_id="r/1")
        with self.assertRaisesRegex(RuntimeError, "not allowed"):
            ledger.transition(1, SlotState.FREE)
        ledger.assert_no_reuse_before_ack()


if __name__ == "__main__":
    unittest.main()
