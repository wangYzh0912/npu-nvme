import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from raw_ring import (KIND_DELTA, KIND_FULL, pack_ring_metadata,
                      pack_ring_slot, select_ab_metadata,
                      select_recovery_chain, unpack_ring_slot)


class RawRingTests(unittest.TestCase):
    def test_whole_frame_and_envelope_corruption_are_rejected(self):
        good = pack_ring_slot(b"header-and-payload", 7, 70, KIND_DELTA, 4096)
        self.assertEqual(unpack_ring_slot(good)["frame"], b"header-and-payload")
        for offset in (8, 64, 70):
            bad = bytearray(good)
            bad[offset] ^= 1
            with self.assertRaisesRegex(ValueError, "CRC"):
                unpack_ring_slot(bad)
        with self.assertRaisesRegex(ValueError, "torn"):
            unpack_ring_slot(good[:40])

    def test_ab_metadata_uses_latest_valid_copy(self):
        old = pack_ring_metadata(4, 12, 0, 8)
        new = pack_ring_metadata(5, 13, 0, 8)
        self.assertEqual(select_ab_metadata(old, new)[0], "B")
        torn = bytearray(new)
        torn[20] ^= 1
        self.assertEqual(select_ab_metadata(old, torn)[0], "A")
        with self.assertRaisesRegex(ValueError, "no valid"):
            select_ab_metadata(torn, torn)

    def test_recovery_selects_latest_full_contiguous_suffix(self):
        slots = [pack_ring_slot(b"old", 1, 1, KIND_FULL)]
        slots += [pack_ring_slot(str(step).encode(), step, step,
                                 KIND_FULL if step == 8 else KIND_DELTA)
                  for step in range(8, 13)]
        chain = select_recovery_chain(slots)
        self.assertEqual([item["step_id"] for item in chain],
                         [8, 9, 10, 11, 12])
        missing = [slot for slot in slots
                   if unpack_ring_slot(slot)["step_id"] != 10]
        with self.assertRaisesRegex(ValueError, "missing generation"):
            select_recovery_chain(missing)

    def test_duplicate_and_reordered_step_are_hard_failures(self):
        duplicate = [pack_ring_slot(b"full", 8, 8, KIND_FULL),
                     pack_ring_slot(b"a", 9, 9, KIND_DELTA),
                     pack_ring_slot(b"b", 9, 9, KIND_DELTA)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_recovery_chain(duplicate)
        reordered = [pack_ring_slot(b"full", 8, 8, KIND_FULL),
                     pack_ring_slot(b"bad-step", 9, 10, KIND_DELTA)]
        with self.assertRaisesRegex(ValueError, "step sequence"):
            select_recovery_chain(reordered)


if __name__ == "__main__":
    unittest.main()
