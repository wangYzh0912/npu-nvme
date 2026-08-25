import os
import struct
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from disk_layout import (  # noqa: E402
    META_SLOT_BYTES,
    make_layout,
    pack_metadata,
    pack_superblock,
    unpack_metadata,
    unpack_superblock,
)


class DiskLayoutTests(unittest.TestCase):
    def test_v2_layout_round_trip_has_disjoint_regions(self):
        layout = make_layout(
            total_bytes=1024 * 1024 * 1024,
            full_slot_bytes=16 * 1024 * 1024,
            full_slot_count=6,
            delta_slot_bytes=4 * 1024 * 1024,
            delta_slot_count=8,
            generation=7,
            active_meta_slot=1,
        )
        self.assertLessEqual(layout.full_end, layout.delta_base)
        self.assertEqual(layout.full_slot_offset(1, 4, 3), layout.full_base + 4 * layout.full_slot_bytes)
        self.assertEqual(layout.delta_slot_offset(7), layout.delta_base + 7 * layout.delta_slot_bytes)
        self.assertEqual(unpack_superblock(pack_superblock(layout)), layout)

    def test_layout_rejects_full_delta_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap|capacity"):
            make_layout(
                total_bytes=64 * 1024 * 1024,
                full_slot_bytes=16 * 1024 * 1024,
                full_slot_count=4,
                delta_slot_bytes=8 * 1024 * 1024,
                delta_slot_count=2,
            )

    def test_metadata_round_trip_and_crc_rejection(self):
        payload = {"checkpoints": {"step_1": {"type": "FULL"}}, "delta_head": 3}
        raw = pack_metadata(payload, generation=9)
        self.assertEqual(len(raw), META_SLOT_BYTES)
        generation, decoded = unpack_metadata(raw)
        self.assertEqual(generation, 9)
        self.assertEqual(decoded, payload)

        corrupted = bytearray(raw)
        corrupted[struct.calcsize("<8sIIQQII")] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            unpack_metadata(corrupted)

    def test_superblock_crc_rejection(self):
        layout = make_layout(
            total_bytes=256 * 1024 * 1024,
            full_slot_bytes=8 * 1024 * 1024,
            full_slot_count=3,
            delta_slot_bytes=4 * 1024 * 1024,
            delta_slot_count=4,
        )
        raw = bytearray(pack_superblock(layout))
        raw[32] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            unpack_superblock(raw)


if __name__ == "__main__":
    unittest.main()
