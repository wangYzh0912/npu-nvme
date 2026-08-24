import os
import struct
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from delta_protocol import (  # noqa: E402
    FileDeltaWriter,
    apply_delta_patches,
    pack_delta_frame,
    unpack_delta_frame,
)
from disk_layout import FRAME_HEADER_SIZE  # noqa: E402


class DeltaProtocolTests(unittest.TestCase):
    def setUp(self):
        self.blocks = [{
            "layer_id": 2,
            "name": "encoder.weight",
            "block_idx": 1,
            "scale": 0.25,
            "int8_data": np.array([-4, 0, 7, 12], dtype=np.int8),
        }]
        self.smalls = [{
            "layer_id": 3,
            "name": "layer_norm.bias",
            "scale": 0.5,
            "int8_data": np.array([2, -2], dtype=np.int8),
        }]

    def test_round_trip_preserves_records(self):
        frame = pack_delta_frame(17, self.blocks, self.smalls)
        step, blocks, smalls = unpack_delta_frame(frame)

        self.assertEqual(step, 17)
        self.assertEqual(blocks[0]["name"], "encoder.weight")
        self.assertEqual(blocks[0]["block_idx"], 1)
        np.testing.assert_array_equal(
            blocks[0]["int8_data"], self.blocks[0]["int8_data"])
        np.testing.assert_array_equal(
            smalls[0]["int8_data"], self.smalls[0]["int8_data"])

    def test_corrupted_payload_is_rejected(self):
        frame = bytearray(pack_delta_frame(17, self.blocks, self.smalls))
        frame[-1] ^= 0x01
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            unpack_delta_frame(frame)

    def test_truncated_frame_is_rejected(self):
        frame = pack_delta_frame(17, self.blocks, self.smalls)
        with self.assertRaisesRegex(ValueError, "frame size"):
            unpack_delta_frame(frame[:-1])

    def test_unparsed_payload_is_rejected_even_with_valid_checksum(self):
        frame = bytearray(pack_delta_frame(17, self.blocks, self.smalls))
        frame.append(0x00)
        total_size = len(frame)
        struct.pack_into("<I", frame, 16, total_size)
        payload = frame[FRAME_HEADER_SIZE:]
        struct.pack_into("<I", frame, 20, sum(payload) & 0xFFFFFFFF)

        with self.assertRaisesRegex(ValueError, "unparsed payload"):
            unpack_delta_frame(frame)

    def test_apply_delta_patches_updates_expected_ranges(self):
        initial = {
            "encoder.weight": np.zeros((3, 4), dtype=np.float32),
            "layer_norm.bias": np.zeros((2,), dtype=np.float32),
        }
        updated = apply_delta_patches(
            initial, self.blocks, self.smalls, block_size=4)

        np.testing.assert_allclose(
            updated["encoder.weight"].reshape(-1)[4:8],
            np.array([-1.0, 0.0, 1.75, 3.0], dtype=np.float32))
        np.testing.assert_allclose(
            updated["layer_norm.bias"],
            np.array([1.0, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(
            initial["encoder.weight"], np.zeros((3, 4), dtype=np.float32))

    def test_file_writer_rotates_slots_and_reads_latest_contents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = FileDeltaWriter(
                temp_dir, delta_slot_count=2, delta_slot_size=1024 * 1024)
            self.assertEqual(writer.write_frame(1, self.blocks, []), 0)
            self.assertEqual(writer.write_frame(2, self.blocks, []), 1)
            self.assertEqual(writer.write_frame(3, self.blocks, []), 0)

            step, _, _ = writer.read_frame(0)
            self.assertEqual(step, 3)
            self.assertEqual(writer.stats["total_frames"], 3)


if __name__ == "__main__":
    unittest.main()
