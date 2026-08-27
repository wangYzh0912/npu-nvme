import sys
import unittest

import numpy as np

sys.path.insert(0, "python")
from incremental_frame import (pack_r0_frame, unpack_r0_frame,
                               pack_r0_frame_prefix,
                               unpack_r0_frame_prefix)  # noqa: E402


class IncrementalFrameTests(unittest.TestCase):
    def frame(self):
        return pack_r0_frame(
            step=4, generation=5, base_full_generation=1,
            base_delta_generation=4, manifest_digest="11" * 32,
            block_records=[{
                "name": "model/x", "state_index": 2, "block_index": 1,
                "element_offset": 4, "element_count": 3,
                "dtype": "float16", "value": np.array([1, 2, 3], np.float16),
            }],
            control_records=[{"name": "data_cursor", "codec": "json-tagged-v1",
                              "payload": b"cursor"}],
        )

    def test_round_trip_native_value_and_control(self):
        info = unpack_r0_frame(self.frame())
        self.assertEqual(info["generation"], 5)
        self.assertEqual(info["manifest_digest"], "11" * 32)
        np.testing.assert_array_equal(info["blocks"][0]["value"], [1, 2, 3])
        self.assertEqual(info["controls"][0]["payload"], b"cursor")

    def test_corruption_is_rejected(self):
        frame = bytearray(self.frame())
        frame[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "payload CRC"):
            unpack_r0_frame(frame)

    def test_generation_lineage_is_rejected(self):
        with self.assertRaises(ValueError):
            pack_r0_frame(1, 2, 1, 2, "22" * 32, [])

    def test_large_descriptor_prefix_is_self_describing(self):
        blocks = [{
            "kind": "block", "name": f"model/p{i}", "state_index": i,
            "block_index": 0, "element_offset": 0, "element_count": 1,
            "dtype": "float32", "encoding": "raw", "payload_offset": i * 4096,
            "payload_bytes": 4, "crc32": i,
        } for i in range(300)]
        prefix = pack_r0_frame_prefix(5, 6, 1, 0, "33" * 32, blocks, [],
                                      300 * 4096, 0)
        info = unpack_r0_frame_prefix(prefix)
        self.assertEqual(len(info["blocks"]), 300)
        self.assertGreater(info["descriptor_bytes"], 4096)


if __name__ == "__main__":
    unittest.main()
