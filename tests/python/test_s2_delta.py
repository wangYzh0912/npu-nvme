import os
import sys
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from delta_protocol import (pack_s2_replacement_frame,
                            unpack_delta_frame_with_meta,
                            unpack_s2_replacement_frame)
from s2_delta import S2DeltaOracle, apply_s2_replacements, build_block_manifest


class S2DeltaOracleTests(unittest.TestCase):
    def setUp(self):
        self.initial = {
            "backbone.blocks.0.weight": np.arange(9, dtype=np.float32),
            "backbone.blocks.1.weight": (np.arange(5, dtype=np.float16) + 10),
            "backbone.layernorm.bias": np.array([1, 2, 3], dtype=np.float32),
        }

    def test_z0_no_change_emits_empty_frame(self):
        oracle = S2DeltaOracle(self.initial, block_size=4, small_threshold=4)
        frame = oracle.observe(0)
        step, blocks, smalls, info = unpack_s2_replacement_frame(frame)
        self.assertEqual((step, len(blocks), len(smalls)), (0, 0, 0))
        self.assertEqual(info["base_generation"], 0)
        oracle.ack(frame)

    def test_z1_single_block_and_z2_repeated_small_change(self):
        oracle = S2DeltaOracle(self.initial, block_size=4, small_threshold=4)
        current = {k: v.copy() for k, v in self.initial.items()}
        current["backbone.blocks.0.weight"][1] = 99
        oracle.set_current(current)
        frame = oracle.observe(1)
        _, blocks, smalls, _ = unpack_s2_replacement_frame(frame)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["element_offset"], 0)
        self.assertEqual(len(smalls), 0)
        oracle.ack(frame)
        current["backbone.layernorm.bias"][0] += 0.5
        oracle.set_current(current)
        frame = oracle.observe(2)
        _, blocks, smalls, _ = unpack_s2_replacement_frame(frame)
        self.assertEqual(len(blocks), 0)
        self.assertEqual(len(smalls), 1)

    def test_z3_dense_and_z4_cold_hot_top_k_are_parameter_local(self):
        oracle = S2DeltaOracle(self.initial, block_size=4, small_threshold=4,
                               top_k=2)
        current = {k: v.copy() for k, v in self.initial.items()}
        current["backbone.blocks.0.weight"] += 1
        current["backbone.blocks.1.weight"] += 100
        oracle.set_current(current)
        frame = oracle.observe(3)
        _, blocks, _, _ = unpack_s2_replacement_frame(frame)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(item["name"] in self.initial for item in blocks))
        # A block index/offset is local to its own parameter, including the
        # tail block of the second parameter.
        self.assertTrue(any(item["name"] == "backbone.blocks.1.weight" and
                            item["element_offset"] == 4 for item in blocks))

    def test_z5_rotation_z6_burst_and_z7_cancellation(self):
        oracle = S2DeltaOracle(self.initial, block_size=4, small_threshold=4)
        current = {k: v.copy() for k, v in self.initial.items()}
        for step in range(5, 8):
            current["backbone.blocks.0.weight"][step - 5] += step
            oracle.set_current(current)
            frame = oracle.observe(step)
            self.assertGreaterEqual(len(unpack_s2_replacement_frame(frame)[1]), 1)
            oracle.ack(frame)
        current["backbone.blocks.0.weight"][0] -= 7
        oracle.set_current(current)
        frame = oracle.observe(8)
        _, blocks, _, _ = unpack_s2_replacement_frame(frame)
        self.assertEqual(len(blocks), 1)

    def test_z8_dynamic_range_z9_nonfinite_values_are_literal(self):
        oracle = S2DeltaOracle(self.initial, block_size=4, small_threshold=4)
        current = {k: v.copy() for k, v in self.initial.items()}
        current["backbone.blocks.0.weight"][0] = np.inf
        current["backbone.blocks.0.weight"][1] = np.nan
        oracle.set_current(current)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            oracle.observe(9)

    def test_ack_is_the_only_reference_advance_and_stale_ack_is_rejected(self):
        oracle = S2DeltaOracle(self.initial, block_size=4, small_threshold=4)
        current = {k: v.copy() for k, v in self.initial.items()}
        current["backbone.blocks.0.weight"][8] = 77
        oracle.set_current(current)
        frame = oracle.observe(10)
        self.assertEqual(oracle.persisted_generation, 0)
        self.assertEqual(len(unpack_s2_replacement_frame(oracle.observe(11))[1]), 1)
        oracle.ack(frame)
        self.assertEqual(oracle.persisted_generation, 1)
        with self.assertRaisesRegex(ValueError, "stale"):
            oracle.ack(frame)

    def test_manifest_and_generation_reject_reorder_or_cross_lineage(self):
        oracle = S2DeltaOracle(self.initial, block_size=4, small_threshold=4)
        current = {k: v.copy() for k, v in self.initial.items()}
        current["backbone.blocks.0.weight"][0] = 5
        oracle.set_current(current)
        first = oracle.observe(20)
        oracle.ack(first)
        second = oracle.observe(21)
        other = S2DeltaOracle({"other": np.zeros(9, dtype=np.float32)}, 4, 4)
        with self.assertRaisesRegex(ValueError, "manifest"):
            other.ack(second)
        with self.assertRaisesRegex(ValueError, "generation"):
            oracle.recover(self.initial, [second])

    def test_protocol_dispatch_and_crc(self):
        oracle = S2DeltaOracle(self.initial, block_size=4, small_threshold=4)
        current = {k: v.copy() for k, v in self.initial.items()}
        current["backbone.blocks.0.weight"][4] = -3
        oracle.set_current(current)
        frame = oracle.observe(30)
        result = unpack_delta_frame_with_meta(frame)
        self.assertEqual(result[3]["version"], 3)
        corrupted = bytearray(frame)
        corrupted[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            unpack_s2_replacement_frame(corrupted)

    def test_apply_replacements_does_not_mutate_input(self):
        manifest = build_block_manifest(self.initial, 4, 4)
        block = manifest["blocks"][0]
        updated = apply_s2_replacements(
            self.initial,
            [{**block, "value": np.full(block["element_count"], 8, dtype=np.float32)}],
            [], manifest)
        self.assertEqual(updated[block["name"]][0], 8)
        self.assertEqual(self.initial[block["name"]][0], 0)


if __name__ == "__main__":
    unittest.main()
