import os
import subprocess
import sys
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from delta_protocol import (pack_s2_replacement_frame,
                            unpack_delta_frame_with_meta,
                            unpack_s2_replacement_frame)
from s2_delta import (S2DeltaOracle, apply_s2_replacements,
                      build_block_manifest, score_manifest_blocks)
from s2_delta import FileS2Ring


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

    def test_vectorized_block_scores_match_parameter_local_reference(self):
        manifest = build_block_manifest(self.initial, block_size=4,
                                        small_threshold=0)
        current = {name: value.copy() for name, value in self.initial.items()}
        current["backbone.blocks.0.weight"][[0, 4, 8]] += 3
        current["backbone.blocks.1.weight"][-1] += 2
        scores = score_manifest_blocks(current, self.initial, manifest)
        by_id = {item["block_id"]: item for item in scores}
        for item in manifest["blocks"]:
            start = item["element_offset"]
            count = item["element_count"]
            a = current[item["name"]].reshape(-1)[start:start + count]
            b = self.initial[item["name"]].reshape(-1)[start:start + count]
            diff = a.astype(np.float64) - b.astype(np.float64)
            self.assertAlmostEqual(by_id[item["block_id"]]["score"],
                                   float(np.linalg.norm(diff)))
            self.assertEqual(by_id[item["block_id"]]["nonzero"],
                             int(np.count_nonzero(diff)))

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

    def test_negative_layer_id_is_preserved_for_large_special_parameter(self):
        initial = {
            "backbone.layernorm.weight": np.zeros(8, dtype=np.float32),
        }
        oracle = S2DeltaOracle(initial, block_size=4, small_threshold=4)
        current = {name: value.copy() for name, value in initial.items()}
        current["backbone.layernorm.weight"][0] = 3.0
        oracle.set_current(current)
        frame = oracle.observe(31)
        _, blocks, _, _ = unpack_s2_replacement_frame(frame)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["layer_id"], -1)
        oracle.ack(frame)

    def test_apply_replacements_does_not_mutate_input(self):
        manifest = build_block_manifest(self.initial, 4, 4)
        block = manifest["blocks"][0]
        updated = apply_s2_replacements(
            self.initial,
            [{**block, "value": np.full(block["element_count"], 8, dtype=np.float32)}],
            [], manifest)
        self.assertEqual(updated[block["name"]][0], 8)
        self.assertEqual(self.initial[block["name"]][0], 0)

    def test_i4_atomic_file_ring_wrap_and_corruption_rejection(self):
        with __import__("tempfile").TemporaryDirectory() as directory:
            oracle = S2DeltaOracle(self.initial, block_size=4, small_threshold=4)
            ring = FileS2Ring(directory, slot_count=2, slot_size=1024 * 1024)
            frames = []
            for step in range(3):
                current = {k: v.copy() for k, v in self.initial.items()}
                current["backbone.blocks.0.weight"][step] = step + 10
                oracle.set_current(current)
                frame = oracle.observe(step + 40)
                frames.append(frame)
                ring.write(frame)
                oracle.ack(frame)
            # Two slots means the first frame is replaced by the third.
            self.assertEqual(unpack_s2_replacement_frame(ring.read(0))[0], 42)
            recovered = oracle.recover(self.initial, frames)
            self.assertEqual(recovered["generation"], 3)
            corrupted = bytearray(ring.read(1))
            corrupted[-1] ^= 1
            with open(ring._path(1), "wb") as stream:
                stream.write(corrupted)
            with self.assertRaisesRegex(ValueError, "CRC"):
                ring.read(1)

    def test_i4_independent_process_reads_complete_slot(self):
        with __import__("tempfile").TemporaryDirectory() as directory:
            oracle = S2DeltaOracle(self.initial, block_size=4, small_threshold=4)
            current = {k: v.copy() for k, v in self.initial.items()}
            current["backbone.blocks.1.weight"][4] = 123
            oracle.set_current(current)
            frame = oracle.observe(77)
            FileS2Ring(directory, slot_count=2, slot_size=1024 * 1024).write(frame)
            child = (
                "import sys; sys.path.insert(0, sys.argv[2]); "
                "from s2_delta import FileS2Ring; "
                "from delta_protocol import unpack_s2_replacement_frame; "
                "f=FileS2Ring(sys.argv[1], 2, 1048576).read(0); "
                "print(unpack_s2_replacement_frame(f)[0])"
            )
            output = subprocess.check_output(
                [sys.executable, "-c", child, directory,
                 os.path.join(REPO_ROOT, "python")], text=True)
            self.assertEqual(output.strip(), "77")


if __name__ == "__main__":
    unittest.main()
