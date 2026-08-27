import sys
import unittest

import numpy as np

sys.path.insert(0, "python")
from s2_policy import S2SelectivePolicy  # noqa: E402


class S2SelectivePolicyTests(unittest.TestCase):
    def test_ack_updates_only_persisted_decoded_blocks(self):
        initial = {"p": np.zeros(8, dtype=np.float32)}
        current = {"p": np.arange(8, dtype=np.float32)}
        policy = S2SelectivePolicy(
            initial, block_size=2, selection_fraction=0.25,
            encoding="fp16", max_age=0, small_threshold=0)
        frame = policy.observe(current, 1)
        self.assertEqual(len(frame["selected"]), 1)
        selected = frame["selected"][0]
        policy.ack(frame["generation"])
        for index, block in enumerate(policy.blocks):
            values = policy.reference[block.name].reshape(-1)[
                block.offset:block.offset + block.count]
            if index == selected:
                expected = current[block.name].reshape(-1)[
                    block.offset:block.offset + block.count]
                self.assertTrue(np.array_equal(values, expected))
            else:
                self.assertTrue(np.array_equal(values, np.zeros(block.count)))

    def test_failed_frame_does_not_advance_reference_or_age(self):
        initial = {"p": np.zeros(8, dtype=np.float32)}
        policy = S2SelectivePolicy(
            initial, 2, 0.25, encoding="int8", max_age=4,
            small_threshold=0)
        frame = policy.observe({"p": np.ones(8, dtype=np.float32)}, 1)
        policy.fail(frame["generation"])
        self.assertEqual(policy.generation, 0)
        self.assertTrue(np.array_equal(policy.age, np.zeros(4)))
        self.assertTrue(np.array_equal(policy.reference["p"], initial["p"]))

    def test_max_age_forces_eventual_refresh(self):
        initial = {"p": np.zeros(8, dtype=np.float32)}
        current = {"p": np.arange(1, 9, dtype=np.float32)}
        policy = S2SelectivePolicy(
            initial, 2, 0.25, encoding="raw", max_age=2,
            small_threshold=0)
        first = policy.observe(current, 1)
        policy.ack(first["generation"])
        second = policy.observe(current, 2)
        first_selected = set(first["selected"])
        self.assertTrue(set(range(4)) - first_selected <=
                        set(second["selected"]))
        policy.ack(second["generation"])
        self.assertLess(int(policy.age.max()), 2)

    def test_int8_reference_is_decoded_value(self):
        initial = {"p": np.zeros(4, dtype=np.float32)}
        current = {"p": np.array([0.1, 0.2, 0.3, 0.7], dtype=np.float32)}
        policy = S2SelectivePolicy(
            initial, 4, 1.0, encoding="int8", max_age=0,
            small_threshold=0)
        frame = policy.observe(current, 1)
        decoded = frame["decoded"][0][1].copy()
        policy.ack(frame["generation"])
        self.assertTrue(np.array_equal(policy.reference["p"], decoded))
        self.assertFalse(np.array_equal(policy.reference["p"], current["p"]))

    def test_error_budget_selects_minimum_energy_prefix(self):
        initial = {"p": np.zeros(4, dtype=np.float32)}
        current = {"p": np.array([4, 0, 1, 0], dtype=np.float32)}
        policy = S2SelectivePolicy(
            initial, 2, 0.80, encoding="raw", small_threshold=0,
            selection_mode="error_budget")
        frame = policy.observe(current, 1)
        self.assertEqual(frame["selected"], [0])

    def test_frozen_threshold_can_emit_no_blocks(self):
        policy = S2SelectivePolicy(
            {"p": np.zeros(4, dtype=np.float32)}, 2, 1.0,
            encoding="raw", small_threshold=0, selection_mode="threshold",
            score_threshold=10.0)
        frame = policy.observe({"p": np.ones(4, dtype=np.float32)}, 1)
        self.assertEqual(frame["selected"], [])


if __name__ == "__main__":
    unittest.main()
