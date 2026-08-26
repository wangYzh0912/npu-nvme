import os
import random
import sys
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from training_state import (decode_control_value, encode_control_value,
                            validate_state_names)


class TrainingStateCodecTests(unittest.TestCase):
    def assert_value_equal(self, expected, actual):
        if isinstance(expected, np.ndarray):
            self.assertIsInstance(actual, np.ndarray)
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertEqual(expected.shape, actual.shape)
            self.assertTrue(np.array_equal(expected, actual))
        elif isinstance(expected, tuple):
            self.assertIsInstance(actual, tuple)
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_value_equal(left, right)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_value_equal(expected[key], actual[key])
        else:
            self.assertEqual(expected, actual)

    def test_rng_and_cursor_roundtrip(self):
        value = {
            "python_rng": random.Random(17).getstate(),
            "numpy_rng": np.random.RandomState(23).get_state(),
            "data_cursor": {"epoch": 3, "sample": 117},
            "loss_scale": np.float32(4096.0),
            "opaque": b"state\x00bytes",
        }
        payload, metadata = encode_control_value(value)
        decoded = decode_control_value(payload, metadata)
        self.assert_value_equal(value, decoded)

    def test_checksum_failure_is_rejected(self):
        payload, metadata = encode_control_value({"step": 10})
        payload[0] ^= np.uint8(1)
        with self.assertRaisesRegex(ValueError, "checksum"):
            decode_control_value(payload, metadata)

    def test_namespace_validation(self):
        validate_state_names({"model": object(), "optimizer": object()},
                             {"global_step": 1})
        with self.assertRaises(ValueError):
            validate_state_names({"bad/name": object()}, {})
        with self.assertRaises(ValueError):
            validate_state_names({"model": object()}, {"bad/name": 1})


if __name__ == "__main__":
    unittest.main()
