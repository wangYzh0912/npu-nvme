import os
import random
import sys
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from training_state import (decode_control_value, encode_control_value,
                            capture_training_controls,
                            restore_training_controls,
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

    def test_capture_and_restore_complete_controls(self):
        class ArrayValue:
            def __init__(self, value):
                self.value = np.asarray(value)

            def asnumpy(self):
                return np.array(self.value, copy=True)

            def set_data(self, value):
                self.value = np.asarray(value.value).copy()

        class TensorValue:
            def __init__(self, value):
                self.value = np.asarray(value)

        class FakeCommon:
            seed = None

            @classmethod
            def set_seed(cls, seed):
                cls.seed = seed

        class FakeMS:
            common = FakeCommon
            rng = ArrayValue(np.array([7, 11], dtype=np.int64))
            restored_rng = None

            @classmethod
            def get_rng_state(cls):
                return cls.rng

            @classmethod
            def Tensor(cls, value):
                return TensorValue(value)

            @classmethod
            def set_rng_state(cls, value):
                cls.restored_rng = np.asarray(value.value).copy()

        class Optimizer:
            global_step = ArrayValue(np.array([9], dtype=np.int32))

        python_before = random.getstate()
        numpy_before = np.random.get_state()
        controls = capture_training_controls(
            FakeMS, Optimizer, {"epoch": 2, "sample": 19}, 1024.0, 31)
        Optimizer.global_step.value[:] = 0
        random.random()
        np.random.random()
        restored = restore_training_controls(FakeMS, Optimizer, controls)
        self.assertTrue(np.array_equal(Optimizer.global_step.value,
                                       np.array([9], dtype=np.int32)))
        self.assertTrue(np.array_equal(FakeMS.restored_rng,
                                       np.array([7, 11], dtype=np.int64)))
        self.assertEqual(FakeCommon.seed, 31)
        self.assertEqual(restored["data_cursor"], {"epoch": 2, "sample": 19})
        random.setstate(python_before)
        np.random.set_state(numpy_before)

    def test_restore_rejects_missing_control(self):
        with self.assertRaisesRegex(ValueError, "control fields"):
            restore_training_controls(object(), object(), {"global_step": 1})


if __name__ == "__main__":
    unittest.main()
