import sys
import unittest

import numpy as np

sys.path.insert(0, "python")
from incremental_manifest import build_training_state_manifest  # noqa: E402


class FakeParameter:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)


class FakeComponent:
    def __init__(self, pairs):
        self.pairs = pairs

    def parameters_and_names(self):
        return tuple(self.pairs)


class IncrementalManifestTests(unittest.TestCase):
    def test_order_and_alias_are_stable(self):
        shared = FakeParameter((9,), np.float32)
        model = FakeComponent([("z", shared), ("a", FakeParameter((5,), np.float16))])
        optimizer = FakeComponent([("weight", shared),
                                   ("moment", FakeParameter((10,), np.float32))])
        first = build_training_state_manifest({"optimizer": optimizer, "model": model},
                                              block_size=4)
        second = build_training_state_manifest({"model": model, "optimizer": optimizer},
                                               block_size=4)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual([field.canonical_name for field in first.fields],
                         ["model/a", "model/z", "optimizer/moment"])
        self.assertTrue(first.fields[0].small)
        self.assertEqual(len(first.fields[0].blocks), 1)
        self.assertEqual(first.fields[0].blocks[-1].element_count, 5)

    def test_manifest_rejects_unsupported_component(self):
        with self.assertRaises(TypeError):
            build_training_state_manifest({"model": object()})

    def test_state_id_changes_with_dtype(self):
        a = build_training_state_manifest(
            {"model": FakeComponent([("x", FakeParameter((4,), np.float16))])})
        b = build_training_state_manifest(
            {"model": FakeComponent([("x", FakeParameter((4,), np.float32))])})
        self.assertNotEqual(a.fields[0].state_id, b.fields[0].state_id)


if __name__ == "__main__":
    unittest.main()
