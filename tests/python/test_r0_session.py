import sys
import unittest

import numpy as np

sys.path.insert(0, "python")
from incremental_manifest import build_training_state_manifest  # noqa: E402
from r0_session import R0Session, state_digest  # noqa: E402


class R0SessionTests(unittest.TestCase):
    def setUp(self):
        self.initial = {
            "model/x": np.arange(9, dtype=np.float16),
            "optimizer/m": np.zeros(9, dtype=np.float32),
        }
        class Component:
            def __init__(self, state):
                self.state = state

            def parameters_and_names(self):
                return tuple((name.split("/", 1)[1], FakeParameter(value))
                             for name, value in self.state.items())

        class FakeParameter:
            def __init__(self, value):
                self.shape = value.shape
                self.dtype = value.dtype

        self.manifest = build_training_state_manifest({
            "model": Component({"model/x": self.initial["model/x"]}),
            "optimizer": Component({"optimizer/m": self.initial["optimizer/m"]}),
        }, block_size=4, small_threshold=0)
        self.session = R0Session(self.initial, self.manifest)

    def test_observe_ack_and_replay(self):
        current = {name: value.copy() for name, value in self.initial.items()}
        current["model/x"][5] = np.float16(42)
        current["optimizer/m"] += 0.25
        self.session.set_current(current)
        frame = self.session.observe(1, 1, [("data_cursor", "raw", b"1")])
        self.assertEqual(self.session.in_flight_generation, 1)
        with self.assertRaises(RuntimeError):
            self.session.observe(2, 2)
        self.session.ack(frame)
        self.assertIsNone(self.session.in_flight_generation)
        recovered = self.session.recover([frame])
        self.assertEqual(recovered["generation"], 1)
        self.assertEqual(state_digest(recovered["state"]), state_digest(current))

    def test_bad_ack_does_not_advance_reference(self):
        current = {name: value.copy() for name, value in self.initial.items()}
        current["model/x"][0] = 9
        self.session.set_current(current)
        frame = self.session.observe(1, 1)
        bad = bytearray(frame)
        bad[-1] ^= 1
        with self.assertRaises(ValueError):
            self.session.ack(bad)
        self.assertEqual(self.session.persisted_generation, 0)
        self.assertEqual(self.session.in_flight_generation, 1)


if __name__ == "__main__":
    unittest.main()
