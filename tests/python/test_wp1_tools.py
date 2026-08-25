import json
import tempfile
import unittest
from pathlib import Path


from experiments.benchmarks.validate_timeline import validate_sample


class Wp1TimelineTests(unittest.TestCase):
    def test_monotonic_events_pass(self):
        sample = {
            "events": [{"name": "start", "monotonic_ns": 10},
                       {"name": "end", "monotonic_ns": 20}],
            "timeline_us": {"end_to_end": 1.0},
        }
        self.assertEqual(validate_sample(sample), [])

    def test_non_monotonic_events_fail(self):
        sample = {
            "events": [{"name": "start", "monotonic_ns": 20},
                       {"name": "end", "monotonic_ns": 10}],
            "timeline_us": {"end_to_end": 1.0},
        }
        self.assertIn("event timestamps are not monotonic",
                      validate_sample(sample))

    def test_negative_duration_fails(self):
        sample = {"events": [], "timeline_us": {"persist": -1}}
        self.assertIn("negative duration: persist", validate_sample(sample))


if __name__ == "__main__":
    unittest.main()
