import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from npu_nvme.types import validate_result_gate


class FullCheckpointProtocolTests(unittest.TestCase):


    def test_none_baseline_is_not_restore_success(self):
        with self.assertRaises(ValueError):
            validate_result_gate({"mode": "none", "restore_verified": True})
        validate_result_gate({"mode": "none", "status": "pass", "restore_verified": None})

    def test_full_success_requires_persistence_and_restore(self):
        with self.assertRaises(ValueError):
            validate_result_gate({"mode": "serial", "status": "pass",
                                  "request_id": 1, "generation": 1,
                                  "persisted": True, "restore_verified": False})
        validate_result_gate({"mode": "serial", "status": "pass",
                              "request_id": 1, "generation": 1,
                              "persisted": True, "restore_verified": True})


if __name__ == "__main__":
    unittest.main()
