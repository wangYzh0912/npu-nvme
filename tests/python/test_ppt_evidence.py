import json
import tempfile
import unittest
from pathlib import Path

from ppt_evidence import EvidenceBundle, RESULT_FIELDS, stats


class PptEvidenceTests(unittest.TestCase):
    def test_stats_suppresses_p99_for_small_samples(self):
        result = stats([1, 2, 3])
        self.assertEqual(result["n"], 3)
        self.assertIsNone(result["p99"])
        self.assertEqual(result["p99_status"], "not reported (n=3<30)")
        self.assertIn("ci95", result)

    def test_stats_reports_p99_at_thirty_samples(self):
        result = stats(range(30))
        self.assertEqual(result["n"], 30)
        self.assertIsNotNone(result["p99"])
        self.assertEqual(result["p99_status"], "reported")

    def test_bundle_writes_required_files_and_null_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = EvidenceBundle(
                "TEST", {"model": "synthetic", "mode": "host"},
                root=temporary, repo_root=Path.cwd(),
                environment={"test": True})
            bundle.add_sample({"request_id": "r0", "latency_ms": 1.0})
            bundle.add_failure({"request_id": "bad", "error": "injected"})
            result = bundle.finalize(metrics={"model": "synthetic"})
            run_dir = Path(temporary) / "TEST" / bundle.run_id
            for name in ("config.json", "environment.json", "commit.json",
                         "samples.jsonl", "timeline.jsonl", "result.json",
                         "failures.jsonl"):
                self.assertTrue((run_dir / name).exists(), name)
            self.assertTrue((run_dir / "raw").is_dir())
            self.assertEqual(result["status"], "fail")
            encoded = json.loads((run_dir / "result.json").read_text())
            for field in RESULT_FIELDS:
                self.assertIn(field, encoded)
            self.assertIsNone(encoded["throughput"])


if __name__ == "__main__":
    unittest.main()
