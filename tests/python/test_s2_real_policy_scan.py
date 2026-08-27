import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, "python")]

from experiments.benchmarks.s2_real_policy_scan import (  # noqa: E402
    candidate_grid, candidate_id, quick_candidates,
)


class RealPolicyScanConfigTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = {
            str(size): {str(value): float(index + 1)
                        for index, value in enumerate((0.01, 0.05, 0.1, 0.2))}
            for size in (65536, 262144, 524288)
        }

    def test_grid_is_complete_and_ids_are_stable(self):
        rows = candidate_grid(self.thresholds)
        self.assertEqual(len(rows), 1152)
        identifiers = [candidate_id(row) for row in rows]
        self.assertEqual(len(set(identifiers)), len(rows))
        self.assertEqual(candidate_id(rows[0]), candidate_id(dict(rows[0])))

    def test_quick_candidates_cover_write_error_and_pareto_lanes(self):
        rows = quick_candidates()
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["selection_mode"] for row in rows},
                         {"topk", "error_budget"})
        self.assertTrue(all(row["strategy"] == "r2" for row in rows))


if __name__ == "__main__":
    unittest.main()
