import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPO_ROOT)

from experiments.benchmarks.s2_incremental_decision import (  # noqa: E402
    make_decision, policy_summary, trajectory_summary,
)


class IncrementalDecisionTests(unittest.TestCase):
    def test_complete_trajectory_and_three_seed_policy_go(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            policies = []
            for seed in (41, 42, 43):
                run = root / f"trajectory-{seed}"
                run.mkdir()
                (run / "config.json").write_text(json.dumps({
                    "seed": seed, "block_sizes": [4]}))
                samples = []
                for step in (1, 2):
                    category = {"l2": 1.0, "relative_l2_median": 0.1,
                                "coverage": {str(k): {"energy_fraction": 0.9}
                                             for k in (1, 5, 10, 20)}}
                    samples.append({"step": step, "block_sizes": {"4": {
                        "adjacent_l2": 1.0, "selected_jaccard": 0.1,
                        "age": {"max": 1}, "categories": {"model": category},
                        "coverage": {str(k): {"energy_fraction": 0.9}
                                     for k in (1, 5, 10, 20, 50)}}}})
                (run / "samples.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in samples))
                runs.append(run)

                policy = root / f"policy-{seed}.json"
                policy.write_text(json.dumps({
                    "status": "PASS", "seed": seed, "rows": [{
                        "candidate_id": "candidate", "config": {
                            "strategy": "r2", "max_age": 4},
                        "write_ratio": 0.1, "errors": [0.005, 0.006],
                        "final_relative_l2_error": 0.006,
                        "recovery_loss_relative_error": 0.001,
                        "max_block_age": 3,
                        "final_category_relative_l2_error": {"model": 0.006},
                    }]}))
                policies.append(policy)
            trajectory = trajectory_summary(runs, expected_steps=2)
            policy_result = policy_summary(policies, expected_seeds=3)
            self.assertTrue(trajectory["complete"])
            self.assertEqual(policy_result["eligible_count"], 1)
            self.assertEqual(make_decision(trajectory, policy_result)["status"],
                             "GO_R2")

    def test_incomplete_trajectory_blocks_decision(self):
        decision = make_decision({"complete": False}, {"eligible_count": 1})
        self.assertEqual(decision["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
