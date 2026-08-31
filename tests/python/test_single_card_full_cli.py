import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "experiments", "benchmarks", "run_single_card_full.py")


class SingleCardFullCliTests(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((ROOT, os.path.join(ROOT, "python")))
        return subprocess.run([sys.executable, SCRIPT, *args],
                              capture_output=True, text=True, env=env,
                              check=False)

    def test_dry_run_does_not_import_mindspore(self):
        result = self.run_cli("--dry-run", "--model", "gpt2",
                             "--checkpoint-steps", "2", "--total-steps", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout)
        self.assertEqual(config["mode"], "serial")
        self.assertEqual(config["checkpoint_steps"], [2])

    def test_unsorted_checkpoint_steps_are_rejected(self):
        result = self.run_cli("--dry-run", "--checkpoint-steps", "5", "2",
                             "--total-steps", "5")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sorted", result.stderr)


if __name__ == "__main__":
    unittest.main()
