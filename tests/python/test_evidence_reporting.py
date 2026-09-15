import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.benchmarks.summarize_p1_p9 import render_report, summarize
from experiments.benchmarks import p2_stack_decompose as p2
from experiments.benchmarks.run_ppt_p1_p9 import p2_evidence_status
from ppt_evidence import EvidenceBundle


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def record(self, relative, experiment, status, **extra):
        directory = self.root / relative
        directory.mkdir(parents=True, exist_ok=True)
        result = {"run_id": experiment + "_example", "status": status, **extra}
        (directory / "result.json").write_text(json.dumps(result))
        (directory / "config.json").write_text(json.dumps({"experiment_id": experiment}))

    def test_legacy_nested_p1_and_smoke_do_not_pass_p2(self):
        self.record("P1/P1/formal", "P1", "pass")
        self.record("P1_busy_smoke/P1/smoke", "P1", "pass")
        self.record("P2/P2/decomposed", "P2", "degraded", closure={"achieved": False})
        self.record("P2/P2/decomposed/nested_p1/P1/child", "P1", "pass")
        report, result = render_report(self.root)
        self.assertEqual(result["P1"]["runs"], 1)
        self.assertEqual(result["P1"]["auxiliary_runs"], 2)
        self.assertEqual(result["P2"]["statuses"], {"degraded": 1})
        self.assertIn("时间闭合通过 0/1", report)
        self.assertNotIn("19%", report)
        self.assertNotIn("G0/G1/G2 正确性门禁", report)
        self.assertEqual(p2_evidence_status(self.root), "degraded")

    def test_p2_exit_success_does_not_replace_time_closure_evidence(self):
        self.record("P2/formal", "P2", "pass", closure={"achieved": False})
        self.assertEqual(p2_evidence_status(self.root), "degraded")
        self.record("P2/formal", "P2", "pass", closure={"achieved": True})
        self.assertEqual(p2_evidence_status(self.root), "pass")

    def test_explicit_child_remains_auxiliary_after_moving(self):
        self.record("P1/relocated", "P1", "pass", run_role="child", parent_run_id="P2_parent")
        _, result = summarize(self.root)
        self.assertEqual(result["P1"]["runs"], 0)
        self.assertEqual(result["P1"]["auxiliary_runs"], 1)

    def test_bundle_persists_identity_and_parent(self):
        with patch.dict("os.environ", {"NPU_NVME_ENVIRONMENT_ID": "env-a",
                                       "NPU_NVME_PARENT_RUN_ID": "parent-a"}):
            bundle = EvidenceBundle("P1", {}, root=self.root)
            result = bundle.finalize(status="pass")
        self.assertEqual(result["environment_id"], "env-a")
        self.assertEqual(result["parent_run_id"], "parent-a")
        self.assertEqual(result["run_role"], "child")
        self.assertEqual(result["experiment_id"], "P1")

    def test_empty_report_does_not_import_historical_conclusions(self):
        report, result = render_report(self.root)
        self.assertTrue(all(row["runs"] == 0 for row in result.values()))
        self.assertIn("无本轮主实验结果", report)
        self.assertNotIn("已完成配置数=1", report)

    def test_missing_trace_instance_permissions_leave_global_trace_untouched(self):
        with patch.object(p2, "tracefs_capabilities", return_value={
                "root": "/sys/kernel/tracing", "available": ["block_rq_issue"], "enabled": False}), \
             patch.object(Path, "mkdir", side_effect=PermissionError("denied")), \
             patch.object(Path, "write_text") as write:
            result = p2.trace_start(True)
        self.assertFalse(result["enabled"])
        write.assert_not_called()

    def test_trace_stop_refuses_global_trace_state(self):
        with patch.object(Path, "write_text") as write:
            p2.trace_stop({"enabled": True, "root": "/sys/kernel/tracing"}, self.root / "trace")
        write.assert_not_called()

    def test_p2_degraded_phase_has_nonzero_exit(self):
        with patch("sys.argv", ["p2", "--modes", "buffered", "--sizes", "4096"]), \
             patch.object(p2, "one", return_value="degraded"):
            self.assertEqual(p2.main(), 2)


if __name__ == "__main__":
    unittest.main()
