"""CPU process tests for H02 orchestration; no ACL/SPDK imports or hardware."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("h02", ROOT / "tests/hardware/stage4_fault_lifecycle.py")
h02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h02)


def child(tmp_path, body, timeout=2, expected_rc=0):
    directory = tmp_path / "run"
    source = "import json,os,time\nfrom pathlib import Path\np=Path(" + repr(str(directory / "worker.json")) + ")\n" + body
    return h02.run_process([sys.executable, "-c", source], directory, timeout, expected_rc)


@pytest.mark.parametrize("body,passed", [
    ("p.write_text(json.dumps({'status':'pass','cleanup_safe':True}))", True),
    ("p.write_text(json.dumps({'status':'fail','cleanup_safe':True}))", False),
    ("p.write_text(json.dumps({'status':'pass','cleanup_safe':False}))", False),
    ("p.write_text(json.dumps({'status':'pass','cleanup_safe':True})); os._exit(7)", False),
    ("os._exit(3)", False),
    ("pass", False),
    ("p.write_text('broken json')", False),
])
def test_process_results(tmp_path, body, passed):
    record = child(tmp_path, body)
    assert (record["status"] == "pass") == passed
    assert json.loads((tmp_path / "run/process.json").read_text())["status"] == record["status"]
    assert (tmp_path / "run/stdout.txt").is_file()
    assert (tmp_path / "run/stderr.txt").is_file()


def test_timeout_even_with_premature_pass_file(tmp_path):
    record = child(tmp_path, "p.write_text(json.dumps({'status':'pass','cleanup_safe':True})); time.sleep(5)", .15)
    assert record["status"] == "fail" and record["timed_out"]
    assert record["returncode"] < 0


def test_expected_crash_requires_marker(tmp_path):
    record = child(tmp_path, "os._exit(86)", expected_rc=86)
    assert record["status"] == "fail"


def test_expected_crash_is_separate_from_cleanup_proof(tmp_path):
    record = child(tmp_path, "p.write_text(json.dumps({'status':'expected_crash','cleanup_safe':False})); os._exit(86)", expected_rc=86)
    assert record["status"] == "pass"
    assert not record["worker"]["cleanup_safe"]


def test_cleanup_failure_stops_reopen():
    calls = []
    def launch(case, verify):
        calls.append(verify)
        return {"status": "fail", "worker": {"cleanup_safe": False}}
    assert not h02.run_pair(launch, "timeout", {})
    assert calls == [False]


def test_verification_failure_propagates():
    calls = []
    def launch(case, verify):
        calls.append(verify)
        return {"status": "fail" if verify else "pass"}
    record = {}
    assert not h02.run_pair(launch, "timeout", record)
    assert record["status"] == "fail" and calls == [False, True]


def test_crash_changed_metadata_fails():
    def launch(case, verify):
        return {"status": "pass", "worker": {"superblock_sha256": str(verify)}}
    assert not h02.run_pair(launch, "before_metadata_commit", {})


def test_old_output_is_not_overwritten(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    marker = directory / "worker.json"
    marker.write_text("original failure")
    with pytest.raises(FileExistsError):
        h02.run_process([sys.executable, "-c", "pass"], directory, 1)
    assert marker.read_text() == "original failure"


def test_parent_import_does_not_load_device_libraries():
    import subprocess
    script = ("import runpy,sys; runpy.run_path(" + repr(str(ROOT / 'tests/hardware/stage4_fault_lifecycle.py')) +
              "); assert 'c_bindings' not in sys.modules; assert 'npu_nvme.storage.bindings' not in sys.modules")
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_protected_device_rejected_before_output_creation(tmp_path):
    import subprocess
    output = tmp_path / 'protected'
    result = subprocess.run([sys.executable, str(ROOT / 'tests/hardware/stage4_fault_lifecycle.py'),
                             '--pci', '0000:84:00.0', '--output', str(output)], capture_output=True, timeout=5)
    assert result.returncode != 0 and not output.exists()


def test_failed_close_keeps_all_caller_buffers():
    from types import SimpleNamespace
    worker = h02.Worker.__new__(h02.Worker)
    worker.ctx = object()
    worker.record = {}
    worker.args = SimpleNamespace(close_timeout_ms=5000)
    worker.snapshot = lambda: {'stats': {}, 'retained_slots': []}
    worker.lib = SimpleNamespace(npu_nvme_close=lambda *args: -110)
    worker.host_buffers, worker.device_buffers, worker.requests = [object()], [object()], [object()]
    with pytest.raises(AssertionError, match='retain all buffers'):
        worker.close()
    assert worker.ctx and worker.host_buffers and worker.device_buffers and worker.requests
    assert not worker.record.get('cleanup_safe')
