import json

import pytest

from experiments.benchmarks.longrun_utils import (atomic_json, checked_stdout,
                                                   completed_result, open_campaign,
                                                   update_entry)


def test_campaign_resume_requires_matching_commit_and_config(tmp_path):
    path = tmp_path / "campaign.json"
    campaign = open_campaign(path, "abc", {"depth": 2})
    assert campaign["entries"] == {}
    assert open_campaign(path, "abc", {"depth": 2}, resume=True)[
        "config_digest"] == campaign["config_digest"]
    with pytest.raises(RuntimeError, match="commit"):
        open_campaign(path, "def", {"depth": 2}, resume=True)
    with pytest.raises(RuntimeError, match="configuration"):
        open_campaign(path, "abc", {"depth": 4}, resume=True)


def test_completed_result_only_reuses_durable_pass(tmp_path):
    campaign_path = tmp_path / "campaign.json"
    campaign = open_campaign(campaign_path, "abc", {})
    result_path = tmp_path / "result.json"
    atomic_json(result_path, {"status": "pass", "value": 7})
    assert completed_result(campaign, "sample", result_path) is None
    update_entry(campaign_path, campaign, "sample", "pass")
    assert completed_result(campaign, "sample", result_path)["value"] == 7
    result_path.write_text("not-json", encoding="utf-8")
    assert completed_result(campaign, "sample", result_path) is None


def test_atomic_json_replaces_complete_document(tmp_path):
    path = tmp_path / "result.json"
    atomic_json(path, {"status": "running", "records": list(range(100))})
    atomic_json(path, {"status": "pass"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "pass"}


def test_checked_stdout_rejects_failed_or_empty_commands():
    assert checked_stdout({"returncode": 0, "stdout": "abc\n"}, "test") == "abc"
    with pytest.raises(RuntimeError, match="failed"):
        checked_stdout({"returncode": 1, "stdout": "", "stderr": "bad"}, "test")
    with pytest.raises(RuntimeError, match="empty"):
        checked_stdout({"returncode": 0, "stdout": ""}, "test")
