import json
from pathlib import Path

import numpy as np
import pytest

from experiments.baselines.repro.host_bridge import (
    SharedSnapshot, snapshot_from_descriptor, snapshot_view_from_descriptor)
from experiments.baselines.repro.cli import (adapter_evidence,
                                             is_formal_semantic_port)
from experiments.baselines.repro.adapters import ADAPTERS
from experiments.baselines.repro.protocol import EventLog, Handle
from experiments.baselines.repro.state_bridge import Snapshot, load_raw_snapshot, save_raw_snapshot
from experiments.baselines.repro.inventory import build_inventory
from python.full_checkpoint_protocol import CheckpointState


def _snapshot():
    arrays = {
        "model/w": np.arange(17, dtype=np.float32),
        "optimizer/m": np.arange(9, dtype=np.int16),
    }
    controls = np.arange(11, dtype=np.uint8)
    schema = {"schema_version": 1, "format": "npu-nvme-repro-raw-v1", "fields": [
        {"name": "model/w", "dtype": "<f4", "shape": [17], "nbytes": 68},
        {"name": "optimizer/m", "dtype": "<i2", "shape": [9], "nbytes": 18},
        {"name": "controls/state", "dtype": "|u1", "shape": [11], "nbytes": 11},
    ]}
    return Snapshot(arrays, controls, {"fields": []}, schema)


def test_raw_snapshot_roundtrip_and_digest(tmp_path):
    original = _snapshot()
    metadata = save_raw_snapshot(original, tmp_path / "generation_1")
    restored = load_raw_snapshot(tmp_path / "generation_1")
    assert metadata["sha256"] == original.digest() == restored.digest()
    assert all(np.array_equal(original.arrays[k], restored.arrays[k]) for k in original.arrays)


def test_shared_snapshot_descriptor_roundtrip():
    original = _snapshot()
    owner, descriptor = SharedSnapshot.from_snapshot(original, prefix="repro_test")
    try:
        restored = snapshot_from_descriptor(descriptor)
        assert restored.digest() == original.digest()
    finally:
        owner.close(unlink=True)


def test_chunked_raw_snapshot_roundtrip(tmp_path):
    original = _snapshot()
    metadata = save_raw_snapshot(
        original, tmp_path / "generation", chunk_bytes=5,
        extra_metadata={"generation": 3})
    restored = load_raw_snapshot(
        tmp_path / "generation", expected_generation=3)
    assert metadata["write_chunk_bytes"] == 5
    assert restored.digest() == original.digest()


def test_shared_snapshot_view_roundtrip_without_copy():
    original = _snapshot()
    owner, descriptor = SharedSnapshot.from_snapshot(original, prefix="repro_view")
    try:
        viewed = snapshot_view_from_descriptor(owner, descriptor)
        assert viewed.digest() == original.digest()
        assert viewed.arrays["model/w"].base is not None
        del viewed
    finally:
        owner.close(unlink=True)


def test_handle_requires_durable_terminal_state(tmp_path):
    events = EventLog(tmp_path / "events.jsonl")
    handle = Handle("test", 1, "req-1", events)
    handle.mark("admitted")
    handle.mark("source_released")
    handle.mark("input_buffer_released")
    handle.mark("data_completed", sha256="abc")
    handle.mark("persisted", sha256="abc")
    assert handle.wait_persisted(0.1).persisted
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "admitted", "source_released", "input_buffer_released", "data_completed", "persisted"
    ]


def test_handle_records_full_state_machine_timestamps(tmp_path):
    events = EventLog(tmp_path / "events.jsonl")
    handle = Handle("test", 7, "req-7", events)
    for state in (
            CheckpointState.SNAPSHOTTING,
            CheckpointState.SNAPSHOT_READY,
            CheckpointState.QUEUED,
            CheckpointState.DMA_COPYING,
            CheckpointState.NVME_WRITING,
            CheckpointState.FLUSHING,
            CheckpointState.METADATA_COMMITTING,
            CheckpointState.PERSISTED):
        handle.transition(state, sha256="abc" if state == CheckpointState.PERSISTED else None)
    record = handle.as_dict()
    assert record["state"] == "PERSISTED"
    assert record["sha256"] == "abc"
    assert record["timestamps_ns"]["CREATED"] <= record["timestamps_ns"]["PERSISTED"]


def test_formal_semantic_port_gate_requires_restore_and_terminal_generations():
    row = {
        "status": "trend_measured",
        "kind": "npu-semantic-port",
        "formal_steps": 30,
        "checkpoint_count": 10,
        "restore": {"status": "pass"},
        "checkpoints": [{"state": "PERSISTED"} for _ in range(10)],
    }
    assert is_formal_semantic_port(row)
    row["checkpoints"][-1]["state"] = "FAILED"
    assert not is_formal_semantic_port(row)


def test_adapter_evidence_discloses_port_substitutions():
    evidence = adapter_evidence("fastpersist_host")
    assert "Locked upstream core invoked: True" in evidence
    assert "FastFileWriter and AIO" in evidence
    assert "GDS disabled" in evidence
    bytecheckpoint = adapter_evidence("bytecheckpoint_host")
    assert "three-component CKPTCounter" in bytecheckpoint
    assert "extra_state workflow" in bytecheckpoint


def test_native_mindspore_save_adapter_is_distinct_from_raw_reference():
    adapter = ADAPTERS["mindspore_native_save"]
    assert adapter.kind == "framework-native-save"
    status = adapter.preflight({"fs_test_dir": "/models"})
    assert status["api"] == "mindspore.save_checkpoint(async_save=False)"
    assert status["storage"] == "durable XFS file backend"


def test_inventory_records_missing_run_without_guessing(tmp_path):
    config = {
        "project_root": str(tmp_path), "project_commit": "locked",
        "fs_test_dir": str(tmp_path / "fs"), "raw_pci": "0000:83:00.0",
        "same_physical_storage_verified": False,
    }
    report = build_inventory(config, {"ours": str(tmp_path / "missing")})
    row = report["methods"][0]
    assert row["status"] == "missing"
    assert row["reason"] == "source.json missing"
    assert report["storage"]["same_physical_storage_verified"] is False


def test_ours_handle_normalizes_native_persisted_event():
    class Native:
        generation = 4
        request_id = "request-4"
        snapshot_state_digest = None
        def as_dict(self):
            return {"state": "PERSISTED", "status": "PERSISTED",
                    "events": [{"state": "PERSISTED", "monotonic_ns": 12345}]}

    row = __import__("experiments.baselines.repro.adapters.ours",
                     fromlist=["_DirectHandle"])._DirectHandle(
        type("Adapter", (), {"name": "ours"})(), Native(),
        EventLog(Path("/tmp/unused-events.jsonl")),
        expected_state_digest="common-digest").as_dict()
    assert row["persisted_ns"] == 12345
    assert row["sha256"] == "common-digest"


def test_ours_restore_checks_generation_and_forwards_verification_mode():
    from experiments.baselines.repro.adapters.ours import OursAdapter

    class Checkpoint:
        meta_dict = {"checkpoints": {"step_8": {"generation": 42}}}
        calls = []
        def load_state(self, components, step=None, verify_checksums=True):
            self.calls.append((components, step, verify_checksums))
            return {"restored": True}

    adapter = object.__new__(OursAdapter)
    adapter.ckpt = Checkpoint()
    destination = {"model": object(), "optimizer": object(), "_step": 8,
                   "_verify_checksums": False}
    assert adapter.restore(42, destination) == {"restored": True}
    assert adapter.ckpt.calls[0][1:] == (8, False)
    with pytest.raises(ValueError, match="generation mismatch"):
        adapter.restore(43, destination)
