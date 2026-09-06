import json
from pathlib import Path

import numpy as np

from experiments.baselines.repro.host_bridge import (
    SharedSnapshot, snapshot_from_descriptor, snapshot_view_from_descriptor)
from experiments.baselines.repro.protocol import EventLog, Handle
from experiments.baselines.repro.state_bridge import Snapshot, load_raw_snapshot, save_raw_snapshot


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
