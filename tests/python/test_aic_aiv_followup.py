import pytest

from tools.aic_aiv_followup import (
    classify_windows,
    compare_variants,
    normalize_hbm_csv,
    normalize_kv_layout,
    record_kv_tensor,
)


def test_kv_layout_checks_strides_and_preserves_block_metadata():
    rows = normalize_kv_layout([{
        "run_id": "base", "layer": 0, "cache_kind": "key", "phase": "decode",
        "shape": [2, 3, 4], "stride": [12, 4, 1], "dtype": "bf16",
        "block_size": 64, "block_table_hash": "abc",
    }])
    assert rows[0]["contiguous"] is True
    assert rows[0]["block_size"] == 64
    assert rows[0]["block_table_hash"] == "abc"
    singleton = normalize_kv_layout([{
        "run_id": "base", "layer": 0, "cache_kind": "key", "phase": "decode",
        "shape": [2, 1, 4], "stride": [4, 99, 1], "dtype": "bf16",
    }])
    assert singleton[0]["contiguous"] is True
    with pytest.raises(ValueError, match="shape/stride"):
        normalize_kv_layout([{"run_id": "bad", "layer": 0, "cache_kind": "key", "phase": "decode",
                              "shape": [2, 3], "stride": [1], "dtype": "bf16"}])


def test_kv_runtime_probe_records_metadata_without_reading_values(tmp_path):
    class TensorMetadata:
        shape = (2, 4)
        dtype = "bf16"

        def stride(self):
            return (4, 1)

        def storage_offset(self):
            return 3

    path = tmp_path / "kv.jsonl"
    row = record_kv_tensor(path, TensorMetadata(), run_id="r0", layer=2,
                           cache_kind="key", phase="decode",
                           source="attention_v1.py:823", cache_layout="BNSD", block_size=64)
    assert row["storage_offset"] == 3
    assert row["phase"] == "decode"
    assert normalize_kv_layout([__import__("json").loads(path.read_text())])[0]["contiguous"]


def test_hbm_import_requires_explicit_units_and_monotonic_time(tmp_path):
    path = tmp_path / "hbm.csv"
    path.write_text("timestamp_us,read_bytes_per_s,write_bytes_per_s,device_id\n1,10,20,4\n2,30,40,4\n")
    rows = normalize_hbm_csv(path)
    assert rows[1]["timestamp_ns"] == 2000
    assert rows[1]["write_bytes_per_s"] == 40
    path.write_text("timestamp,read,write\n1,10,20\n")
    with pytest.raises(ValueError, match="requires"):
        normalize_hbm_csv(path)


def test_window_classifier_does_not_infer_readiness_without_dependency_data():
    windows = [{"window_start_ns": 0, "window_end_ns": 10,
                "aic_activity_pct": 2, "aiv_activity_pct": 80, "valid": True}]
    tasks = [{"task_id": "a", "engine": "aic", "start_ns": 1, "end_ns": 4}]
    assert classify_windows(windows, tasks, [])[0]["classification"] == "dependency_unknown"
    ready = [{"task_id": "a", "state": "ready", "source": "runtime_event"}]
    result = classify_windows(windows, tasks, ready)[0]
    assert result["classification"] == "aic_ready_candidate"
    assert result["confidence"] == "dependency_evidence"


def test_variant_screen_requires_paired_correct_replicates():
    rows = [
        {"variant": "baseline", "replicate": "0", "latency_ms": 100,
         "output_sha256": "same", "valid": True},
        {"variant": "kv-control", "replicate": "0", "latency_ms": 90,
         "output_sha256": "same", "valid": True},
    ]
    result = compare_variants(rows)["results"][0]
    assert result["verdict"] == "retain_for_confirmation"
    rows[1]["output_sha256"] = "different"
    assert compare_variants(rows)["results"][0]["verdict"] == "reject_incomplete_or_incorrect"


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_hbm_rejects_nonfinite_bandwidth(tmp_path, value):
    path = tmp_path / 'hbm.csv'
    path.write_text(f'timestamp_ns,read_bytes_per_s,write_bytes_per_s\n1,{value},0\n')
    with pytest.raises(ValueError, match='non-finite'):
        normalize_hbm_csv(path)


def test_hbm_tracks_monotonic_time_per_device_and_preserves_fractional_us(tmp_path):
    path = tmp_path / 'hbm.csv'
    path.write_text('timestamp_us,read_bytes_per_s,write_bytes_per_s,device_id\n1.001,2,3,4\n1.001,4,5,5\n1.002,6,7,4\n')
    assert [r['timestamp_ns'] for r in normalize_hbm_csv(path)] == [1001, 1001, 1002]


def test_compare_rejects_duplicate_replicates():
    b = dict(variant='baseline', replicate=0, latency_ms=100, output_sha256='x', valid=True)
    v = dict(b, variant='control', latency_ms=90)
    with pytest.raises(ValueError, match='duplicate experiment'):
        compare_variants([b, v, v])


@pytest.mark.parametrize('field,value', [('valid', 'false'), ('latency_ms', 0), ('latency_ms', float('nan'))])
def test_compare_rejects_invalid_boundary_values(field, value):
    b = dict(variant='baseline', replicate=0, latency_ms=100, output_sha256='x', valid=True)
    with pytest.raises(ValueError):
        compare_variants([b, dict(b, variant='control', **{field: value})])


def test_compare_does_not_pair_different_workloads():
    b = dict(case='b1', variant='baseline', replicate=0, latency_ms=100, output_sha256='x', valid=True)
    result = compare_variants([b, dict(b, case='b8', variant='control')])
    assert result['results'][0]['verdict'] == 'reject_incomplete_or_incorrect'


def test_multiple_baseline_cases_do_not_count_as_candidate_variants():
    b = dict(case='b1', variant='baseline', replicate=0, latency_ms=100,
             output_sha256='x', valid=True)
    with pytest.raises(ValueError, match='at least one candidate'):
        compare_variants([b, dict(b, case='b8')])
