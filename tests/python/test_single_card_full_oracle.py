import pytest

from experiments.benchmarks.run_single_card_full import (
    continuation_oracle,
    request_timing,
)


def test_continuation_oracle_prefers_source_process_trajectory():
    losses, name = continuation_oracle(
        {"continuous_losses": [1.0, 2.0, 3.0, 4.0]},
        {"losses": [10.0, 20.0, 30.0, 40.0]},
        2,
    )
    assert losses == [3.0, 4.0]
    assert name == "source_process_continuation"


def test_continuation_oracle_supports_pre_change_results():
    losses, name = continuation_oracle(
        {}, {"losses": [10.0, 20.0, 30.0]}, 1)
    assert losses == [20.0, 30.0]
    assert name == "independent_process_baseline"


def test_request_timing_uses_persisted_event_not_wait_time():
    timing = request_timing({
        "api_enter_ns": 1_000_000_000,
        "events": [
            {"state": "CREATED", "monotonic_ns": 2_000_000_000},
            {"state": "PERSISTED", "monotonic_ns": 3_500_000_000},
        ],
    })
    assert timing == {"persist_seconds": 2.5, "state_machine_seconds": 1.5}


def test_request_timing_rejects_missing_or_reversed_events():
    with pytest.raises(ValueError, match="missing"):
        request_timing({"api_enter_ns": 1, "events": []})
    with pytest.raises(ValueError, match="not monotonic"):
        request_timing({
            "api_enter_ns": 10,
            "events": [
                {"state": "CREATED", "monotonic_ns": 20},
                {"state": "PERSISTED", "monotonic_ns": 15},
            ],
        })
