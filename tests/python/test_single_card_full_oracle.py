from experiments.benchmarks.run_single_card_full import continuation_oracle


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
