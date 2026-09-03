from experiments.benchmarks.io4_bottleneck_campaign import aggregate, reactor_decision


def record(path, throughput, reactor_cpu_us=0, queue_wait_ns=0,
           outstanding=0, depth=4):
    elapsed_ns = 1_000_000_000
    return {
        "path": path, "producers": 1, "payload": 1024, "chunk": 4096,
        "depth": depth, "numa_node": 4, "slow_delay_ms": 0.0,
        "sample": 0,
        "result": {
            "status": "pass", "byte_exact": True, "producers": 1,
            "elapsed_ns": elapsed_ns,
            "throughput_bytes_per_second": throughput,
            "coordinator_queue_wait_ns": queue_wait_ns,
            "spdk_stats": {"reactor_cpu_us": reactor_cpu_us,
                           "nvme_outstanding_peak": outstanding},
        },
    }


def test_aggregate_keeps_real_outstanding_and_byte_gate():
    groups = aggregate([record("B0", 100), record("B0", 120,
                                                  outstanding=3)])
    assert groups[0]["throughput_mean"] == 110
    assert groups[0]["nvme_outstanding_peak"] == 3
    assert groups[0]["all_byte_exact"] is True


def test_multi_reactor_never_passes_without_controlled_owner_gain():
    groups = aggregate([
        record("B0", 1000),
        record("B4", 500, reactor_cpu_us=950_000,
               queue_wait_ns=500_000_000, outstanding=1),
    ])
    decision = reactor_decision(groups)
    assert decision["gates"]["single_reactor_cpu_saturated"] is True
    assert decision["gates"]["controlled_multi_owner_gain"] is False
    assert decision["implement_multi_reactor"] is False
