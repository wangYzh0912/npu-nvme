from direct_checkpoint import DirectCheckpoint


def test_live_async_reports_strict_unsupported_reason():
    capability = DirectCheckpoint.live_async_capability()
    assert capability["supported"] is False
    assert capability["code"] == "BOUNDED_DMA_POOL_HAS_NO_DEVICE_AGGREGATE_FENCE"
    assert "events" in capability["detail"]
