from direct_checkpoint import DirectCheckpoint


def test_live_async_reports_generation_staging_capability():
    capability = DirectCheckpoint.live_async_capability()
    assert capability["supported"] is True
    assert capability["code"] == "SUPPORTED_GENERATION_PINNED_STAGING"
    assert "split" in capability["required"]
