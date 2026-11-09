from direct_checkpoint import DirectCheckpoint

def test_live_async_explicitly_retired():
    capability = DirectCheckpoint.live_async_capability()
    assert capability['supported'] is False
    assert capability['code'] == 'RETIRED_NONSTRICT_FULL'
