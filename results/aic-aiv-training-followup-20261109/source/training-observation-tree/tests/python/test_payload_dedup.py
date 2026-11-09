from npu_nvme.runtime.payload_dedup import deduplicate_completed


def test_completed_identical_payloads_share_storage_and_preserve_bytes(tmp_path):
    first=tmp_path/'first';second=tmp_path/'second';first.mkdir();second.mkdir()
    data=b'a'*(1<<20)
    a=first/'model.safetensors';b=second/'model.safetensors'
    a.write_bytes(data);b.write_bytes(data)
    cache=tmp_path/'cache'
    deduplicate_completed(first,cache);deduplicate_completed(second,cache)
    assert a.stat().st_ino==b.stat().st_ino
    assert a.read_bytes()==b.read_bytes()==data
