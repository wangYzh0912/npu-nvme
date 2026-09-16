from test_blocking_state import state


def test_one_export_per_tensor_and_release_at_end(monkeypatch):
    value,param=state(monkeypatch)
    original=param.asnumpy;calls=[]
    def export():calls.append(1);return original()
    param.asnumpy=export
    chunks=[]
    for offset in range(0,20,4):
        with value.read_chunk('x',offset,4) as data:chunks.append(bytes(data))
    assert b''.join(chunks)==param.value.tobytes()
    assert len(calls)==1 and value.host_source is None
