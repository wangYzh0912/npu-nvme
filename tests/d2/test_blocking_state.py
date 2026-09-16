from types import SimpleNamespace
from contextlib import contextmanager
import hashlib
import numpy as np
import pytest
from npu_nvme.framework.d2_state import BlockingState
from npu_nvme.framework.parameters import TensorAddress

class Parameter:
    name='x';shape=(5,);dtype=np.dtype('float32')
    def __init__(self):self.value=np.arange(5,dtype=self.dtype)
    def asnumpy(self):return self.value.copy()
    def set_data(self,value):self.value=np.asarray(value).copy()


def state(monkeypatch,pointer=0,budget=20):
    address=TensorAddress('device' if pointer else 'host',pointer,0 if pointer else None,20,'mindspore.DeviceAddress')
    monkeypatch.setattr('npu_nvme.framework.d2_state.describe_tensor',lambda *a,**kw:address)
    param=Parameter();network=SimpleNamespace(parameters_and_names=lambda:[('x',param)])
    ms=SimpleNamespace(hal=SimpleNamespace(synchronize=lambda:None),dtype_to_nptype=lambda dtype:dtype,
                       Tensor=lambda value,dtype:np.asarray(value,dtype=dtype))
    return BlockingState(ms,network,rank=0,executor=SimpleNamespace(npu_id=0),host_tensor_budget=budget,partitions={'x':'replicated'}),param

def test_host_capture_restore_exact(monkeypatch):
    s,p=state(monkeypatch);raw=p.value.tobytes()
    with s.read_chunk('x',4,12) as data:assert data.tobytes()==raw[4:16]
    p.value.fill(0)
    for offset in range(0,len(raw),8):
        chunk=raw[offset:offset+8];s.apply_chunk('x',offset,chunk,hashlib.sha256(chunk).hexdigest())
    s.verify_finished();assert p.value.tobytes()==raw

def test_host_budget_before_capture(monkeypatch):
    with pytest.raises(MemoryError):state(monkeypatch,budget=19)

def test_corrupt_chunk_rejected_without_mutation(monkeypatch):
    s,p=state(monkeypatch);before=p.value.tobytes()
    with pytest.raises(ValueError):s.apply_chunk('x',0,b'abcd','0'*64)
    assert p.value.tobytes()==before
