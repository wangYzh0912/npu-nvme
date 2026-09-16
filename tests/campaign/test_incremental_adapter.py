from types import SimpleNamespace
import numpy as np
import pytest
from npu_nvme.framework.incremental import IncrementalState

class State:
    def __init__(self):self.value=np.arange(4,dtype=np.float32);self.shape=self.value.shape;self.dtype=self.value.dtype
    def asnumpy(self):return self.value.copy()


def test_durable_failure_never_advances_npu_reference(monkeypatch):
    parameter=State();component=SimpleNamespace(parameters_and_names=lambda:[('x',parameter)])
    class Store:
        ledger=None
        def __init__(self,*args,**kwargs):pass
        def save(self,*args,**kwargs):raise OSError('flush failed')
    monkeypatch.setattr('npu_nvme.framework.incremental.PersistentR0',Store)
    calls=[];capture=SimpleNamespace(capture=lambda:np.array([True]),commit_ack=lambda:calls.append('ack'))
    ms=SimpleNamespace(hal=SimpleNamespace(synchronize=lambda:None))
    state=IncrementalState(ms,{'model':component},None,host_budget_bytes=1024,npu_capture=capture)
    with pytest.raises(OSError):state.save({},step=1)
    assert calls==[] and state.failed
    with pytest.raises(RuntimeError):state.save({},step=1)
