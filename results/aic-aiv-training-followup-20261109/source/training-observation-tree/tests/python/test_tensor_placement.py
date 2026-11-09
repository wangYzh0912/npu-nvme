import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from npu_nvme.framework.parameters import UnknownTensorAddress, validate_allocation


def allocation(**overrides):
    return dict(dict(placement='device', pointer=4096, allocation_bytes=16,
                     device_id=0, provenance='mindspore.DeviceAddress'), **overrides)


def test_nonzero_host_address_is_never_a_device_pointer():
    row=allocation(placement='host', pointer=0x378d5260)
    result=validate_allocation(row, required_bytes=8, expected_device=0)
    assert result.placement=='host' and result.pointer==0


@pytest.mark.parametrize('change', [dict(placement='unknown'), dict(pointer=0),
    dict(pointer=True), dict(allocation_bytes=7), dict(device_id=1),
    dict(pointer=(1<<64)-4), dict(provenance='nonzero_data_ptr')])
def test_unsafe_allocation_is_rejected_before_dma(change):
    with pytest.raises(UnknownTensorAddress):
        validate_allocation(allocation(**change), required_bytes=8, expected_device=0)


def test_actual_device_allocation_keeps_direct_pointer():
    assert validate_allocation(allocation(), required_bytes=8, expected_device=0).pointer==4096


def test_host_control_save_and_restore_never_call_dma(monkeypatch):
    from npu_nvme.framework import d2_state
    class Parameter:
        name='dropout.offset'; shape=(); dtype=np.int64
        def __init__(self): self.value=np.array(0,dtype=np.int64); self.data=self
        def _data_ptr(self): return int(self.value.ctypes.data)
        def asnumpy(self): return self.value
        def set_data(self,value): self.value=value
    parameter=Parameter()
    address=validate_allocation(allocation(placement='host'),required_bytes=8)
    monkeypatch.setattr(d2_state,'describe_tensor',lambda *a,**kw:address)
    framework=SimpleNamespace(hal=SimpleNamespace(synchronize=lambda:None),
        dtype_to_nptype=lambda d:d, Tensor=lambda value,dtype:np.asarray(value,dtype=dtype))
    state=d2_state.BlockingState(framework,
        SimpleNamespace(parameters_and_names=lambda:[(parameter.name,parameter)]),
        rank=0,executor=SimpleNamespace(npu_id=0),host_tensor_budget=8,
        partitions={parameter.name:'per_rank_control'})
    with state.read_chunk(parameter.name,0,8) as raw: assert bytes(raw)==bytes(8)
    changed=np.array(9,dtype=np.int64).tobytes()
    state.apply_chunk(parameter.name,0,changed,hashlib.sha256(changed).hexdigest())
    state.verify_finished()
    assert parameter.value==9


def test_changed_allocation_is_rejected_before_copy(monkeypatch):
    from npu_nvme.framework import d2_state
    original=validate_allocation(allocation(),required_bytes=8)
    changed=validate_allocation(allocation(pointer=8192),required_bytes=8)
    monkeypatch.setattr(d2_state,'describe_tensor',lambda *a,**kw:changed)
    state=object.__new__(d2_state.BlockingState)
    state.executor=SimpleNamespace(npu_id=0)
    with pytest.raises(ValueError,match='allocation changed'):
        state._check_address(dict(param=SimpleNamespace(name='weight'),bytes=8,address=original))
