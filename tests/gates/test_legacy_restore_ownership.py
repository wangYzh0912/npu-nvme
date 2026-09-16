"""Run the extracted framework target and transport with bounded Host buffers."""
import ctypes
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from npu_nvme.framework.capture import FrozenCapture
from npu_nvme.framework.restore import LegacyStateTarget
from npu_nvme.runtime.restore import LegacyRestore
from npu_nvme.storage.transport import LegacyBatchTransport
from training_state import encode_control_value


class Parameter:
    shape=(2,)
    dtype='float32'
    def __init__(self): self.array=np.zeros(2,dtype=np.float32)
    def asnumpy(self): return self.array
    def value(self): return self


@pytest.mark.parametrize('fault', ['none','shape','missing','read','checksum'])
def test_legacy_restore_uses_target_and_transport_interfaces(fault):
    param=Parameter()
    framework=SimpleNamespace(dtype_to_nptype=lambda x:x,Tensor=lambda x,dtype:np.asarray(x,dtype=dtype),
                              hal=SimpleNamespace(synchronize=lambda:None))
    operations=SimpleNamespace(assign=lambda p,value:setattr(p,'array',value.copy()))
    capture=FrozenCapture(rank_id=0,device_id=0,framework=framework,acl=None,pointer_of=lambda p:0)
    component=SimpleNamespace(parameters_and_names=lambda:[('x',param)])
    target=LegacyStateTarget({'model':component},capture,framework,operations)
    expected=np.asarray([1,2],dtype=np.float32)
    payload,control=encode_control_value(8)
    saved={'model/x':dict(offset=4096,size=8,shape=[2],dtype='float32',sha256=hashlib.sha256(expected.tobytes()).hexdigest()),
           'control/global_step':dict(offset=8192,size=payload.nbytes,shape=[payload.nbytes],dtype='uint8',**control)}
    record=dict(type='TRAINING_STATE_FULL',schema_version=1,state_step=8,components=['model'],
                control_names=['global_step'],params=saved,chunk_size=4096,checksum='sha256')
    if fault=='shape':saved['model/x']['shape']=[3]
    if fault=='missing':del saved['model/x']
    calls=[]
    def read(ctx,ptrs,offsets,sizes,count):
        calls.append(count)
        if fault=='read':return -5
        for i in range(count):
            raw=expected.tobytes() if offsets[i]==4096 else payload.tobytes()
            if fault=='checksum' and offsets[i]==4096:raw=np.asarray([3,4],np.float32).tobytes()
            ctypes.memmove(ptrs[i],raw,sizes[i])
        return 0
    binding=SimpleNamespace(npu_nvme_read_batch_host=read)
    runtime=LegacyRestore(select_record=lambda step:(8,record),transport=LegacyBatchTransport(binding,None),
                          rank_id=0,chunk_size=4096,schema_version=1)
    if fault=='none':
        assert runtime.load_state(target)=={'global_step':8}
        np.testing.assert_array_equal(param.array,expected)
    else:
        with pytest.raises((ValueError,RuntimeError)):runtime.load_state(target)
        if fault in ('shape','missing'):assert not calls
        if fault=='checksum':
            # B preserves the old in-place apply-before-verify behavior;
            # this is precisely why D1 needs a separate unready target.
            np.testing.assert_array_equal(param.array,np.asarray([3,4],np.float32))


def test_weights_adapter_supplies_pointer_provider(monkeypatch):
    import mindspore as ms
    from npu_nvme.framework.weights import LegacyWeightsRestore
    parameter = ms.Parameter(ms.Tensor([0.0, 0.0]), name='x')
    model = SimpleNamespace(parameters_and_names=lambda: [('x', parameter)])
    records = {'checkpoints': {'step_2': {'params': {
        'x': dict(offset=4096, size=8, shape=[2], dtype='float32')
    }}}}
    commit = SimpleNamespace(state=SimpleNamespace(meta_dict=records), mount=lambda *args: None)
    wanted = np.asarray([4, 7], np.float32)
    def read(ctx, ptrs, offsets, sizes, count):
        assert count == 1 and offsets[0] == 4096 and sizes[0] == wanted.nbytes
        ctypes.memmove(ptrs[0], wanted.ctypes.data, wanted.nbytes)
        return 0
    adapter = LegacyWeightsRestore(binding=SimpleNamespace(npu_nvme_read_batch_host=read),
        context=None, commit=commit, total_bytes=16384, rank_id=0, chunk_size=4096,
        pointer_of=lambda p: 0)
    result = adapter.load(model, step=2)
    assert result[0:2] == (8, 1)
    np.testing.assert_array_equal(parameter.asnumpy(), wanted)
