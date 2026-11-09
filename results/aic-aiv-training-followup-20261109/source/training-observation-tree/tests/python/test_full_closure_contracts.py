"""Admission and publication ordering without ACL, SPDK or a framework."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_abi2_is_rejected_before_native_binding_or_device_initialization(monkeypatch):
    from npu_nvme.storage import bindings
    acl,library=MagicMock(),MagicMock()
    library.npu_nvme_abi_version.return_value=2
    monkeypatch.setattr(bindings.ctypes,'CDLL',MagicMock(side_effect=[acl,library]))
    with pytest.raises(bindings.BackendUnavailable,match='ABI major mismatch'):
        bindings.load_backend('historical-abi2.so')
    assert library.mock_calls==[('npu_nvme_abi_version',(),{})]
    acl.aclrtSetDevice.assert_not_called()


@pytest.mark.parametrize('fail_prepare',[False,True])
def test_all_rank_prepare_barrier_precedes_payload(monkeypatch,tmp_path,fail_prepare):
    from npu_nvme.framework import training_checkpoint as module
    events=[]
    state=SimpleNamespace(schema=[dict(bytes=4)],placement={'x':'device'},
                          read_chunk=object())
    monkeypatch.setattr(module,'BlockingState',lambda *args,**kwargs:state)
    monkeypatch.setattr(module,'digest',lambda value:'schema')
    monkeypatch.setattr(module,'validate_rank',lambda *args:None)
    session=module.TrainingCheckpoint(dict(strategy={},identity={'strategy_sha256':'schema'},
        host_tensor_budget_bytes=4),rank=0,npu_id=0,output=tmp_path)
    monkeypatch.setattr(session,'_open',lambda:None)
    session.executor=SimpleNamespace(transport=SimpleNamespace(capabilities=SimpleNamespace(dma_pool_bytes=4096)))
    def save(*args,**kwargs):
        assert events==['prepare','barrier']
        events.append('payload');return {'generation':1}
    session.client=SimpleNamespace(sequence=0,epoch='epoch',save=save)
    def prepare(backend):
        assert backend['request_id']=='epoch:0' and backend['seconds'] is None
        events.append('prepare')
        if fail_prepare:raise OSError('journal fsync failed')
    def transfer():
        return session.transfer(None,None,operation='save',step=4,partitions={},controls={'step':4},
            prepare=prepare,barrier=lambda:events.append('barrier'))
    if fail_prepare:
        with pytest.raises(OSError,match='journal fsync failed'):transfer()
        assert events==['prepare'] and session.client.failed
    else:
        assert transfer()['receipt']=={'generation':1}
        assert events==['prepare','barrier','payload']


def test_static_rejection_precedes_framework_and_environment_probe(monkeypatch):
    from npu_nvme.cli.qwen_training import preflight
    monkeypatch.setattr('npu_nvme.runtime.resources.qwen_capacity',lambda config:{'admitted':False})
    with pytest.raises(MemoryError,match='preflight resource admission rejected'):
        preflight({'method':'ours'})


def test_runtime_rejection_uses_geometry_without_copying_payload(monkeypatch,tmp_path):
    import ctypes
    import json
    import numpy as np
    from npu_nvme.framework.checkpoint_controller import CheckpointController
    from npu_nvme.runtime import training_catalog as catalog
    strategy=tmp_path/'strategy.json'
    strategy.write_text(json.dumps({'tensors':[{'name':'x','partition':'sharded'}]}))
    parameter=SimpleNamespace(name='x',asnumpy=MagicMock(side_effect=AssertionError('payload copy')))
    monkeypatch.setattr('npu_nvme.framework.parameters.tensor_inventory',
        lambda *args:iter([(parameter,np.dtype('float32'),[1],4)]))
    monkeypatch.setattr('npu_nvme.d2.tp_schema.validate_rank',lambda *args:None)
    def meminfo(kind,free,total):
        ctypes.cast(free,ctypes.POINTER(ctypes.c_size_t))[0]=0
        return 0
    monkeypatch.setattr('npu_nvme.runtime.resources.qwen_capacity',
        lambda *args,**kwargs:{'admitted':False,'hbm_free':kwargs['hbm_free']})
    controller=CheckpointController(dict(method='ours',checkpoint_root=str(tmp_path/'checkpoints'),
        strategy=str(strategy)),rank=0,npu_id=0,output=tmp_path)
    controller.d2=SimpleNamespace(executor=SimpleNamespace(
        backend=SimpleNamespace(acl_lib=SimpleNamespace(aclrtGetMemInfo=meminfo)),
        transport=SimpleNamespace(capabilities=SimpleNamespace(dma_pool_bytes=4096))))
    with pytest.raises(MemoryError,match='resource admission rejected'):
        controller.admit(None,None,step=4,operation='save',barrier=lambda:None)
    parameter.asnumpy.assert_not_called()
    assert catalog.read_checked(tmp_path/'rank_0/resource-save-4.json')['observed_state_bytes']==4
