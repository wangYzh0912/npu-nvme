"""FAKE native executor: verify routing and ownership, no device initialization."""
import ctypes as C
import gc
import hashlib
import threading
import weakref
from types import SimpleNamespace
import pytest
from npu_nvme.storage.bindings import (NPUNVMEContext,NPUNVMERequest,NPUNVMETransferSpec,
    NPUNVMECopySpec,NPUNVMETransferReceipt,NPUNVMETransferDigest)
from npu_nvme.storage.full_transport import FullTransport,TransferFailure
from npu_nvme.framework.full_state import MindSporeRestoreTarget


class Native:
    def __init__(self): self.calls=[];self.disk={};self.pending=False;self.releases=0
    def _accept(self,spec,out):
        self.spec=spec
        handle=C.pointer(NPUNVMERequest())
        C.cast(out,C.POINTER(C.POINTER(NPUNVMERequest)))[0]=handle
        self.handle=handle
        return 0
    def npu_nvme_submit_transfer(self,ctx,raw,out):
        spec=C.cast(raw,C.POINTER(NPUNVMETransferSpec)).contents
        self.calls.append(spec.operation)
        if spec.item_count:
            item=spec.items[0]
            if spec.operation==0: self.disk[item.offset]=C.string_at(item.address,item.length)
            if spec.operation==1: C.memmove(item.address,self.disk[item.offset],item.length)
            self.payload=C.string_at(item.address,item.length)
        return self._accept(spec,out)
    def npu_nvme_submit_copy(self,ctx,raw,out):
        spec=C.cast(raw,C.POINTER(NPUNVMECopySpec)).contents
        self.calls.append(spec.operation)
        if not self.pending:C.memmove(spec.destination,spec.source,spec.length)
        return self._accept(spec,out)
    def npu_nvme_wait_request(self,*args): return -110 if self.pending else 0
    def npu_nvme_poll_request(self,request,done):
        C.cast(done,C.POINTER(C.c_int))[0]=int(not self.pending);return 0
    def npu_nvme_get_transfer_receipt(self,request,out,size):
        r=C.cast(out,C.POINTER(NPUNVMETransferReceipt)).contents
        r.done=r.source_safe=r.transport_safe=1;r.operation=self.spec.operation;r.data_durable=self.spec.operation==4
        return 0
    def npu_nvme_get_transfer_digests(self,request,out,count):
        C.cast(out,C.POINTER(NPUNVMETransferDigest)).contents.sha256[:]=hashlib.sha256(self.payload).digest();return 0
    def npu_nvme_release_request(self,request):self.releases+=1
    def npu_nvme_wait_quiescent(self,*args):return -110 if self.pending else 0


def transport():
    t=FullTransport.__new__(FullTransport);t.lib=Native();t.ctx=C.pointer(NPUNVMEContext())
    t.retained=[];t._buffers_lock=threading.Lock()
    t.acl=SimpleNamespace(aclrtMemcpy=lambda *args:pytest.fail('synchronous ACL path invoked'))
    return t


def test_frozen_hbm_hash_write_and_restore_use_native_requests():
    t=transport();source=C.create_string_buffer(b'payload',7);target=C.create_string_buffer(7)
    item=dict(ptr=C.addressof(source),snapshot_dev_ptr=C.addressof(source),placement='device',owner=source)
    assert t.frozen_bytes(item,0,7)==b'payload'
    assert t.write_frozen(4096,item,0,7)==hashlib.sha256(b'payload').hexdigest()
    t.flush();data=t.read(4096,7);t.copy_h2d(C.addressof(target),data,owner=target)
    assert target.raw==b'payload' and t.lib.calls==[5,0,4,1,6]
    assert t.lib.releases==5 and not t.retained


def test_timeout_retains_both_borrowers_after_handle_release():
    t=transport();t.lib.pending=True
    source=C.create_string_buffer(b'payload',7);destination=C.create_string_buffer(7)
    source_ref=weakref.ref(source);dest_ref=weakref.ref(destination)
    spec=NPUNVMECopySpec(C.sizeof(NPUNVMECopySpec),1,5,0,C.addressof(source),C.addressof(destination),7)
    with pytest.raises(TransferFailure) as caught:t._request(spec,(source,destination),copy=True)
    assert not caught.value.transport_safe and t.lib.releases==1
    del source,destination;gc.collect()
    assert source_ref() is not None and dest_ref() is not None and not t.quiescent()


def test_unproven_copy_prevents_target_discard():
    target=MindSporeRestoreTarget(framework=None,acl=None,npu=7,model=None,optimizer=None,cell=None,identity={})
    target._params={'model/x':dict(ptr=4096,size=7)}
    def fail(*args,**kwargs):raise TransferFailure('unknown DMA',transport_safe=False)
    target.set_copy_executor(fail)
    with pytest.raises(TransferFailure):target.apply_chunk('model/x',0,b'payload')
    assert not target.transport_safe and not target.ready
    with pytest.raises(RuntimeError,match='unproven'):target.discard()


def test_async_config_accepts_real_chunk_and_depth_bounds(tmp_path):
    import json
    from pathlib import Path
    from npu_nvme.cli.contracts import load_config
    root=Path(__file__).resolve().parents[2]
    config=json.loads((root/'config/c2/pilot.json').read_text())
    config['runtime'].update(chunk_bytes=16*1024**2,dma_depth=64)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    assert load_config(path)['runtime']['dma_depth']==64
    config['runtime']['dma_depth']=65;path.write_text(json.dumps(config))
    with pytest.raises(ValueError):load_config(path)
