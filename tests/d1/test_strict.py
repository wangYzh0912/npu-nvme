from copy import deepcopy
import threading
import time
import pytest
from npu_nvme.runtime.d1_restore import StrictRestoreSession, StrictRestoreError, RestoreHandle, ObservationTimeout
from npu_nvme.runtime.d1_schema import manifest_digest, validate_record
from conftest import publish


class Target:
    def __init__(self): self.ready=False; self.discarded=False; self.data=[]
    def prepare_restore(self,spec): pass
    def apply_chunk(self,n,o,d): self.data.append((n,o,d))
    def finish_restore(self,spec): pass
    def verify_controls(self,spec): return True
    def mark_ready(self): self.ready=True
    def discard(self): self.discarded=True; self.ready=False


def test_stream_success(coordinator,spec):
    r,_,record,disk,_=publish(coordinator,spec,4)
    target=Target()
    session=StrictRestoreSession(coordinator=coordinator,layout=coordinator.state.layout,reader=lambda o,n:disk[o])
    out,receipt=session.restore_full_state(lambda s:target,spec,4)
    assert out is target and target.ready and receipt.ready
    assert receipt.bytes_read==sum(map(len,disk.values())) and not coordinator._pins


@pytest.mark.parametrize('field',['identity','shape','dtype','controls','checksum','overlap','offset','legacy','whole_digest'])
def test_planning_rejects_before_target_or_read(coordinator,spec,field):
    r,_,record,disk,_=publish(coordinator,spec,4)
    stored=coordinator.state.meta_dict['checkpoints']['generation_1']
    if field=='identity': spec['identity']['model']='wrong'
    elif field=='shape': spec['parameters']['model/x']['shape']=[4096,2]
    elif field=='dtype': spec['parameters']['model/x'].update(dtype='uint16',shape=[4096])
    elif field=='controls': spec['control_names']=[]
    elif field=='checksum': stored['params']['model/x']['chunks'][0].pop('sha256')
    elif field=='overlap': stored['params']['model/x']['offset']=stored['params']['optimizer/m']['offset']
    elif field=='offset': stored['params']['model/x']['chunks'][0]['offset']=4096
    elif field=='legacy': stored.pop('strict_contract')
    else: stored['params']['model/x']['sha256']='z'*64
    stored['manifest_sha256']=manifest_digest(stored)
    calls=[]
    session=StrictRestoreSession(coordinator=coordinator,layout=coordinator.state.layout,reader=lambda *a:calls.append('read'))
    with pytest.raises(ValueError): session.restore_full_state(lambda s:calls.append('target'),spec,4)
    assert calls==[] and not coordinator._pins


@pytest.mark.parametrize('kind',['last_chunk','whole_tensor','controls','apply','mark_ready'])
def test_failure_never_returns_ready_target(coordinator,spec,kind):
    r,_,record,disk,_=publish(coordinator,spec,4)
    target=Target()
    if kind=='last_chunk': disk[max(disk)]=b'y'*len(disk[max(disk)])
    if kind=='whole_tensor':
        stored=coordinator.state.meta_dict['checkpoints']['generation_1']
        stored['params']['optimizer/m']['sha256']='0'*64
        stored['manifest_sha256']=manifest_digest(stored)
    if kind=='controls': target.verify_controls=lambda s:False
    if kind=='apply': target.apply_chunk=lambda *a:(_ for _ in ()).throw(RuntimeError('apply failed'))
    if kind=='mark_ready':
        def fail(): target.ready=True; raise RuntimeError('mark failed')
        target.mark_ready=fail
    session=StrictRestoreSession(coordinator=coordinator,layout=coordinator.state.layout,reader=lambda o,n:disk[o])
    with pytest.raises(RuntimeError): session.restore_full_state(lambda s:target,spec,4)
    assert target.discarded and not target.ready and not coordinator._pins


def test_timeout_only_ends_observation():
    event=threading.Event()
    handle=RestoreHandle(lambda rid:event.wait(5) or True).start()
    try:
        with pytest.raises(ObservationTimeout) as caught: handle.result(time.monotonic()+0.001)
        assert caught.value.handle is handle and not handle.done
    finally: event.set()
    assert handle.result(time.monotonic()+2) is True


def test_unsafe_failure_retains_target_and_pin(coordinator,spec):
    _,_,_,disk,_=publish(coordinator,spec,4); target=Target()
    target.transport_safe=False
    target.apply_chunk=lambda *a:(_ for _ in ()).throw(RuntimeError('DMA error'))
    session=StrictRestoreSession(coordinator=coordinator,layout=coordinator.state.layout,reader=lambda o,n:disk[o])
    with pytest.raises(RuntimeError): session.restore_full_state(lambda s:target,spec,4)
    assert session.quarantined==[target] and not target.ready and not target.discarded
    assert coordinator._pins[1]==1 and coordinator.poisoned
