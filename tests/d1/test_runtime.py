import hashlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from npu_nvme.runtime.d1_runtime import FullRuntime
from npu_nvme.runtime.d1_restore import ObservationTimeout
from npu_nvme.runtime.scheduler import CheckpointBusyError
from test_strict import Target


class Transport:
    def __init__(self):
        self.disk={}; self.flushes=0; self.safe=True; self.closed=False
        self.gate=None; self.entered=threading.Event(); self.fail=False
    def frozen_bytes(self,item,offset,size): return item['raw'][offset:offset+size]
    def write(self,offset,data):
        self.entered.set()
        if self.gate is not None: assert self.gate.wait(5)
        if self.fail: raise IOError('write failed')
        self.disk[offset]=data
    def read(self,offset,size): return self.disk[offset]
    def flush(self): self.flushes+=1
    def quiescent(self): return self.safe
    def close(self,timeout): self.closed=True


def runtime(c,spec):
    transport=Transport(); released=[]
    def prepare(components,controls,expected):
        params=[dict(info,name=name,raw=b'x'*info['size']) for name,info in spec['parameters'].items()]
        params += [dict(name='control/'+n,shape=[4],dtype='uint8',size=4,raw=b'abcd') for n in spec['control_names']]
        return params
    return FullRuntime(c,transport,prepare=prepare,freeze=lambda p,g:p,
                       release=lambda p:released.append(p),chunk_size=4096),transport,released


def test_save_close_reopen_restore(coordinator,spec):
    rt,t,released=runtime(coordinator,spec)
    h=rt.save({}, {}, 4,spec).wait(5)
    assert h.status=='PERSISTED' and len(released)==1 and t.flushes==1
    from npu_nvme.runtime.d1_commit import D1CommitCoordinator, OutcomeUnknown
    reopened=D1CommitCoordinator(metadata_io=coordinator.io,state=coordinator.state)
    with pytest.raises(OutcomeUnknown): reopened.resolve(h.request_id)
    restored,t2,unused=runtime(reopened,spec); t2.disk=t.disk
    target,receipt=restored.begin_restore(lambda s:Target(),spec,4).result(time.monotonic()+5)
    assert target.ready and receipt.ready and receipt.digest==h.snapshot_state_digest
    rt.close(); restored.close()
    assert t.closed and t2.closed


def test_busy_timeout_and_late_success(coordinator,spec):
    rt,t,released=runtime(coordinator,spec); t.gate=threading.Event()
    h=rt.save({}, {}, 4,spec)
    try:
        assert t.entered.wait(2)
        with pytest.raises(CheckpointBusyError): rt.save({}, {}, 5,spec,admission='try')
        with pytest.raises(ObservationTimeout): h.wait(0.001)
        assert not released and h.status=='DISPATCHED'
        with pytest.raises(TimeoutError): rt.drain(0.001)
    finally: t.gate.set()
    h.wait(2)
    assert h.status=='PERSISTED' and len(released)==1
    rt.close()


def test_start_failure_releases_exactly_once(coordinator,spec):
    rt,t,released=runtime(coordinator,spec)
    with patch('threading.Thread.start',side_effect=RuntimeError('cannot start')):
        with pytest.raises(RuntimeError): rt.save({}, {}, 4,spec)
    assert len(released)==1 and not rt._busy and not rt._handles and not coordinator._reservations
    rt.save({}, {}, 5,spec).wait(2)


def test_unsafe_write_failure_retains_frozen_buffers(coordinator,spec):
    rt,t,released=runtime(coordinator,spec); t.fail=True; t.safe=False
    h=rt.save({}, {}, 4,spec)
    with pytest.raises(IOError): h.wait(2)
    assert rt.quarantined and not released and coordinator.poisoned
    with pytest.raises(RuntimeError): rt.close()
    assert not t.closed


def test_close_wakes_blocked_submitter(coordinator,spec):
    rt,t,released=runtime(coordinator,spec); t.gate=threading.Event()
    h=rt.save({}, {}, 4,spec); errors=[]
    def submit():
        try: rt.save({}, {}, 5,spec)
        except RuntimeError as e: errors.append(e)
    thread=threading.Thread(target=submit); thread.start()
    try:
        with pytest.raises(TimeoutError): rt.close(0.001)
        thread.join(2)
        assert not thread.is_alive() and errors
    finally: t.gate.set()
    h.wait(2); rt.close()
