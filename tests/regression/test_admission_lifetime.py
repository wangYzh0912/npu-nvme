"""Exercise production save preparation, actual threads, and lease ownership."""
import threading
from types import SimpleNamespace
import time

import pytest
import direct_checkpoint as checkpoint
from direct_checkpoint import DirectCheckpoint, CheckpointHandle
from full_checkpoint_protocol import CheckpointState
from npu_nvme.runtime.scheduler import CheckpointScheduler
from npu_nvme.runtime.commit import LegacyCommitCoordinator


def manager(monkeypatch, slots=1):
    m = object.__new__(DirectCheckpoint)
    m._handle_services = checkpoint.HandleServices()
    m._binding = checkpoint.lib
    m._framework = checkpoint.ms
    m._acl = checkpoint.acl_lib
    from npu_nvme.framework.capture import FrozenCapture
    m._capture = FrozenCapture(rank_id=0, device_id=7, framework=m._framework,
                               acl=m._acl, pointer_of=lambda _: 0)
    m._commit = LegacyCommitCoordinator(None)
    m.admission = 'block'; m._closed = False
    m.checkpoint_slots = m.request_slots = slots
    m._scheduler = CheckpointScheduler(slots, slots)
    m._io_mutex = threading.Lock(); m._sequence_lock = threading.Lock()
    m._snapshot_generation = 0
    m._io_error = None; m.io_thread = None; m.rank_id = 0; m.keep_last_n = 2
    m.total_bytes = m.slot_bytes = 1 << 20; m.chunk_size = 4096
    m.ctx = None; m._spdk_initialized = False; m.meta_dict = {}
    m._get_current_slot_base_offset = lambda step: 4096
    m._prepare_params = lambda model: [dict(name='x',ptr=4096,size=4)]
    m._snapshot_params = lambda params,generation: params
    m.released = []
    m._release_snapshot = lambda params: m.released.append(params)
    m.flush_nvme = lambda: None
    monkeypatch.setattr(checkpoint.ms.hal,'synchronize',lambda: None)
    return m


def test_second_budget_failure_leaves_request_budget_available(monkeypatch):
    m=manager(monkeypatch)
    m._scheduler.snapshot_budget.acquire()
    with pytest.raises(checkpoint.CheckpointBusyError): m._admit_checkpoint(admission='try')
    assert m._scheduler.request_budget.acquire(blocking=False)
    assert not m._scheduler.leases


def test_lease_cannot_release_twice(monkeypatch):
    m=manager(monkeypatch)
    lease=m._admit_checkpoint()
    m._release_checkpoint_slot(lease)
    with pytest.raises(RuntimeError): m._release_checkpoint_slot(lease)
    assert m._scheduler.snapshot_budget.acquire(blocking=False)
    assert not m._scheduler.snapshot_budget.acquire(blocking=False)


def test_close_wakes_submitter_and_retains_accepted_lease(monkeypatch):
    m=manager(monkeypatch)
    lease=m._admit_checkpoint()
    entered=threading.Event(); errors=[]
    def submit():
        entered.set()
        try: m._admit_checkpoint(timeout=30)
        except Exception as e: errors.append(e)
    t=threading.Thread(target=submit);t.start();assert entered.wait(1)
    with pytest.raises(TimeoutError): m.cleanup(timeout=0.02)
    t.join(1)
    assert not t.is_alive() and len(errors)==1 and 'closed' in str(errors[0])
    assert not lease.released and not m._closed
    m._release_checkpoint_slot(lease)
    m.cleanup(timeout=0)


def test_admission_timeout_does_not_release_other_request(monkeypatch):
    m=manager(monkeypatch);lease=m._admit_checkpoint()
    start=time.monotonic()
    with pytest.raises(TimeoutError): m._admit_checkpoint(timeout=0.02)
    assert time.monotonic()-start < 1
    assert m._scheduler.leases=={lease}
    m._release_checkpoint_slot(lease)


@pytest.mark.parametrize('fault',['prepare','chunks','thread_start'])
def test_preworker_failures_release_only_owned_resources(monkeypatch,fault):
    m=manager(monkeypatch)
    def fail(*args,**kwargs): raise RuntimeError('injected '+fault)
    if fault=='prepare': m._prepare_params=fail
    if fault=='chunks': monkeypatch.setattr(checkpoint,'build_chunks',fail)
    if fault=='thread_start': monkeypatch.setattr(checkpoint.threading.Thread,'start',fail)
    with pytest.raises(RuntimeError,match='injected'): m.save(None,1,commit_meta=False)
    assert not m._scheduler.leases and not m._scheduler.active_handles and not m._scheduler.handle_threads
    assert m._scheduler.request_budget.acquire(blocking=False) and m._scheduler.snapshot_budget.acquire(blocking=False)
    assert len(m.released)==(0 if fault=='prepare' else 1)


def test_cancelled_worker_does_not_unlock_foreign_mutex(monkeypatch):
    m=manager(monkeypatch)
    m._io_mutex.acquire()
    actual_start=threading.Thread.start
    def poison_then_start(thread):
        m._poison_checkpoint_queue(None)
        actual_start(thread)
    monkeypatch.setattr(threading.Thread,'start',poison_then_start)
    handle=m.save(None,1,commit_meta=False)
    m.wait_for_io_completion(timeout=1)
    assert m._io_mutex.locked()
    assert handle.status==handle.CANCELLED
    m._io_mutex.release()


def test_barrier_includes_capture_not_yet_registered_as_worker(monkeypatch):
    m=manager(monkeypatch)
    entered=threading.Event();release=threading.Event();errors=[]
    def prepare(model):
        entered.set();assert release.wait(2);raise RuntimeError('capture interrupted')
    m._prepare_params=prepare
    def save():
        try: m.save(None,1)
        except Exception as e: errors.append(e)
    t=threading.Thread(target=save);t.start();assert entered.wait(1)
    with pytest.raises(TimeoutError): m.wait_for_io_completion(timeout=0.01)
    release.set();t.join(1)
    assert not t.is_alive() and len(errors)==1 and not m._scheduler.leases


def test_poison_does_not_publish_running_dma_as_cancelled(monkeypatch):
    m=manager(monkeypatch)
    h=CheckpointHandle(m,'active',1,1);h.state=CheckpointState.DMA_COPYING
    m._scheduler.active_handles.add(h)
    m._poison_checkpoint_queue(None)
    assert h.state==CheckpointState.DMA_COPYING and not h.done()


def test_concurrent_submitters_have_unique_order_and_balanced_leases(monkeypatch):
    m=manager(monkeypatch,slots=8)
    gate=threading.Barrier(8); leases=[];lock=threading.Lock()
    def submit():
        gate.wait(timeout=2)
        lease=m._admit_checkpoint(timeout=1)
        with lock: leases.append(lease)
    threads=[threading.Thread(target=submit) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join(2);assert not t.is_alive()
    assert sorted(x.sequence for x in leases)==list(range(1,9))
    assert sorted(x.generation for x in leases)==list(range(1,9))
    for lease in leases: m._release_checkpoint_slot(lease)
    assert not m._scheduler.leases


def test_reset_returns_acknowledged_failure(monkeypatch):
    m=manager(monkeypatch);m.metadata_generation=0
    failure=RuntimeError('prior write failed');m._io_error=failure;m._scheduler.poisoned=True
    assert m.reset_checkpoint_queue() is failure
    lease=m._admit_checkpoint(timeout=0)
    m._release_checkpoint_slot(lease)


def test_quarantine_has_bounded_owned_byte_ledger(monkeypatch):
    m=manager(monkeypatch)
    lease=m._admit_checkpoint();lease.quarantined=True
    lease.params=[dict(size=4096)];lease.handle=CheckpointHandle(m,'q1',1,1)
    lease.handle.error=RuntimeError('DMA stop unknown')
    report=m.retained_resource_report()
    assert report['snapshots']==[dict(request_id='q1',generation=1,owner='DirectCheckpoint',bytes=4096,
        reason='DMA stop unknown',release_condition='proven native and live DMA stop, then no remaining borrowers')]
    assert report['is_stop_proof'] is False
    with pytest.raises(TimeoutError):m.cleanup(timeout=0)
    assert m._scheduler.leases=={lease} and not m.released
