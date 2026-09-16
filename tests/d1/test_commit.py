from copy import deepcopy
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import pytest
from conftest import publish, make_record
from npu_nvme.runtime.d1_commit import NoSpace, D1CommitError, OutcomeUnknown, D1CommitCoordinator


def test_duplicate_conflict_and_identity(coordinator,spec):
    r,t,record,disk,receipt=publish(coordinator,spec,3)
    before=len(coordinator.io.history)
    assert coordinator.publish(r,t,record) is receipt
    assert len(coordinator.io.history)==before
    with pytest.raises(D1CommitError): coordinator.publish(r,replace(t,digest='0'*64),record)
    assert coordinator.resolve(r.request_id)==receipt
    with pytest.raises(OutcomeUnknown): coordinator.resolve('missing')


def test_wraparound_retires_fallback_before_reuse(coordinator,spec):
    previous=[]
    for step in range(7):
        r,t,record,disk,receipt=publish(coordinator,spec,step)
        before,new=coordinator.io.history[-2:]
        assert r.slot not in {v['slot'] for v in before['checkpoints'].values()}
        assert before['checkpoints']=={k:v for k,v in previous}
        assert receipt.metadata_generation==2*(step+1)
        assert len(new['checkpoints'])==min(2,step+1)
        previous=list(new['checkpoints'].items())
    assert receipt.retained_generations==(6,7)


def test_pins_are_reference_counted(coordinator,spec):
    publish(coordinator,spec,1); publish(coordinator,spec,2)
    with coordinator.selected(1):
        with coordinator.selected(1):
            assert coordinator._pins[1]==2
            with pytest.raises(NoSpace): coordinator.reserve(step=3)
        with pytest.raises(NoSpace): coordinator.reserve(step=3)
    publish(coordinator,spec,3)
    with pytest.raises(FileNotFoundError):
        with coordinator.selected(1): pass


def test_reservation_blocks_late_pin_and_other_reservations(coordinator,spec):
    publish(coordinator,spec,1); publish(coordinator,spec,2)
    r=coordinator.reserve(step=3)
    with pytest.raises(NoSpace):
        with coordinator.selected(1): pass
    with pytest.raises(NoSpace): coordinator.reserve(step=4)
    coordinator.abort(r)


def test_publish_failure_does_not_mutate_committed_catalog(coordinator,spec):
    publish(coordinator,spec,1)
    r=coordinator.reserve(step=2); record,_=make_record(coordinator,r,spec)
    before=deepcopy(coordinator.state.meta_dict)
    from npu_nvme.types import TransferReceipt
    t=TransferReceipt(r.request_id,r.generation,sum(p['size'] for p in record['params'].values()),
                      sum(len(p['chunks']) for p in record['params'].values()),True,record['data_sha256'])
    coordinator.io.fail=True
    with pytest.raises(IOError): coordinator.publish(r,t,record)
    assert coordinator.state.meta_dict==before and coordinator.poisoned
    with pytest.raises(OutcomeUnknown): coordinator.resolve(r.request_id)
    with pytest.raises(D1CommitError): coordinator.reserve(step=3)


def test_terminal_retention_and_restart_unknown(coordinator,spec):
    first=coordinator.reserve(step=1); coordinator.abort(first)
    for i in range(1024):
        r=coordinator.reserve(step=i); coordinator.abort(r)
    with pytest.raises(OutcomeUnknown): coordinator.resolve(first.request_id)
    assert len(coordinator._terminal)==1024
    reopened=D1CommitCoordinator(metadata_io=coordinator.io,state=coordinator.state)
    assert reopened.writer_epoch!=coordinator.writer_epoch
    with pytest.raises(OutcomeUnknown): reopened.resolve(r.request_id)


def test_concurrent_reservations_have_one_winner(coordinator):
    def reserve(i):
        try: return coordinator.reserve(step=i)
        except NoSpace: return None
    with ThreadPoolExecutor(4) as pool: results=list(pool.map(reserve,range(4)))
    winners=[r for r in results if r]
    assert len(winners)==1 and winners[0].writer_epoch==coordinator.writer_epoch
    coordinator.abort(winners[0])
