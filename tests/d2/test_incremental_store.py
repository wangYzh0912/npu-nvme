"""FAKE disk tests of real D2 commit ordering and reference advancement."""
import numpy as np
import pytest
from incremental_manifest import build_training_state_manifest
from npu_nvme.d2.format import Region,SHARED_PAYLOAD
from npu_nvme.d2.incremental import PersistentR0

class Disk:
    def __init__(self):self.data=bytearray(16<<20);self.durable=self.data[:];self.fail=False;self.writes=[]
    def read(self,o,n):return bytes(self.data[o:o+n])
    def write(self,o,data):self.data[o:o+len(data)]=data;self.writes.append((o,len(data)))
    def flush(self):
        if self.fail:raise OSError('flush fault')
        self.durable[:]=self.data
    def crash(self):self.data[:]=self.durable

def setup():
    class P:shape=(4096,);dtype=np.float32
    class M:
        def parameters_and_names(self):return [('weight',P())]
    manifest=build_training_state_manifest({'model':M()},block_size=1024,small_threshold=0)
    disk=Disk();r=Region(disk,offset=0,length=len(disk.data),retention=3);r.format(region_id='F1',features=[SHARED_PAYLOAD])
    return disk,manifest,PersistentR0(r,manifest,chunk_bytes=4096,max_chain_length=2)

def remount(disk,manifest):
    r=Region(disk,offset=0,length=len(disk.data),retention=3);assert r.mount()==[]
    return PersistentR0(r,manifest,chunk_bytes=4096,max_chain_length=2)

def test_lossless_delta_fresh_mount_compaction_and_retention():
    disk,manifest,store=setup();state={'model/weight':np.zeros(4096,np.float32)}
    store.save(state,{'cursor':1},step=1)
    root=store.rows[0]['payload']['offset']
    for step in range(2,7):
        state['model/weight'][step]=-0.0 if step==2 else step
        store.save(state,{'cursor':step},step=step)
        disk.crash();store=remount(disk,manifest);result=store.recover()
        assert result['state']['model/weight'].tobytes()==state['model/weight'].tobytes()
        assert result['controls']=={'cursor':step}
    # Compaction at step 4 starts a new FULL, never rewrites the original root
    # while old retained/fallback generations still reference it.
    assert store.ledger.base_full_generation==4
    assert store.region.current['sequence']==6

def test_failed_flush_cannot_ack_reference():
    disk,manifest,store=setup();state={'model/weight':np.zeros(4096,np.float32)}
    store.save(state,{'cursor':1},step=1);before=store.ledger.persisted['model/weight'].tobytes()
    state['model/weight'][0]=1;disk.fail=True
    with pytest.raises(OSError):store.save(state,{'cursor':2},step=2)
    assert store.ledger.persisted['model/weight'].tobytes()==before
    assert store.ledger.in_flight_generation==1 and store.failed
    disk.fail=False;disk.crash();restored=remount(disk,manifest).recover()
    assert restored['state']['model/weight'].tobytes()==before

def test_retained_old_generation_has_complete_chain():
    disk,manifest,store=setup();state={'model/weight':np.zeros(4096,np.float32)}
    saved={}
    for step in range(1,5):
        state['model/weight'][0]=step;saved[step]=state['model/weight'].tobytes()
        store.save(state,{'cursor':step},step=step)
    for generation in (2,3,4):
        recovered=remount(disk,manifest).recover(generation)
        assert recovered['state']['model/weight'].tobytes()==saved[generation]


def test_wrong_receipt_does_not_advance_ledger(monkeypatch):
    disk,manifest,store=setup();state={'model/weight':np.zeros(4096,np.float32)}
    store.save(state,{'cursor':1},step=1)
    real=store.region.commit
    def wrong(*args,**kwargs):
        receipt=real(*args,**kwargs);receipt['request_id']='foreign';return receipt
    monkeypatch.setattr(store.region,'commit',wrong)
    state['model/weight'][1]=2
    with pytest.raises(ValueError,match='receipt'):store.save(state,{'cursor':2},step=2)
    assert store.ledger.persisted_generation==0 and store.ledger.in_flight_generation==1
    # Durable storage may have succeeded, so a fresh owner resolves it.
    restored=remount(disk,manifest).recover()
    assert restored['state']['model/weight'].tobytes()==state['model/weight'].tobytes()


def test_corrupt_frame_payload_rejects_without_publishing_reference():
    disk,manifest,store=setup();state={'model/weight':np.zeros(4096,np.float32)}
    store.save(state,{},step=1);state['model/weight'][0]=1;store.save(state,{},step=2)
    ref=next(row['payload'] for row in store.rows if row['name'].startswith('delta/'))
    disk.data[ref['offset']]^=1
    reader=remount(disk,manifest)
    with pytest.raises(ValueError,match='integrity'):reader.recover()
    assert reader.ledger is None
