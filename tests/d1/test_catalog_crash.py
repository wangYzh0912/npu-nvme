"""Faults around the real V2 catalog write/flush sequence and payload reuse."""
from copy import deepcopy
import ctypes
import hashlib
import pytest
from conftest import make_record
from npu_nvme.runtime.commit import MetadataState
from npu_nvme.runtime.d1_commit import D1CommitCoordinator
from npu_nvme.runtime.d1_schema import validate_record
from npu_nvme.storage.metadata import MetadataIO
from npu_nvme.storage.format import pack_metadata,pack_superblock
from npu_nvme.storage.layout import META_SLOT_A_OFFSET,META_SLOT_B_OFFSET,SUPERBLOCK_OFFSET
from npu_nvme.types import TransferReceipt


class Disk:
    def __init__(self, layout):
        meta={'strict_contract':'D1','catalog_revision':0,'checkpoints':{}}
        self.blocks={META_SLOT_A_OFFSET:pack_metadata(meta,0),META_SLOT_B_OFFSET:pack_metadata(meta,0),
                     SUPERBLOCK_OFFSET:pack_superblock(layout)}
        self.durable=deepcopy(self.blocks)
        self.point=None; self.events=0
    def event(self):
        self.events+=1
        if self.events==self.point: raise IOError('injected process interruption')
    def npu_nvme_sync_meta_io(self,ctx,offset,size,read,pointer):
        if read: ctypes.memmove(pointer,self.blocks[offset],size); return 0
        self.event(); self.blocks[offset]=ctypes.string_at(pointer,size); self.event(); return 0
    def npu_nvme_flush(self,ctx):
        self.event(); self.durable=deepcopy(self.blocks); self.event(); return 0
    def crash(self): self.blocks=deepcopy(self.durable); self.point=None


def commit_payload(c,disk,spec,step):
    r=c.reserve(step=step); record,payload=make_record(c,r,spec)
    disk.blocks.update(payload)
    disk.npu_nvme_flush(None)
    total,count=validate_record(record,c.state.layout)
    return c.publish(r,TransferReceipt(r.request_id,r.generation,total,count,True,record['data_sha256']),record)


@pytest.mark.parametrize('stage',['retire','publish'])
@pytest.mark.parametrize('point',range(1,9))
def test_all_catalog_write_flush_crash_windows(coordinator,spec,stage,point):
    layout=coordinator.state.layout
    disk=Disk(layout)
    io=MetadataIO(disk)
    state=MetadataState(); io.mount(state,layout.total_bytes,0)
    c=D1CommitCoordinator(metadata_io=io,state=state)
    commit_payload(c,disk,spec,1); commit_payload(c,disk,spec,2); commit_payload(c,disk,spec,3)
    # Both replicas now refer to all three physical slots; the next reserve
    # must retire fallback generation 1 before reusing its payload slot.
    disk.events=0
    if stage=='retire':
        disk.point=point
        with pytest.raises(IOError): c.reserve(step=4)
    else:
        r=c.reserve(step=4); record,payload=make_record(c,r,spec)
        disk.blocks.update(payload); disk.npu_nvme_flush(None)
        disk.events=0; disk.point=point
        total,count=validate_record(record,layout)
        with pytest.raises(IOError):
            c.publish(r,TransferReceipt(r.request_id,r.generation,total,count,True,record['data_sha256']),record)
    disk.crash()
    mounted=MetadataState()
    io.mount(mounted,layout.total_bytes,0)
    fresh=D1CommitCoordinator(metadata_io=io,state=mounted)
    assert fresh.snapshot().active_generation in (3,4)
    for rec in fresh.snapshot().records:
        for info in rec['params'].values():
            for chunk in info['chunks']:
                data=disk.blocks[info['offset']+chunk['offset']]
                assert hashlib.sha256(data).hexdigest()==chunk['sha256']


def test_corrupt_active_replica_never_falls_back_to_reused_payload(coordinator,spec):
    layout=coordinator.state.layout; disk=Disk(layout); io=MetadataIO(disk)
    state=MetadataState(); io.mount(state,layout.total_bytes,0)
    c=D1CommitCoordinator(metadata_io=io,state=state)
    for step in range(1,7): commit_payload(c,disk,spec,step)
    c.reserve(step=7)  # retirement is durable before overwriting the spare slot
    slot=next(iter(c._reservations.values())).slot
    start=layout.full_base+slot*layout.full_slot_bytes
    for offset in list(disk.blocks):
        if start<=offset<start+layout.full_slot_bytes: disk.blocks[offset]=b'corrupt'
    active=META_SLOT_A_OFFSET if state.active_meta_slot==0 else META_SLOT_B_OFFSET
    disk.blocks[active]=b'\0'*len(disk.blocks[active])
    mounted=MetadataState(); io.mount(mounted,layout.total_bytes,0)
    fresh=D1CommitCoordinator(metadata_io=io,state=mounted)
    assert fresh.snapshot().active_generation==6
    assert slot not in {r['slot'] for r in fresh.snapshot().records}
