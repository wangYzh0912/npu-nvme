import copy
import pytest
from npu_nvme.d2.format import Region,SHARED_PAYLOAD

class Disk:
    def __init__(self):self.data=bytearray(8<<20);self.writes=[]
    def read(self,o,n):return bytes(self.data[o:o+n])
    def write(self,o,data):self.data[o:o+len(data)]=data;self.writes.append((o,len(data)))
    def flush(self):pass

def region(disk):return Region(disk,offset=0,length=len(disk.data),retention=3)
def row(name,ref):return dict(rank=0,name=name,shape=[ref['logical_bytes']],dtype='uint8',partition='replicated',logical_offset=0,logical_bytes=ref['logical_bytes'],payload=ref)

def test_shared_full_root_survives_retention_and_remount():
    disk=Disk();r=region(disk);r.format(region_id='F1',features=[SHARED_PAYLOAD])
    r.begin('full',{});root=r.payload(b'FULL');rows=[row('root',root)];r.commit(rows,step=1,topology={'world_size':1})
    for i in range(2,9):
        r.begin(str(i),{});ref=r.payload(bytes([i])*513)
        r.commit([*rows,row('delta',ref)],step=i,topology={'world_size':1},inherited_payloads=[root])
        r=region(disk);assert r.mount()==[]
        assert r.verify_payloads(r.current['state']['generations'][0])
    assert disk.read(root['offset'],4)==b'FULL'
    assert sum(o==root['offset'] for o,n in disk.writes)==1

def test_unformatted_feature_cannot_share():
    disk=Disk();r=region(disk);r.format(region_id='old')
    r.begin('full',{});ref=r.payload(b'FULL');r.commit([row('root',ref)],step=1,topology={})
    r.begin('delta',{})
    with pytest.raises(ValueError,match='feature'):r.commit([row('root',ref)],step=2,topology={},inherited_payloads=[ref])

def test_uncommitted_or_modified_inherited_ref_rejected():
    disk=Disk();r=region(disk);r.format(region_id='F1',features=[SHARED_PAYLOAD])
    r.begin('full',{});ref=r.payload(b'FULL');r.commit([row('root',ref)],step=1,topology={})
    changed=copy.deepcopy(ref);changed['logical_sha256']='0'*64
    r.begin('delta',{})
    with pytest.raises(ValueError,match='not committed'):r.commit([row('root',changed)],step=2,topology={},inherited_payloads=[changed])
    fresh=region(disk);assert fresh.mount()==[] and fresh.current['sequence']==1
