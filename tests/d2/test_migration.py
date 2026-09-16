"""FAKE source validator: test migration copying/order, not D1 validation."""
import copy
import hashlib
import pytest
from npu_nvme.d2.format import Region
from npu_nvme.d2.v2_migration import migrate

class Disk:
    def __init__(self):self.data=bytearray(8<<20);self.writes=[]
    def read(self,o,n):return bytes(self.data[o:o+n])
    def write(self,o,data):self.data[o:o+len(data)]=data;self.writes.append((o,len(data)))
    def flush(self):pass


def fixture():
    disk=Disk();data=bytes(range(256))*33;disk.data[:len(data)]=data
    chunks=[dict(offset=o,size=len(data[o:o+4096]),sha256=hashlib.sha256(data[o:o+4096]).hexdigest()) for o in range(0,len(data),4096)]
    record=dict(strict_contract='D1',type='TRAINING_STATE_FULL',state_step=8,data_sha256=hashlib.sha256(data).hexdigest(),
        params={'x':dict(offset=0,size=len(data),shape=[len(data)],dtype='uint8',sha256=hashlib.sha256(data).hexdigest(),chunks=chunks)})
    region=Region(disk,offset=1<<20,length=7<<20,retention=3);region.format(region_id='migration')
    return disk,data,record,region

def perform(disk,record,region):return migrate(source=disk,record=record,destination=region,request_id='migration',identity={'scope':'test'},chunk_bytes=8192,source_layout=None,validate_source=lambda r,l:None)

def test_source_unchanged_destination_verified():
    disk,data,record,region=fixture();perform(disk,record,region)
    assert disk.read(0,len(data))==data
    assert region.verify_payloads(region.current['state']['generations'][0])
    assert all(o>=region.base for o,n in disk.writes)

def test_overlap_rejected_before_any_destination_write():
    disk,data,record,region=fixture();record['params']['x']['offset']=region.base
    before=len(disk.writes)
    with pytest.raises(ValueError,match='overlaps'):perform(disk,record,region)
    assert len(disk.writes)==before and region.pending is None

def test_corruption_never_publishes():
    disk,data,record,region=fixture();disk.data[0]^=1
    with pytest.raises(ValueError,match='corruption'):perform(disk,record,region)
    assert region.current is None and region.poisoned
