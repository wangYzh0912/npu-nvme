import copy
import pytest
from npu_nvme.d2.format import Region, BLOCK, pack, unpack


class CrashDisk:
    def __init__(self,size=8<<20):
        self.data=bytearray(size);self.durable=bytearray(size)
        self.flush_count=0;self.fail_flush=None;self.writes=[]
    def read(self,offset,length):return bytes(self.data[offset:offset+length])
    def write(self,offset,data):
        assert 0<=offset<=len(self.data)-len(data)
        self.data[offset:offset+len(data)]=data;self.writes.append((offset,len(data)))
    def flush(self):
        self.flush_count+=1
        if self.flush_count==self.fail_flush:raise OSError('injected flush failure')
        self.durable[:]=self.data
    def crash(self):self.data[:]=self.durable


def fresh(disk):return Region(disk,offset=0,length=len(disk.data),retention=3)


def commit(region,number):
    region.begin('request-'+str(number),{'step':number})
    ref=region.payload(bytes([number])*513)
    rows=[dict(rank=0,name='x',shape=[513],dtype='uint8',partition='replicated',logical_offset=0,logical_bytes=513,payload=ref)]
    return region.commit(rows,step=number,topology={'world_size':1},rank_controls=[{'rank':0,'step':number,'controls':{'cursor':number}}])


def test_fresh_mount_retention_and_idempotent_retry():
    disk=CrashDisk();r=fresh(disk);r.format(region_id='unit')
    for n in range(1,6):commit(r,n)
    disk.crash();r=fresh(disk);assert r.mount()==[]
    assert [g['generation'] for g in r.current['state']['generations']]==[5,4,3]
    assert r.begin('request-5',{'step':5},retry=True)['replayed']
    with pytest.raises(ValueError):r.begin('request-5',{'step':6},retry=True)
    with pytest.raises(KeyError):r.begin('expired',{},retry=True)


@pytest.mark.parametrize('flush_phase',[1,2,3])
def test_crash_before_publication_recovers_previous_generation(flush_phase):
    disk=CrashDisk();r=fresh(disk);r.format(region_id='unit');commit(r,1)
    disk.fail_flush=disk.flush_count+flush_phase
    with pytest.raises(OSError):commit(r,2)
    assert r.poisoned
    disk.crash();r=fresh(disk);r.mount()
    assert r.current['sequence']==1


@pytest.mark.parametrize('point,sequence',[
    ('after_payload_flush',1),('after_metadata_flush',1),('after_anchor_flush',2),
])
def test_named_post_flush_faults_have_a_single_recoverable_generation(point,sequence):
    disk=CrashDisk();base=fresh(disk);base.format(region_id='unit');commit(base,1)
    def injected(observed):
        if observed==point:raise OSError(observed)
    failing=Region(disk,offset=0,length=len(disk.data),retention=3,fault_hook=injected)
    failing.mount()
    with pytest.raises(OSError,match=point):commit(failing,2)
    disk.crash();mounted=fresh(disk);mounted.mount()
    assert mounted.current['sequence']==sequence
    assert mounted.verify_payloads(mounted.current['state']['generations'][0])


def test_corrupted_latest_anchor_falls_back_without_overwriting_payload():
    disk=CrashDisk();r=fresh(disk);r.format(region_id='unit');commit(r,1);commit(r,2)
    disk.data[(r.current['slot']+1)*BLOCK+5]^=1
    mounted=fresh(disk);errors=mounted.mount()
    assert errors and mounted.current['sequence']==1
    assert mounted.verify_payloads(mounted.current['state']['generations'][0])


def test_reader_pin_keeps_old_extents_through_retention_rollover():
    disk=CrashDisk();r=fresh(disk);r.format(region_id='unit');commit(r,1)
    with r.selected(1) as selected:
        refs=copy.deepcopy(selected['extents'])
        for n in range(2,9):commit(r,n)
        assert all(disk.read(ref['offset'],ref['length'])[:ref['logical_bytes']]==bytes([1])*513 for ref in refs)
        assert r.pins
    assert not r.pins


def test_no_implicit_reformat():
    disk=CrashDisk();r=fresh(disk);r.format(region_id='unit')
    count=len(disk.writes)
    with pytest.raises(ValueError):fresh(disk).format(region_id='different')
    assert len(disk.writes)==count


def test_header_length_is_integrity_protected():
    raw=bytearray(pack('anchor',{'x':1},BLOCK))
    raw[16]^=1
    with pytest.raises(ValueError):unpack('anchor',raw)


def test_reservation_fails_before_any_payload_write():
    disk=CrashDisk();r=fresh(disk);r.format(region_id='unit');before=len(disk.writes)
    with pytest.raises(BufferError):r.begin('too-big',{},reserve_bytes=len(disk.data))
    assert r.pending is None and len(disk.writes)==before


def test_admitted_identity_cannot_be_mutated_by_caller():
    disk=CrashDisk();r=fresh(disk);r.format(region_id='unit');identity={'scope':['full']}
    r.begin('r',identity,reserve_bytes=1<<20);identity['scope'][0]='wrong'
    assert r.pending['identity']=={'scope':['full']}
