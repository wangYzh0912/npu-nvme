from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import socket
import time
import pytest
from npu_nvme.d2.session import OwnerSession,RankSession
from npu_nvme.d2.format import Region

class Disk:
    def __init__(self):self.data=bytearray(32<<20)
    def read(self,o,n):return bytes(self.data[o:o+n])
    def write(self,o,data):self.data[o:o+len(data)]=data
    def flush(self):pass

@pytest.mark.parametrize('world',[2,4])
def test_save_fresh_mount_restore(world):
    disk=Disk();r=Region(disk,offset=0,length=len(disk.data),retention=3);r.format(region_id='test')
    def run(operation,region):
        links=[socket.socketpair() for _ in range(world)]
        identity={'workload':'cpu-fixture'};deadline=time.monotonic()+5
        owner=OwnerSession(region,{rank:p[0] for rank,p in enumerate(links)},epoch=operation,request_id='request',
            topology={'world_size':world},chunk_bytes=4096,deadline=deadline,identity=identity)
        def worker(rank):
            schema=[dict(rank=rank,name='tensor',shape=[8197],dtype='uint8',partition='sharded',bytes=8197)]
            session=RankSession(links[rank][1],rank=rank,epoch=operation,identity=identity,schema=schema,chunk_bytes=4096,deadline=deadline)
            expected=bytes([rank+1])*8197
            if operation=='save':
                @contextmanager
                def read(name,offset,length):yield expected[offset:offset+length]
                return session.save(read,{'cursor':8,'rank':rank},step=8)
            observed=bytearray(8197)
            def apply(name,offset,data,digest):observed[offset:offset+len(data)]=data
            def controls(value,step):
                assert value=={'cursor':8,'rank':rank} and step==8 and bytes(observed)==expected
            return session.restore(apply,controls)
        try:
            with ThreadPoolExecutor(max_workers=world+1) as executor:
                workers=[executor.submit(worker,rank) for rank in range(world)]
                result=owner.save(step=8) if operation=='save' else owner.restore()
                results=[f.result() for f in workers]
                assert all(v==result for v in results)
        finally:
            for pair in links:
                for sock in pair:sock.close()
    run('save',r)
    fresh=Region(disk,offset=0,length=len(disk.data),retention=3);assert fresh.mount()==[]
    run('restore',fresh)


def test_bad_chunk_wakes_idle_peer_without_waiting_deadline():
    from npu_nvme.d2.rank import Collective
    from npu_nvme.d2.socket_service import RankService
    from npu_nvme.d2 import wire
    disk=Disk();region=Region(disk,offset=0,length=len(disk.data),retention=3)
    region.format(region_id='failure');region.begin('req',{})
    collective=Collective(world_size=2,epoch='epoch',schema=[dict(name='x',partition='sharded',bytes_per_rank=[1,1])],
        chunk_bytes=4096,credits=2,timeout_seconds=10)
    collective.begin('req',1)
    pairs=[socket.socketpair() for _ in range(2)]
    service=RankService(collective,region,wire,max_payload=4096,deadline=time.monotonic()+10,
        tensor_schema=[dict(rank=r,name='x',shape=[1],dtype='uint8',partition='sharded',bytes=1) for r in range(2)])
    try:
        wire.send(pairs[0][1],dict(kind='chunk',rank=0,epoch='epoch',request_id='req',name='x',
            offset=0,length=1,sha256='0'*64,lease=0),b'x',deadline=time.monotonic()+1,max_payload=4096)
        start=time.monotonic()
        with pytest.raises(ValueError,match='digest'):service.run({r:p[0] for r,p in enumerate(pairs)},step=1,topology={'world_size':2})
        assert time.monotonic()-start<2
        assert region.current is None
    finally:
        for pair in pairs:
            for connection in pair:connection.close()
