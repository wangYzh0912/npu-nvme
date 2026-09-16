from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import socket
import time

import pytest

from npu_nvme.d2.format import Region
from npu_nvme.d2.training_session import TrainingOwner, TrainingRank, operation_message
from test_session import Disk


@pytest.mark.parametrize('world',[2,4])
def test_multiple_saves_restore_and_save_on_same_connections(world):
    disk=Disk();region=Region(disk,offset=0,length=len(disk.data),retention=3)
    region.format(region_id='persistent-training')
    links=[socket.socketpair() for _ in range(world)]
    common=dict(epoch='job',identity={'model':'fixture'},chunk_bytes=4096,
                deadline=time.monotonic()+10,operation_timeout=5)
    owner=TrainingOwner(region,{r:p[0] for r,p in enumerate(links)},**common)
    def worker(rank):
        client=TrainingRank(links[rank][1],rank=rank,**common)
        schema=[dict(rank=rank,name='x',shape=[8197],dtype='uint8',partition='sharded',bytes=8197)]
        state=bytearray([rank+4])*8197
        @contextmanager
        def read(name,offset,length): yield state[offset:offset+length]
        def apply(name,offset,data,digest):state[offset:offset+len(data)]=data
        def verify(control,step):
            assert step==4 and control=={'step':4}
            assert state==bytearray([rank+4])*8197
        first=client.save(schema,read,{'step':4},step=4)
        state[:]=bytes([rank+8])*len(state)
        second=client.save(schema,read,{'step':8},step=8)
        client.restore(schema,apply,verify,step=4,generation=first['generation'])
        state[:]=bytes([rank+12])*len(state)
        third=client.save(schema,read,{'step':12},step=12)
        client.close()
        return first,second,third
    try:
        with ThreadPoolExecutor(max_workers=world+1) as pool:
            owner_future=pool.submit(owner.run)
            futures=[pool.submit(worker,r) for r in range(world)]
            results=[f.result(timeout=12) for f in futures]
            assert owner_future.result(timeout=12)==4
        assert all(row==results[0] for row in results)
        assert [row['generation'] for row in results[0]]==[1,2,3]
        fresh=Region(disk,offset=0,length=len(disk.data),retention=3)
        assert fresh.mount()==[]
        with fresh.selected(3) as value: assert value['step']==12
    finally:
        for pair in links:
            for connection in pair:connection.close()


@pytest.mark.parametrize('fields',[dict(operation='restore',step=4,generation=None),
    dict(operation='save',step=0),dict(operation='save',step=4,generation=8),
    dict(operation='save',step=True),dict(operation='other',step=4)])
def test_invalid_operation_rejected_before_admission(fields):
    with pytest.raises(ValueError):operation_message(epoch='job',sequence=0,rank=0,**fields)
