"""CPU failure boundaries for the actual socket/collective implementation."""
import hashlib
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from npu_nvme.d2 import wire
from npu_nvme.d2.rank import Collective
from npu_nvme.d2.ready_protocol import ReadyDecision,coordinate


def collective(world=4,clock=lambda:0):
    c=Collective(world_size=world,epoch='epoch',schema=[dict(name='x',partition='sharded',bytes_per_rank=[4]*world)],
                 chunk_bytes=4,credits=1,timeout_seconds=5,clock=clock)
    c.begin('request',8)
    return c

def admit(c,rank=0,**change):
    args=dict(epoch='epoch',request_id='request',rank=rank,name='x',offset=0,length=4,
              sha256=hashlib.sha256(b'abcd').hexdigest(),lease=rank)
    args.update(change)
    return c.admit(**args)

@pytest.mark.parametrize('change',[dict(epoch='stale'),dict(request_id='old'),dict(rank=True),dict(rank=4),
    dict(offset=4),dict(length=3),dict(sha256='z'*64)])
def test_invalid_rank_message(change):
    with pytest.raises(ValueError):admit(collective(),**change)

def test_credits_and_duplicates():
    c=collective();key=admit(c)
    assert admit(c)=='inflight'
    with pytest.raises(ValueError):admit(c,lease=9)
    with pytest.raises(BlockingIOError):admit(c,rank=1)
    c.complete(key,b'abcd');assert admit(c)=='already_received'
    assert isinstance(admit(c,rank=1),tuple)

def test_corruption_poison_retains_inflight():
    c=collective();key=admit(c)
    with pytest.raises(ValueError):c.complete(key,b'abce')
    assert c.poisoned and key in c.current['inflight']
    with pytest.raises(RuntimeError):c.manifest()

def test_deadline_and_disconnect_poison():
    now=[0];c=collective(clock=lambda:now[0]);key=admit(c);now[0]=6
    with pytest.raises(TimeoutError):admit(c,rank=1)
    assert key in c.current['inflight'] and c.poisoned
    c=collective();c.disconnect(1)
    with pytest.raises(RuntimeError):admit(c)

@pytest.mark.parametrize('world',[2,4])
def test_missing_rank_never_commits(world):
    c=collective(world)
    for rank in range(world):
        key=admit(c,rank);c.complete(key,b'abcd')
        if rank==world-1:
            with pytest.raises(RuntimeError):c.manifest()
        c.rank_complete(epoch='epoch',request_id='request',rank=rank,step=8,controls={'rank':rank})
    assert len(c.manifest()['ranks'])==world
    for rank in range(world):assert c.restore_applied(rank,verified=True)==(rank==world-1)

def test_wire_oversize_rejected_before_payload():
    a,b=socket.socketpair()
    try:
        a.sendall(wire.HEADER.pack(wire.MAGIC,3,0,2**63,bytes(32)))
        with pytest.raises(ValueError):wire.receive(b,deadline=time.monotonic()+1,max_payload=4096)
    finally:a.close();b.close()

def test_ready_failure_sends_no_release():
    links=[socket.socketpair() for _ in range(2)]
    d=ReadyDecision(world_size=2,epoch='e',generation=1,manifest_sha256='a'*64)
    deadline=time.monotonic()+1
    try:
        with ThreadPoolExecutor() as pool:
            f=pool.submit(coordinate,{r:x[0] for r,x in enumerate(links)},d,wire,deadline=deadline)
            proof=dict(kind='prepared',rank=0,epoch='e',generation=1,manifest_sha256='a'*64,
                       transport_safe=True,schema_verified=True,controls_verified=True)
            wire.send(links[0][1],proof,b'',deadline=deadline,max_payload=0)
            links[1][1].close()
            with pytest.raises(EOFError):f.result()
        links[0][1].settimeout(.02)
        with pytest.raises(socket.timeout):links[0][1].recv(1)
        assert d.failed
    finally:
        for pair in links:
            for s in pair:s.close()
