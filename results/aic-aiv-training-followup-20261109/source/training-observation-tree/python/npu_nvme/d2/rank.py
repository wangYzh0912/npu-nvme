"""Draft bounded collective assembly; rank messages contain offsets, never HBM pointers."""
from __future__ import annotations
import hashlib
import json
import threading
import time


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


class Collective:
    def __init__(self,*,world_size,epoch,schema,chunk_bytes,credits,timeout_seconds,clock=time.monotonic):
        if world_size not in (2,4) or not epoch or chunk_bytes<=0 or credits<=0 or timeout_seconds<=0:raise ValueError('invalid collective bounds')
        self.world_size=world_size;self.epoch=epoch;self.schema=schema;self.chunk=chunk_bytes;self.credits=credits
        self.timeout=timeout_seconds;self.clock=clock;self.lock=threading.RLock();self.current=None
        self.poisoned=False;self.ready=set();self.applied=set();self.rows={}
        for rank in range(world_size):
            expected={}
            for tensor in schema:
                if tensor['partition'] not in ('replicated','sharded','per_rank_control'):raise ValueError('partition')
                name=tensor['name'];size=tensor['bytes_per_rank'][rank]
                if name in expected or type(size) is not int or size<=0:raise ValueError('invalid schema')
                expected[name]=size
            self.rows[rank]=expected

    def begin(self,request_id,step):
        with self.lock:
            if self.poisoned or self.current:raise RuntimeError('collective unavailable')
            self.current=dict(request_id=request_id,step=step,started=self.clock(),ranks={},inflight={},received={})
            self.ready=set();self.applied=set()

    def _check(self,epoch,request_id,rank):
        if self.poisoned or self.current is None:raise RuntimeError('collective unavailable')
        if self.clock()-self.current['started']>self.timeout:
            self.poisoned=True;raise TimeoutError('collective deadline; owner retained')
        if epoch!=self.epoch or request_id!=self.current['request_id'] or type(rank) is not int or rank not in self.rows:raise ValueError('stale/wrong rank message')

    def admit(self,*,epoch,request_id,rank,name,offset,length,sha256,lease):
        with self.lock:
            self._check(epoch,request_id,rank)
            if name not in self.rows[rank] or offset<0 or offset%self.chunk or length!=min(self.chunk,self.rows[rank][name]-offset) or length<=0:raise ValueError('invalid chunk geometry')
            if not isinstance(sha256,str) or len(sha256)!=64 or any(c not in '0123456789abcdef' for c in sha256):raise ValueError('invalid digest')
            key=(rank,name,offset)
            descriptor=dict(rank=rank,name=name,offset=offset,length=length,sha256=sha256,lease=lease)
            if key in self.current['received']:
                if self.current['received'][key]!=descriptor:raise ValueError('conflicting duplicate chunk')
                return 'already_received'
            if key in self.current['inflight']:
                if self.current['inflight'][key]!=descriptor:raise ValueError('conflicting in-flight chunk')
                return 'inflight'
            if len(self.current['inflight'])>=self.credits:raise BlockingIOError('transport credits exhausted')
            self.current['inflight'][key]=descriptor;return key

    def complete(self,key,payload):
        with self.lock:
            descriptor=self.current['inflight'][key]
            if len(payload)!=descriptor['length'] or hashlib.sha256(payload).hexdigest()!=descriptor['sha256']:
                self.poisoned=True;raise ValueError('payload failed integrity; lease retained')
            self.current['received'][key]=descriptor;del self.current['inflight'][key]

    def rank_complete(self,*,epoch,request_id,rank,step,controls):
        with self.lock:
            self._check(epoch,request_id,rank)
            if step!=self.current['step']:raise ValueError('rank step differs')
            expected={(rank,name,off) for name,size in self.rows[rank].items() for off in range(0,size,self.chunk)}
            received={key for key in self.current['received'] if key[0]==rank}
            if expected!=received:raise ValueError('rank has missing/extra chunks')
            if any(key[0]==rank for key in self.current['inflight']):raise RuntimeError('rank DMA not source safe')
            # Controls are rank-local; cross-rank byte equality is not required.
            value=dict(step=step,controls=controls)
            if rank in self.current['ranks'] and self.current['ranks'][rank]!=value:raise ValueError('conflicting rank completion')
            self.current['ranks'][rank]=value

    def manifest(self):
        with self.lock:
            if self.poisoned or not self.current or len(self.current['ranks'])!=self.world_size or self.current['inflight']:raise RuntimeError('not globally complete')
            return dict(epoch=self.epoch,request_id=self.current['request_id'],step=self.current['step'],world_size=self.world_size,
                        ranks=self.current['ranks'],chunks=[self.current['received'][k] for k in sorted(self.current['received'])])

    def restore_applied(self,rank,*,verified):
        with self.lock:
            self.manifest()
            if type(rank) is not int or rank not in self.rows or verified is not True:raise ValueError('rank not verified')
            self.applied.add(rank)
            if len(self.applied)==self.world_size:self.ready=set(self.applied)
            return len(self.ready)==self.world_size

    def disconnect(self,rank):
        with self.lock:
            if type(rank) is not int or rank not in self.rows:raise ValueError('rank')
            self.poisoned=True;self.ready.clear()
            # In-flight leases remain owned, never auto-taken over.
