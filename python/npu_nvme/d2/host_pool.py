"""Single-owner shared memzone and rank slice geometry; no remote HBM pointers."""
from dataclasses import dataclass
import ctypes as C
import hashlib
import json
import threading

@dataclass(frozen=True)
class PoolPlan:
    world_size:int
    slots_per_rank:int
    chunk_bytes:int
    epoch:str
    def __post_init__(self):
        if type(self.world_size) is not int or self.world_size not in (2,4):raise ValueError('world size')
        if type(self.slots_per_rank) is not int or not 1<=self.slots_per_rank<=64:raise ValueError('slot count')
        if type(self.chunk_bytes) is not int or self.chunk_bytes<=0 or self.chunk_bytes%4096:raise ValueError('chunk geometry')
        if type(self.epoch) is not str or not 0<len(self.epoch)<=128:raise ValueError('epoch')
        if self.total_bytes>2**63-1:raise ValueError('pool overflow')
    @property
    def rank_bytes(self):return self.slots_per_rank*self.chunk_bytes
    @property
    def total_bytes(self):return self.world_size*self.rank_bytes
    @property
    def name(self):return 'npu_rank_'+hashlib.sha256(self.epoch.encode()).hexdigest()[:32]
    def descriptor(self,rank,shm_id):
        if type(rank) is not int or not 0<=rank<self.world_size or type(shm_id) is not int or shm_id<0:raise ValueError('rank/shm')
        return dict(epoch=self.epoch,memzone=self.name,shm_id=shm_id,rank=rank,offset=rank*self.rank_bytes,length=self.rank_bytes,slots=self.slots_per_rank,chunk_bytes=self.chunk_bytes)

class PrimaryPool:
    def __init__(self,lib,plan,*,host_budget_bytes,shm_id):
        if plan.total_bytes>host_budget_bytes:raise MemoryError('shared pool exceeds Host admission')
        self.lib=lib;self.plan=plan;self.shm_id=shm_id;self.closed=False;self.retained=False;self.safe_ranks=set();self.lock=threading.RLock()
        lib.spdk_memzone_reserve.argtypes=[C.c_char_p,C.c_size_t,C.c_int,C.c_uint32];lib.spdk_memzone_reserve.restype=C.c_void_p
        lib.spdk_memzone_free.argtypes=[C.c_char_p];lib.spdk_memzone_free.restype=C.c_int
        self.address=lib.spdk_memzone_reserve(plan.name.encode(),plan.total_bytes,-1,0)
        if not self.address:raise MemoryError('SPDK shared pool reservation failed')
    def view(self,lease):
        if self.closed or self.retained:raise RuntimeError('pool unavailable')
        rank=lease['rank'];slot=lease['slot'];length=lease['length']
        if lease['epoch']!=self.plan.epoch or type(rank) is not int or not 0<=rank<self.plan.world_size or type(slot) is not int or not 0<=slot<self.plan.slots_per_rank or type(length) is not int or not 0<length<=self.plan.chunk_bytes:raise ValueError('lease bounds')
        offset=rank*self.plan.rank_bytes+slot*self.plan.chunk_bytes
        if lease['offset']!=offset:raise ValueError('global shared offset differs')
        return (C.c_ubyte*length).from_address(self.address+offset)
    def rank_closed(self,rank,*,epoch,transport_safe):
        with self.lock:
            if epoch!=self.plan.epoch or type(rank) is not int or not 0<=rank<self.plan.world_size or transport_safe is not True:raise ValueError('rank lacks close proof')
            self.safe_ranks.add(rank)
    def close(self):
        with self.lock:
            if self.closed:return
            if len(self.safe_ranks)!=self.plan.world_size:
                self.retained=True;raise RuntimeError('shared pool retained: missing rank DMA stop proof')
            if self.lib.spdk_memzone_free(self.plan.name.encode()):raise RuntimeError('memzone release failed')
            self.closed=True
