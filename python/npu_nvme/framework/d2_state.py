"""Explicit blocking full-state capture for fixed-topology D2 training sessions.

Device-resident tensors use the rank's private DMA pool. Parameters with no
exported device address are copied through one tensor-sized Host allocation;
this placement is recorded before capture, never selected after a DMA failure.
"""
from contextlib import contextmanager
import hashlib
import math
import numpy as np
from .parameters import get_dev_ptr


class BlockingState:
    def __init__(self,framework,network,*,rank,executor,host_tensor_budget,partitions):
        self.framework=framework;self.executor=executor;self.rank=rank
        self.entries={};self.schema=[];seen=set();self.host_tensor_budget=host_tensor_budget
        self.host_source=None;self.host_source_name=None
        self.host_target=None;self.host_target_name=None
        framework.hal.synchronize()
        for _,param in network.parameters_and_names():
            if id(param) in seen:continue
            seen.add(id(param));name=param.name
            if name in self.entries:raise ValueError('duplicate parameter '+name)
            dtype=np.dtype(framework.dtype_to_nptype(param.dtype))
            shape=list(getattr(param,'sliced_shape',None) or param.shape)
            if hasattr(param,'data') and math.prod(param.data.shape)<math.prod(shape):shape=list(param.data.shape)
            size=math.prod(shape)*dtype.itemsize
            if size<=0:raise ValueError('empty state tensor '+name)
            pointer=get_dev_ptr(param)
            if not pointer and size>host_tensor_budget:raise MemoryError('Host tensor capture budget exceeded: '+name)
            partition=partitions.get(name)
            if partition not in ('replicated','sharded','per_rank_control'):raise ValueError('missing explicit tensor partition: '+name)
            self.entries[name]=dict(param=param,pointer=pointer,dtype=dtype,shape=shape,bytes=size)
            self.schema.append(dict(rank=rank,name=name,shape=shape,dtype=dtype.name,bytes=size,
                                    partition=partition))
        self.schema.sort(key=lambda row:row['name'])
        if not self.entries:raise ValueError('empty state registry')
        self.placement={name:('device' if entry['pointer'] else 'host') for name,entry in self.entries.items()}

    @contextmanager
    def read_chunk(self,name,offset,length):
        entry=self.entries[name]
        if offset<0 or length<=0 or offset+length>entry['bytes']:raise ValueError('capture bounds')
        if entry['pointer']:
            with self.executor.d2h(entry['pointer']+offset,length,owners=(entry['param'],)) as data:yield data
        else:
            # Only one tensor copy lives across a chunk callback. Capture is
            # explicitly blocking, so subsequent chunks see the same state.
            if self.host_source_name!=name:
                self.host_source=None
                self.host_source=np.ascontiguousarray(entry['param'].asnumpy())
                self.host_source_name=name
            array=self.host_source
            if array.nbytes!=entry['bytes'] or list(array.shape)!=entry['shape'] or array.dtype!=entry['dtype']:
                raise ValueError('Host placement changed')
            try:
                yield memoryview(array).cast('B')[offset:offset+length]
            finally:
                if offset+length==entry['bytes']:
                    self.host_source=None;self.host_source_name=None

    def apply_chunk(self,name,offset,data,digest):
        entry=self.entries[name]
        if offset<0 or not data or offset+len(data)>entry['bytes'] or hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError('restore chunk bounds/digest')
        if entry['pointer']:
            self.executor.h2d(entry['pointer']+offset,data,expected_sha256=digest,owners=(entry['param'],))
        else:
            if offset==0:
                if self.host_target is not None:raise ValueError('previous Host tensor incomplete')
                self.host_target=bytearray(entry['bytes']);self.host_target_name=name
            if self.host_target_name!=name:raise ValueError('Host tensor sequence differs')
            self.host_target[offset:offset+len(data)]=data
            if offset+len(data)==entry['bytes']:
                value=np.frombuffer(self.host_target,dtype=entry['dtype']).reshape(entry['shape']).copy()
                entry['param'].set_data(self.framework.Tensor(value,dtype=entry['param'].dtype))
                self.host_target=None;self.host_target_name=None

    def verify_finished(self):
        if self.host_target is not None:raise ValueError('incomplete Host target')
        self.framework.hal.synchronize()
