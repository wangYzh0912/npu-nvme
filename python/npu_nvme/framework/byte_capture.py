"""Exact NPU byte mirrors for R0 capture; optimizer remains blocked until ACK.

Synchronous D2D copies preserve floating-point bit patterns. No numerical cast
or floating equality is used for change detection. This is a correctness path,
not an asynchronous codec performance claim.
"""
import ctypes
import numpy as np
from .parameters import get_dev_ptr


class ByteCapture:
    def __init__(self,framework,acl,manifest,registry,*,hbm_budget_bytes):
        self.ms=framework;self.acl=acl;self.manifest=manifest;self.registry=registry
        size=sum(f.byte_count for f in manifest.fields)
        if size*3>hbm_budget_bytes:raise MemoryError('NPU byte mirrors/comparison exceed budget')
        self.current={};self.persisted={};self.initialized=False
        for field in manifest.fields:
            if not get_dev_ptr(registry[field.canonical_name]):raise ValueError('NPU byte capture requires device state')
            for group in (self.current,self.persisted):
                value=framework.ops.zeros((field.byte_count,),framework.uint8)
                parameter=framework.Parameter(value,requires_grad=False)
                framework.ops.assign(parameter,value)
                group[field.canonical_name]=parameter
        framework.hal.synchronize()
        self.copy=acl.aclrtMemcpy
        self.copy.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
        self.copy.restype=ctypes.c_int

    def _copy(self,destination,source,size):
        dst,src=get_dev_ptr(destination),get_dev_ptr(source)
        if not dst or not src:raise ValueError('missing device byte address')
        rc=self.copy(dst,size,src,size,3)
        if rc:raise RuntimeError('ACL byte mirror D2D failed: '+str(rc))

    def capture(self):
        self.ms.hal.synchronize();flags=[]
        for field in self.manifest.fields:
            name=field.canonical_name
            self._copy(self.current[name],self.registry[name],field.byte_count)
            if not self.initialized:
                flags.extend([True]*len(field.blocks));continue
            width=np.dtype(field.dtype).itemsize
            for block in field.blocks:
                start=block.element_offset*width;end=start+block.element_count*width
                changed=self.ms.ops.any(self.ms.ops.not_equal(self.current[name][start:end],self.persisted[name][start:end]))
                flags.append(bool(changed.asnumpy()))
        self.ms.hal.synchronize()
        return np.asarray(flags,dtype=np.bool_)

    def commit_ack(self):
        for field in self.manifest.fields:
            self._copy(self.persisted[field.canonical_name],self.current[field.canonical_name],field.byte_count)
        self.ms.hal.synchronize();self.initialized=True
