"""Explicit submit/wait helpers for finite standalone diagnostic workloads.

The native ABI contains no blocking batch entry point. Raw-pointer diagnostics
keep their call stack (and all caller allocations) alive on an observation
expiry until the request is terminal. Unknown DMA can therefore retain a worker
indefinitely; its orchestrator must record the owner and stop without killing it.
Product paths use FullTransport's owned-buffer bounded observation instead.
"""
import ctypes as C
import errno
import time


def transfer_wait(lib,operation,memory_kind,ctx,ptrs,offsets,sizes,count,*,timeout_ms=0):
    from .bindings import NPUNVMERequest,NPUNVMETransferItem,NPUNVMETransferSpec
    if type(count) is not int or not 0<count<=65536:return -errno.EINVAL
    items=(NPUNVMETransferItem*count)()
    for i,item in enumerate(items):
        item.address=ptrs[i];item.offset=offsets[i];item.length=sizes[i]
    spec=NPUNVMETransferSpec(C.sizeof(NPUNVMETransferSpec),1,operation,memory_kind,0,count,items)
    req=C.POINTER(NPUNVMERequest)()
    rc=lib.npu_nvme_submit_transfer(ctx,C.byref(spec),C.byref(req))
    if rc:return rc
    try:
        rc=lib.npu_nvme_wait_request(req,timeout_ms)
        if rc==-errno.ETIMEDOUT:
            done=C.c_int()
            while not done.value:
                lib.npu_nvme_poll_request(req,C.byref(done))
                if not done.value:time.sleep(.001)
        return rc
    finally:lib.npu_nvme_release_request(req)
