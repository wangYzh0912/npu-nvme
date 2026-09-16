"""Rank-local private-buffer socket bridge copy wrapper; requests retain all owners until DMA safe."""
from contextlib import contextmanager
import ctypes as C
import threading
_RETAINED=[]

class CopyFailure(RuntimeError):
    def __init__(self,message,*,transport_safe):super().__init__(message);self.transport_safe=transport_safe

class RankCopyExecutor:
    def __init__(self,backend,descriptor,*,npu_id,depth,timeout_ms):
        from npu_nvme.storage.full_transport import FullTransport
        import os
        self.backend=backend;self.lib=backend.lib;self.descriptor=dict(descriptor);self.timeout_ms=timeout_ms
        self.lock=threading.RLock();self.unsafe=False;self.closed=False;self.owners=[]
        os.environ['SPDK_SHM_ID']=str(descriptor['shm_id'])
        self.transport=FullTransport(backend,pci=None,npu=npu_id,depth=depth,
            chunk_size=descriptor['chunk_bytes'],profiling_dir=descriptor.get('profiling_dir','.'),role='copy_rank')
        self.ctx=self.transport.ctx
        self.buffer=C.create_string_buffer(descriptor['chunk_bytes'])
        self.base=C.addressof(self.buffer)
    def _copy(self,operation,device,length,*,expected_sha256=None,owners=()):
        from npu_nvme.storage.bindings import NPUNVMECopySpec,NPUNVMERequest,NPUNVMETransferReceipt,NPUNVMETransferDigest
        if self.unsafe or self.closed:raise RuntimeError('rank executor unavailable')
        if type(length) is not int or not 0<length<=self.descriptor['chunk_bytes']:raise ValueError('copy length')
        spec=NPUNVMECopySpec(struct_size=C.sizeof(NPUNVMECopySpec),version=1,operation=operation,
            source=device if operation==5 else self.base,destination=self.base if operation==5 else device,length=length,checksum_flags=2)
        if operation==6:
            if type(expected_sha256) is not str or len(expected_sha256)!=64:raise ValueError('restore requires digest')
            spec.expected_sha256[:]=bytes.fromhex(expected_sha256)
        request=C.POINTER(NPUNVMERequest)();self.owners=list(owners)
        rc=self.lib.npu_nvme_submit_copy(self.ctx,C.byref(spec),C.byref(request))
        if rc:self.owners=[];raise CopyFailure('copy rejected: '+str(rc),transport_safe=True)
        self.lib.npu_nvme_wait_request(request,self.timeout_ms)
        receipt=NPUNVMETransferReceipt()
        observed=self.lib.npu_nvme_get_transfer_receipt(request,C.byref(receipt),C.sizeof(receipt))
        if observed or not receipt.done or not receipt.transport_safe:
            self.unsafe=True;_RETAINED.append(self)
            self.lib.npu_nvme_release_request(request)
            raise CopyFailure('copy lacks terminal DMA stop proof',transport_safe=False)
        try:
            if receipt.result:raise CopyFailure('copy failed: '+str(receipt.result),transport_safe=True)
            digest=NPUNVMETransferDigest()
            rc=self.lib.npu_nvme_get_transfer_digests(request,C.byref(digest),1)
            if rc:raise CopyFailure('copy digest unavailable',transport_safe=True)
            return bytes(digest.sha256).hex()
        finally:self.lib.npu_nvme_release_request(request);self.owners=[]
    @contextmanager
    def d2h(self,device,length,*,owners=()):
        with self.lock:
            self._copy(5,device,length,owners=owners)
            # Slot not reused until the consumer leaves this context.
            yield memoryview((C.c_ubyte*length).from_address(self.base)).cast('B')
    def h2d(self,device,data,*,expected_sha256,owners=()):
        with self.lock:
            if self.unsafe or self.closed:raise RuntimeError('rank executor unavailable')
            if not 0<len(data)<=self.descriptor['chunk_bytes']:raise ValueError('copy length')
            raw=bytes(data);C.memmove(self.base,raw,len(raw))
            return self._copy(6,device,len(raw),expected_sha256=expected_sha256,owners=(*owners,raw))
    def close(self):
        with self.lock:
            if self.closed:return True
            if self.unsafe:return False
            if self.lib.npu_nvme_close(self.ctx,self.timeout_ms):
                self.unsafe=True;_RETAINED.append(self);return False
            self.transport.close(self.timeout_ms/1000);self.closed=True;return True
