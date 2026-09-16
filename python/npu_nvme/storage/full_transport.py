"""Strict FULL uses one native asynchronous transfer/copy executor."""
import ctypes as C
import threading
from .bindings import (NPUNVMEContext, NPUNVMERequest, NPUNVMETransferItem,
    NPUNVMETransferSpec, NPUNVMETransferReceipt, NPUNVMETransferDigest,
    NPUNVMECopySpec, NPUNVMECapabilities, NPUNVMEInitOptions)
from .metadata import MetadataIO
from .layout import BLOCK_SIZE


_RETAINED_TRANSPORTS=set()


class TransferFailure(RuntimeError):
    def __init__(self,message,*,transport_safe):
        super().__init__(message)
        self.transport_safe=transport_safe


class FullTransport:
    def __init__(self, backend, *, pci, npu, depth, chunk_size, profiling_dir, profiling=False, role="combined"):
        self.lib, self.acl = backend.lib, backend.acl_lib
        for name in ('npu_nvme_submit_transfer','npu_nvme_submit_copy','npu_nvme_get_capabilities'):
            if not hasattr(self.lib,name): raise RuntimeError('async FULL capability missing: '+name)
        self.npu = npu
        self.ctx = C.POINTER(NPUNVMEContext)()
        self.retained = []
        self._buffers_lock = threading.Lock()
        if role == 'combined':
            rc = self.lib.npu_nvme_init(C.byref(self.ctx), pci.encode(), npu, depth,
                                       chunk_size, profiling, str(profiling_dir).encode())
        else:
            if role not in ('host_owner','copy_rank'): raise ValueError('unknown native role')
            if not hasattr(self.lib,'npu_nvme_init_ex'): raise RuntimeError('role ABI missing')
            options=NPUNVMEInitOptions(C.sizeof(NPUNVMEInitOptions),1,
                1 if role=='host_owner' else 2,depth,chunk_size,npu,0)
            rc=self.lib.npu_nvme_init_ex(C.byref(self.ctx),pci.encode() if pci else None,
                C.byref(options),str(profiling_dir).encode())
        if rc != 0: raise RuntimeError(f'native open rejected: {rc}')
        _RETAINED_TRANSPORTS.add(self)
        self.capabilities = NPUNVMECapabilities()
        rc = self.lib.npu_nvme_get_capabilities(self.ctx,C.byref(self.capabilities),C.sizeof(self.capabilities))
        if rc or self.capabilities.chunk_size != chunk_size or self.capabilities.pipe_depth != depth:
            close = self.lib.npu_nvme_close(self.ctx,120000)
            if not close:
                self.lib.npu_nvme_cleanup(self.ctx); self.ctx=None
                _RETAINED_TRANSPORTS.discard(self)
            raise RuntimeError(f'effective native configuration differs: {rc}; close={close}')
        self.metadata = MetadataIO(self.lib, self.ctx)
        self.total_bytes = self.capabilities.namespace_bytes

    def quiescent(self):
        return bool(self.ctx) and not self.retained and self.lib.npu_nvme_wait_quiescent(self.ctx,1)==0

    def _request(self, spec, owners, *, copy=False):
        request=C.POINTER(NPUNVMERequest)()
        submit=self.lib.npu_nvme_submit_copy if copy else self.lib.npu_nvme_submit_transfer
        rc=submit(self.ctx,C.byref(spec),C.byref(request))
        if rc: raise TransferFailure(f'async admission failed: {rc}',transport_safe=True)
        try:
            rc=self.lib.npu_nvme_wait_request(request,0)
            done=C.c_int()
            self.lib.npu_nvme_poll_request(request,C.byref(done))
            if not done.value:
                # Releasing a request handle never releases its borrowed buffers.
                with self._buffers_lock: self.retained.append(tuple(owners))
                raise TransferFailure(f'async transfer has no stop proof: {rc}; buffers retained',transport_safe=False)
            if rc: raise TransferFailure(f'async transfer failed: {rc}',transport_safe=True)
            receipt=NPUNVMETransferReceipt()
            rc=self.lib.npu_nvme_get_transfer_receipt(request,C.byref(receipt),C.sizeof(receipt))
            if rc or not receipt.transport_safe:
                with self._buffers_lock: self.retained.append(tuple(owners))
                raise TransferFailure(f'invalid transfer receipt: {rc}',transport_safe=False)
            digest=None
            if spec.checksum_flags:
                digest=NPUNVMETransferDigest()
                rc=self.lib.npu_nvme_get_transfer_digests(request,C.byref(digest),1)
                if rc: raise TransferFailure(f'transfer digest unavailable: {rc}',transport_safe=True)
            return receipt,digest
        finally:
            self.lib.npu_nvme_release_request(request)

    def _io(self, offset, size, data=None):
        buffer=C.create_string_buffer((size+BLOCK_SIZE-1)//BLOCK_SIZE*BLOCK_SIZE)
        if data is not None: C.memmove(buffer,data,size)
        item=NPUNVMETransferItem(C.addressof(buffer),offset,size)
        spec=NPUNVMETransferSpec(C.sizeof(NPUNVMETransferSpec),1,int(data is None),1,0,1,C.pointer(item))
        self._request(spec,(buffer,))
        return buffer.raw[:size] if data is None else None

    def read(self, offset, size): return self._io(offset,size)
    def write(self, offset, data): return self._io(offset,len(data),data)
    def flush(self):
        spec=NPUNVMETransferSpec(C.sizeof(NPUNVMETransferSpec),1,4,1,0,0,None)
        receipt,_=self._request(spec,())
        if not receipt.data_durable: raise RuntimeError('flush did not publish durable completion')

    def frozen_bytes(self, item, offset, size):
        if item.get('placement')=='host' or (item.get('snapshot_dev_ptr') is None and not item.get('borrowed_device')):
            return memoryview(item['np_arr']).cast('B')[offset:offset+size].tobytes()
        buffer=C.create_string_buffer(size)
        spec=NPUNVMECopySpec(C.sizeof(NPUNVMECopySpec),1,5,0,item['ptr']+offset,C.addressof(buffer),size)
        self._request(spec,(item,buffer),copy=True)
        return buffer.raw

    def write_frozen(self, disk_offset, item, offset, size):
        owners=[item]
        host=item.get('placement')=='host' or (item.get('snapshot_dev_ptr') is None and not item.get('borrowed_device'))
        if host:
            data=memoryview(item['np_arr']).cast('B')[offset:offset+size].tobytes()
            buffer=C.create_string_buffer(data,size);owners.append(buffer);pointer=C.addressof(buffer)
        else: pointer=item['ptr']+offset
        part=NPUNVMETransferItem(pointer,disk_offset,size)
        spec=NPUNVMETransferSpec(C.sizeof(NPUNVMETransferSpec),1,0,int(host),3,1,C.pointer(part))
        _,digest=self._request(spec,owners)
        return bytes(digest.sha256).hex()

    def copy_h2d(self, pointer, data, *, owner):
        buffer=C.create_string_buffer(data,len(data))
        spec=NPUNVMECopySpec(C.sizeof(NPUNVMECopySpec),1,6,0,C.addressof(buffer),pointer,len(data))
        self._request(spec,(buffer,owner),copy=True)

    def apply_chunk(self, target, name, offset, data):
        if hasattr(target,'set_copy_executor'): target.set_copy_executor(self.copy_h2d)
        target.apply_chunk(name,offset,data)

    def close(self, timeout):
        if not self.ctx: return
        if self.retained: raise RuntimeError('transport resources quarantined; stop proof required')
        rc=self.lib.npu_nvme_close(self.ctx,max(1,int(timeout*1000)))
        if rc: raise RuntimeError(f'close incomplete: {rc}')
        self.lib.npu_nvme_cleanup(self.ctx);self.ctx=None
        _RETAINED_TRANSPORTS.discard(self)
