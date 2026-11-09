"""Owner-only NVMe transport uses the registered SPDK rank pool directly."""
import ctypes as C
import threading
_RETAINED=[]
class SharedTransport:
    def __init__(self,transport,pool):
        self.transport=transport;self.pool=pool;self.lock=threading.RLock();self.unsafe=False
        lib=transport.lib
        lib.npu_nvme_register_shared_host.argtypes=[C.c_void_p,C.c_char_p];lib.npu_nvme_register_shared_host.restype=C.c_int
        rc=lib.npu_nvme_register_shared_host(transport.ctx,pool.plan.name.encode())
        if rc:raise RuntimeError('shared pool registration failed: '+str(rc))
    def transfer(self,*,read,disk_offset,lease,expected_sha256=None):
        from npu_nvme.storage.bindings import NPUNVMETransferItem,NPUNVMETransferSpec,NPUNVMERequest,NPUNVMETransferReceipt,NPUNVMETransferDigest
        if self.unsafe:raise RuntimeError('shared transport quarantined')
        with self.lock:
            view=self.pool.view(lease);pointer=C.addressof(view);length=lease['length']
            item=NPUNVMETransferItem(address=pointer,offset=disk_offset,length=length)
            if read:
                if type(expected_sha256) is not str or len(expected_sha256)!=64:raise ValueError('restore requires digest')
                item.expected_sha256[:]=bytes.fromhex(expected_sha256)
            spec=NPUNVMETransferSpec(struct_size=C.sizeof(NPUNVMETransferSpec),version=1,operation=int(read),memory_kind=2,checksum_flags=2,item_count=1,items=C.pointer(item))
            req=C.POINTER(NPUNVMERequest)();lib=self.transport.lib
            rc=lib.npu_nvme_submit_transfer(self.transport.ctx,C.byref(spec),C.byref(req))
            if rc:raise RuntimeError('shared transfer admission failed: '+str(rc))
            try:
                lib.npu_nvme_wait_request(req,0)
                receipt=NPUNVMETransferReceipt()
                if lib.npu_nvme_get_transfer_receipt(req,C.byref(receipt),C.sizeof(receipt)) or not receipt.done or not receipt.transport_safe:
                    self.unsafe=True;self.pool.retained=True;_RETAINED.append((self,view,lease))
                    raise RuntimeError('shared NVMe transfer lacks stop proof')
                if receipt.result:raise RuntimeError('shared NVMe transfer failed: '+str(receipt.result))
                digest=NPUNVMETransferDigest()
                if lib.npu_nvme_get_transfer_digests(req,C.byref(digest),1):raise RuntimeError('shared digest missing')
                return bytes(digest.sha256).hex()
            finally:lib.npu_nvme_release_request(req)
