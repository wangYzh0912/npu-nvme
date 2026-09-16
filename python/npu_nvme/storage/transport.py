"""Compatibility names backed exclusively by versioned async requests."""
import ctypes as C
from .bindings import NPUNVMERequest,NPUNVMETransferItem,NPUNVMETransferSpec,NPUNVMECapabilities
from .chunks import iter_chunk_windows,build_ctypes_arrays


class LegacyBatchTransport:
    def __init__(self,binding,context):
        self.binding,self.context=binding,context
        self.retained=[]
        self.capabilities=NPUNVMECapabilities()
        rc=binding.npu_nvme_get_capabilities(context,C.byref(self.capabilities),C.sizeof(self.capabilities))
        if rc:raise RuntimeError(f'capabilities unavailable: {rc}')

    def _transfer(self,ptrs,offsets,sizes,count,*,host,read=False,serial=False,owners=()):
        start=0
        while start<count:
            stop=start;total=0
            while stop<count and stop-start<(1 if serial else self.capabilities.max_request_items):
                size=int(sizes[stop]);padded=(size+4095)//4096*4096
                if padded>self.capabilities.max_request_bytes-total:break
                total+=padded;stop+=1
            if stop==start:raise ValueError('item exceeds request capability')
            items=(NPUNVMETransferItem*(stop-start))()
            for item,i in zip(items,range(start,stop)):
                item.address=ptrs[i];item.offset=offsets[i];item.length=sizes[i]
            spec=NPUNVMETransferSpec(C.sizeof(NPUNVMETransferSpec),1,int(read),int(host),0,len(items),items)
            req=C.POINTER(NPUNVMERequest)()
            rc=self.binding.npu_nvme_submit_transfer(self.context,C.byref(spec),C.byref(req))
            if rc:raise RuntimeError(f'transfer admission failed: {rc}')
            try:
                rc=self.binding.npu_nvme_wait_request(req,0)
                if rc:
                    done=C.c_int();self.binding.npu_nvme_poll_request(req,C.byref(done))
                    if not done.value:self.retained.append((ptrs,offsets,sizes,owners))
                    raise RuntimeError(f'transfer failed: {rc}; source owner must remain until quiescent')
            finally:self.binding.npu_nvme_release_request(req)
            start=stop

    def read_state(self,dev_buffers,host_buffers,chunk_size):
        for buffers,host in ((dev_buffers,False),(host_buffers,True)):
            for chunks in iter_chunk_windows(buffers,chunk_size,max_items=self.capabilities.max_request_items,
                    max_bytes=self.capabilities.max_request_bytes):
                self._transfer(*build_ctypes_arrays(chunks),len(chunks),host=host,read=True,owners=buffers)

    def write_device(self,ptrs,offsets,sizes,count,io_mode):
        self._transfer(ptrs,offsets,sizes,count,host=False,serial=io_mode=='serial')

    def write_host(self,ptrs,offsets,sizes,count,io_mode):
        self._transfer(ptrs,offsets,sizes,count,host=True,serial=io_mode=='serial')

    def quiescent(self):
        return not self.context or (not self.retained and self.binding.npu_nvme_wait_quiescent(self.context,1)==0)
