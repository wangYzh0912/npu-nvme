"""Bounded Host staging over the existing C ABI, with explicit buffer ownership."""
import ctypes
import threading
from .bindings import NPUNVMEContext
from .metadata import MetadataIO
from .layout import BLOCK_SIZE


class FullTransport:
    def __init__(self, backend, *, pci, npu, depth, chunk_size, profiling_dir, profiling=False):
        self.lib, self.acl = backend.lib, backend.acl_lib
        self.npu = npu
        self.ctx = ctypes.POINTER(NPUNVMEContext)()
        self.retained = []
        self._buffers_lock = threading.Lock()
        rc = self.lib.npu_nvme_init(ctypes.byref(self.ctx), pci.encode(), npu, depth,
                                     chunk_size, profiling, str(profiling_dir).encode())
        if rc != 0: raise RuntimeError(f'native open rejected: {rc}')
        self.metadata = MetadataIO(self.lib, self.ctx)
        self.total_bytes = self.lib.npu_nvme_get_total_blocks(self.ctx)

    def quiescent(self):
        return bool(self.ctx) and not self.retained and self.lib.npu_nvme_wait_quiescent(self.ctx, 1) == 0

    def _io(self, offset, size, data=None):
        allocation = (size + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        buffer = ctypes.create_string_buffer(allocation)
        if data is not None: ctypes.memmove(buffer, data, size)
        ptrs = (ctypes.c_void_p * 1)(ctypes.addressof(buffer))
        offsets = (ctypes.c_uint64 * 1)(offset)
        sizes = (ctypes.c_size_t * 1)(size)
        call = self.lib.npu_nvme_read_batch_host if data is None else self.lib.npu_nvme_write_batch_host
        rc = call(self.ctx, ptrs, offsets, sizes, 1)
        if rc != 0:
            # Even when the wrapper returns, lack of a quiescence proof keeps
            # the caller-owned staging allocation alive with the store.
            if self.lib.npu_nvme_wait_quiescent(self.ctx, 1) != 0:
                with self._buffers_lock: self.retained.append(buffer)
            raise RuntimeError(f'FULL {"read" if data is None else "write"} failed: {rc}')
        return buffer.raw[:size] if data is None else None

    def read(self, offset, size): return self._io(offset, size)
    def write(self, offset, data): return self._io(offset, len(data), data)
    def flush(self): self.metadata.flush_nvme()

    def frozen_bytes(self, item, offset, size):
        if item.get('placement') == 'host' or item.get('snapshot_dev_ptr') is None:
            array = item['np_arr']
            return memoryview(array).cast('B')[offset:offset+size].tobytes()
        buffer = ctypes.create_string_buffer(size)
        if self.acl.aclrtSetDevice(self.npu) != 0: raise RuntimeError('cannot bind capture device')
        rc = self.acl.aclrtMemcpy(ctypes.byref(buffer), size,
                                ctypes.c_void_p(item['ptr'] + offset), size, 2)
        if rc != 0:
            with self._buffers_lock: self.retained.append(buffer)
            raise RuntimeError(f'frozen D2H failed: {rc}; resources retained')
        return buffer.raw

    def close(self, timeout):
        if not self.ctx: return
        if self.retained: raise RuntimeError('transport resources quarantined; stop proof required')
        rc = self.lib.npu_nvme_close(self.ctx, max(1, int(timeout * 1000)))
        if rc != 0: raise RuntimeError(f'close incomplete: {rc}')
        self.lib.npu_nvme_cleanup(self.ctx)
        self.ctx = None
