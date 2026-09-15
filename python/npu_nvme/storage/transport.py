"""Existing blocking batch ABI adapter; no commit/target ownership.

Bounded request migration and removal of legacy drain behavior belong to B2/C2.
"""
import ctypes
from .bindings import NPUNVMERequest
from .chunks import build_chunks, build_ctypes_arrays


class LegacyBatchTransport:
    def __init__(self, binding, context):
        self.binding = binding
        self.context = context

    def read_state(self, dev_buffers, host_buffers, chunk_size):
        dev_chunks, _ = build_chunks(dev_buffers, chunk_size)
        host_chunks, _ = build_chunks(host_buffers, chunk_size)
        if dev_chunks:
            arrays = build_ctypes_arrays(dev_chunks)
            rc = self.binding.npu_nvme_read_batch(self.context, *arrays, len(dev_chunks))
            if rc != 0:
                raise RuntimeError(f"training-state device read failed: {rc}")
        if host_chunks:
            arrays = build_ctypes_arrays(host_chunks)
            rc = self.binding.npu_nvme_read_batch_host(self.context, *arrays, len(host_chunks))
            if rc != 0:
                raise RuntimeError(f"training-state host read failed: {rc}")

    def write_device(self, ptrs, offsets, sizes, count, io_mode):
        if count <= 0:
            return
        if io_mode == "serial":
            for index in range(count):
                one_ptr = (ctypes.c_void_p * 1)(ptrs[index])
                one_off = (ctypes.c_uint64 * 1)(offsets[index])
                one_size = (ctypes.c_size_t * 1)(sizes[index])
                rc = self.binding.npu_nvme_write_batch(
                    self.context, one_ptr, one_off, one_size, 1)
                if rc != 0:
                    raise RuntimeError(f"serial write failed (rc={rc})")
            return
        if io_mode == "async" and hasattr(
                self.binding, "npu_nvme_submit_write_batch"):
            request = ctypes.POINTER(NPUNVMERequest)()
            rc = self.binding.npu_nvme_submit_write_batch(
                self.context, ptrs, offsets, sizes, count,
                ctypes.byref(request))
            if rc != 0:
                raise RuntimeError(f"async submit failed (rc={rc})")
            try:
                rc = self.binding.npu_nvme_wait_request(request, 0)
                if rc != 0:
                    raise RuntimeError(f"async request failed (rc={rc})")
            finally:
                self.binding.npu_nvme_release_request(request)
            return
        rc = self.binding.npu_nvme_write_batch(
            self.context, ptrs, offsets, sizes, count)
        if rc != 0:
            raise RuntimeError(f"write_batch failed (rc={rc})")


    def write_host(self, ptrs, offsets, sizes, count, io_mode):
        if not hasattr(self.binding, "npu_nvme_write_batch_host"):
            raise RuntimeError(
                "C library missing npu_nvme_write_batch_host")
        if io_mode == "live_host":
            request = ctypes.POINTER(NPUNVMERequest)()
            rc = self.binding.npu_nvme_submit_write_batch_host(
                self.context, ptrs, offsets, sizes, count,
                ctypes.byref(request))
            if rc == 0:
                try:
                    rc = self.binding.npu_nvme_wait_request(request, 0)
                finally:
                    self.binding.npu_nvme_release_request(request)
        else:
            rc = self.binding.npu_nvme_write_batch_host(
                self.context, ptrs, offsets, sizes, count)
        if rc != 0:
            raise RuntimeError(f"write_batch_host failed (rc={rc})")

    def quiescent(self):
        return not self.context or (hasattr(self.binding, 'npu_nvme_wait_quiescent') and
            self.binding.npu_nvme_wait_quiescent(self.context, 1) == 0)
