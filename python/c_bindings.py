"""C library ctypes bindings for libnpu_nvme.so and libascendcl.so.

Import this module to access the C library handle (``lib``), the ACL
helper (``acl_lib``), and the opaque context type (``NPUNVMEContext``).

All argtypes / restype declarations are centralised here — no other file
should declare them.
"""

import ctypes
import os

# -- Library path --
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_PATH = os.path.join(_REPO_ROOT, "build_out", "lib", "libnpu_nvme.so")


# -- ACL runtime (Ascend) --
try:
    acl_lib = ctypes.CDLL("libascendcl.so")
    acl_lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_int]
    acl_lib.aclrtMemcpy.restype = ctypes.c_int
    acl_lib.aclrtMemcpyAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    acl_lib.aclrtMemcpyAsync.restype = ctypes.c_int
    acl_lib.aclrtMallocHost.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    acl_lib.aclrtMallocHost.restype = ctypes.c_int
    acl_lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    acl_lib.aclrtFreeHost.restype = ctypes.c_int
    acl_lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    acl_lib.aclrtCreateStream.restype = ctypes.c_int
    acl_lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    acl_lib.aclrtDestroyStream.restype = ctypes.c_int
    acl_lib.aclrtCreateEvent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    acl_lib.aclrtCreateEvent.restype = ctypes.c_int
    acl_lib.aclrtDestroyEvent.argtypes = [ctypes.c_void_p]
    acl_lib.aclrtDestroyEvent.restype = ctypes.c_int
    acl_lib.aclrtRecordEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    acl_lib.aclrtRecordEvent.restype = ctypes.c_int
    acl_lib.aclrtSynchronizeEvent.argtypes = [ctypes.c_void_p]
    acl_lib.aclrtSynchronizeEvent.restype = ctypes.c_int
    acl_lib.aclrtQueryEventStatus.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    acl_lib.aclrtQueryEventStatus.restype = ctypes.c_int
    acl_lib.aclrtStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    acl_lib.aclrtStreamWaitEvent.restype = ctypes.c_int
    acl_lib.aclrtEventElapsedTime.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    acl_lib.aclrtEventElapsedTime.restype = ctypes.c_int
    _set_device = getattr(acl_lib, "aclrtSetDevice", None)
    if _set_device is not None:
        _set_device.argtypes = [ctypes.c_int]
        _set_device.restype = ctypes.c_int
except Exception as e:
    print(f"[DirectCkpt] Warning: Failed to load libascendcl.so for probe: {e}")
    acl_lib = None


# -- libnpu_nvme.so --
class NPUNVMEContext(ctypes.Structure):
    """Opaque context handle — Python sees this as an opaque pointer."""
    pass


class NPUNVMERequest(ctypes.Structure):
    """Opaque asynchronous write request."""
    pass


class NPUNVMEStats(ctypes.Structure):
    _fields_ = [
        ("nvme_submit_count", ctypes.c_uint64),
        ("nvme_complete_count", ctypes.c_uint64),
        ("nvme_outstanding", ctypes.c_uint32),
        ("nvme_outstanding_peak", ctypes.c_uint32),
        ("dma_inflight", ctypes.c_uint32),
        ("dma_inflight_peak", ctypes.c_uint32),
        ("request_ring_depth", ctypes.c_uint32),
        ("request_ring_peak", ctypes.c_uint32),
        ("async_dma_submit_count", ctypes.c_uint64),
        ("async_event_query_count", ctypes.c_uint64),
        ("async_event_query_error_count", ctypes.c_uint64),
        ("stream_sync_fallback_count", ctypes.c_uint64),
        ("spdk_retry_count", ctypes.c_uint64),
        ("completion_error_count", ctypes.c_uint64),
        ("reactor_cpu_us", ctypes.c_uint64),
    ]


try:
    lib = ctypes.CDLL(_LIB_PATH)

    # -- FaF listener control --
    if hasattr(lib, "npu_nvme_set_probe_flag_ptr"):
        lib.npu_nvme_set_probe_flag_ptr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.npu_nvme_set_probe_flag_ptr.restype = ctypes.c_int
    if hasattr(lib, "npu_nvme_set_probe_flag_value"):
        lib.npu_nvme_set_probe_flag_value.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.c_uint32]
        lib.npu_nvme_set_probe_flag_value.restype = ctypes.c_int
    if hasattr(lib, "npu_nvme_get_probe_flag_dev_ptr"):
        lib.npu_nvme_get_probe_flag_dev_ptr.argtypes = [ctypes.c_void_p]
        lib.npu_nvme_get_probe_flag_dev_ptr.restype = ctypes.c_void_p
    if hasattr(lib, "npu_nvme_set_step_ptr"):
        lib.npu_nvme_set_step_ptr.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.npu_nvme_set_step_ptr.restype = ctypes.c_int

    # -- Task registration --
    lib.npu_nvme_register_tasks.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]
    lib.npu_nvme_register_tasks.restype = ctypes.c_int

    # -- Init / cleanup --
    lib.npu_nvme_init.argtypes = [
        ctypes.POINTER(ctypes.POINTER(NPUNVMEContext)),
        ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_bool, ctypes.c_char_p,
    ]
    lib.npu_nvme_init.restype = ctypes.c_int

    lib.npu_nvme_cleanup.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_cleanup.restype = None

    if hasattr(lib, "npu_nvme_submit_write_batch"):
        lib.npu_nvme_submit_write_batch.argtypes = [
            ctypes.POINTER(NPUNVMEContext),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(NPUNVMERequest)),
        ]
        lib.npu_nvme_submit_write_batch.restype = ctypes.c_int
        lib.npu_nvme_submit_write_batch_host.argtypes = [
            ctypes.POINTER(NPUNVMEContext),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(NPUNVMERequest)),
        ]
        lib.npu_nvme_submit_write_batch_host.restype = ctypes.c_int
        lib.npu_nvme_poll_request.argtypes = [
            ctypes.POINTER(NPUNVMERequest), ctypes.POINTER(ctypes.c_int)]
        lib.npu_nvme_poll_request.restype = ctypes.c_int
        lib.npu_nvme_wait_request.argtypes = [
            ctypes.POINTER(NPUNVMERequest), ctypes.c_uint32]
        lib.npu_nvme_wait_request.restype = ctypes.c_int
        lib.npu_nvme_release_request.argtypes = [
            ctypes.POINTER(NPUNVMERequest)]
        lib.npu_nvme_release_request.restype = None

    # -- Query --
    lib.npu_nvme_get_max_transfer.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_get_max_transfer.restype = ctypes.c_int

    lib.npu_nvme_get_total_blocks.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_get_total_blocks.restype = ctypes.c_uint64
    if hasattr(lib, "npu_nvme_get_stats"):
        lib.npu_nvme_get_stats.argtypes = [ctypes.POINTER(NPUNVMEContext),
                                           ctypes.POINTER(NPUNVMEStats)]
        lib.npu_nvme_get_stats.restype = ctypes.c_int

    # -- Synchronous metadata I/O --
    lib.npu_nvme_sync_meta_io.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.c_uint64,
        ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.npu_nvme_sync_meta_io.restype = ctypes.c_int
    if hasattr(lib, "npu_nvme_flush"):
        lib.npu_nvme_flush.argtypes = [ctypes.POINTER(NPUNVMEContext)]
        lib.npu_nvme_flush.restype = ctypes.c_int

    # -- Batch data transfer --
    lib.npu_nvme_write_batch.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]
    lib.npu_nvme_write_batch.restype = ctypes.c_int
    if hasattr(lib, "npu_nvme_write_batch_crc"):
        lib.npu_nvme_write_batch_crc.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ]
        lib.npu_nvme_write_batch_crc.restype = ctypes.c_int

    lib.npu_nvme_read_batch.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]
    lib.npu_nvme_read_batch.restype = ctypes.c_int

    if hasattr(lib, "npu_nvme_read_batch_host"):
        lib.npu_nvme_read_batch_host.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        lib.npu_nvme_read_batch_host.restype = ctypes.c_int

    if hasattr(lib, "npu_nvme_write_batch_host"):
        lib.npu_nvme_write_batch_host.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        lib.npu_nvme_write_batch_host.restype = ctypes.c_int

    # -- Delta frame ring-buffer layout (bookkeeping only, no I/O) --
    if hasattr(lib, "npu_nvme_delta_init"):
        lib.npu_nvme_delta_init.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_uint32]
        lib.npu_nvme_delta_init.restype = ctypes.c_int
        lib.npu_nvme_delta_get_area_offset.argtypes = [ctypes.c_void_p]
        lib.npu_nvme_delta_get_area_offset.restype = ctypes.c_uint64
    if hasattr(lib, "npu_nvme_delta_get_slot_size"):
        lib.npu_nvme_delta_get_slot_size.argtypes = [ctypes.c_void_p]
        lib.npu_nvme_delta_get_slot_size.restype = ctypes.c_uint64
    if hasattr(lib, "npu_nvme_delta_get_slot_count"):
        lib.npu_nvme_delta_get_slot_count.argtypes = [ctypes.c_void_p]
        lib.npu_nvme_delta_get_slot_count.restype = ctypes.c_uint32
    if hasattr(lib, "npu_nvme_set_io_timeout_ms"):
        lib.npu_nvme_set_io_timeout_ms.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32]
        lib.npu_nvme_set_io_timeout_ms.restype = ctypes.c_int
        lib.npu_nvme_get_io_timeout_ms.argtypes = [ctypes.c_void_p]
        lib.npu_nvme_get_io_timeout_ms.restype = ctypes.c_uint32
    if hasattr(lib, "npu_nvme_wait_quiescent"):
        lib.npu_nvme_wait_quiescent.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.npu_nvme_wait_quiescent.restype = ctypes.c_int

except OSError as e:
    print(f"[Warning] Failed to load {_LIB_PATH}. Error: {e}")
    lib = None
    acl_lib = None
