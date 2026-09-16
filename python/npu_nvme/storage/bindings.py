"""Canonical ABI declarations and explicit, lazy library loading."""
import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

class BackendUnavailable(RuntimeError):
    """A selected runtime library or required ABI symbol is unavailable."""
    exit_code = 3


class NPUNVMEContext(ctypes.Structure):
    """Opaque context handle — Python sees this as an opaque pointer."""
    pass


class NPUNVMERequest(ctypes.Structure):
    """Opaque asynchronous write request."""
    pass


class NPUNVMECapabilities(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in
        ('struct_size','version','operation_mask','block_size','chunk_size','pipe_depth',
         'max_request_items','max_pending_requests','copy_bytes_per_tick','checksum_bytes_per_tick',
         'submit_items_per_tick','quantum_items')] + [(name, ctypes.c_uint64) for name in
         ('max_request_bytes','namespace_bytes','dma_pool_bytes')]


class NPUNVMECopySpec(ctypes.Structure):
    _fields_ = [(name,ctypes.c_uint32) for name in ('struct_size','version','operation','checksum_flags')] + [
        ('source',ctypes.c_void_p),('destination',ctypes.c_void_p),('length',ctypes.c_uint64),
        ('expected_crc32',ctypes.c_uint32),('reserved',ctypes.c_uint32),('expected_sha256',ctypes.c_uint8*32)]


class NPUNVMETransferItem(ctypes.Structure):
    _fields_ = [("address", ctypes.c_void_p), ("offset", ctypes.c_uint64),
                ("length", ctypes.c_uint64), ("expected_crc32", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32), ("expected_sha256", ctypes.c_uint8 * 32)]


class NPUNVMETransferSpec(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in
                ("struct_size", "version", "operation", "memory_kind", "checksum_flags", "item_count")] + [
                ("items", ctypes.POINTER(NPUNVMETransferItem))]


class NPUNVMETransferReceipt(ctypes.Structure):
    _fields_ = [("request_id", ctypes.c_uint64), ("logical_bytes", ctypes.c_uint64),
                ("item_count", ctypes.c_uint32), ("operation", ctypes.c_uint32),
                ("result", ctypes.c_int32)] + [(name, ctypes.c_uint32) for name in
                ("done", "source_safe", "transport_safe", "data_durable", "reserved")]


class NPUNVMETransferDigest(ctypes.Structure):
    _fields_ = [("crc32", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                ("sha256", ctypes.c_uint8 * 32)]


class NPUNVMERetainedSlot(ctypes.Structure):
    _fields_ = [("slot", ctypes.c_uint32), ("reason", ctypes.c_uint32),
                ("request_id", ctypes.c_uint64), ("bytes", ctypes.c_uint64),
                ("nvme_offset", ctypes.c_uint64)]


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



def load_backend(library_path=None, acl_library="libascendcl.so"):
    """Load libraries only when explicitly opened; do not initialize devices."""
    _LIB_PATH = str(library_path or os.environ.get("NPU_NVME_LIBRARY_PATH") or (Path(__file__).resolve().parents[3] / "build_out/lib/libnpu_nvme.so"))
    try:
        acl_lib = ctypes.CDLL(acl_library)
        acl_lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
        acl_lib.aclrtMalloc.restype = ctypes.c_int
        acl_lib.aclrtFree.argtypes = [ctypes.c_void_p]
        acl_lib.aclrtFree.restype = ctypes.c_int
        acl_lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
        acl_lib.aclrtSynchronizeStream.restype = ctypes.c_int
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
        raise BackendUnavailable(f"ACL backend unavailable: {e}") from e


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
        if hasattr(lib, 'npu_nvme_register_tasks'):
            lib.npu_nvme_register_tasks.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_int,
            ]
            lib.npu_nvme_register_tasks.restype = ctypes.c_int

        # B2 additive ABI; absence is explicit at the selected transport boundary.
        if hasattr(lib, "npu_nvme_get_capabilities"):
            lib.npu_nvme_get_capabilities.argtypes = [ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(NPUNVMECapabilities), ctypes.c_uint32]
            lib.npu_nvme_get_capabilities.restype = ctypes.c_int
        if hasattr(lib, "npu_nvme_submit_copy"):
            lib.npu_nvme_submit_copy.argtypes = [ctypes.POINTER(NPUNVMEContext),ctypes.POINTER(NPUNVMECopySpec),ctypes.POINTER(ctypes.POINTER(NPUNVMERequest))]
            lib.npu_nvme_submit_copy.restype = ctypes.c_int
        if hasattr(lib, "npu_nvme_submit_transfer"):
            lib.npu_nvme_submit_transfer.argtypes = [ctypes.POINTER(NPUNVMEContext),
                ctypes.POINTER(NPUNVMETransferSpec), ctypes.POINTER(ctypes.POINTER(NPUNVMERequest))]
            lib.npu_nvme_submit_transfer.restype = ctypes.c_int
            lib.npu_nvme_get_transfer_receipt.argtypes = [ctypes.POINTER(NPUNVMERequest),
                ctypes.POINTER(NPUNVMETransferReceipt), ctypes.c_uint32]
            lib.npu_nvme_get_transfer_receipt.restype = ctypes.c_int
            lib.npu_nvme_get_transfer_digests.argtypes = [ctypes.POINTER(NPUNVMERequest),
                ctypes.POINTER(NPUNVMETransferDigest), ctypes.c_uint32]
            lib.npu_nvme_get_transfer_digests.restype = ctypes.c_int

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
        if hasattr(lib, "npu_nvme_get_retained_slots"):
            lib.npu_nvme_get_retained_slots.argtypes = [ctypes.POINTER(NPUNVMEContext),
                ctypes.POINTER(NPUNVMERetainedSlot), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
            lib.npu_nvme_get_retained_slots.restype = ctypes.c_int
        if hasattr(lib, "npu_nvme_close"):
            lib.npu_nvme_close.argtypes = [ctypes.POINTER(NPUNVMEContext), ctypes.c_uint32]
            lib.npu_nvme_close.restype = ctypes.c_int

        if hasattr(lib, "npu_nvme_submit_write_batch"):
            lib.npu_nvme_submit_write_batch.argtypes = [
                ctypes.POINTER(NPUNVMEContext),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_size_t), ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(NPUNVMERequest)),
            ]
            lib.npu_nvme_submit_write_batch.restype = ctypes.c_int
        if hasattr(lib, "npu_nvme_submit_write_batch_host"):
            lib.npu_nvme_submit_write_batch_host.argtypes = [
                ctypes.POINTER(NPUNVMEContext),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_size_t), ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(NPUNVMERequest)),
            ]
            lib.npu_nvme_submit_write_batch_host.restype = ctypes.c_int
        if hasattr(lib, "npu_nvme_poll_request"):
            lib.npu_nvme_poll_request.argtypes = [
                ctypes.POINTER(NPUNVMERequest), ctypes.POINTER(ctypes.c_int)]
            lib.npu_nvme_poll_request.restype = ctypes.c_int
        if hasattr(lib, "npu_nvme_wait_request"):
            lib.npu_nvme_wait_request.argtypes = [
                ctypes.POINTER(NPUNVMERequest), ctypes.c_uint32]
            lib.npu_nvme_wait_request.restype = ctypes.c_int
        if hasattr(lib, "npu_nvme_release_request"):
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

        # -- Delta frame ring-buffer layout (bookkeeping only, no I/O) --
        if hasattr(lib, "npu_nvme_delta_init"):
            lib.npu_nvme_delta_init.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_uint32]
            lib.npu_nvme_delta_init.restype = ctypes.c_int
        if hasattr(lib, "npu_nvme_delta_get_area_offset"):
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
        if hasattr(lib, "npu_nvme_get_io_timeout_ms"):
            lib.npu_nvme_get_io_timeout_ms.argtypes = [ctypes.c_void_p]
            lib.npu_nvme_get_io_timeout_ms.restype = ctypes.c_uint32
        if hasattr(lib, "npu_nvme_wait_quiescent"):
            lib.npu_nvme_wait_quiescent.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            lib.npu_nvme_wait_quiescent.restype = ctypes.c_int

        lib.npu_nvme_get_last_io_us.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.npu_nvme_get_last_io_us.restype = ctypes.c_uint64

    except (OSError, AttributeError) as e:
        raise BackendUnavailable(f"NVMe backend unavailable: {_LIB_PATH}: {e}") from e
    for symbol in ('npu_nvme_flush','npu_nvme_close','npu_nvme_wait_quiescent',
                   'npu_nvme_submit_transfer','npu_nvme_submit_copy','npu_nvme_get_capabilities','npu_nvme_poll_request','npu_nvme_wait_request','npu_nvme_release_request'):
        if not hasattr(lib,symbol): raise BackendUnavailable('required FULL symbol missing: '+symbol)
    if not hasattr(acl_lib,'aclrtSetDevice'): raise BackendUnavailable('required ACL symbol missing: aclrtSetDevice')
    if not hasattr(lib,'npu_nvme_abi_version'): raise BackendUnavailable('ABI2 version symbol missing')
    lib.npu_nvme_abi_version.argtypes=[];lib.npu_nvme_abi_version.restype=ctypes.c_uint32
    if lib.npu_nvme_abi_version()!=2: raise BackendUnavailable('ABI major mismatch: expected2')
    return SimpleNamespace(lib=lib, acl_lib=acl_lib, library_path=_LIB_PATH)
