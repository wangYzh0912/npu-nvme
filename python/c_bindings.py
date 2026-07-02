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
_LIB_PATH = os.path.join(_REPO_ROOT, "install", "lib", "libnpu_nvme.so")


# -- ACL runtime (Ascend) --
try:
    acl_lib = ctypes.CDLL("libascendcl.so")
    acl_lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_int]
    acl_lib.aclrtMemcpy.restype = ctypes.c_int
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

    # -- Query --
    lib.npu_nvme_get_max_transfer.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_get_max_transfer.restype = ctypes.c_int

    lib.npu_nvme_get_total_blocks.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_get_total_blocks.restype = ctypes.c_uint64

    # -- Synchronous metadata I/O --
    lib.npu_nvme_sync_meta_io.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.c_uint64,
        ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.npu_nvme_sync_meta_io.restype = ctypes.c_int

    # -- Batch data transfer --
    lib.npu_nvme_write_batch.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]
    lib.npu_nvme_write_batch.restype = ctypes.c_int

    lib.npu_nvme_read_batch.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]
    lib.npu_nvme_read_batch.restype = ctypes.c_int

    if hasattr(lib, "npu_nvme_raw_write_batch_host"):
        lib.npu_nvme_raw_write_batch_host.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        lib.npu_nvme_raw_write_batch_host.restype = ctypes.c_int
    if hasattr(lib, "npu_nvme_raw_read_batch_host"):
        lib.npu_nvme_raw_read_batch_host.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        lib.npu_nvme_raw_read_batch_host.restype = ctypes.c_int
    if hasattr(lib, "npu_nvme_raw_write_batch"):
        lib.npu_nvme_raw_write_batch.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        lib.npu_nvme_raw_write_batch.restype = ctypes.c_int
    if hasattr(lib, "npu_nvme_raw_read_batch"):
        lib.npu_nvme_raw_read_batch.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        lib.npu_nvme_raw_read_batch.restype = ctypes.c_int

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
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
        lib.npu_nvme_delta_init.restype = ctypes.c_int
        lib.npu_nvme_delta_get_area_offset.argtypes = [ctypes.c_void_p]
        lib.npu_nvme_delta_get_area_offset.restype = ctypes.c_uint64
    if hasattr(lib, "npu_nvme_delta_get_slot_size"):
        lib.npu_nvme_delta_get_slot_size.argtypes = [ctypes.c_void_p]
        lib.npu_nvme_delta_get_slot_size.restype = ctypes.c_uint64
    if hasattr(lib, "npu_nvme_delta_get_slot_count"):
        lib.npu_nvme_delta_get_slot_count.argtypes = [ctypes.c_void_p]
        lib.npu_nvme_delta_get_slot_count.restype = ctypes.c_uint32

except OSError as e:
    print(f"[Warning] Failed to load {_LIB_PATH}. Error: {e}")
    lib = None
    acl_lib = None
