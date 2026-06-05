"""ctypes bindings for the pure transfer API (no checkpoint/probe).

Usage:
- from transfer_api import TransferAPI

Inputs:
- NVMe PCI address, NPU device id, transfer sizes.
Outputs:
- Returns status codes from native library calls.
"""
import ctypes
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_PATH = os.path.join(_REPO_ROOT, "build_out", "lib", "libnpu_nvme.so")

class NPUNVMEContext(ctypes.Structure):
    pass

class TransferAPI:
    def __init__(self, lib_path=_LIB_PATH):
        self._lib = ctypes.CDLL(lib_path)
        self._ctx = ctypes.POINTER(NPUNVMEContext)()
        self._bind_symbols()

    def _bind_symbols(self):
        lib = self._lib
        lib.npu_nvme_transfer_init.argtypes = [
            ctypes.POINTER(ctypes.POINTER(NPUNVMEContext)), ctypes.c_char_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_char_p,
        ]
        lib.npu_nvme_transfer_init.restype = ctypes.c_int

        lib.npu_nvme_transfer_cleanup.argtypes = [ctypes.POINTER(NPUNVMEContext)]
        lib.npu_nvme_transfer_cleanup.restype = None

        lib.npu_nvme_transfer_get_max_transfer.argtypes = [ctypes.POINTER(NPUNVMEContext)]
        lib.npu_nvme_transfer_get_max_transfer.restype = ctypes.c_int

        lib.npu_nvme_transfer_get_total_blocks.argtypes = [ctypes.POINTER(NPUNVMEContext)]
        lib.npu_nvme_transfer_get_total_blocks.restype = ctypes.c_uint64

        lib.npu_nvme_transfer_sync_meta_io.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p
        ]
        lib.npu_nvme_transfer_sync_meta_io.restype = ctypes.c_int

        lib.npu_nvme_transfer_write_batch.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
        ]
        lib.npu_nvme_transfer_write_batch.restype = ctypes.c_int

        lib.npu_nvme_transfer_read_batch.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
        ]
        lib.npu_nvme_transfer_read_batch.restype = ctypes.c_int

        lib.npu_nvme_transfer_write_batch_host.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
        ]
        lib.npu_nvme_transfer_write_batch_host.restype = ctypes.c_int

    def init(self, pci_addr, npu_id=0, pipe_depth=8, chunk_size=4 * 1024 * 1024,
             enable_profiling=False, prof_dir=b"."):
        if isinstance(pci_addr, str):
            pci_addr = pci_addr.encode("utf-8")
        if isinstance(prof_dir, str):
            prof_dir = prof_dir.encode("utf-8")
        rc = self._lib.npu_nvme_transfer_init(
            ctypes.byref(self._ctx), pci_addr, npu_id,
            pipe_depth, chunk_size, enable_profiling, prof_dir,
        )
        return rc

    def cleanup(self):
        if self._ctx:
            self._lib.npu_nvme_transfer_cleanup(self._ctx)
            self._ctx = ctypes.POINTER(NPUNVMEContext)()

    @property
    def ctx(self):
        return self._ctx
