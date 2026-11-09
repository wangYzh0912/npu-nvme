"""Compatibility references; importing this module never opens a library."""
from threading import Lock
from npu_nvme.storage.bindings import (NPUNVMEContext, NPUNVMERequest,
    NPUNVMERetainedSlot, NPUNVMEStats, BackendUnavailable, load_backend)
_backend = None
_lock = Lock()

def _open():
    global _backend
    with _lock:
        if _backend is None: _backend = load_backend()
    return _backend

class _Library:
    def __init__(self, name): self.name=name
    def __getattr__(self, name): return getattr(getattr(_open(), self.name), name)

lib = _Library('lib')
acl_lib = _Library('acl_lib')

def __getattr__(name):
    if name == '_LIB_PATH': return _open().library_path
    raise AttributeError(name)
