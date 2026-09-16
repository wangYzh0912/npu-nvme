"""Legacy eager-open facade; new callers explicitly open storage.bindings."""
import ctypes
from npu_nvme.storage.bindings import (NPUNVMEContext, NPUNVMERequest,
    NPUNVMERetainedSlot, NPUNVMEStats, load_backend)

_backend = load_backend()
lib = _backend.lib
acl_lib = _backend.acl_lib
_LIB_PATH = _backend.library_path
