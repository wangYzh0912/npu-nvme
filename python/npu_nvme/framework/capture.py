"""Framework allocation adapter with an explicit device-pointer provider."""
import math
import ctypes
from typing import Dict
import numpy as np
from npu_nvme.storage.chunks import (build_chunks, _integer, _chunk_size,
                                     MAX_BATCH_ITEMS, MAX_BATCH_BYTES)



class FrozenCapture:
    """Own framework parameter registry and frozen buffers; dependencies are explicit."""

    def __init__(self, *, rank_id, device_id, framework, acl, pointer_of):
        self.rank_id = rank_id
        self.npu_device_id = device_id
        self.framework = framework
        self.acl = acl
        self.pointer_of = pointer_of





    def snapshot(self, params, generation):
        """Freeze device/host parameters before starting background I/O."""
        if self.acl is None:
            raise RuntimeError("Ascend ACL library is required for snapshots")
        if hasattr(self.acl, "aclrtSetDevice"):
            rc = self.acl.aclrtSetDevice(self.npu_device_id)
            if rc != 0:
                raise RuntimeError(f"aclrtSetDevice failed during snapshot: {rc}")

        frozen = []
        allocated = []
        try:
            for item in params:
                copy_item = dict(item)
                copy_item["generation"] = generation
                if item["ptr"]:
                    device_ptr = ctypes.c_void_p()
                    rc = self.acl.aclrtMalloc(
                        ctypes.byref(device_ptr), item["size"], 0)
                    if rc != 0 or not device_ptr.value:
                        raise RuntimeError(
                            f"aclrtMalloc snapshot failed for {item['name']}: {rc}")
                    allocated.append(device_ptr)
                    rc = self.acl.aclrtMemcpy(
                        device_ptr, item["size"],
                        ctypes.c_void_p(item["ptr"]), item["size"], 3)
                    if rc != 0:
                        raise RuntimeError(
                            f"device snapshot copy failed for {item['name']}: {rc}")
                    copy_item["ptr"] = int(device_ptr.value)
                    copy_item["snapshot_dev_ptr"] = device_ptr
                else:
                    # The array was copied in prepare; copy again so
                    # callers cannot mutate the source while the I/O thread
                    # is running.
                    copy_item["np_arr"] = np.ascontiguousarray(
                        item["np_arr"]).copy()
                    copy_item["ptr"] = int(copy_item["np_arr"].ctypes.data)
                frozen.append(copy_item)
        except BaseException:
            for ptr in allocated:
                self.acl.aclrtFree(ptr)
            raise
        return frozen


    def release(self, params):
        if self.acl is None:
            return
        for item in params or []:
            ptr = item.get("snapshot_dev_ptr")
            if ptr is not None and ptr.value:
                self.acl.aclrtFree(ptr)
                item["snapshot_dev_ptr"] = ctypes.c_void_p()

    def ordered_components(self, components):
        preferred = [name for name in ("model", "optimizer")
                     if name in components]
        preferred.extend(sorted(name for name in components
                                if name not in {"model", "optimizer"}))
        return preferred

    def prepare_state_components(self, components):
        """Build namespaced descriptors, excluding aliased model weights.

        MindSpore optimizers commonly expose the model weights alongside
        moment slots.  Object-identity de-duplication keeps those weights in
        ``model/`` and stores only optimizer-owned state in ``optimizer/``.
        """
        params = []
        seen_objects = set()
        seen_names = set()
        for component in self.ordered_components(components):
            obj = components[component]
            if obj is None or not hasattr(obj, "parameters_and_names"):
                raise TypeError(
                    f"component {component!r} has no parameters_and_names()")
            for source_name, parameter in obj.parameters_and_names():
                identity = id(parameter)
                if identity in seen_objects:
                    continue
                seen_objects.add(identity)
                name = f"{component}/{source_name}"
                if name in seen_names:
                    raise ValueError(f"duplicate training-state field: {name}")
                seen_names.add(name)

                dtype_np = np.dtype(self.framework.dtype_to_nptype(parameter.dtype))
                local_shape = tuple(parameter.shape)
                if hasattr(parameter, "sliced_shape") and parameter.sliced_shape:
                    local_shape = tuple(parameter.sliced_shape)
                elif hasattr(parameter, "data") and hasattr(parameter.data, "shape"):
                    data_shape = tuple(parameter.data.shape)
                    if np.prod(data_shape) < np.prod(local_shape):
                        local_shape = data_shape
                if int(np.prod(local_shape)) == 0:
                    continue

                ptr = self.pointer_of(parameter)
                host_arr = None
                if ptr == 0:
                    host_arr = np.ascontiguousarray(
                        parameter.asnumpy(), dtype=dtype_np).reshape(local_shape)
                params.append({
                    "name": name,
                    "source_name": source_name,
                    "component": component,
                    "category": "parameter",
                    "placement": "device" if ptr else "host",
                    "ptr": ptr,
                    "size": int(np.prod(local_shape)) * dtype_np.itemsize,
                    "shape": list(local_shape),
                    "dtype": dtype_np.name,
                    "np_arr": host_arr,
                    "param_ref": parameter,
                })
        if not params:
            raise ValueError("components contain no persistable parameters")
        return params

    def validate(self, params, chunk_size):
        from npu_nvme.storage.chunks import validate_descriptors
        # Reject malformed or excessive input before any snapshot allocation.
        descriptors = []
        for item in params:
            ptr = item["ptr"]
            if not ptr and item.get("np_arr") is not None:
                array = item["np_arr"]
                if array.nbytes != item["size"] or not array.flags.c_contiguous:
                    raise ValueError("host buffer length/contiguity differs from descriptor")
                ptr = int(array.ctypes.data)
            if "shape" in item and "dtype" in item:
                dtype = np.dtype(item["dtype"])
                if dtype.hasobject or math.prod(item["shape"]) * dtype.itemsize != item["size"]:
                    raise ValueError("snapshot shape/dtype differs from descriptor")
            descriptors.append(dict(item, ptr=ptr, offset=0))
        validate_descriptors(descriptors, chunk_size)

    def synchronize(self):
        """A framework barrier is required before describing/freeze-copying state."""
        self.framework.hal.synchronize()
