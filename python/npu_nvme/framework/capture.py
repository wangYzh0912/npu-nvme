"""Framework allocation adapter with an explicit device-pointer provider."""
import math
import hashlib
import ctypes
from typing import Dict
import numpy as np
from npu_nvme.storage.chunks import (build_chunks, _integer, _chunk_size,
                                     MAX_BATCH_ITEMS, MAX_BATCH_BYTES)

def rebuild_chunks_from_meta(models, params_meta: Dict, chunk_size: int, *, pointer_of):
    """Rebuild device + host chunk lists from checkpoint metadata for load().

    Args:
        models:      a model or list of models with parameters_and_names()
        params_meta: dict of {param_name: {offset, size, shape, dtype}}
        chunk_size:  max bytes per DMA chunk
    Returns:
        (dev_chunks, host_chunks, buffers) — two chunk lists + detail list
    """
    _chunk_size(chunk_size)
    if len(params_meta) > MAX_BATCH_ITEMS:
        raise ValueError("too many metadata descriptors")
    if not isinstance(models, (list, tuple)):
        models = [models]
    planned = []
    total = count = 0
    for model in models:
        if model is None or not hasattr(model, "parameters_and_names"):
            continue
        for name, param in model.parameters_and_names():
            if name not in params_meta:
                continue
            info = params_meta[name]
            if len(info["shape"]) > 32:
                raise ValueError("tensor rank exceeds limit")
            shape = tuple(_integer(x, "shape", minimum=1) for x in info["shape"])
            if not isinstance(info["dtype"], str) or len(info["dtype"]) > 64:
                raise ValueError("invalid dtype descriptor")
            try:
                dtype = np.dtype(info["dtype"])
            except TypeError as error:
                raise ValueError("invalid dtype descriptor") from error
            if dtype.hasobject or dtype.fields or dtype.kind not in 'biufc':
                raise ValueError("unsupported tensor dtype")
            size = _integer(info["size"], "size", minimum=1, maximum=MAX_BATCH_BYTES)
            if math.prod(shape) * dtype.itemsize != size:
                raise ValueError("shape/dtype does not match DMA size")
            actual_shape = tuple(getattr(param, "sliced_shape", None) or param.shape)
            if hasattr(param, "data") and hasattr(param.data, "shape"):
                if math.prod(param.data.shape) < math.prod(actual_shape):
                    actual_shape = tuple(param.data.shape)
            if shape != actual_shape:
                raise ValueError("checkpoint shape does not match target allocation")
            if hasattr(param, "dtype"):
                import mindspore as ms
                if np.dtype(ms.dtype_to_nptype(param.dtype)) != dtype:
                    raise ValueError("checkpoint dtype does not match target allocation")
            offset = _integer(info["offset"], "offset")
            if offset % 4096 or offset > (1 << 64) - 1 - ((size + 4095) // 4096 * 4096):
                raise ValueError("invalid checkpoint extent")
            total += ((size + 4095) // 4096) * 4096
            count += (size + chunk_size - 1) // chunk_size
            if total > MAX_BATCH_BYTES or count > MAX_BATCH_ITEMS:
                raise ValueError("restore exceeds item or byte budget")
            planned.append((name, param, shape, dtype, size, offset))
    # Validate every descriptor before allocation or device pointer lookup.
    buffers = []
    for name, param, shape, dtype, size, offset in planned:
        dev_ptr = pointer_of(param)
        np_arr = None if dev_ptr else np.empty(shape, dtype=dtype)
        buffers.append({"name": name, "ptr": dev_ptr or np_arr.ctypes.data,
                        "size": size, "offset": offset, "np_arr": np_arr,
                        "param_ref": param, "use_dev": bool(dev_ptr)})

    buffers.sort(key=lambda x: x["offset"])

    dev_buffers = [b for b in buffers if b["use_dev"]]
    host_buffers = [b for b in buffers if not b["use_dev"]]

    dev_chunks, _ = build_chunks(dev_buffers, chunk_size)
    host_chunks, _ = build_chunks(host_buffers, chunk_size)

    return dev_chunks, host_chunks, buffers


class FrozenCapture:
    """Own framework parameter registry and frozen buffers; dependencies are explicit."""

    def __init__(self, *, rank_id, device_id, framework, acl, pointer_of):
        self.rank_id = rank_id
        self.npu_device_id = device_id
        self.framework = framework
        self.acl = acl
        self.pointer_of = pointer_of
        self.local_valid_param_names = None

    def build_registry(self, models):
        if not isinstance(models, (list, tuple)):
            models = [models]
        self.local_valid_param_names = set()

        print(f"[DirectCkpt] Rank {self.rank_id} building parameter registry...",
              flush=True)

        for model in models:
            if model is None or not hasattr(model, "parameters_and_names"):
                continue
            for name, p in model.parameters_and_names():
                self.local_valid_param_names.add(name)

        print(f"[DirectCkpt] Rank {self.rank_id} registry: "
              f"{len(self.local_valid_param_names)} valid params.", flush=True)


    def prepare(self, models):
        if getattr(self, "local_valid_param_names", None) is None:
            self.build_registry(models)

        if not isinstance(models, (list, tuple)):
            models = [models]
        params = []
        for model in models:
            if model is None or not hasattr(model, "parameters_and_names"):
                continue
            for name, p in model.parameters_and_names():
                if name not in self.local_valid_param_names:
                    continue

                ptr = self.pointer_of(p)

                dtype_np = np.dtype(self.framework.dtype_to_nptype(p.dtype))
                local_shape = p.shape

                if hasattr(p, "sliced_shape") and p.sliced_shape:
                    local_shape = p.sliced_shape
                elif hasattr(p, "data") and hasattr(p.data, "shape"):
                    if np.prod(p.data.shape) < np.prod(p.shape):
                        local_shape = p.data.shape

                if np.prod(local_shape) == 0:
                    continue
                size = int(np.prod(local_shape)) * dtype_np.itemsize

                host_arr = None
                if ptr == 0:
                    if not hasattr(p, "asnumpy"):
                        raise RuntimeError(
                            f"Host parameter {name} has no asnumpy() accessor")
                    host_arr = np.asarray(p.asnumpy(), dtype=dtype_np).copy()
                    if np.prod(local_shape) != np.prod(host_arr.shape):
                        host_arr = host_arr.reshape(tuple(local_shape)).copy()

                params.append({
                    "name": name, "ptr": ptr, "size": size,
                    "shape": list(p.shape), "dtype": dtype_np.name,
                    "np_arr": host_arr,
                    "param_ref": p,
                })
        return params


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

    def prepare_state_components(self, components, with_checksums=True):
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
                checksum = None
                if with_checksums:
                    checksum_arr = (host_arr if host_arr is not None else
                                    np.ascontiguousarray(parameter.value().asnumpy(),
                                                         dtype=dtype_np))
                    checksum = hashlib.sha256(
                        np.ascontiguousarray(checksum_arr, dtype=dtype_np)
                        .reshape(-1).tobytes()).hexdigest()
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
                    "sha256": checksum,
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
        if hasattr(self.framework.hal, "synchronize"):
            self.framework.hal.synchronize()
        elif hasattr(self.framework, "runtime") and hasattr(self.framework.runtime, "synchronize"):
            self.framework.runtime.synchronize()
        elif self.acl is not None and hasattr(self.acl, "aclrtSynchronizeStream"):
            self.acl.aclrtSynchronizeStream(None)
