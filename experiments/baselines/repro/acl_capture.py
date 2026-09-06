"""FULL training-state capture from real MindSpore NPU addresses."""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass

import numpy as np

from experiments.baselines import two_phase_common as tpc
from python.direct_checkpoint import get_dev_ptr
from python.training_state import encode_control_value

from .state_bridge import Snapshot


@dataclass
class ACLPinnedSlot:
    slot_id: int
    ptr: int
    size: int

    def close(self):
        if self.ptr:
            tpc.free_pinned_host_buffer(self.ptr)
            self.ptr = 0


def device_state_layout(components):
    """Describe every unique model/optimizer tensor without copying to Host."""
    seen_objects = set()
    seen_ptrs = {}
    fields = []
    host_fields = []
    offset = 0
    for category, component in components.items():
        for name, parameter in component.parameters_and_names():
            if id(parameter) in seen_objects:
                continue
            seen_objects.add(id(parameter))
            ptr = int(get_dev_ptr(parameter))
            dtype = np.dtype(str(parameter.dtype).replace("Float", "float").replace(
                "Int", "int").replace("UInt", "uint").lower())
            # MindSpore's dtype string is not guaranteed to be NumPy spelling.
            try:
                import mindspore as ms
                dtype = np.dtype(ms.dtype_to_nptype(parameter.dtype))
            except Exception:
                pass
            size = int(parameter.size) * int(dtype.itemsize)
            canonical = f"{category}/{name}"
            if not ptr:
                host_fields.append({
                    "name": canonical, "category": category,
                    "dtype": dtype.str, "shape": list(parameter.shape),
                    "nbytes": size, "device_kind": "host",
                    "address": None, "host_offset": None,
                    "alias_group": canonical, "parameter": parameter,
                })
                continue
            alias = seen_ptrs.get(ptr)
            if alias is not None:
                continue
            seen_ptrs[ptr] = canonical
            fields.append({
                "name": canonical, "category": category, "dtype": dtype.str,
                "shape": list(parameter.shape), "nbytes": size,
                "device_kind": "npu", "address": ptr,
                "host_offset": offset, "alias_group": canonical,
            })
            offset += size
    if not fields:
        raise RuntimeError("FULL ACL capture found no NPU tensors")
    return fields, host_fields, offset


def allocate_slot(slot_id, size):
    return ACLPinnedSlot(int(slot_id), tpc.allocate_pinned_host_buffer(int(size)),
                         int(size))


def capture_to_slot(fields, host_fields, slot, controls, device_id):
    """Synchronize the training boundary, then copy all fields with ACL D2H."""
    import mindspore as ms

    tpc._ensure_acl_device(int(device_id))
    sync_begin = time.monotonic_ns()
    if hasattr(ms, "runtime") and hasattr(ms.runtime, "synchronize"):
        ms.runtime.synchronize()
    else:
        ms.hal.synchronize()
    sync_end = time.monotonic_ns()
    dma_begin = time.monotonic_ns()
    chunks = []
    for field in fields:
        submit_ns = time.monotonic_ns()
        rc = tpc.acl_lib.aclrtMemcpy(
            ctypes.c_void_p(slot.ptr + int(field["host_offset"])),
            int(field["nbytes"]), ctypes.c_void_p(int(field["address"])),
            int(field["nbytes"]), tpc.ACL_MEMCPY_DEVICE_TO_HOST)
        tpc._check_acl_ret(rc, f"FULL D2H {field['name']}")
        chunks.append({"name": field["name"], "bytes": field["nbytes"],
                       "submit_ns": submit_ns,
                       "complete_ns": time.monotonic_ns()})
    dma_end = time.monotonic_ns()
    payload, controls_metadata = encode_control_value(controls)
    schema_fields = [{key: value for key, value in field.items()
                      if key != "address"} for field in fields]
    schema_fields.append({
        "name": "controls/state", "category": "control",
        "dtype": payload.dtype.str, "shape": list(payload.shape),
        "nbytes": int(payload.nbytes), "device_kind": "host",
        "host_offset": None, "alias_group": "controls/state",
    })
    arrays = {}
    backing = (ctypes.c_uint8 * slot.size).from_address(slot.ptr)
    byte_view = np.ctypeslib.as_array(backing)
    for field in fields:
        begin = int(field["host_offset"])
        end = begin + int(field["nbytes"])
        arrays[field["name"]] = np.frombuffer(
            byte_view[begin:end], dtype=np.dtype(field["dtype"])).reshape(
                tuple(field["shape"]))
    for field in host_fields:
        arrays[field["name"]] = np.ascontiguousarray(
            field["parameter"].value().asnumpy())
        schema_fields.append({key: value for key, value in field.items()
                              if key not in ("address", "parameter")})
    snapshot = Snapshot(arrays, payload, controls_metadata, {
        "schema_version": 2, "format": "npu-semantic-port-full-v1",
        "capture_backend": "aclrtMemcpy-from-real-npu-address",
        "fields": sorted(schema_fields, key=lambda item: item["name"]),
    })
    return snapshot, {
        "sync_begin_ns": sync_begin, "sync_end_ns": sync_end,
        "dma_begin_ns": dma_begin, "dma_end_ns": dma_end,
        "dma_chunks": chunks, "capture_backend": "aclrtMemcpy",
    }
