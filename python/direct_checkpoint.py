"""Direct checkpoint manager with MindSpore probe wrapper.

Usage:
- Import DirectCheckpoint and ProbeTrainOneStepCell from this module.
- Used by training scripts under python/.

Inputs:
- NVMe PCI address, NPU device id, chunk size, profiling directory.
Outputs:
- Writes checkpoints and metadata to NVMe and optional profiling CSV under output/.

Compatibility note:
  This module depends on the PRIVATE MindSpore API _data_ptr() to obtain
  NPU device pointers.  The API is available in MS 2.4+ but may break on
  any version upgrade since it is not part of the public contract.  All
  device-pointer access is routed through the get_dev_ptr() helper — if
  a future MS release removes _data_ptr(), only that function needs to
  be updated.

Sub-modules (importable independently):
  c_bindings      — ctypes wrappers for libnpu_nvme.so + libascendcl.so
  disk_layout     — raw block-device byte-offset constants
  chunk_helpers   — build_chunks / build_chunks_host / rebuild_chunks_from_meta
  delta_protocol  — pack/unpack/apply delta frames + FileDeltaWriter
  noop_init       — NoOpInitializer for fast NVMe checkpoint restore
  training_cell   — ProbeTrainOneStepCell with FaF step_counter injection
"""

import ctypes
from dataclasses import replace
import hashlib
import math
import os
import pickle
import json
import struct
import time
import threading
from typing import List, Dict

import mindspore as ms
from mindspore import ops, nn, Tensor
import numpy as np

import atexit

# -- Re-exports from sub-modules (backward-compatible surface) ---------------
import c_bindings  # keep module reference for _LIB_PATH
from c_bindings import lib, acl_lib, NPUNVMEContext
from disk_layout import (SUPERBLOCK_OFFSET, SUPERBLOCK_HEADER_BYTES,
                          META_SLOT_A_OFFSET, META_SLOT_B_OFFSET,
                          META_SLOT_BYTES, MAGIC_NUMBER, UINT32_BYTES,
                          DELTA_MAGIC, FRAME_HEADER_SIZE, BLOCK_SIZE,
                          DiskLayout, make_layout, pack_metadata,
                          unpack_metadata, pack_superblock,
                          unpack_superblock)
from chunk_helpers import (build_chunks, build_chunks_host,
                            build_ctypes_arrays, rebuild_chunks_from_meta)
from delta_protocol import (pack_delta_frame, pack_lossless_delta_frame,
                             pack_s2_replacement_frame,
                             unpack_delta_frame, unpack_delta_frame_with_meta,
                             apply_delta_patches, FileDeltaWriter)
from noop_init import NoOpInitializer, replace_with_noop_initializer
from training_cell import ProbeTrainOneStepCell
from training_state import (TRAINING_STATE_SCHEMA_VERSION,
                            decode_control_value, encode_control_value,
                            validate_state_names)


# -- Device pointer helper (single entry point for all MS pointer access) ----

def get_dev_ptr(tensor):
    """Return the NPU device pointer for a MindSpore Parameter or Tensor.

    Uses the private _data_ptr() API (available MS 2.4+).  Falls back to
    the undocumented data_ptr() on older builds.  Returns 0 on failure.

    NOTE: _data_ptr() is a PRIVATE MindSpore API and may break on any
    version upgrade.  ALL device-pointer access in this codebase should
    go through this function so there is a single point to fix.
    """
    ptr = 0
    try:
        data_obj = tensor.data if hasattr(tensor, "data") else tensor
        if hasattr(data_obj, "_data_ptr"):
            if isinstance(tensor, ms.Parameter) and hasattr(tensor, "is_inited") and not tensor.is_inited:
                tensor.init_data()
            ptr = int(data_obj._data_ptr())
        elif hasattr(data_obj, "data_ptr"):
            ptr = int(data_obj.data_ptr())
    except Exception:
        ptr = 0
    return ptr


# -- Import-time API availability check --------------------------------------

_MS_HAS_DATA_PTR = False
try:
    _dummy = ms.Tensor([0])
    _MS_HAS_DATA_PTR = hasattr(_dummy, "_data_ptr") or hasattr(_dummy, "data_ptr")
except Exception:
    pass
if not _MS_HAS_DATA_PTR:
    import warnings
    warnings.warn(
        "MindSpore Tensor has neither _data_ptr() nor data_ptr().  "
        "DirectCheckpoint will NOT be able to obtain NPU device addresses.  "
        "Upgrade MindSpore or use MS 2.4+.")


# -- Completion handle -------------------------------------------------------

class CheckpointHandle:
    """Observable completion state for one frozen checkpoint generation."""

    DISPATCHED = "DISPATCHED"
    PERSISTED = "PERSISTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def __init__(self, owner, request_id, generation, step):
        self.owner = owner
        self.request_id = request_id
        self.generation = generation
        self.step = step
        self.status = self.DISPATCHED
        self.error = None
        self._done = threading.Event()

    def _complete(self):
        self.status = self.PERSISTED
        self._done.set()

    def _fail(self, error):
        self.status = self.FAILED
        self.error = error
        self._done.set()

    def wait(self, timeout=None):
        """Wait for durable completion and raise the original failure."""
        self.owner.wait_for_io_completion(timeout=timeout)
        if self.status == self.FAILED:
            raise RuntimeError("checkpoint persistence failed") from self.error
        if self.status != self.PERSISTED:
            raise TimeoutError("checkpoint did not reach PERSISTED state")
        return self

    def __iter__(self):
        """Legacy tuple compatibility for existing benchmark callers."""
        yield 0
        yield getattr(self.owner, "_last_chunk_count", 0)
        yield 0.0
        yield 0.0
        yield getattr(self.owner, "_last_save_stats", {})


# -- DirectCheckpoint: NVMe-backed training checkpoint manager ----------------

class DirectCheckpoint:
    _ms_warmed_up = False

    def __init__(
        self, nvme_addr: str = "0000:83:00.0", npu_device_id: int = 0,
        pipeline_depth: int = 4, requested_chunk_size: int = 4 * 1024 * 1024,
        enable_profiling: bool = False, profiling_dir: str = "./output/profiling",
        rank_id: int = 0, world_size: int = 1,
        base_offset_bytes: int = 0, shard_span_bytes: int = None,
        spdk_shm_id: int = 1, keep_last_n: int = 3, slot_size_gb: int = 10,
        warmup_fn: callable = None,
    ):
        self.ctx = ctypes.POINTER(NPUNVMEContext)()
        self.npu_device_id = npu_device_id
        self.enable_profiling = enable_profiling
        self.profiling_dir = profiling_dir
        self.rank_id = rank_id
        self.world_size = world_size
        self.base_offset_bytes = base_offset_bytes
        self.shard_span_bytes = shard_span_bytes
        os.environ.setdefault("SPDK_SHM_ID", str(spdk_shm_id))

        self.keep_last_n = keep_last_n
        self.slot_bytes = slot_size_gb * 1024**3
        self.active_meta_slot = 0
        self.metadata_generation = 0
        self.layout = None
        self.stack_start_bytes = 0
        self.meta_dict = {"checkpoints": {}}
        self.last_layout = []
        self._meta_pkl = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "experiments", "output", "checkpoint_meta.pkl")

        if self.enable_profiling:
            os.makedirs(self.profiling_dir, exist_ok=True)

        if warmup_fn is not None and not DirectCheckpoint._ms_warmed_up:
            print("[DirectCheckpoint] Running MS runtime warmup before SPDK init...",
                  flush=True)
            warmup_fn()
            DirectCheckpoint._ms_warmed_up = True
            print("[DirectCheckpoint] MS runtime warmup complete.", flush=True)

        print(f"[DirectCheckpoint] loading so from {c_bindings._LIB_PATH}")

        rc = lib.npu_nvme_init(
            ctypes.byref(self.ctx),
            nvme_addr.encode(),
            npu_device_id,
            pipeline_depth,
            requested_chunk_size,
            enable_profiling,
            self.profiling_dir.encode("utf-8"),
        )
        if rc != 0:
            raise RuntimeError("npu_nvme_init failed")

        self.total_bytes = lib.npu_nvme_get_total_blocks(self.ctx)
        if self.total_bytes == 0:
            raise RuntimeError("Failed to get NVMe total bytes from hardware.")

        self.chunk_size = requested_chunk_size
        effective = lib.npu_nvme_get_max_transfer(self.ctx)
        print(f"[DirectCheckpoint] init ok. chunk={self.chunk_size/1024/1024:.2f}MB "
              f"(effective={effective/1024/1024:.2f}MB), rank={self.rank_id}/{self.world_size}")

        self._spdk_initialized = True
        self._closed = False
        self.io_thread = None
        self._io_error = None
        self._active_handle = None
        self._request_counter = 0
        self._snapshot_generation = 0
        self._last_chunk_count = 0
        self._last_save_stats = {}
        atexit.register(self.close)

        # Set up ctypes signature for C-layer profiling
        lib.npu_nvme_get_last_io_us.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.npu_nvme_get_last_io_us.restype = ctypes.c_uint64

        self._mount_filesystem()

    # -- Filesystem mount ----------------------------------------------------

    def _mount_filesystem(self):
        sb_buf = ctypes.create_string_buffer(4096)
        rc = lib.npu_nvme_sync_meta_io(
            self.ctx, SUPERBLOCK_OFFSET, 4096, 1,
            ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc != 0:
            raise RuntimeError("Failed to read Superblock.")

        try:
            self.layout = unpack_superblock(sb_buf.raw)
        except ValueError as error:
            raise RuntimeError(
                f"Superblock verification failed: {error}. "
                "Run format_npu_disk.py with the V2 layout to initialize the disk.") from error
        if self.layout.total_bytes != self.total_bytes:
            raise RuntimeError("superblock capacity does not match NVMe capacity")
        sb_active_meta_slot = self.layout.active_meta_slot
        sb_generation = self.layout.generation
        self.active_meta_slot = sb_active_meta_slot
        self.metadata_generation = sb_generation
        self.stack_start_bytes = self.layout.full_base

        valid = []
        for slot, target_offset in enumerate((META_SLOT_A_OFFSET,
                                               META_SLOT_B_OFFSET)):
            meta_buf = ctypes.create_string_buffer(META_SLOT_BYTES)
            rc = lib.npu_nvme_sync_meta_io(
                self.ctx, target_offset, META_SLOT_BYTES, 1,
                ctypes.c_void_p(ctypes.addressof(meta_buf)))
            if rc != 0:
                continue
            try:
                generation, payload = unpack_metadata(meta_buf.raw)
                valid.append((generation, slot, payload))
            except (ValueError, json.JSONDecodeError):
                continue
        # The superblock is the commit point.  A valid-looking inactive
        # replica may be a torn/future write and must not become visible just
        # because it has a larger generation.  The designated slot must
        # match the superblock generation; the other slot is only a usable
        # previous committed replica.
        committed = [item for item in valid
                     if item[1] == sb_active_meta_slot
                     and item[0] == sb_generation]
        if committed:
            generation, slot, payload = committed[0]
        else:
            previous = [item for item in valid
                        if item[1] != sb_active_meta_slot
                        and item[0] < sb_generation]
            if not previous:
                raise RuntimeError(
                    "no metadata replica matches the committed superblock")
            generation, slot, payload = max(previous, key=lambda item: item[0])
        self.metadata_generation = generation
        self.active_meta_slot = slot
        self.meta_dict = payload
        self.layout = replace(self.layout, generation=generation,
                              active_meta_slot=slot)

        print(f"[Rank {self.rank_id}] FileSystem Mounted. "
              f"V{2} generation={generation} "
              f"Active Slot: {'A' if self.active_meta_slot == 0 else 'B'}")

    # -- Slot layout ---------------------------------------------------------

    def _get_current_slot_base_offset(self, step: int):
        if self.layout is None:
            raise RuntimeError("disk layout is not mounted")
        if self.slot_bytes != self.layout.full_slot_bytes:
            raise RuntimeError(
                "configured full slot size differs from formatted disk")
        return self.layout.full_slot_offset(
            self.rank_id, step, self.keep_last_n)

    # -- Metadata commit -----------------------------------------------------

    def _commit_metadata(self, step: int, layout: List[Dict],
                         checkpoint_meta: Dict = None):
        if self.rank_id != 0:
            return

        if self.layout is None:
            raise RuntimeError("cannot commit metadata before mounting layout")
        next_generation = self.metadata_generation + 1
        ckpt_key = f"step_{step}"
        param_records = {}
        for p in layout:
            record = {
                "offset": p["offset"], "size": p["size"],
                "shape": p["shape"], "dtype": p["dtype"],
            }
            # Keep the on-disk record compact.  The namespace in the key
            # already identifies model/optimizer/control and source_name is
            # the suffix after the first slash; placement is known from the
            # target object at load time.
            for field in ("sha256", "codec"):
                if field in p:
                    record[field] = p[field]
            param_records[p["name"]] = record

        checkpoint_record = {
            "type": "FULL",
            "generation": next_generation,
            "chunk_size": self.chunk_size,
            "rank_id": self.rank_id,
            "world_size": self.world_size,
            "params": param_records,
        }
        if checkpoint_meta:
            checkpoint_record.update(checkpoint_meta)
        self.meta_dict["checkpoints"][ckpt_key] = checkpoint_record

        saved_steps = []
        for k in self.meta_dict["checkpoints"].keys():
            if k.startswith("step_"):
                try:
                    saved_steps.append(int(k.split('_')[1]))
                except ValueError:
                    pass
        saved_steps = sorted(saved_steps)

        delta_keys = []
        for k in self.meta_dict.get("delta_chain", {}).keys():
            if k.startswith("step_"):
                try:
                    delta_keys.append(int(k.split('_')[1]))
                except ValueError:
                    pass
        for ds in delta_keys:
            if saved_steps and ds < saved_steps[0]:
                del self.meta_dict["delta_chain"][f"step_{ds}"]

        while len(saved_steps) > self.keep_last_n:
            oldest_step = saved_steps.pop(0)
            old_key = f"step_{oldest_step}"
            if old_key in self.meta_dict.get("checkpoints", {}):
                del self.meta_dict["checkpoints"][old_key]
            if old_key in self.meta_dict.get("delta_chain", {}):
                del self.meta_dict["delta_chain"][old_key]

        import pickle as _pickle
        os.makedirs(os.path.dirname(self._meta_pkl), exist_ok=True)
        with open(self._meta_pkl, "wb") as _f:
            _pickle.dump(self.meta_dict, _f)

        def _dump_meta_pkl():
            with open(self._meta_pkl, "wb") as _f:
                _pickle.dump(self.meta_dict, _f)
        self._dump_meta_pkl = _dump_meta_pkl

        self._persist_metadata(next_generation)
        print(f"[DirectCkpt] Rank 0 Meta committed safely to "
              f"Slot {'B' if self.active_meta_slot == 1 else 'A'} "
              "(Superblock updated).",
              flush=True)

    def _persist_metadata(self, generation=None):
        """Persist the current metadata dict using the A/B commit protocol."""
        if self.layout is None:
            raise RuntimeError("cannot persist metadata before mounting layout")
        if generation is None:
            generation = self.metadata_generation + 1
        next_slot = 1 if self.active_meta_slot == 0 else 0
        target_offset = (META_SLOT_B_OFFSET if next_slot == 1
                         else META_SLOT_A_OFFSET)
        meta_buf = ctypes.create_string_buffer(
            pack_metadata(self.meta_dict, generation), META_SLOT_BYTES)
        rc = lib.npu_nvme_sync_meta_io(
            self.ctx, target_offset, META_SLOT_BYTES, 0,
            ctypes.c_void_p(ctypes.addressof(meta_buf)))
        if rc != 0:
            raise RuntimeError(f"metadata replica write failed (rc={rc})")
        self.flush_nvme()
        new_layout = replace(self.layout, generation=generation,
                             active_meta_slot=next_slot)
        sb_buf = ctypes.create_string_buffer(pack_superblock(new_layout), 4096)
        rc = lib.npu_nvme_sync_meta_io(
            self.ctx, SUPERBLOCK_OFFSET, 4096, 0,
            ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc != 0:
            raise RuntimeError(f"superblock commit failed (rc={rc})")
        self.flush_nvme()
        self.active_meta_slot = next_slot
        self.metadata_generation = generation
        self.layout = new_layout

    def flush_nvme(self):
        """Wait for the namespace persistence barrier used by R0 commits."""
        if not hasattr(lib, "npu_nvme_flush"):
            raise RuntimeError("C library lacks npu_nvme_flush; rebuild required")
        rc = lib.npu_nvme_flush(self.ctx)
        if rc != 0:
            raise RuntimeError(f"NVMe flush failed (rc={rc})")

    # -- Probe flag helpers --------------------------------------------------

    def set_probe_flag_ptr(self, flag_tensor: Tensor = None):
        if not hasattr(lib, "npu_nvme_set_probe_flag_ptr"):
            raise RuntimeError(
                "npu_nvme_set_probe_flag_ptr is not available in the C library.")

        ptr = get_dev_ptr(flag_tensor) if flag_tensor is not None else 0
        rc = lib.npu_nvme_set_probe_flag_ptr(self.ctx, ctypes.c_void_p(ptr))
        if rc != 0:
            raise RuntimeError(
                f"Failed to set probe flag pointer. C API returned {rc}")

        if hasattr(lib, "npu_nvme_get_probe_flag_dev_ptr"):
            actual_ptr = lib.npu_nvme_get_probe_flag_dev_ptr(self.ctx)
            if actual_ptr:
                self.probe_flag_ptr = actual_ptr
                return
        self.probe_flag_ptr = ptr

    def read_probe_flag_dev(self) -> int:
        if acl_lib is None:
            raise RuntimeError("acl_lib not available for device read")
        if not hasattr(self, "probe_flag_ptr") or self.probe_flag_ptr == 0:
            raise RuntimeError("probe_flag_ptr is not set")

        ret = acl_lib.aclrtSetDevice(self.npu_device_id)
        if ret != 0:
            raise RuntimeError(f"aclrtSetDevice failed, ret={ret}")

        host_buf = ctypes.create_string_buffer(UINT32_BYTES)
        ret = acl_lib.aclrtMemcpy(
            ctypes.byref(host_buf), UINT32_BYTES,
            ctypes.c_void_p(self.probe_flag_ptr), UINT32_BYTES, 2)
        if ret != 0:
            raise RuntimeError(f"aclrtMemcpy device->host failed, ret={ret}")
        return int.from_bytes(host_buf.raw[:UINT32_BYTES],
                              byteorder="little", signed=False)

    def write_probe_flag_dev(self, value: int):
        if acl_lib is None:
            raise RuntimeError("acl_lib not available for device write")
        if not hasattr(self, "probe_flag_ptr") or self.probe_flag_ptr == 0:
            raise RuntimeError("probe_flag_ptr is not set")
        if value < 0:
            raise ValueError("probe flag value must be non-negative")

        ret = acl_lib.aclrtSetDevice(self.npu_device_id)
        if ret != 0:
            raise RuntimeError(f"aclrtSetDevice failed, ret={ret}")

        host_buf = ctypes.create_string_buffer(
            int(value).to_bytes(UINT32_BYTES, byteorder="little", signed=False))
        ret = acl_lib.aclrtMemcpy(
            ctypes.c_void_p(self.probe_flag_ptr), UINT32_BYTES,
            ctypes.byref(host_buf), UINT32_BYTES, 1)
        if ret != 0:
            raise RuntimeError(f"aclrtMemcpy host->device failed, ret={ret}")

    def probe_flag_selftest(self):
        orig = self.read_probe_flag_dev()
        self.write_probe_flag_dev(1)
        after = self.read_probe_flag_dev()
        self.write_probe_flag_dev(orig)
        print(f"[DirectCkpt] probe flag selftest: orig={orig}, after={after}")

    def set_probe_flag_value(self, value: int):
        if not hasattr(lib, "npu_nvme_set_probe_flag_value"):
            raise RuntimeError(
                "npu_nvme_set_probe_flag_value is not available in the C library.")
        if value < 0:
            raise ValueError("probe flag value must be non-negative")
        rc = lib.npu_nvme_set_probe_flag_value(
            self.ctx, ctypes.c_uint32(value))
        if rc != 0:
            raise RuntimeError(
                f"npu_nvme_set_probe_flag_value failed with rc={rc}")

    # -- Cleanup / lifecycle -------------------------------------------------

    def cleanup(self):
        io_error = None
        try:
            self.wait_for_io_completion()
            if self.ctx and hasattr(lib, "npu_nvme_wait_quiescent"):
                # A timed-out public batch call can leave a detached request
                # owned by the Reactor.  Do not release ACL context/HBM until
                # that request has reached a terminal state.
                rc = lib.npu_nvme_wait_quiescent(self.ctx, 600000)
                if rc != 0:
                    raise RuntimeError(
                        f"Reactor did not become quiescent (rc={rc})")
        except RuntimeError as error:
            io_error = error
        finally:
            if getattr(self, '_spdk_initialized', False) and self.ctx:
                lib.npu_nvme_cleanup(self.ctx)
                self.ctx = None
                self._spdk_initialized = False
        if io_error is not None:
            raise io_error

    def get_last_io_us(self, is_read: bool = False) -> int:
        """C-layer I/O latency in microseconds (DMA + SPDK only, no Python overhead).
        Returns 0 if no I/O has been performed."""
        if not self.ctx:
            return 0
        return lib.npu_nvme_get_last_io_us(self.ctx, 1 if is_read else 0)

    def close(self):
        if not getattr(self, '_closed', False) and hasattr(self, 'ctx') and self.ctx:
            print(f"[DirectCkpt] Rank {self.rank_id} safely tearing down "
                  f"NPUNVME context...", flush=True)
            try:
                self.cleanup()
            finally:
                self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _build_local_param_registry(self, models):
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

    def _prepare_params(self, models):
        if getattr(self, "local_valid_param_names", None) is None:
            self._build_local_param_registry(models)

        if not isinstance(models, (list, tuple)):
            models = [models]
        params = []
        for model in models:
            if model is None or not hasattr(model, "parameters_and_names"):
                continue
            for name, p in model.parameters_and_names():
                if name not in self.local_valid_param_names:
                    continue

                ptr = get_dev_ptr(p)

                dtype_np = np.dtype(ms.dtype_to_nptype(p.dtype))
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

    def _snapshot_params(self, params, generation):
        """Freeze device/host parameters before starting background I/O."""
        if acl_lib is None:
            raise RuntimeError("Ascend ACL library is required for snapshots")
        if not hasattr(acl_lib, "aclrtMalloc"):
            acl_lib.aclrtMalloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
            acl_lib.aclrtMalloc.restype = ctypes.c_int
            acl_lib.aclrtFree.argtypes = [ctypes.c_void_p]
            acl_lib.aclrtFree.restype = ctypes.c_int
        if hasattr(acl_lib, "aclrtSetDevice"):
            rc = acl_lib.aclrtSetDevice(self.npu_device_id)
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
                    rc = acl_lib.aclrtMalloc(
                        ctypes.byref(device_ptr), item["size"], 0)
                    if rc != 0 or not device_ptr.value:
                        raise RuntimeError(
                            f"aclrtMalloc snapshot failed for {item['name']}: {rc}")
                    allocated.append(device_ptr)
                    rc = acl_lib.aclrtMemcpy(
                        device_ptr, item["size"],
                        ctypes.c_void_p(item["ptr"]), item["size"], 3)
                    if rc != 0:
                        raise RuntimeError(
                            f"device snapshot copy failed for {item['name']}: {rc}")
                    copy_item["ptr"] = int(device_ptr.value)
                    copy_item["snapshot_dev_ptr"] = device_ptr
                else:
                    # The array was copied in _prepare_params; copy again so
                    # callers cannot mutate the source while the I/O thread
                    # is running.
                    copy_item["np_arr"] = np.ascontiguousarray(
                        item["np_arr"]).copy()
                    copy_item["ptr"] = int(copy_item["np_arr"].ctypes.data)
                frozen.append(copy_item)
        except BaseException:
            for ptr in allocated:
                acl_lib.aclrtFree(ptr)
            raise
        return frozen

    def _release_snapshot(self, params):
        if acl_lib is None:
            return
        for item in params or []:
            ptr = item.get("snapshot_dev_ptr")
            if ptr is not None and ptr.value:
                acl_lib.aclrtFree(ptr)
                item["snapshot_dev_ptr"] = ctypes.c_void_p()

    # -- I/O synchronisation -------------------------------------------------

    def wait_for_io_completion(self, timeout=None):
        thread = getattr(self, 'io_thread', None)
        if thread is not None:
            t_wait_start = time.perf_counter()
            if thread.is_alive():
                print(f"[Timeline][Rank {self.rank_id}] I/O Barrier: Waiting for "
                      f"background SPDK flush to finish...", flush=True)
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise TimeoutError("background checkpoint I/O timed out")
            self.io_thread = None
            t_wait_end = time.perf_counter()
            print(f"[Timeline][Rank {self.rank_id}] I/O Barrier Cleared! "
                  f"Wait time: {(t_wait_end - t_wait_start):.3f}s", flush=True)

        if self._io_error is not None:
            error = self._io_error
            self._io_error = None
            raise RuntimeError("Background checkpoint persistence failed") from error

    # DEPRECATED: kept as no-op for backward compatibility.
    def wait_async_io(self):
        pass

    # -- Layout --------------------------------------------------------------

    def build_layout(self, models, step: int = 0):
        if not isinstance(models, (list, tuple)):
            models = [models]

        params = self._prepare_params(models)
        base_offset_bytes = self._get_current_slot_base_offset(step)
        current_offset = base_offset_bytes

        chunks = []
        for p in params:
            remaining = p["size"]
            inner_off = 0
            nvme_offset_bytes = current_offset

            while remaining > 0:
                take = min(remaining, self.chunk_size)
                aligned_take = int(math.ceil(take / 4096.0)) * 4096

                if nvme_offset_bytes + aligned_take > self.total_bytes:
                    raise MemoryError(
                        "CRITICAL: DMA write exceeds disk capacity "
                        "during build_layout.")

                if (nvme_offset_bytes - base_offset_bytes) + aligned_take > self.slot_bytes:
                    raise MemoryError(
                        f"OOM: Tensor {p['name']} exceeds slot size "
                        f"during build_layout.")

                chunks.append({
                    "name": p["name"],
                    "param_ref": p.get("param_ref"),
                    "npu_ptr": p["ptr"] + inner_off,
                    "base_ptr": p["ptr"],
                    "inner_off": inner_off,
                    "size": take,
                    "nvme_offset": nvme_offset_bytes,
                })

                remaining -= take
                inner_off += take
                nvme_offset_bytes += aligned_take

            current_offset = nvme_offset_bytes

        self.chunks = chunks
        return chunks

    def register_tasks(self, model: ms.nn.Cell, step: int = 0):
        print(f"[DirectCkpt] Rank {self.rank_id} Registering Layout to "
              f"NPU-NVMe Background Thread...")

        if not hasattr(self, 'chunks') or len(self.chunks) == 0:
            self.build_layout(model, step=step)

        num_items = len(self.chunks)
        if num_items == 0:
            print("[DirectCkpt] Warning: No chunks to register!")
            return

        npu_ptrs = (ctypes.c_void_p * num_items)()
        nvme_offsets = (ctypes.c_uint64 * num_items)()
        sizes = (ctypes.c_size_t * num_items)()

        for i, chunk in enumerate(self.chunks):
            param = chunk.get("param_ref")
            if param is None:
                raise ValueError(
                    f"[DirectCkpt] Chunk {i} is missing 'param_ref'.")

            ptr = chunk.get("npu_ptr", 0)
            if ptr == 0:
                base = get_dev_ptr(param)
                inner = chunk.get("inner_off", 0)
                ptr = base + inner if base else 0

            if ptr == 0 or ptr is None:
                raise RuntimeError(
                    f"\n[Fatal Error] Parameter '{param.name}' has an invalid "
                    f"physical address (0x0).\n"
                    f"-> Reason: MindSpore uses lazy memory allocation.\n"
                    f"-> Solution: You MUST call register_tasks() AFTER "
                    f"model.build() or after the first forward pass.")

            npu_ptrs[i] = ptr
            nvme_offsets[i] = chunk["nvme_offset"]
            sizes[i] = chunk["size"]

        rc = lib.npu_nvme_register_tasks(
            self.ctx, npu_ptrs, nvme_offsets, sizes, num_items)
        if rc != 0:
            raise RuntimeError(
                f"[DirectCkpt] Failed to register tasks! C API returned {rc}")

        print(f"[DirectCkpt] Rank {self.rank_id} Successfully registered "
              f"{num_items} tensor pointers to SPDK background thread.")

    # -- Delta-buffer layout (for DeltaTrainCell) --------------------------------

    def build_layout_for_delta(self, delta_cell):
        """Build NVMe layout for DeltaTrainCell output buffers.

        Maps the delta_quant_buf, delta_scale_buf, and delta_idx_buf
        HBM Parameters to NVMe byte offsets within the delta ring area.
        Populates ``self.chunks`` for subsequent FaF registration.

        Args:
            delta_cell: a DeltaTrainCell instance (must be compiled).

        Returns:
            list[dict] — chunk descriptors suitable for C-layer registration.
        """
        if not hasattr(self, '_delta_slot_count'):
            self.delta_init()

        from delta_cell import DeltaTrainCell
        if not isinstance(delta_cell, DeltaTrainCell):
            raise TypeError("build_layout_for_delta expects a DeltaTrainCell")

        # Store the delta block size for correct recovery later.
        # Must match the block_size used by DeltaTrainCell (default 524288).
        self.delta_block_size = delta_cell.bs

        slot_offset = (lib.npu_nvme_delta_get_area_offset(self.ctx)
                       + self._delta_next_slot % self._delta_slot_count
                       * self._delta_slot_size)

        # Use get_dev_ptr for HBM addresses
        quant_ptr = get_dev_ptr(delta_cell.delta_quant_buf)
        scale_ptr = get_dev_ptr(delta_cell.delta_scale_buf)
        idx_ptr = get_dev_ptr(delta_cell.delta_idx_buf)
        if quant_ptr == 0 or scale_ptr == 0 or idx_ptr == 0:
            raise RuntimeError(
                "DeltaTrainCell buffers have null device pointers. "
                "Ensure the cell has been compiled (one forward pass) "
                "before calling build_layout_for_delta.")

        k = delta_cell.k
        bs = delta_cell.bs
        quant_bytes = int(k * bs * np.dtype(np.int8).itemsize)
        scale_bytes = int(k * np.dtype(np.float32).itemsize)
        idx_bytes = int(k * np.dtype(np.int32).itemsize)

        buffers = [
            {'name': 'delta_quant_buf', 'npu_ptr': quant_ptr,
             'nvme_offset': slot_offset,
             'size': quant_bytes, 'param_ref': delta_cell.delta_quant_buf},
            {'name': 'delta_scale_buf', 'npu_ptr': scale_ptr,
             'nvme_offset': slot_offset + int(math.ceil(quant_bytes / 4096.0)) * 4096,
             'size': scale_bytes, 'param_ref': delta_cell.delta_scale_buf},
            {'name': 'delta_idx_buf', 'npu_ptr': idx_ptr,
             'nvme_offset': (slot_offset
                              + int(math.ceil(quant_bytes / 4096.0)) * 4096
                              + int(math.ceil(scale_bytes / 4096.0)) * 4096),
             'size': idx_bytes, 'param_ref': delta_cell.delta_idx_buf},
        ]

        chunks = []
        for buf in buffers:
            remaining = buf['size']
            inner_off = 0
            nvme_offset = buf['nvme_offset']
            while remaining > 0:
                take = min(remaining, self.chunk_size)
                chunks.append({
                    **buf,
                    'name': f"{buf['name']}@{inner_off}",
                    'npu_ptr': buf['npu_ptr'] + inner_off,
                    'nvme_offset': nvme_offset,
                    'size': take,
                })
                remaining -= take
                inner_off += take
                nvme_offset += int(math.ceil(take / 4096.0)) * 4096

        slot_end = max(
            ch['nvme_offset'] + int(math.ceil(ch['size'] / 4096.0)) * 4096
            for ch in chunks)
        if slot_end > slot_offset + self._delta_slot_size:
            raise MemoryError(
                f"Delta buffers require {slot_end - slot_offset} bytes, "
                f"exceeding slot size {self._delta_slot_size}")

        self.chunks = chunks
        return chunks

    def register_delta_tasks(self, delta_cell, ckpt_interval: int = 5):
        """Register DeltaTrainCell output buffers with the FaF listener.

        Also wires the step_counter to the C-layer poller and initialises
        the delta ring area if not already done.

        Args:
            delta_cell:  a compiled DeltaTrainCell.
            ckpt_interval: trigger delta write every N steps.

        Returns:
            (dev_flag: int, dev_step: int) — HBM addresses.
        """
        if not hasattr(self, 'chunks') or len(self.chunks) == 0:
            self.build_layout_for_delta(delta_cell)

        num_items = len(self.chunks)
        npu_ptrs = (ctypes.c_void_p * num_items)()
        nvme_offsets = (ctypes.c_uint64 * num_items)()
        sizes = (ctypes.c_size_t * num_items)()

        for i, ch in enumerate(self.chunks):
            npu_ptrs[i] = ch['npu_ptr']
            nvme_offsets[i] = ch['nvme_offset']
            sizes[i] = ch['size']

        rc = lib.npu_nvme_register_tasks(
            self.ctx, npu_ptrs, nvme_offsets, sizes, num_items)
        if rc != 0:
            raise RuntimeError(
                f"register_delta_tasks: C API returned {rc}")

        # Wire step counter
        dev_step = get_dev_ptr(delta_cell.step_counter)
        rc = lib.npu_nvme_set_step_ptr(
            self.ctx, ctypes.c_void_p(dev_step), ckpt_interval)
        if rc != 0:
            raise RuntimeError(f"set_step_ptr failed: {rc}")

        # Wire probe flag
        dev_flag = get_dev_ptr(getattr(delta_cell, 'flag',
                                        delta_cell.step_counter))
        if hasattr(delta_cell, 'flag'):
            dev_flag = get_dev_ptr(delta_cell.flag)
            lib.npu_nvme_set_probe_flag_ptr(self.ctx, ctypes.c_void_p(dev_flag))
        else:
            lib.npu_nvme_set_probe_flag_ptr(self.ctx, ctypes.c_void_p(0))

        if dev_flag == 0 and hasattr(lib, "npu_nvme_get_probe_flag_dev_ptr"):
            dev_flag = lib.npu_nvme_get_probe_flag_dev_ptr(self.ctx)
        self.probe_flag_ptr = dev_flag

        print(f"[DirectCkpt] Rank {self.rank_id} Registered {num_items} delta "
              f"buffers to FaF listener. step_counter={hex(dev_step)} "
              f"flag={hex(dev_flag)} interval={ckpt_interval}",
              flush=True)
        return dev_flag, dev_step

    # -- Save / Load ---------------------------------------------------------

    def save(self, model: ms.nn.Cell, step: int,
             meta_path: str = "checkpoint_meta.pkl", commit_meta: bool = True,
             _prepared_params=None, _checkpoint_meta=None):
        # A previous generation owns its snapshot buffers until it reaches a
        # terminal state.  Never overwrite/reuse those buffers implicitly.
        self.wait_for_io_completion()
        t_start = time.perf_counter()

        # -- T_Prep --
        t_prep_start = time.perf_counter()
        params = (_prepared_params if _prepared_params is not None
                  else self._prepare_params(model))
        t_prep_end = time.perf_counter()
        T_Prep = t_prep_end - t_prep_start

        self._request_counter += 1
        self._snapshot_generation += 1
        request_id = self._request_counter
        generation = self._snapshot_generation
        # Freeze the graph before copying any parameter address.  A D2D copy
        # submitted while the optimizer is still running would otherwise
        # produce a mixed-step checkpoint.
        if hasattr(ms.hal, "synchronize"):
            ms.hal.synchronize()
        elif hasattr(ms, "runtime") and hasattr(ms.runtime, "synchronize"):
            ms.runtime.synchronize()
        elif acl_lib is not None and hasattr(acl_lib, "aclrtSynchronizeStream"):
            acl_lib.aclrtSynchronizeStream(None)
        params = self._snapshot_params(params, generation)
        handle = CheckpointHandle(self, request_id, generation, step)

        # -- T_Layout --
        base_offset_bytes = self._get_current_slot_base_offset(step)
        print(f"[DirectCkpt] Rank {self.rank_id} saving step {step} to offset "
              f"{base_offset_bytes / 1024**3:.2f} GB ...", flush=True)

        current_offset = base_offset_bytes
        layout, dev_params, host_params = [], [], []

        for p in params:
            aligned_bytes = int(math.ceil(p["size"] / 4096.0)) * 4096

            if current_offset + aligned_bytes > self.total_bytes:
                raise MemoryError(
                    f"Rank {self.rank_id} CRITICAL: DMA Write will exceed disk "
                    f"physical capacity! Offset: "
                    f"{(current_offset + aligned_bytes)/1024**3:.2f}GB > "
                    f"Total: {self.total_bytes/1024**3:.2f}GB")

            if (current_offset - base_offset_bytes) + aligned_bytes > self.slot_bytes:
                raise MemoryError(
                    f"Rank {self.rank_id} OOM! Tensor {p['name']} exceeds slot size.")

            p_record = {**p, "offset": current_offset}
            layout.append(p_record)

            if p.get("np_arr") is not None:
                host_params.append(p_record)
            else:
                dev_params.append(p_record)

            current_offset += aligned_bytes

        total_written = 0

        dev_chunks, dev_sz = build_chunks(dev_params, self.chunk_size)
        if dev_chunks:
            with open(f"task_mapping_rank_{self.rank_id}.txt", "w") as f:
                for i, chunk in enumerate(dev_chunks):
                    f.write(f"TaskIdx: {i} | Name: {chunk[3]} | "
                            f"Size: {chunk[2].value}\n")
            c_ptrs_dev, c_offs_dev, c_sizes_dev = build_ctypes_arrays(dev_chunks)
        else:
            c_ptrs_dev = c_offs_dev = c_sizes_dev = None

        host_chunks, host_sz = build_chunks(host_params, self.chunk_size)
        if host_chunks:
            c_ptrs_host, c_offs_host, c_sizes_host = build_ctypes_arrays(host_chunks)
        else:
            c_ptrs_host = c_offs_host = c_sizes_host = None

        t_layout_end = time.perf_counter()
        T_Layout = t_layout_end - t_prep_end

        def background_io_worker(c_ptrs_d, c_offs_d, c_sizes_d, n_dev, d_sz,
                                 c_ptrs_h, c_offs_h, c_sizes_h, n_host, h_sz):
            try:
                t_spdk_start = time.perf_counter()
                total_written = 0

                if n_dev > 0:
                    rc = lib.npu_nvme_write_batch(
                        self.ctx, c_ptrs_d, c_offs_d, c_sizes_d, n_dev)
                    if rc != 0:
                        raise RuntimeError(f"write_batch failed (rc={rc})")
                    total_written += d_sz

                if n_host > 0:
                    if not hasattr(lib, "npu_nvme_write_batch_host"):
                        raise RuntimeError(
                            "C library missing npu_nvme_write_batch_host")
                    rc = lib.npu_nvme_write_batch_host(
                        self.ctx, c_ptrs_h, c_offs_h, c_sizes_h, n_host)
                    if rc != 0:
                        raise RuntimeError(
                            f"write_batch_host failed (rc={rc})")
                    total_written += h_sz

                t_spdk_end = time.perf_counter()
                T_SPDK = t_spdk_end - t_spdk_start

                t_meta_start = time.perf_counter()
                self.last_layout = layout
                if commit_meta:
                    self._commit_metadata(
                        step, layout, checkpoint_meta=_checkpoint_meta)

                with open(meta_path, "wb") as f:
                    pickle.dump(self.meta_dict, f)

                t_meta_end = time.perf_counter()
                T_Meta = t_meta_end - t_meta_start

                real_time = time.perf_counter() - t_start
                bw = (total_written / 1024 / 1024 / real_time
                      if real_time > 0 else 0)

                print(f"\n{'='*54}")
                print(f"[Timeline][Rank {self.rank_id}] Step {step} | "
                      f"Background SPDK Flush ENDED at {time.perf_counter():.3f}s")
                print(f"[Breakdown][Rank {self.rank_id}] "
                      f"Prep: {T_Prep*1000:.2f}ms | Layout: {T_Layout*1000:.2f}ms | "
                      f"SPDK(H/W): {T_SPDK*1000:.2f}ms | Meta: {T_Meta*1000:.2f}ms")
                print(f"[DirectCkpt][Rank {self.rank_id}] Background Safe Write: "
                      f"{total_written/1024/1024:.2f} MB | BW: {bw:.2f} MB/s")
                print(f"{'='*54}\n", flush=True)
                handle._complete()
            except BaseException as error:
                self._io_error = error
                handle._fail(error)
                print(f"[Fatal][Rank {self.rank_id}] Background checkpoint "
                      f"failed: {error}", flush=True)
            finally:
                self._release_snapshot(params)

        num_dev_val = len(dev_chunks) if dev_chunks else 0
        dev_sz_val = dev_sz if dev_chunks else 0
        num_host_val = len(host_chunks) if host_chunks else 0
        host_sz_val = host_sz if host_chunks else 0

        self._io_error = None
        self._active_handle = handle
        self._last_chunk_count = num_dev_val + num_host_val
        self._last_save_stats = {"prep_time": T_Prep,
                                 "layout_time": T_Layout,
                                 "request_id": request_id,
                                 "generation": generation}

        self.io_thread = threading.Thread(
            target=background_io_worker,
            args=(c_ptrs_dev, c_offs_dev, c_sizes_dev, num_dev_val, dev_sz_val,
                  c_ptrs_host, c_offs_host, c_sizes_host, num_host_val, host_sz_val))
        self.io_thread.start()

        t_return = time.perf_counter()
        print(f"[Timeline][Rank {self.rank_id}] Step {step} | Python save() "
              f"dispatched to background thread. "
              f"Layout cost: {T_Layout*1000:.2f}ms", flush=True)

        return handle

    def _ordered_components(self, components):
        preferred = [name for name in ("model", "optimizer")
                     if name in components]
        preferred.extend(sorted(name for name in components
                                if name not in {"model", "optimizer"}))
        return preferred

    def _prepare_state_components(self, components, with_checksums=True):
        """Build namespaced descriptors, excluding aliased model weights.

        MindSpore optimizers commonly expose the model weights alongside
        moment slots.  Object-identity de-duplication keeps those weights in
        ``model/`` and stores only optimizer-owned state in ``optimizer/``.
        """
        params = []
        seen_objects = set()
        seen_names = set()
        for component in self._ordered_components(components):
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

                dtype_np = np.dtype(ms.dtype_to_nptype(parameter.dtype))
                local_shape = tuple(parameter.shape)
                if hasattr(parameter, "sliced_shape") and parameter.sliced_shape:
                    local_shape = tuple(parameter.sliced_shape)
                elif hasattr(parameter, "data") and hasattr(parameter.data, "shape"):
                    data_shape = tuple(parameter.data.shape)
                    if np.prod(data_shape) < np.prod(local_shape):
                        local_shape = data_shape
                if int(np.prod(local_shape)) == 0:
                    continue

                ptr = get_dev_ptr(parameter)
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
                        np.ascontiguousarray(checksum_arr).tobytes()).hexdigest()
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

    def save_state(self, components, control_state, step: int,
                   meta_path: str = "checkpoint_meta.pkl",
                   commit_meta: bool = True, verify_checksums: bool = True):
        """Freeze and persist a versioned complete training state.

        ``components`` maps namespaces such as ``model`` and ``optimizer`` to
        MindSpore objects exposing ``parameters_and_names``.  ``control_state``
        contains JSON-tagged Python/NumPy state and is returned by
        :meth:`load_state` for the caller to re-apply.
        """
        validate_state_names(components, control_state)
        if hasattr(ms.hal, "synchronize"):
            ms.hal.synchronize()
        params = self._prepare_state_components(
            components, with_checksums=verify_checksums)
        for name in sorted(control_state):
            payload, control_meta = encode_control_value(control_state[name])
            params.append({
                "name": f"control/{name}",
                "source_name": name,
                "component": "control",
                "category": "control",
                "placement": "host",
                "ptr": 0,
                "size": int(payload.nbytes),
                "shape": [int(payload.nbytes)],
                "dtype": "uint8",
                "np_arr": payload,
                "param_ref": None,
                "codec": control_meta["codec"],
                "sha256": control_meta["sha256"],
            })
        checkpoint_meta = {
            "type": "TRAINING_STATE_FULL",
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "state_step": int(step),
            "checksum": "sha256" if verify_checksums else "none",
            "components": self._ordered_components(components),
            "control_names": sorted(control_state),
        }
        return self.save(
            None, step=step, meta_path=meta_path, commit_meta=commit_meta,
            _prepared_params=params, _checkpoint_meta=checkpoint_meta)

    def _select_checkpoint_record(self, step):
        self._mount_filesystem()
        if step is None:
            candidates = []
            for key, record in self.meta_dict.get("checkpoints", {}).items():
                if not key.startswith("step_"):
                    continue
                try:
                    candidates.append((int(key.split("_", 1)[1]), record))
                except ValueError:
                    continue
            if not candidates:
                raise FileNotFoundError("No checkpoints found in metadata")
            return max(candidates, key=lambda item: item[0])
        key = f"step_{int(step)}"
        try:
            return int(step), self.meta_dict["checkpoints"][key]
        except KeyError as error:
            raise FileNotFoundError(f"Checkpoint for {key} not found") from error

    def load_state(self, components, step: int = None,
                   verify_checksums: bool = True):
        """Restore namespaced parameters and return decoded control state."""
        validate_state_names(components, {})
        selected_step, record = self._select_checkpoint_record(step)
        if record.get("type") == "MULTI_TRAINING_STATE_FULL":
            rank_record = record.get("ranks", {}).get(str(self.rank_id))
            if rank_record is None:
                raise ValueError(
                    f"checkpoint has no shard for rank {self.rank_id}")
            # Coordinator metadata carries the global commit fields at the
            # outer level and a normal TRAINING_STATE_FULL-shaped manifest in
            # each rank record.  Keep all subsequent validation and DMA code
            # shared with the single-rank path.
            record = {
                **rank_record,
                "type": "TRAINING_STATE_FULL",
                "schema_version": record.get("schema_version"),
                "state_step": record.get("state_step"),
                "chunk_size": record.get("chunk_size", self.chunk_size),
            }
        if record.get("type") != "TRAINING_STATE_FULL":
            raise ValueError(
                f"step {selected_step} is not a complete training-state checkpoint")
        if record.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported training-state schema version")
        if record.get("state_step") != selected_step:
            raise ValueError("training-state step does not match metadata key")
        if record.get("components") != self._ordered_components(components):
            raise ValueError("training-state component manifest mismatch")

        saved = record.get("params", {})
        saved_control_names = {
            name.split("/", 1)[1] for name in saved
            if name.startswith("control/")
        }
        if saved_control_names != set(record.get("control_names", [])):
            raise ValueError("training-state control manifest mismatch")
        target_items = self._prepare_state_components(
            components, with_checksums=False)
        targets = {item["name"]: item for item in target_items}
        saved_parameter_names = {
            name for name in saved
            if not name.startswith("control/")
        }
        if set(targets) != saved_parameter_names:
            missing = sorted(saved_parameter_names - set(targets))
            extra = sorted(set(targets) - saved_parameter_names)
            raise ValueError(
                f"training-state parameter set mismatch: missing={missing[:3]} "
                f"extra={extra[:3]}")

        dev_buffers, host_buffers, control_buffers = [], [], {}
        for name, info in saved.items():
            if name.startswith("control/"):
                if info.get("dtype") != "uint8" or info.get("size", 0) <= 0:
                    raise ValueError(f"invalid control-state record: {name}")
                array = np.empty(int(info["size"]), dtype=np.uint8)
                item = {"name": name, "ptr": int(array.ctypes.data),
                        "size": int(info["size"]), "offset": int(info["offset"])}
                host_buffers.append(item)
                control_buffers[name] = (array, info)
                continue

            target = targets[name]
            saved_shape = list(info.get("shape", []))
            target_shape = list(target["shape"])
            # MindSpore may materialize scalar optimizer hyperparameters as
            # either [] or [1] in fresh processes.  They are byte-compatible
            # when dtype and payload size agree; keep strict shape checks for
            # all non-singleton tensors.
            singleton_shape = (
                int(info.get("size", -1)) == int(target["size"]) and
                np.prod(saved_shape or [1]) == 1 and
                np.prod(target_shape or [1]) == 1)
            if ((saved_shape != target_shape and not singleton_shape) or
                    np.dtype(info.get("dtype")) != np.dtype(target["dtype"]) or
                    int(info.get("size", -1)) != int(target["size"])):
                raise ValueError(f"shape/dtype/size mismatch for {name}")
            item = dict(target)
            item.update(offset=int(info["offset"]), size=int(info["size"]))
            if target["ptr"]:
                dev_buffers.append(item)
            else:
                array = np.empty(target["shape"], dtype=np.dtype(target["dtype"]))
                item["np_arr"] = array
                item["ptr"] = int(array.ctypes.data)
                host_buffers.append(item)

        chunk_size = min(int(record.get("chunk_size", self.chunk_size)),
                         self.chunk_size)
        dev_chunks, _ = build_chunks(dev_buffers, chunk_size)
        host_chunks, _ = build_chunks(host_buffers, chunk_size)
        if dev_chunks:
            arrays = build_ctypes_arrays(dev_chunks)
            rc = lib.npu_nvme_read_batch(self.ctx, *arrays, len(dev_chunks))
            if rc != 0:
                raise RuntimeError(f"training-state device read failed: {rc}")
        if host_chunks:
            arrays = build_ctypes_arrays(host_chunks)
            rc = lib.npu_nvme_read_batch_host(
                self.ctx, *arrays, len(host_chunks))
            if rc != 0:
                raise RuntimeError(f"training-state host read failed: {rc}")

        for item in host_buffers:
            if item["name"].startswith("control/"):
                continue
            parameter = item["param_ref"]
            ops.assign(parameter, ms.Tensor(item["np_arr"], dtype=parameter.dtype))
        if hasattr(ms.hal, "synchronize"):
            ms.hal.synchronize()

        if verify_checksums and record.get("checksum") == "sha256":
            for name, target in targets.items():
                expected = saved[name].get("sha256")
                if not expected:
                    raise ValueError(f"missing checksum for {name}")
                actual = hashlib.sha256(np.ascontiguousarray(
                    target["param_ref"].value().asnumpy()).tobytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"parameter checksum mismatch for {name}")

        controls = {}
        for qualified_name, (array, info) in control_buffers.items():
            source_name = qualified_name.split("/", 1)[1]
            if not source_name or qualified_name != f"control/{source_name}":
                raise ValueError(f"invalid control-state namespace: {qualified_name}")
            controls[source_name] = decode_control_value(array, info)
        return controls

    def load(self, model: ms.nn.Cell, step: int = None,
             meta_path: str = "checkpoint_meta.pkl"):
        t_start = time.time()

        # NVMe metadata is the source of truth.  The pickle is retained only
        # as a diagnostic sidecar and must never make a restarted process
        # appear to have a checkpoint that was not persisted to the device.
        self._mount_filesystem()

        if step is not None:
            ckpt_key = f"step_{step}"
        else:
            valid_keys = [k for k in self.meta_dict.get("checkpoints", {}).keys()
                          if "complete" not in k]
            if not valid_keys:
                raise FileNotFoundError(
                    "No checkpoints found in Meta Dictionary!")
            ckpt_key = sorted(valid_keys,
                              key=lambda x: int(x.split('_')[1]))[-1]

        if ckpt_key not in self.meta_dict.get("checkpoints", {}):
            raise FileNotFoundError(
                f"Checkpoint for {ckpt_key} not found!")

        meta_info = self.meta_dict["checkpoints"][ckpt_key]
        chunk_size = min(meta_info.get("chunk_size", self.chunk_size),
                         self.chunk_size)

        t_rebuild = time.time()
        dev_chunks, host_chunks, buffers = rebuild_chunks_from_meta(
            model, meta_info["params"], chunk_size)
        t_rebuild_end = time.time()

        t0 = time.time()
        total_read = 0

        # Read NPU-resident parameters via DMA.
        if dev_chunks:
            c_ptrs, c_offs, c_sizes = build_ctypes_arrays(dev_chunks)
            rc = lib.npu_nvme_read_batch(
                self.ctx, c_ptrs, c_offs, c_sizes, len(dev_chunks))
            if rc != 0:
                raise RuntimeError("read_batch failed")
            total_read += sum(c[2].value for c in dev_chunks)

        # Read CPU-resident parameters through the explicit Host path.
        if host_chunks:
            if not hasattr(lib, "npu_nvme_read_batch_host"):
                raise RuntimeError(
                    "C library missing npu_nvme_read_batch_host")
            c_ptrs, c_offs, c_sizes = build_ctypes_arrays(host_chunks)
            rc = lib.npu_nvme_read_batch_host(
                self.ctx, c_ptrs, c_offs, c_sizes, len(host_chunks))
            if rc != 0:
                raise RuntimeError("read_batch_host failed")
            total_read += sum(c[2].value for c in host_chunks)

        t1 = time.time()
        t_update = time.time()

        for buf in buffers:
            if buf.get("use_dev", False):
                continue
            param = buf["param_ref"]
            if buf["np_arr"] is not None:
                tensor = ms.Tensor(buf["np_arr"], dtype=param.dtype)
                ops.assign(param, tensor)
        t_end = time.time()

        pure_read_time = t1 - t0
        total_time = t_end - t_start
        bw_e2e = total_read / 1024 / 1024 / total_time if total_time > 0 else 0

        stats = {
            "prepare_time": t_rebuild_end - t_rebuild,
            "read_time": pure_read_time,
            "set_data_time": t_end - t_update,
            "total_time": total_time,
            "bw_pure": (total_read / 1024 / 1024 / pure_read_time
                       if pure_read_time > 0 else 0),
            "bw_e2e": bw_e2e,
        }
        return total_read, len(dev_chunks) + len(host_chunks), total_time, bw_e2e, stats

    # -- Delta frame I/O --------------------------------------------------------

    def delta_init(self, slot_size_mb: int = 256, slot_count: int = 128):
        slot_bytes = slot_size_mb * 1024 * 1024
        if self.layout is None:
            raise RuntimeError("disk layout is not mounted")
        if (slot_bytes != self.layout.delta_slot_bytes or
                slot_count != self.layout.delta_slot_count):
            raise RuntimeError(
                "requested Delta geometry differs from formatted disk: "
                f"requested {slot_count}x{slot_bytes}, formatted "
                f"{self.layout.delta_slot_count}x{self.layout.delta_slot_bytes}")
        if not hasattr(lib, "npu_nvme_delta_init"):
            raise RuntimeError(
                "C library missing npu_nvme_delta_init — rebuild required.")
        rc = lib.npu_nvme_delta_init(
            self.ctx, ctypes.c_uint64(self.layout.delta_base),
            ctypes.c_uint64(slot_bytes), ctypes.c_uint32(slot_count))
        if rc != 0:
            raise RuntimeError(f"Delta init failed (rc={rc})")
        self._delta_slot_size = slot_bytes
        self._delta_slot_count = slot_count
        self._delta_next_slot = int(self.meta_dict.get("delta_head", 0))
        self._delta_step_map = {}
        self._delta_frame_sizes = []
        if "delta_chain" not in self.meta_dict:
            self.meta_dict["delta_chain"] = {}
        if not hasattr(self, '_dump_meta_pkl'):
            import pickle as _pickle
            os.makedirs(os.path.dirname(self._meta_pkl), exist_ok=True)
            def _dump_meta_pkl():
                with open(self._meta_pkl, "wb") as _f:
                    _pickle.dump(self.meta_dict, _f)
            self._dump_meta_pkl = _dump_meta_pkl
        print(f"[DirectCkpt] Delta area initialized: "
              f"{slot_count} slots x {slot_size_mb}MB", flush=True)

    def delta_save(self, step: int, block_patches: list, small_patches: list,
                   lossless: bool = False, base_generation: int = None):
        if not hasattr(self, '_delta_slot_count'):
            self.delta_init()

        self.wait_for_io_completion()
        self.wait_async_io()

        if lossless:
            if base_generation is None:
                full_steps = []
                for key, value in self.meta_dict.get("checkpoints", {}).items():
                    if key.startswith("step_") and value.get("type", "FULL") == "FULL":
                        try:
                            full_step = int(key.split("_")[1])
                        except (IndexError, ValueError):
                            continue
                        if full_step <= step:
                            full_steps.append((full_step, int(value.get("generation", 0))))
                if not full_steps:
                    raise RuntimeError(
                        "lossless Delta requires a persisted FULL base checkpoint")
                base_generation = max(full_steps)[1]
            frame = pack_lossless_delta_frame(
                step, block_patches, small_patches,
                base_generation=base_generation,
                generation=self.metadata_generation + 1)
            encoding = "fp16"
        else:
            frame = pack_delta_frame(step, block_patches, small_patches)
            encoding = "int8"
        total_bytes = len(frame)

        if total_bytes > self._delta_slot_size:
            raise RuntimeError(
                f"Delta frame {total_bytes} bytes > slot {self._delta_slot_size}")

        slot_idx = self._delta_next_slot % self._delta_slot_count
        slot_offset = self.layout.delta_slot_offset(slot_idx)

        # Use the chunk pipeline (supports >64MB) instead of the deprecated
        # sync_meta_io path via npu_nvme_write_delta.
        frame_buf = ctypes.create_string_buffer(frame, total_bytes)
        chunks, _ = build_chunks_host(
            ctypes.addressof(frame_buf), slot_offset, total_bytes, self.chunk_size)
        c_ptrs, c_offs, c_sizes = build_ctypes_arrays(chunks)

        if not hasattr(lib, "npu_nvme_write_batch_host"):
            raise RuntimeError("C library missing npu_nvme_write_batch_host")
        rc = lib.npu_nvme_write_batch_host(
            self.ctx, c_ptrs, c_offs, c_sizes, len(chunks))
        if rc != 0:
            raise RuntimeError(f"Delta write failed at slot {slot_idx} (rc={rc})")

        # A ring slot has one authoritative frame.  Remove metadata for the
        # frame that is about to be overwritten; retaining it would make a
        # post-restart chain point at a newer frame and fail only much later.
        for old_key, old_record in list(self.meta_dict["delta_chain"].items()):
            if old_record.get("slot") == slot_idx:
                del self.meta_dict["delta_chain"][old_key]

        self._delta_step_map[step] = slot_idx
        self._delta_next_slot += 1
        self._delta_frame_sizes.append(total_bytes)

        self.meta_dict["delta_chain"][f"step_{step}"] = {
            "type": "DELTA",
            "generation": self.metadata_generation + 1,
            "slot": slot_idx,
            "frame_size": total_bytes,
            "n_blocks": len(block_patches),
            "n_small": len(small_patches),
            "encoding": encoding,
        }
        if lossless:
            self.meta_dict["delta_chain"][f"step_{step}"]["base_generation"] = int(base_generation)
        self.meta_dict["delta_head"] = self._delta_next_slot
        self.meta_dict["delta_tail"] = max(
            0, self._delta_next_slot - self._delta_slot_count)
        self._persist_metadata(self.metadata_generation + 1)
        if hasattr(self, '_dump_meta_pkl'):
            self._dump_meta_pkl()

        return slot_idx

    def delta_save_lossless(self, step: int, block_patches: list,
                            small_patches: list, base_generation: int = None):
        """Persist one R0 self-described FP16 Delta frame."""
        return self.delta_save(
            step, block_patches, small_patches, lossless=True,
            base_generation=base_generation)

    def write_host_frame(self, frame: bytes, byte_offset: int):
        """Write one self-described frame through the Host-SPDK path.

        This deliberately does not modify the live metadata ledger.  It is
        the I5 frame-byte loopback primitive; callers commit lineage only
        after the frame has been read back and validated.
        """
        if not isinstance(frame, (bytes, bytearray)) or not frame:
            raise ValueError("frame must be non-empty bytes")
        if byte_offset % BLOCK_SIZE:
            raise ValueError("frame offset must be 4 KiB aligned")
        if byte_offset < 0 or byte_offset + len(frame) > self.total_bytes:
            raise ValueError("frame exceeds NVMe capacity")
        aligned_size = (len(frame) + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        frame_buf = ctypes.create_string_buffer(aligned_size)
        ctypes.memmove(frame_buf, bytes(frame), len(frame))
        chunks, _ = build_chunks_host(ctypes.addressof(frame_buf), byte_offset,
                                      len(frame), self.chunk_size)
        c_ptrs, c_offs, c_sizes = build_ctypes_arrays(chunks)
        rc = lib.npu_nvme_write_batch_host(
            self.ctx, c_ptrs, c_offs, c_sizes, len(chunks))
        if rc != 0:
            raise RuntimeError(f"Host-SPDK frame write failed: {rc}")
        return {"offset": byte_offset, "bytes": len(frame),
                "aligned_bytes": aligned_size, "chunks": len(chunks),
                "c_io_us": int(lib.npu_nvme_get_last_io_us(self.ctx, 0))}

    def read_host_frame(self, byte_offset: int, frame_size: int):
        """Read exactly one previously written self-described frame."""
        if frame_size <= 0 or byte_offset % BLOCK_SIZE:
            raise ValueError("invalid frame offset or size")
        if byte_offset < 0 or byte_offset + frame_size > self.total_bytes:
            raise ValueError("frame exceeds NVMe capacity")
        aligned_size = (frame_size + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        frame_buf = ctypes.create_string_buffer(aligned_size)
        chunks, _ = build_chunks_host(ctypes.addressof(frame_buf), byte_offset,
                                      frame_size, self.chunk_size)
        c_ptrs, c_offs, c_sizes = build_ctypes_arrays(chunks)
        rc = lib.npu_nvme_read_batch_host(
            self.ctx, c_ptrs, c_offs, c_sizes, len(chunks))
        if rc != 0:
            raise RuntimeError(f"Host-SPDK frame read failed: {rc}")
        return bytes(frame_buf.raw[:frame_size])

    def delta_load_slot(self, slot_idx: int, return_meta: bool = False):
        if not hasattr(self, '_delta_slot_size'):
            raise RuntimeError(
                "Delta not initialized. Call delta_init() first.")

        slot_offset = self.layout.delta_slot_offset(slot_idx)

        # Read header to determine actual frame size, then full data.
        header_buf = ctypes.create_string_buffer(FRAME_HEADER_SIZE)
        h_chunks, _ = build_chunks_host(
            ctypes.addressof(header_buf), slot_offset,
            FRAME_HEADER_SIZE, self.chunk_size)
        h_ptrs, h_offs, h_sizes = build_ctypes_arrays(h_chunks)
        if not hasattr(lib, "npu_nvme_read_batch_host"):
            raise RuntimeError("C library missing npu_nvme_read_batch_host")
        rc = lib.npu_nvme_read_batch_host(
            self.ctx, h_ptrs, h_offs, h_sizes, len(h_chunks))
        if rc != 0:
            raise RuntimeError(
                f"Delta header read failed at slot {slot_idx} (rc={rc})")

        magic = struct.unpack_from("<I", header_buf.raw, 0)[0]
        total_sz = struct.unpack_from("<I", header_buf.raw, 16)[0]
        if magic != DELTA_MAGIC:
            raise RuntimeError(
                f"Delta read at slot {slot_idx}: bad magic 0x{magic:08x}")
        if total_sz > self._delta_slot_size:
            raise RuntimeError(
                f"Delta slot {slot_idx}: frame size {total_sz} > "
                f"slot {self._delta_slot_size}")

        data_buf = ctypes.create_string_buffer(total_sz)
        d_chunks, _ = build_chunks_host(
            ctypes.addressof(data_buf), slot_offset,
            total_sz, self.chunk_size)
        d_ptrs, d_offs, d_sizes = build_ctypes_arrays(d_chunks)
        rc = lib.npu_nvme_read_batch_host(
            self.ctx, d_ptrs, d_offs, d_sizes, len(d_chunks))
        if rc != 0:
            raise RuntimeError(
                f"Delta data read failed at slot {slot_idx} (rc={rc})")

        decoded = unpack_delta_frame_with_meta(data_buf.raw[:total_sz])
        if return_meta:
            return decoded
        return decoded[:3]

    def delta_load_chain(self, from_step: int, to_step: int):
        chain = []
        for s in range(from_step + 1, to_step + 1):
            key = f"step_{s}"
            if key not in self.meta_dict.get("delta_chain", {}):
                raise FileNotFoundError(
                    f"Delta frame for step {s} not found in metadata")
            record = self.meta_dict["delta_chain"][key]
            slot = record["slot"]
            sid, blocks, smalls, frame_info = self.delta_load_slot(
                slot, return_meta=True)
            if sid != s:
                raise RuntimeError(
                    f"Delta slot {slot} step_id={sid} != expected {s}")
            expected_encoding = record.get("encoding")
            if expected_encoding == "fp16" and frame_info.get("version") != 2:
                raise RuntimeError(
                    f"Delta step {s} metadata expects FP16 frame, got {frame_info}")
            if expected_encoding == "fp16":
                expected_base = int(record.get("base_generation", -1))
                if frame_info.get("base_generation") != expected_base:
                    raise RuntimeError(
                        f"Delta step {s} base generation mismatch: "
                        f"frame={frame_info.get('base_generation')} metadata={expected_base}")
            chain.append((sid, blocks, smalls))
        return chain

    def _find_nearest_full(self, target_step: int):
        def _scan():
            best = None
            for k, v in self.meta_dict.get("checkpoints", {}).items():
                if not k.startswith("step_"):
                    continue
                if v.get("type", "FULL") != "FULL":
                    continue
                try:
                    s = int(k.split("_")[1])
                except ValueError:
                    continue
                if s <= target_step and (best is None or s > best):
                    best = s
            return best

        best = _scan()
        if best is not None:
            return best

        print("[DirectCkpt] Re-reading meta from NVMe to find FULL checkpoint...",
              flush=True)
        self._mount_filesystem()
        best = _scan()
        if best is not None:
            print(f"[DirectCkpt] Found FULL checkpoint step_{best} "
                  f"after disk re-read.", flush=True)
            return best

        raise FileNotFoundError(
            f"No FULL checkpoint found <= step {target_step}")

    def recover(self, model: "ms.nn.Cell", target_step: int):
        t_start = time.perf_counter()

        base_step = self._find_nearest_full(target_step)
        print(f"[DirectCkpt] Recover target=step_{target_step}, "
              f"base=FULL step_{base_step}", flush=True)

        self.load(model, step=base_step)

        if base_step == target_step:
            dt = time.perf_counter() - t_start
            print(f"[DirectCkpt] Recovery done (FULL only, no deltas): {dt:.2f}s",
                  flush=True)
            return {"base_step": base_step, "n_deltas": 0, "total_time": dt}

        chain = self.delta_load_chain(base_step, target_step)
        print(f"[DirectCkpt] Loaded {len(chain)} delta frames "
              f"(step {base_step + 1}->{target_step})", flush=True)

        param_dtypes = {}
        host_weights = {}
        for name, p in model.parameters_and_names():
            param_dtypes[name] = p.dtype
            host_weights[name] = p.value().asnumpy().copy()

        for sid, blocks, smalls in chain:
            host_weights = apply_delta_patches(
                host_weights, blocks, smalls,
                getattr(self, 'delta_block_size', self.chunk_size))

        for name, p in model.parameters_and_names():
            if name in host_weights:
                ms_dtype = param_dtypes.get(name, ms.float16)
                np_dtype = np.dtype(ms.dtype_to_nptype(ms_dtype))
                p.set_data(Tensor(host_weights[name].astype(np_dtype), ms_dtype))

        dt = time.perf_counter() - t_start
        n_deltas = len(chain)
        print(f"[DirectCkpt] Recovery complete: {n_deltas} deltas applied, "
              f"total {dt:.2f}s", flush=True)
        return {"base_step": base_step, "n_deltas": n_deltas, "total_time": dt}
