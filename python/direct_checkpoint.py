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
  _legacy_compat  — DEPRECATED WaitProbe-era symbols (do NOT use in new code)
"""

import ctypes
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
                          DELTA_MAGIC, FRAME_HEADER_SIZE, BLOCK_SIZE)
from chunk_helpers import (build_chunks, build_chunks_host,
                            build_ctypes_arrays, rebuild_chunks_from_meta)
from delta_protocol import (pack_delta_frame, unpack_delta_frame,
                             apply_delta_patches, FileDeltaWriter)
from noop_init import NoOpInitializer, replace_with_noop_initializer
from training_cell import ProbeTrainOneStepCell

# DEPRECATED — only for backward compatibility with legacy scripts
from _legacy_compat import bind_depend_op, wait_op_info, trigger_op_info


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


# -- DirectCheckpoint: NVMe-backed training checkpoint manager ----------------

class DirectCheckpoint:
    _ms_warmed_up = False

    def __init__(
        self, nvme_addr: str = "0000:83:00.0", npu_device_id: int = 0,
        pipeline_depth: int = 4, requested_chunk_size: int = 4 * 1024 * 1024,
        enable_profiling: bool = False, profiling_dir: str = "./output/profiling",
        rank_id: int = 0, world_size: int = 1,
        base_offset_bytes: int = 0, shard_span_bytes: int = None,
        spdk_shm_id: int = 1, keep_last_n: int = 3, slot_size_gb: int = 50,
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
        atexit.register(self.close)

        self._mount_filesystem()

        self.io_thread = None

    # -- Filesystem mount ----------------------------------------------------

    def _mount_filesystem(self):
        sb_buf = ctypes.create_string_buffer(4096)
        rc = lib.npu_nvme_sync_meta_io(
            self.ctx, SUPERBLOCK_OFFSET, 4096, 1,
            ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc != 0:
            raise RuntimeError("Failed to read Superblock.")

        header = struct.unpack("<8s I Q Q", sb_buf.raw[:SUPERBLOCK_HEADER_BYTES])
        magic = header[0]

        if magic != MAGIC_NUMBER:
            raise RuntimeError(
                f"Superblock verification failed: unexpected header "
                f"0x{magic.hex()} (expected NPUNVME1). "
                f"Run format_npu_disk.py to initialize the disk.")

        self.active_meta_slot = header[1]
        self.stack_start_bytes = header[3]

        total_stack_bytes = self.world_size * self.keep_last_n * self.slot_bytes
        if self.stack_start_bytes == 0 or (
            self.stack_start_bytes + total_stack_bytes > self.total_bytes):
            print(f"[Warning] Rank {self.rank_id}: Stale Superblock! "
                  f"Current config needs {total_stack_bytes/1024**3:.2f} GB, "
                  f"but disk bounds exceeded. Recalculating...", flush=True)
            self.stack_start_bytes = self.total_bytes - total_stack_bytes
            print(f"[Info] Stack dynamically re-allocated at disk end. "
                  f"Start Offset: {self.stack_start_bytes / 1024**3:.2f} GB")

        target_offset = (META_SLOT_A_OFFSET if self.active_meta_slot == 0
                         else META_SLOT_B_OFFSET)
        meta_buf = ctypes.create_string_buffer(META_SLOT_BYTES)
        lib.npu_nvme_sync_meta_io(self.ctx, target_offset, META_SLOT_BYTES,
                                   1, ctypes.c_void_p(ctypes.addressof(meta_buf)))

        meta_str = meta_buf.value.decode('utf-8', errors='ignore').rstrip('\x00')
        if meta_str:
            try:
                self.meta_dict = json.loads(meta_str)
            except json.JSONDecodeError:
                self.meta_dict = {"checkpoints": {}}
                print("[Warning] Parsed metadata JSON was invalid. Starting fresh.")

        print(f"[Rank {self.rank_id}] FileSystem Mounted. "
              f"Active Slot: {'A' if self.active_meta_slot == 0 else 'B'}")

    # -- Slot layout ---------------------------------------------------------

    def _get_current_slot_base_offset(self, step: int):
        slot_idx = step % self.keep_last_n
        rank_offset = self.rank_id * self.keep_last_n * self.slot_bytes
        slot_offset = slot_idx * self.slot_bytes
        return self.stack_start_bytes + rank_offset + slot_offset

    # -- Metadata commit -----------------------------------------------------

    def _commit_metadata(self, step: int, layout: List[Dict]):
        if self.rank_id != 0:
            return

        ckpt_key = f"step_{step}"
        self.meta_dict["checkpoints"][ckpt_key] = {
            "type": "FULL",
            "chunk_size": self.chunk_size,
            "rank_id": self.rank_id,
            "world_size": self.world_size,
            "params": {p["name"]: {
                "offset": p["offset"], "size": p["size"],
                "shape": p["shape"], "dtype": p["dtype"],
            } for p in layout},
        }

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
            if ds < saved_steps[0]:
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

        next_slot = 1 if self.active_meta_slot == 0 else 0
        target_offset = META_SLOT_B_OFFSET if next_slot == 1 else META_SLOT_A_OFFSET

        meta_json = json.dumps(self.meta_dict).encode('utf-8')
        if len(meta_json) > META_SLOT_BYTES:
            raise RuntimeError(
                f"Metadata JSON exceeds allocated {META_SLOT_BYTES} bytes!")

        meta_buf = ctypes.create_string_buffer(meta_json, META_SLOT_BYTES)
        rc1 = lib.npu_nvme_sync_meta_io(
            self.ctx, target_offset, META_SLOT_BYTES, 0,
            ctypes.c_void_p(ctypes.addressof(meta_buf)))
        if rc1 != 0:
            raise RuntimeError(
                f"Fatal: Meta JSON write failed! C layer returned {rc1}.")

        sb_buf = ctypes.create_string_buffer(4096)
        struct.pack_into("<8s I Q Q", sb_buf, 0,
                         MAGIC_NUMBER, next_slot, self.total_bytes,
                         self.stack_start_bytes)
        rc2 = lib.npu_nvme_sync_meta_io(
            self.ctx, SUPERBLOCK_OFFSET, 4096, 0,
            ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc2 != 0:
            raise RuntimeError("Fatal: Superblock write failed!")

        self.active_meta_slot = next_slot
        print(f"[DirectCkpt] Rank 0 Meta committed safely to "
              f"Slot {'B' if next_slot == 1 else 'A'} (Superblock updated).",
              flush=True)

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

        host_buf = ctypes.create_string_buffer(UINT32_BYTES)
        host_buf.raw = int(value).to_bytes(UINT32_BYTES,
                                           byteorder="little", signed=False)
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
        self.wait_for_io_completion()
        if getattr(self, '_spdk_initialized', False) and self.ctx:
            lib.npu_nvme_cleanup(self.ctx)
            self.ctx = None
            self._spdk_initialized = False

    def close(self):
        if not getattr(self, '_closed', False) and hasattr(self, 'ctx') and self.ctx:
            print(f"[DirectCkpt] Rank {self.rank_id} safely tearing down "
                  f"NPUNVME context...", flush=True)
            self.cleanup()
            self._closed = True

    def __del__(self):
        self.close()

    def _build_local_param_registry(self, models):
        if not isinstance(models, (list, tuple)):
            models = [models]
        self.local_valid_param_names = set()

        print(f"[DirectCkpt] Rank {self.rank_id} running dynamic hardware "
              f"memory probe...", flush=True)

        dummy_dst = ctypes.create_string_buffer(1)

        for model in models:
            if model is None or not hasattr(model, "parameters_and_names"):
                continue
            for name, p in model.parameters_and_names():
                ptr = get_dev_ptr(p)
                if ptr == 0:
                    continue

                is_valid_on_this_rank = False

                if acl_lib is not None:
                    ret = acl_lib.aclrtMemcpy(
                        ctypes.byref(dummy_dst), 1,
                        ctypes.c_void_p(ptr), 1, 2)
                    if ret == 0:
                        is_valid_on_this_rank = True

                if not is_valid_on_this_rank:
                    if ("step" in name.lower() or "scale" in name.lower()
                            or np.prod(p.shape) <= 8):
                        try:
                            _ = p.asnumpy()
                            is_valid_on_this_rank = True
                        except Exception:
                            pass

                if is_valid_on_this_rank:
                    self.local_valid_param_names.add(name)

        print(f"[DirectCkpt] Rank {self.rank_id} registry dynamically built: "
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
                if ptr == 0:
                    continue

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

                params.append({
                    "name": name, "ptr": ptr, "size": size,
                    "shape": list(p.shape), "dtype": dtype_np.name,
                    "np_arr": None,
                    "param_ref": p,
                })
        return params

    # -- I/O synchronisation -------------------------------------------------

    def wait_for_io_completion(self):
        if getattr(self, 'io_thread', None) is not None and self.io_thread.is_alive():
            t_wait_start = time.perf_counter()
            print(f"[Timeline][Rank {self.rank_id}] I/O Barrier: Waiting for "
                  f"background SPDK flush to finish...", flush=True)
            self.io_thread.join()
            t_wait_end = time.perf_counter()
            print(f"[Timeline][Rank {self.rank_id}] I/O Barrier Cleared! "
                  f"Wait time: {(t_wait_end - t_wait_start):.3f}s", flush=True)
            self.io_thread = None

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

        chunks = [
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
             meta_path: str = "checkpoint_meta.pkl", commit_meta: bool = True):
        t_start = time.perf_counter()

        # -- T_Prep --
        t_prep_start = time.perf_counter()
        params = self._prepare_params(model)
        t_prep_end = time.perf_counter()
        T_Prep = t_prep_end - t_prep_start

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

        # -- T_SPDK: background I/O --
        # Synchronise device before DMA reads.  Prefer ms.hal.synchronize
        # (public API, MS 2.3+); fall back to ms.runtime.synchronize
        # (internal, may move).  ops.functional.depend is NOT a runtime
        # barrier — if neither exists, use ACL stream synchronisation.
        if hasattr(ms.hal, "synchronize"):
            ms.hal.synchronize()
        elif hasattr(ms, "runtime") and hasattr(ms.runtime, "synchronize"):
            ms.runtime.synchronize()
        else:
            if acl_lib is not None:
                acl_lib.aclrtSynchronizeStream(None)

        def background_io_worker(c_ptrs_d, c_offs_d, c_sizes_d, n_dev, d_sz,
                                 c_ptrs_h, c_offs_h, c_sizes_h, n_host, h_sz):
            t_spdk_start = time.perf_counter()
            total_written = 0

            if n_dev > 0:
                rc = lib.npu_nvme_write_batch(
                    self.ctx, c_ptrs_d, c_offs_d, c_sizes_d, n_dev)
                if rc != 0:
                    print(f"[Fatal] write_batch failed (rc={rc})")
                total_written += d_sz

            if n_host > 0:
                if hasattr(lib, "npu_nvme_write_batch_host"):
                    rc = lib.npu_nvme_write_batch_host(
                        self.ctx, c_ptrs_h, c_offs_h, c_sizes_h, n_host)
                    if rc != 0:
                        print(f"[Fatal] write_batch_host failed (rc={rc})")
                    total_written += h_sz

            t_spdk_end = time.perf_counter()
            T_SPDK = t_spdk_end - t_spdk_start

            t_meta_start = time.perf_counter()
            self.last_layout = layout
            if commit_meta:
                self._commit_metadata(step, layout)

            with open(meta_path, "wb") as f:
                pickle.dump(self.meta_dict, f)

            t_meta_end = time.perf_counter()
            T_Meta = t_meta_end - t_meta_start

            real_time = time.perf_counter() - t_start
            bw = total_written / 1024 / 1024 / real_time if real_time > 0 else 0

            print(f"\n{'='*54}")
            print(f"[Timeline][Rank {self.rank_id}] Step {step} | "
                  f"Background SPDK Flush ENDED at {time.perf_counter():.3f}s")
            print(f"[Breakdown][Rank {self.rank_id}] "
                  f"Prep: {T_Prep*1000:.2f}ms | Layout: {T_Layout*1000:.2f}ms | "
                  f"SPDK(H/W): {T_SPDK*1000:.2f}ms | Meta: {T_Meta*1000:.2f}ms")
            print(f"[DirectCkpt][Rank {self.rank_id}] Background Safe Write: "
                  f"{total_written/1024/1024:.2f} MB | BW: {bw:.2f} MB/s")
            print(f"{'='*54}\n", flush=True)

        num_dev_val = len(dev_chunks) if dev_chunks else 0
        dev_sz_val = dev_sz if dev_chunks else 0
        num_host_val = len(host_chunks) if host_chunks else 0
        host_sz_val = host_sz if host_chunks else 0

        self.wait_for_io_completion()

        self.io_thread = threading.Thread(
            target=background_io_worker,
            args=(c_ptrs_dev, c_offs_dev, c_sizes_dev, num_dev_val, dev_sz_val,
                  c_ptrs_host, c_offs_host, c_sizes_host, num_host_val, host_sz_val))
        self.io_thread.start()

        t_return = time.perf_counter()
        print(f"[Timeline][Rank {self.rank_id}] Step {step} | Python save() "
              f"dispatched to background thread. "
              f"Layout cost: {T_Layout*1000:.2f}ms", flush=True)

        return (0, len(dev_chunks) + len(host_chunks), 0.0, 0.0,
                {"prep_time": T_Prep, "layout_time": T_Layout})

    def load(self, model: ms.nn.Cell, step: int = None,
             meta_path: str = "checkpoint_meta.pkl"):
        t_start = time.time()

        if step is not None:
            ckpt_key = f"step_{step}"
        else:
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.meta_dict = pickle.load(f)
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

        # NOTE: host_chunks (parameters without NPU device memory, e.g.
        # very small CPU-resident tensors) are NOT read from NVMe here
        # because there is no npu_nvme_read_batch_host C API.  Their
        # np_arr buffers remain zero-filled.  In normal operation (NPU
        # training), all model parameters have device pointers and this
        # path is never exercised.

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
        if not hasattr(lib, "npu_nvme_delta_init"):
            raise RuntimeError(
                "C library missing npu_nvme_delta_init — rebuild required.")
        rc = lib.npu_nvme_delta_init(
            self.ctx, ctypes.c_uint64(slot_bytes), ctypes.c_uint32(slot_count))
        if rc != 0:
            raise RuntimeError(f"Delta init failed (rc={rc})")
        self._delta_slot_size = slot_bytes
        self._delta_slot_count = slot_count
        self._delta_next_slot = 0
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

    def delta_save(self, step: int, block_patches: list, small_patches: list):
        if not hasattr(self, '_delta_slot_count'):
            self.delta_init()

        self.wait_for_io_completion()
        self.wait_async_io()

        frame = pack_delta_frame(step, block_patches, small_patches)
        total_bytes = len(frame)

        if total_bytes > self._delta_slot_size:
            raise RuntimeError(
                f"Delta frame {total_bytes} bytes > slot {self._delta_slot_size}")

        slot_idx = self._delta_next_slot % self._delta_slot_count
        slot_offset = (lib.npu_nvme_delta_get_area_offset(self.ctx)
                       + slot_idx * self._delta_slot_size)

        # Use the chunk pipeline (supports >64MB) instead of the deprecated
        # sync_meta_io path via npu_nvme_write_delta.
        frame_buf = ctypes.create_string_buffer(frame, total_bytes)
        chunks, _ = build_chunks_host(
            ctypes.addressof(frame_buf), slot_offset, total_bytes, self.chunk_size)
        c_ptrs, c_offs, c_sizes = build_ctypes_arrays(chunks)

        if hasattr(lib, "npu_nvme_write_batch_host"):
            rc = lib.npu_nvme_write_batch_host(
                self.ctx, c_ptrs, c_offs, c_sizes, len(chunks))
        else:
            rc = lib.npu_nvme_write_batch(
                self.ctx, c_ptrs, c_offs, c_sizes, len(chunks))
        if rc != 0:
            raise RuntimeError(f"Delta write failed at slot {slot_idx} (rc={rc})")

        self._delta_step_map[step] = slot_idx
        self._delta_next_slot += 1
        self._delta_frame_sizes.append(total_bytes)

        self.meta_dict["delta_chain"][f"step_{step}"] = {
            "type": "DELTA",
            "slot": slot_idx,
            "frame_size": total_bytes,
            "n_blocks": len(block_patches),
            "n_small": len(small_patches),
        }
        if hasattr(self, '_dump_meta_pkl'):
            self._dump_meta_pkl()

        return slot_idx

    def delta_load_slot(self, slot_idx: int):
        if not hasattr(self, '_delta_slot_size'):
            raise RuntimeError(
                "Delta not initialized. Call delta_init() first.")

        slot_offset = (lib.npu_nvme_delta_get_area_offset(self.ctx)
                       + slot_idx * self._delta_slot_size)

        # Read header to determine actual frame size, then full data.
        header_buf = ctypes.create_string_buffer(FRAME_HEADER_SIZE)
        h_chunks, _ = build_chunks_host(
            ctypes.addressof(header_buf), slot_offset,
            FRAME_HEADER_SIZE, self.chunk_size)
        h_ptrs, h_offs, h_sizes = build_ctypes_arrays(h_chunks)
        if hasattr(lib, "npu_nvme_read_batch_host"):
            rc = lib.npu_nvme_read_batch_host(
                self.ctx, h_ptrs, h_offs, h_sizes, len(h_chunks))
        else:
            rc = lib.npu_nvme_read_batch(
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
        if hasattr(lib, "npu_nvme_read_batch_host"):
            rc = lib.npu_nvme_read_batch_host(
                self.ctx, d_ptrs, d_offs, d_sizes, len(d_chunks))
        else:
            rc = lib.npu_nvme_read_batch(
                self.ctx, d_ptrs, d_offs, d_sizes, len(d_chunks))
        if rc != 0:
            raise RuntimeError(
                f"Delta data read failed at slot {slot_idx} (rc={rc})")

        step_id, blocks, smalls = unpack_delta_frame(data_buf.raw[:total_sz])
        return step_id, blocks, smalls

    def delta_load_chain(self, from_step: int, to_step: int):
        chain = []
        for s in range(from_step + 1, to_step + 1):
            key = f"step_{s}"
            if key not in self.meta_dict.get("delta_chain", {}):
                raise FileNotFoundError(
                    f"Delta frame for step {s} not found in metadata")
            slot = self.meta_dict["delta_chain"][key]["slot"]
            sid, blocks, smalls = self.delta_load_slot(slot)
            if sid != s:
                print(f"[DirectCkpt] WARNING: delta slot {slot} "
                      f"step_id={sid} != expected {s}")
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
