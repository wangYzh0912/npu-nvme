"""Direct checkpoint manager with MindSpore probe wrapper.

Usage:
- Import DirectCheckpoint and ProbeTrainOneStepCell from this module.
- Used by training scripts under python/.

Inputs:
- NVMe PCI address, NPU device id, chunk size, profiling directory.
Outputs:
- Writes checkpoints and metadata to NVMe and optional profiling CSV under output/.
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
from mindspore.ops import MultitypeFuncGraph, HyperMap, CustomRegOp, DataType
from mindspore.common.initializer import Initializer, _register, initializer
import numpy as np

import atexit

# -- Disk layout constants (byte-addressed raw block device) --
SUPERBLOCK_OFFSET      = 0
SUPERBLOCK_HEADER_BYTES = 28  # "<8s I Q Q" = magic(8) + slot(4) + capacity(8) + stack(8)
META_SLOT_A_OFFSET     = 4096
META_SLOT_B_OFFSET     = 4096 + 400 * 1024
META_SLOT_BYTES        = 400 * 1024
MAGIC_NUMBER           = b"NPUNVME1"
UINT32_BYTES           = 4

# -- Delta frame binary protocol constants --
DELTA_MAGIC      = 0x414C5444   # "DLTA"
FRAME_HEADER_SIZE = 4096

# -- C library bindings --
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_PATH  = os.path.join(_REPO_ROOT, "build_out", "lib", "libnpu_nvme.so")

try:
    acl_lib = ctypes.CDLL("libascendcl.so")
    # aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, int kind);
    acl_lib.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    acl_lib.aclrtMemcpy.restype = ctypes.c_int
except Exception as e:
    print(f"[DirectCkpt] Warning: Failed to load libascendcl.so for probe: {e}")
    acl_lib = None

try:
    lib = ctypes.CDLL(_LIB_PATH)
    class NPUNVMEContext(ctypes.Structure): pass

    if hasattr(lib, "npu_nvme_set_probe_flag_ptr"):
        lib.npu_nvme_set_probe_flag_ptr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.npu_nvme_set_probe_flag_ptr.restype = ctypes.c_int
    if hasattr(lib, "npu_nvme_trigger_probe"):
        lib.npu_nvme_trigger_probe.argtypes = [ctypes.POINTER(NPUNVMEContext)]
        lib.npu_nvme_trigger_probe.restype = ctypes.c_int
    if hasattr(lib, "npu_nvme_set_probe_flag_value"):
        lib.npu_nvme_set_probe_flag_value.argtypes = [ctypes.POINTER(NPUNVMEContext), ctypes.c_uint32]
        lib.npu_nvme_set_probe_flag_value.restype = ctypes.c_int
    if hasattr(lib, "npu_nvme_get_probe_flag_dev_ptr"):
        lib.npu_nvme_get_probe_flag_dev_ptr.argtypes = [ctypes.c_void_p]
        lib.npu_nvme_get_probe_flag_dev_ptr.restype = ctypes.c_void_p
    if hasattr(lib, "npu_nvme_set_step_ptr"):
        lib.npu_nvme_set_step_ptr.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.npu_nvme_set_step_ptr.restype = ctypes.c_int

    lib.npu_nvme_register_tasks.argtypes = [
        ctypes.c_void_p,                     # ctx
        ctypes.POINTER(ctypes.c_void_p),     # npu_ptrs
        ctypes.POINTER(ctypes.c_uint64),     # nvme_offsets
        ctypes.POINTER(ctypes.c_size_t),     # sizes
        ctypes.c_int                         # num_items
    ]
    lib.npu_nvme_register_tasks.restype = ctypes.c_int

    lib.npu_nvme_init.argtypes = [
        ctypes.POINTER(ctypes.POINTER(NPUNVMEContext)), ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_char_p,
    ]
    lib.npu_nvme_init.restype = ctypes.c_int

    lib.npu_nvme_cleanup.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_cleanup.restype = None

    lib.npu_nvme_get_max_transfer.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_get_max_transfer.restype = ctypes.c_int

    lib.npu_nvme_get_total_blocks.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_get_total_blocks.restype = ctypes.c_uint64

    lib.npu_nvme_sync_meta_io.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p
    ]
    lib.npu_nvme_sync_meta_io.restype = ctypes.c_int

    lib.npu_nvme_write_batch.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
    ]
    lib.npu_nvme_write_batch.restype = ctypes.c_int

    lib.npu_nvme_read_batch.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
    ]
    lib.npu_nvme_read_batch.restype = ctypes.c_int

    if hasattr(lib, "npu_nvme_write_batch_host"):
        lib.npu_nvme_write_batch_host.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
        ]
        lib.npu_nvme_write_batch_host.restype = ctypes.c_int

    # Delta frame I/O bindings
    if hasattr(lib, "npu_nvme_delta_init"):
        lib.npu_nvme_delta_init.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
        lib.npu_nvme_delta_init.restype  = ctypes.c_int
        lib.npu_nvme_write_delta.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        lib.npu_nvme_write_delta.restype  = ctypes.c_int
        lib.npu_nvme_read_delta.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        lib.npu_nvme_read_delta.restype  = ctypes.c_int
        lib.npu_nvme_delta_get_area_offset.argtypes = [ctypes.c_void_p]
        lib.npu_nvme_delta_get_area_offset.restype  = ctypes.c_uint64

except OSError as e:
    print(f"[Warning] Failed to load {_LIB_PATH}. Error: {e}")

# -- Chunk builder and layout helpers --
def build_chunks(params: List[Dict], chunk_size: int):
    chunks = []
    total_size = 0
    for p in params:
        ptr = p["ptr"]
        remaining = p["size"]
        inner_off = 0
        nvme_offset_bytes = p["offset"] 
        name = p.get("name", "unknown") # chunk metadata: parameter name for debugging
        
        while remaining > 0:
            take = min(remaining, chunk_size)
            chunks.append((
                ctypes.c_void_p(ptr + inner_off),
                ctypes.c_uint64(nvme_offset_bytes),
                ctypes.c_size_t(take),
                name 
            ))
            remaining -= take
            inner_off += take
            nvme_offset_bytes += int(math.ceil(take / 4096.0)) * 4096
            total_size += take
            
    return chunks, total_size


def build_chunks_host(ptr_base: int, start_offset: int, total_size: int,
                      chunk_size: int, name: str = ""):
    """build_chunks variant for a single host-side buffer (no MindSpore param).

    Wraps a raw pointer + offset + size into the params dict format expected
    by build_chunks().  Returns (chunks_list, total_size) — same shape as
    build_chunks().

    Args:
        ptr_base:     base Python integer address of the host buffer
        start_offset: NVMe byte offset where writes begin
        total_size:   total bytes to write
        chunk_size:   max bytes per chunk (4K-aligned in build_chunks)
        name:         label for debugging (default "")
    Returns:
        (list of (ptr: c_void_p, offset: c_uint64, size: c_size_t, name: str),
         total_size: int)
    """
    params = [{"ptr": ptr_base, "size": total_size, "offset": start_offset, "name": name}]
    return build_chunks(params, chunk_size)


def rebuild_chunks_from_meta(models, params_meta: Dict, chunk_size: int):
    if not isinstance(models, (list, tuple)): models = [models]
    buffers = []

    def get_dev_ptr(p):
        ptr = 0
        try:
            data_obj = p.data if hasattr(p, "data") else p
            if hasattr(data_obj, "_data_ptr"):
                 if isinstance(p, ms.Parameter) and hasattr(p, "is_inited") and not p.is_inited:
                     p.init_data()
                 ptr = int(data_obj._data_ptr())
        except Exception:
            ptr = 0
        return ptr

    for model in models:
        if model is None or not hasattr(model, "parameters_and_names"): continue
        for name, param in model.parameters_and_names():
            if name not in params_meta: continue
            info = params_meta[name]

            dev_ptr = get_dev_ptr(param)
            use_dev = (dev_ptr != 0)
            np_arr = None
            ptr_val = dev_ptr

            if not use_dev:
                # restore exact element count without 4 KB padding
                np_arr = np.empty(info["shape"], dtype=np.dtype(info["dtype"]))
                ptr_val = np_arr.ctypes.data

            buffers.append({
                "name": name, "ptr": ptr_val, "size": info["size"],
                "offset": info["offset"], "np_arr": np_arr,
                "param_ref": param, "use_dev": use_dev
            })

    buffers.sort(key=lambda x: x["offset"])

    dev_buffers = [b for b in buffers if b["use_dev"]]
    host_buffers = [b for b in buffers if not b["use_dev"]]

    dev_chunks, _ = build_chunks(dev_buffers, chunk_size)
    host_chunks, _ = build_chunks(host_buffers, chunk_size)

    return dev_chunks, host_chunks, buffers


# -- Delta frame binary protocol (serialization) ------------------------------------------------

def pack_delta_frame(step_id, block_patches, small_patches):
    """Serialize I3 delta to binary frame. Returns bytes.

    Frame layout:
      Header (28 bytes, zero-padded to 4KB):
        magic(4) + step_id(4) + n_blocks(4) + n_small(4) + total_sz(4) + checksum(4)
      Block Records (variable):
        layer_id(i2) + name_len(H) + name + block_idx(i4) + data_len(i4) + scale(f4) + data(i1[])
      Small Records (variable):
        layer_id(i2) + name_len(H) + name + data_len(i4) + scale(f4) + data(i1[])
    """
    buf = bytearray(FRAME_HEADER_SIZE)
    payload = bytearray()

    # Pack block records
    for bp in block_patches:
        lid = bp["layer_id"]
        name = bp["name"].encode('utf-8')
        bidx = bp["block_idx"]
        i8_data = bp["int8_data"]
        i8_bytes = i8_data.tobytes() if isinstance(i8_data, np.ndarray) else bytes(i8_data)
        scale = float(bp["scale"])
        data_len = len(i8_bytes)

        payload += struct.pack(f"<hH{len(name)}s i f", lid, len(name), name, bidx, scale)
        payload += struct.pack(f"<i{len(i8_bytes)}s", data_len, i8_bytes)

    # Pack small records
    for sp in small_patches:
        lid = sp["layer_id"]
        name = sp["name"].encode('utf-8')
        i8_data = sp["int8_data"]
        i8_bytes = i8_data.tobytes() if isinstance(i8_data, np.ndarray) else bytes(i8_data)
        scale = float(sp["scale"])
        data_len = len(i8_bytes)

        payload += struct.pack(f"<hH{len(name)}s f", lid, len(name), name, scale)
        payload += struct.pack(f"<i{len(i8_bytes)}s", data_len, i8_bytes)

    total_sz = FRAME_HEADER_SIZE + len(payload)
    checksum = sum(payload) & 0xFFFFFFFF  # simple checksum

    # Write header
    struct.pack_into(f"<I I I I I I", buf, 0, DELTA_MAGIC, step_id,
                     len(block_patches), len(small_patches), total_sz, checksum)

    frame = bytes(buf) + bytes(payload)
    return frame


def unpack_delta_frame(frame_bytes):
    """Deserialize binary frame back to (step_id, block_patches, small_patches)."""
    if len(frame_bytes) < FRAME_HEADER_SIZE:
        raise ValueError(f"Frame too short: {len(frame_bytes)} < {FRAME_HEADER_SIZE}")

    magic, step_id, n_blocks, n_small, total_sz, checksum = \
        struct.unpack_from("<I I I I I I", frame_bytes, 0)

    if magic != DELTA_MAGIC:
        raise ValueError(f"Invalid delta magic: 0x{magic:08x} (expected 0x{DELTA_MAGIC:08x})")

    pos = FRAME_HEADER_SIZE
    block_patches, small_patches = [], []

    def _read_block(pos):
        lid, name_len = struct.unpack_from("<hH", frame_bytes, pos)
        pos += 4
        name = frame_bytes[pos:pos+name_len].decode('utf-8')
        pos += name_len
        bidx, scale = struct.unpack_from("<i f", frame_bytes, pos)
        pos += 8
        data_len = struct.unpack_from("<i", frame_bytes, pos)[0]
        pos += 4
        i8_data = np.frombuffer(frame_bytes[pos:pos+data_len], dtype=np.int8)
        pos += data_len
        return pos, {"layer_id": lid, "name": name, "block_idx": bidx,
                      "int8_data": i8_data, "scale": scale}

    def _read_small(pos):
        lid, name_len = struct.unpack_from("<hH", frame_bytes, pos)
        pos += 4
        name = frame_bytes[pos:pos+name_len].decode('utf-8')
        pos += name_len
        scale = struct.unpack_from("<f", frame_bytes, pos)[0]
        pos += 4
        data_len = struct.unpack_from("<i", frame_bytes, pos)[0]
        pos += 4
        i8_data = np.frombuffer(frame_bytes[pos:pos+data_len], dtype=np.int8)
        pos += data_len
        return pos, {"layer_id": lid, "name": name, "int8_data": i8_data, "scale": scale}

    for _ in range(n_blocks):
        pos, bp = _read_block(pos)
        block_patches.append(bp)
    for _ in range(n_small):
        pos, sp = _read_small(pos)
        small_patches.append(sp)

    return step_id, block_patches, small_patches


def apply_delta_patches(init_weights, block_patches, small_patches, block_size):
    """Apply delta patches to a weight dictionary. Returns modified copy."""
    import copy
    w = copy.deepcopy(init_weights)
    for bp in block_patches:
        name = bp["name"]
        bidx = bp["block_idx"]
        i8 = bp["int8_data"]
        s = bp["scale"]
        if isinstance(i8, np.ndarray):
            fp32 = i8.astype(np.float32) * s
        else:
            fp32 = np.frombuffer(i8, dtype=np.int8).astype(np.float32) * s
        start = bidx * block_size
        end = min(start + len(fp32), int(np.prod(w[name].shape)))
        wv = w[name].astype(np.float32).flatten()
        wv[start:end] = fp32[:end-start]
        w[name] = wv.reshape(w[name].shape)
    for sp in small_patches:
        name = sp["name"]
        i8 = sp["int8_data"]
        s = sp["scale"]
        if isinstance(i8, np.ndarray):
            fp32 = i8.astype(np.float32) * s
        else:
            fp32 = np.frombuffer(i8, dtype=np.int8).astype(np.float32) * s
        w[name] = fp32[:int(np.prod(w[name].shape))].reshape(w[name].shape)
    return w


# -- Delta frame I/O classes --------------------------------------------------------------------

class I3DeltaWriter:
    """SPDK-backed incremental checkpoint writer.

    Uses host-side buffer → npu_nvme_write_delta → NVMe delta ring.
    Reuses the module-level C library bindings.
    """
    def __init__(self, ctx, delta_slot_size=256*1024*1024, delta_slot_count=128):
        self.ctx = ctx
        self.slot_size = delta_slot_size
        self.slot_count = delta_slot_count

        # Reuse module-level lib (no separate CDLL load)
        rc = lib.npu_nvme_delta_init(ctx, delta_slot_size, delta_slot_count)
        if rc != 0:
            raise RuntimeError(f"Delta init failed (rc={rc})")

        self.next_slot = 0
        self.step_map = {}   # step_id → slot_idx
        self.frame_sizes = []

    @property
    def area_offset(self):
        return lib.npu_nvme_delta_get_area_offset(self.ctx)

    def write_frame(self, step_id, block_patches, small_patches):
        """Serialize and write one delta frame to next slot. Returns slot_idx."""
        frame = pack_delta_frame(step_id, block_patches, small_patches)
        total_bytes = len(frame)

        if total_bytes > self.slot_size:
            raise ValueError(f"Frame {total_bytes} bytes > slot {self.slot_size} bytes!")

        slot_idx = self.next_slot % self.slot_count

        buf = ctypes.create_string_buffer(frame, total_bytes)
        rc = lib.npu_nvme_write_delta(self.ctx, slot_idx,
                                       ctypes.c_void_p(ctypes.addressof(buf)),
                                       total_bytes)
        if rc != 0:
            raise RuntimeError(f"Delta write failed at slot {slot_idx} (rc={rc})")

        self.step_map[step_id] = slot_idx
        self.next_slot += 1
        self.frame_sizes.append(total_bytes)

        return slot_idx

    def read_frame(self, slot_idx):
        """Read a delta frame from NVMe. Returns (step_id, block_patches, small_patches)."""
        buf = ctypes.create_string_buffer(self.slot_size)
        rc = lib.npu_nvme_read_delta(
            self.ctx, slot_idx,
            ctypes.c_void_p(ctypes.addressof(buf)),
            self.slot_size)
        if rc != 0:
            raise RuntimeError(f"Delta read failed at slot {slot_idx} (rc={rc})")

        # Parse header to get actual frame size
        magic, total_sz = struct.unpack_from("<I", buf.raw, 0)[0], struct.unpack_from("<I", buf.raw, 16)[0]
        if magic != DELTA_MAGIC:
            raise RuntimeError(f"Delta read at slot {slot_idx}: bad magic 0x{magic:08x}")
        frame = buf.raw[:total_sz]
        return unpack_delta_frame(frame)

    def get_slot_range(self, start_step, end_step):
        """Get list of slot indices between start_step and end_step (inclusive)."""
        slots = []
        for s in range(start_step, end_step + 1):
            if s in self.step_map:
                slots.append(self.step_map[s])
        return slots

    @property
    def stats(self):
        return {
            "total_frames": len(self.frame_sizes),
            "total_bytes": sum(self.frame_sizes),
            "total_mb": sum(self.frame_sizes) / (1024*1024),
            "avg_kb": (sum(self.frame_sizes) / max(len(self.frame_sizes), 1)) / 1024,
            "max_kb": max(self.frame_sizes) / 1024 if self.frame_sizes else 0,
            "slots_used": self.next_slot,
            "slot_capacity": self.slot_count,
        }

    def close(self):
        pass


class FileDeltaWriter:
    """Filesystem-backed delta writer. Same API as I3DeltaWriter.

    Uses a ring of files under delta_dir. No SPDK/NVMe dependency.
    """
    def __init__(self, delta_dir, delta_slot_count=128, delta_slot_size=256*1024*1024):
        self.delta_dir = delta_dir
        self.slot_count = delta_slot_count
        self.slot_size = delta_slot_size
        os.makedirs(delta_dir, exist_ok=True)
        self.next_slot = 0
        self.step_map = {}
        self.frame_sizes = []

    def write_frame(self, step_id, block_patches, small_patches):
        frame = pack_delta_frame(step_id, block_patches, small_patches)
        total_bytes = len(frame)
        if total_bytes > self.slot_size:
            raise ValueError(f"Frame {total_bytes} > slot {self.slot_size}")

        slot_idx = self.next_slot % self.slot_count
        fpath = os.path.join(self.delta_dir, f"delta_slot_{slot_idx:04d}.bin")
        with open(fpath, "wb") as f:
            f.write(frame)

        self.step_map[step_id] = slot_idx
        self.next_slot += 1
        self.frame_sizes.append(total_bytes)
        return slot_idx

    def read_frame(self, slot_idx):
        fpath = os.path.join(self.delta_dir, f"delta_slot_{slot_idx:04d}.bin")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Delta slot {slot_idx} not found: {fpath}")
        with open(fpath, "rb") as f:
            frame = f.read()
        return unpack_delta_frame(frame)

    @property
    def stats(self):
        return {
            "total_frames": len(self.frame_sizes),
            "total_bytes": sum(self.frame_sizes),
            "total_mb": sum(self.frame_sizes) / (1024*1024),
            "avg_kb": (sum(self.frame_sizes) / max(len(self.frame_sizes), 1)) / 1024,
            "max_kb": max(self.frame_sizes) / 1024 if self.frame_sizes else 0,
            "slots_used": self.next_slot,
            "slot_capacity": self.slot_count,
            "backend": "file",
        }

    def close(self):
        pass


# -- Fast initialization (skip random init when loading from NVMe) -------------------------------

@_register('noop')
class NoOpInitializer(Initializer):
    """A 'Fake' Initializer that does nothing.

    Used to bypass time-consuming random initialization when parameters will
    be immediately overwritten with data loaded from an NVMe checkpoint via
    DirectCheckpoint.load().

    Note: ``arr`` retains uninitialized memory (garbage values).  This is
    safe only when every parameter is overwritten by a subsequent load() call.
    """
    def _initialize(self, arr):
        # Do nothing — arr retains garbage values.
        # These will be overwritten by DirectCheckpoint.load().
        pass


def replace_with_noop_initializer(model):
    """Replace all parameter initializers with NoOpInitializer.

    Iterates over model.parameters_and_names() and sets each parameter's
    init_mode to a NoOpInitializer MetaTensor.  The init_mode must be a
    Tensor (wrapped initializer), not the Initializer object itself, so
    that copy/clone operations work correctly.

    Args:
        model: a MindSpore nn.Cell instance
    """
    print("[FastInit] Replacing original initializers with NoOpInitializer...", flush=True)
    count = 0
    for param in model.get_parameters():
        if param.init_mode is not None:
            # init_mode must be a Tensor (wrapped initializer), not the
            # Initializer object itself.  Use the initializer factory function
            # to create a MetaTensor that holds the config.
            param.init_mode = initializer(NoOpInitializer(), shape=param.shape, dtype=param.dtype)
            count += 1
    print(f"[FastInit] Replaced {count} parameter initializers.", flush=True)


# DEPRECATED: WaitProbe-era bind_depend_op.  Used only by
# cell_overhead_analysis.py and operator_microbenchmarks.py.
# New code should use the FaF step_counter path (ProbeTrainOneStepCell.enable_probe=False).
bind_depend_op = MultitypeFuncGraph("bind_depend_op")
@bind_depend_op.register("Tensor", "Tensor")
def _bind_depend_op(sig, grad):
    return ops.depend(grad, sig)

def get_dev_ptr(tensor):
    ptr = 0
    try:
        data_obj = tensor.data if hasattr(tensor, "data") else tensor
        if hasattr(data_obj, "_data_ptr"):
            if isinstance(tensor, ms.Parameter) and hasattr(tensor, "is_inited") and not tensor.is_inited:
                tensor.init_data()
            ptr = int(data_obj._data_ptr())
        elif hasattr(tensor, "data_ptr"):
            ptr = int(tensor.data_ptr())
    except Exception:
        ptr = 0
    return ptr

# DEPRECATED: WaitProbe AICPU custom-op registrations.
# These were part of the original I2 design (graph-injected synchronisation
# primitive).  The GE compiler cannot load custom AICPU kernels in sink=TRUE
# mode, so the WaitProbe path was replaced by the FaF step_counter listener.
# Kept for backward compatibility with legacy experiment scripts.
wait_op_info = CustomRegOp("WaitProbe") \
    .input(0, "flag") \
    .input(1, "expected") \
    .output(0, "y") \
    .dtype_format(DataType.U32_Default, DataType.U32_Default, DataType.U32_Default) \
    .target("Ascend") \
    .get_op_info()

trigger_op_info = CustomRegOp("TriggerProbe") \
    .input(0, "step") \
    .input(1, "interval") \
    .input(2, "trigger_buf") \
    .input(3, "expected") \
    .output(0, "y") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default,
                  DataType.U32_Default, DataType.U32_Default,
                  DataType.I32_Default) \
    .target("Ascend") \
    .get_op_info()

class ProbeTrainOneStepCell(nn.Cell):
    """Training cell with optional step_counter injection for the FaF listener.

    DEPRECATED (WaitProbe params): ``flag``, ``expected``, ``wait_probe``,
    and ``trigger_probe`` are created when ``enable_probe=True`` but are NOT
    used in the GE graph.  They exist only as HBM scratch space for legacy
    experiment scripts that poll them from Python callbacks.  These will be
    removed once all callers migrate to the FaF step_counter-only path.
    """

    def __init__(self, network, optimizer, so_path, flag_addr, enable_probe=True, probe_mode="full",
                 ckpt_interval=10):
        super().__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.grad_fn = ops.value_and_grad(self.network, grad_position=None, weights=self.optimizer.parameters)

        self.enable_probe = enable_probe
        self.depend = ops.Depend()
        self.hyper_map = HyperMap()

        self.probe_mode = probe_mode
        self.ckpt_interval = ckpt_interval

        if self.enable_probe:
            # DEPRECATED: WaitProbe flag/expected — unused in graph, kept for
            # legacy callers that poll them from Python (train_gpt2_spdk.py et al.)
            self.flag = ms.Parameter(ms.Tensor([0], dtype=ms.uint32), requires_grad=False, name="probe_flag")
            self.expected = ms.Parameter(ms.Tensor([0], dtype=ms.uint32), requires_grad=False, name="probe_expected")
            self.wait_probe = ops.Custom("WaitProbe", out_shape=[1], out_dtype=ms.uint32,
                                         func_type="aicpu", reg_info=wait_op_info)

            # FaF step_counter: injected into the GE graph so the C-layer
            # listener can poll it via aclrtMemcpy.  This is the active path.
            self.step_counter = ms.Parameter(
                ms.Tensor([0], dtype=ms.int32), requires_grad=False, name="step_counter")
            self.one_i32 = Tensor([1], dtype=ms.int32)
            self.interval_i32 = Tensor([ckpt_interval], dtype=ms.int32)

            # DEPRECATED: WaitProbe trigger_buf — unused in graph
            self.trigger_buf = ms.Parameter(
                ms.Tensor([0], dtype=ms.uint32), requires_grad=False, name="trigger_buf")
            self.trigger_probe = ops.Custom(
                "TriggerProbe", out_shape=[1], out_dtype=ms.int32,
                func_type="aicpu", reg_info=trigger_op_info)

    def construct(self, *inputs):
        if not self.enable_probe:
            loss, grads = self.grad_fn(*inputs)
            opt_res = self.optimizer(grads)
            loss = self.depend(loss, opt_res)
            return loss

        loss, grads = self.grad_fn(*inputs)

        # Fire-and-Forget: step_counter auto-increments each step.
        # C layer listener polls step_counter directly (no AICPU kernel needed).
        # This avoids the GE aclnn wrapper issue that prevents custom AICPU
        # kernels from launching in sink=TRUE fused graphs.
        # CRITICAL: depend() ensures step_counter update is included in the fused graph.
        step = ops.assign_add(self.step_counter, self.one_i32)
        loss = self.depend(loss, step)

        opt_res = self.optimizer(grads)
        loss = self.depend(loss, opt_res)

        return loss

# -- DirectCheckpoint: NVMe-backed training checkpoint manager --
class DirectCheckpoint:
    _ms_warmed_up = False

    def __init__(
        self, nvme_addr: str = "0000:83:00.0", npu_device_id: int = 0, pipeline_depth: int = 4,
        requested_chunk_size: int = 4 * 1024 * 1024, enable_profiling: bool = False,
        profiling_dir: str = "./output/profiling", rank_id: int = 0, world_size: int = 1,
        base_offset_bytes: int = 0, shard_span_bytes: int = None, spdk_shm_id: int = 1,
        keep_last_n: int = 3, slot_size_gb: int = 50,
        warmup_fn: callable = None
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
        # Path for auto-saved meta pickle (used by recovery)
        self._meta_pkl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "experiments", "output", "checkpoint_meta.pkl")
        
        if self.enable_profiling:
            os.makedirs(self.profiling_dir, exist_ok=True)

        # MS runtime warmup (must complete before SPDK init)
        if warmup_fn is not None and not DirectCheckpoint._ms_warmed_up:
            print("[DirectCheckpoint] Running MS runtime warmup before SPDK init...", flush=True)
            warmup_fn()
            DirectCheckpoint._ms_warmed_up = True
            print("[DirectCheckpoint] MS runtime warmup complete.", flush=True)

        print(f"[DirectCheckpoint] loading so from {_LIB_PATH}")

        rc = lib.npu_nvme_init(
            ctypes.byref(self.ctx),
            nvme_addr.encode(),
            npu_device_id,
            pipeline_depth,
            requested_chunk_size,
            enable_profiling,
            self.profiling_dir.encode("utf-8")
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
        
        self.async_thread = None
        self.async_lock = threading.Lock()

        self._mount_filesystem()

        self._closed = False
        atexit.register(self.close)

        self.io_thread = None

    def _mount_filesystem(self):
        sb_buf = ctypes.create_string_buffer(4096)
        rc = lib.npu_nvme_sync_meta_io(self.ctx, SUPERBLOCK_OFFSET, 4096, 1, ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc != 0:
            raise RuntimeError("Failed to read Superblock.")

        header = struct.unpack("<8s I Q Q", sb_buf.raw[:SUPERBLOCK_HEADER_BYTES])
        magic = header[0]
        
        if magic != MAGIC_NUMBER:
            raise RuntimeError(f"Superblock verification failed: unexpected header 0x{magic.hex()} (expected NPUNVME1). Run format_npu_disk.py to initialize the disk.")
            
        self.active_meta_slot = header[1]
        self.stack_start_bytes = header[3]
        
        # Verify the stored stack start  still fits the current multi-rank configuration
        total_stack_bytes = self.world_size * self.keep_last_n * self.slot_bytes
        if self.stack_start_bytes == 0 or (self.stack_start_bytes + total_stack_bytes > self.total_bytes):
            print(f"[Warning] Rank {self.rank_id}: Stale Superblock! Current config needs {total_stack_bytes/1024**3:.2f} GB, but disk bounds exceeded. Recalculating...", flush=True)
            self.stack_start_bytes = self.total_bytes - total_stack_bytes
            print(f"[Info] Stack dynamically re-allocated at disk end. Start Offset: {self.stack_start_bytes / 1024**3:.2f} GB")

        target_offset = META_SLOT_A_OFFSET if self.active_meta_slot == 0 else META_SLOT_B_OFFSET
        meta_buf = ctypes.create_string_buffer(META_SLOT_BYTES)
        lib.npu_nvme_sync_meta_io(self.ctx, target_offset, META_SLOT_BYTES, 1, ctypes.c_void_p(ctypes.addressof(meta_buf)))
        
        meta_str = meta_buf.value.decode('utf-8', errors='ignore').rstrip('\x00')
        if meta_str:
            try:
                self.meta_dict = json.loads(meta_str)
            except json.JSONDecodeError:
                self.meta_dict = {"checkpoints": {}}
                print("[Warning] Parsed metadata JSON was invalid. Starting fresh.")
            
        print(f"[Rank {self.rank_id}] FileSystem Mounted. Active Slot: {'A' if self.active_meta_slot==0 else 'B'}")

    def _get_current_slot_base_offset(self, step: int):
        slot_idx = step % self.keep_last_n
        rank_offset = self.rank_id * self.keep_last_n * self.slot_bytes
        slot_offset = slot_idx * self.slot_bytes
        return self.stack_start_bytes + rank_offset + slot_offset

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
                "offset": p["offset"],
                "size": p["size"],
                "shape": p["shape"],
                "dtype": p["dtype"],
            } for p in layout}
        }
        
        saved_steps = []
        for k in self.meta_dict["checkpoints"].keys():
            if k.startswith("step_"):
                try:
                    saved_steps.append(int(k.split('_')[1]))
                except ValueError:
                    pass
        saved_steps = sorted(saved_steps)

        # Also prune delta chain entries > keep_last_n (delta and FULL share step space)
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

        # Write the FULL checkpoint meta BEFORE committing NVMe JSON.
        # Save to pickle for recovery — includes both checkpoints + delta_chain.
        import pickle as _pickle
        os.makedirs(os.path.dirname(self._meta_pkl), exist_ok=True)
        with open(self._meta_pkl, "wb") as _f:
            _pickle.dump(self.meta_dict, _f)

        # ALSO save after each delta to keep pickle up-to-date.
        # We define a helper for internal use.
        def _dump_meta_pkl():
            with open(self._meta_pkl, "wb") as _f:
                _pickle.dump(self.meta_dict, _f)
        self._dump_meta_pkl = _dump_meta_pkl

        next_slot = 1 if self.active_meta_slot == 0 else 0
        target_offset = META_SLOT_B_OFFSET if next_slot == 1 else META_SLOT_A_OFFSET
        
        meta_json = json.dumps(self.meta_dict).encode('utf-8')
        if len(meta_json) > META_SLOT_BYTES:
            raise RuntimeError(f"Metadata JSON exceeds allocated {META_SLOT_BYTES} bytes!")
            
        meta_buf = ctypes.create_string_buffer(meta_json, META_SLOT_BYTES)
        
        rc1 = lib.npu_nvme_sync_meta_io(self.ctx, target_offset, META_SLOT_BYTES, 0, ctypes.c_void_p(ctypes.addressof(meta_buf)))
        if rc1 != 0:
            raise RuntimeError(f"Fatal: Meta JSON write failed! C layer returned {rc1}.")
        
        sb_buf = ctypes.create_string_buffer(4096)
        struct.pack_into("<8s I Q Q", sb_buf, 0, MAGIC_NUMBER, next_slot, self.total_bytes, self.stack_start_bytes)
        rc2 = lib.npu_nvme_sync_meta_io(self.ctx, SUPERBLOCK_OFFSET, 4096, 0, ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc2 != 0:
            raise RuntimeError("Fatal: Superblock write failed!")
        
        self.active_meta_slot = next_slot
        print(f"[DirectCkpt] Rank 0 Meta committed safely to Slot {'B' if next_slot == 1 else 'A'} (Superblock updated).", flush=True)

    def set_probe_flag_ptr(self, flag_tensor: Tensor = None):
        """Set the probe flag device pointer for the C-layer listener.

        When flag_tensor is provided with a valid device pointer, that pointer
        is used directly.  When flag_tensor is None or its device pointer is 0
        (typical in sink=TRUE graphs where MS never allocates the tensor), the
        C layer self-allocates a 4-byte HBM buffer via aclrtMalloc instead.
        """
        if not hasattr(lib, "npu_nvme_set_probe_flag_ptr"):
            raise RuntimeError("npu_nvme_set_probe_flag_ptr is not available in the C library.")

        ptr = get_dev_ptr(flag_tensor) if flag_tensor is not None else 0

        rc = lib.npu_nvme_set_probe_flag_ptr(self.ctx, ctypes.c_void_p(ptr))
        if rc != 0:
            raise RuntimeError(f"Failed to set probe flag pointer. C API returned {rc}")

        # Always refresh probe_flag_ptr from the C layer — it may have self-allocated
        if hasattr(lib, "npu_nvme_get_probe_flag_dev_ptr"):
            actual_ptr = lib.npu_nvme_get_probe_flag_dev_ptr(self.ctx)
            if actual_ptr:
                self.probe_flag_ptr = actual_ptr
                return
        self.probe_flag_ptr = ptr

    def read_probe_flag_dev(self) -> int:
        """Read probe flag directly from device memory via ACL."""
        if acl_lib is None:
            raise RuntimeError("acl_lib not available for device read")
        if not hasattr(self, "probe_flag_ptr") or self.probe_flag_ptr == 0:
            raise RuntimeError("probe_flag_ptr is not set")

        ret = acl_lib.aclrtSetDevice(self.npu_device_id)
        if ret != 0:
            raise RuntimeError(f"aclrtSetDevice failed, ret={ret}")

        host_buf = ctypes.create_string_buffer(UINT32_BYTES)
        ret = acl_lib.aclrtMemcpy(ctypes.byref(host_buf), UINT32_BYTES,
                                  ctypes.c_void_p(self.probe_flag_ptr), UINT32_BYTES, 2)
        if ret != 0:
            raise RuntimeError(f"aclrtMemcpy device->host failed, ret={ret}")
        return int.from_bytes(host_buf.raw[:UINT32_BYTES], byteorder="little", signed=False)

    def write_probe_flag_dev(self, value: int):
        """Write probe flag directly to device memory via ACL."""
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
        host_buf.raw = int(value).to_bytes(UINT32_BYTES, byteorder="little", signed=False)
        ret = acl_lib.aclrtMemcpy(ctypes.c_void_p(self.probe_flag_ptr), UINT32_BYTES,
                                  ctypes.byref(host_buf), UINT32_BYTES, 1)
        if ret != 0:
            raise RuntimeError(f"aclrtMemcpy host->device failed, ret={ret}")

    def probe_flag_selftest(self):
        """One-shot selftest to verify device pointer is writable."""
        orig = self.read_probe_flag_dev()
        self.write_probe_flag_dev(1)
        after = self.read_probe_flag_dev()
        self.write_probe_flag_dev(orig)
        print(f"[DirectCkpt] probe flag selftest: orig={orig}, after={after}")

    def trigger_probe(self):
        """DEPRECATED: trigger the C-layer listener via probe_flags[0]=1.

        This is the old WaitProbe synchronisation path used by
        train_gpt2_spdk.py, spdk_end_to_end.py, and sink_test.py.
        New code should use the FaF step_counter listener instead.
        """
        if not hasattr(lib, "npu_nvme_trigger_probe"):
            raise RuntimeError("npu_nvme_trigger_probe is not available in the C library.")
        rc = lib.npu_nvme_trigger_probe(self.ctx)
        if rc != 0:
            raise RuntimeError(f"npu_nvme_trigger_probe failed with rc={rc}")

    def set_probe_flag_value(self, value: int):
        """Set probe flag via C to avoid cross-stream races."""
        if not hasattr(lib, "npu_nvme_set_probe_flag_value"):
            raise RuntimeError("npu_nvme_set_probe_flag_value is not available in the C library.")
        if value < 0:
            raise ValueError("probe flag value must be non-negative")
        rc = lib.npu_nvme_set_probe_flag_value(self.ctx, ctypes.c_uint32(value))
        if rc != 0:
            raise RuntimeError(f"npu_nvme_set_probe_flag_value failed with rc={rc}")

    def cleanup(self):
        self.wait_for_io_completion()
        if self.async_thread and self.async_thread.is_alive():
             self.async_thread.join()
        if self.ctx:
            lib.npu_nvme_cleanup(self.ctx)
            self.ctx = None
            
    def _build_local_param_registry(self, models):
        """
        Probe each parameter with a 1-byte aclrtMemcpy to determine
        whether it resides on this rank.  Parameters that fail the probe
        but are very small or named like 'step'/'scale' are rescued
        via asnumpy() as a fallback.
        """
        if not isinstance(models, (list, tuple)): models = [models]
        self.local_valid_param_names = set()

        print(f"[DirectCkpt] Rank {self.rank_id} running dynamic hardware memory probe...", flush=True)
        
        # allocate a 1-byte host buffer as a probe target
        dummy_dst = ctypes.create_string_buffer(1)

        for model in models:
            if model is None or not hasattr(model, "parameters_and_names"): continue
            for name, p in model.parameters_and_names():
                ptr = p._data_ptr() if hasattr(p, "_data_ptr") else 0
                if ptr == 0: continue

                is_valid_on_this_rank = False

                # 1. hardware probe: 1-byte D2H copy to test if the pointer is valid HBM
                if acl_lib is not None:
                    
                    ret = acl_lib.aclrtMemcpy(ctypes.byref(dummy_dst), 1, ctypes.c_void_p(ptr), 1, 2)
                    if ret == 0:
                        is_valid_on_this_rank = True
                
                # 2. fallback: rescue small CPU-resident tensors (e.g. global_step)
                # If the hardware probe fails the pointer is either on another rank or CPU-only
                if not is_valid_on_this_rank:
                    # very small tensors or known names: try asnumpy as last resort
                    if "step" in name.lower() or "scale" in name.lower() or np.prod(p.shape) <= 8:
                        try:
                            _ = p.asnumpy()
                            is_valid_on_this_rank = True
                        except:
                            pass
                
                # 3. record valid parameters
                if is_valid_on_this_rank:
                    self.local_valid_param_names.add(name)

        print(f"[DirectCkpt] Rank {self.rank_id} registry dynamically built: {len(self.local_valid_param_names)} valid params.", flush=True)

    def _prepare_params(self, models):
        # build registry on first call
        if getattr(self, "local_valid_param_names", None) is None:
            self._build_local_param_registry(models)

        if not isinstance(models, (list, tuple)): models = [models]
        params = []
        for model in models:
            if model is None or not hasattr(model, "parameters_and_names"): continue
            for name, p in model.parameters_and_names():
                
                # 1. skip parameters that belong to other ranks
                if name not in self.local_valid_param_names:
                    continue

                ptr = p._data_ptr() if hasattr(p, "_data_ptr") else 0
                if ptr == 0: continue

                # 2. get the local shard size to avoid out-of-bounds access
                dtype_np = np.dtype(ms.dtype_to_nptype(p.dtype))
                local_shape = p.shape
                
                if hasattr(p, "sliced_shape") and p.sliced_shape:
                    local_shape = p.sliced_shape
                elif hasattr(p, "data") and hasattr(p.data, "shape"):
                    if np.prod(p.data.shape) < np.prod(p.shape):
                        local_shape = p.data.shape

                if np.prod(local_shape) == 0: continue
                size = int(np.prod(local_shape)) * dtype_np.itemsize
                
                params.append({
                    "name": name, "ptr": ptr, "size": size,
                    "shape": list(p.shape), "dtype": dtype_np.name, 
                    "np_arr": None,  # all params use the device path (HBM -> NVMe DMA)
                    "param_ref": p
                })
        return params

    def wait_for_io_completion(self):
        """
        Wait for the background I/O thread to finish, so the next
        optimizer step does not overwrite parameters being written.
        """
        if getattr(self, 'io_thread', None) is not None and self.io_thread.is_alive():
            t_wait_start = time.perf_counter()
            print(f"[Timeline][Rank {self.rank_id}] I/O Barrier: Waiting for background SPDK flush to finish...", flush=True)
            
            self.io_thread.join()  # block the main thread (releases GIL) until the C worker finishes
            
            t_wait_end = time.perf_counter()
            print(f"[Timeline][Rank {self.rank_id}] I/O Barrier Cleared! Wait time: {(t_wait_end - t_wait_start):.3f}s", flush=True)
            self.io_thread = None    

    def build_layout(self, models, step: int = 0):
        """
        Build a chunked layout (split by chunk_size, aligned to 4 KB)
        and cache it in self.chunks.  The layout is bound to a specific
        slot identified by the step number.
        """
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
                    raise MemoryError("CRITICAL: DMA write exceeds disk capacity during build_layout.")

                if (nvme_offset_bytes - base_offset_bytes) + aligned_take > self.slot_bytes:
                    raise MemoryError(f"OOM: Tensor {p['name']} exceeds slot size during build_layout.")

                chunks.append({
                    "name": p["name"],
                    "param_ref": p.get("param_ref"),
                    "npu_ptr": p["ptr"] + inner_off,
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
        """
        Register all parameter device pointers with the C-layer background listener.
        Must be called after graph compilation and memory allocation
        (e.g. at the end of step 1, or after an explicit model.build()).
        """
        print(f"[DirectCkpt] Rank {self.rank_id} Registering Layout to NPU-NVMe Background Thread...")
        
        # 1. ensure the NVMe layout has been built
        if not hasattr(self, 'chunks') or len(self.chunks) == 0:
            self.build_layout(model, step=step)
            
        num_items = len(self.chunks)
        if num_items == 0:
            print("[DirectCkpt] Warning: No chunks to register!")
            return

        # 2. 构造 ctypes C 语言兼容的数组
        npu_ptrs = (ctypes.c_void_p * num_items)()
        nvme_offsets = (ctypes.c_uint64 * num_items)()
        sizes = (ctypes.c_size_t * num_items)()

        # 3. 提取真实的物理指针
        for i, chunk in enumerate(self.chunks):
            # 获取 MindSpore 的 Parameter 对象
            param = chunk.get("param_ref")
            if param is None:
                raise ValueError(f"[DirectCkpt] Chunk {i} is missing 'param_ref'.")

            # ---------------------------------------------------------
            # 核心：获取 NPU 显存物理地址
            # ---------------------------------------------------------
            ptr = 0
            
            # 方法 A: 尝试通过用户原有的自定义扩展获取（如果你在 build_layout 中已经拿到了）
            if "npu_ptr" in chunk and chunk["npu_ptr"] != 0:
                ptr = chunk["npu_ptr"]
            else:
                # 方法 B: 使用 MindSpore 原生 API 获取设备物理地址 (支持 MS 2.x)
                try:
                    # 获取内部张量对象的底层设备地址
                    ptr = param.data_ptr() 
                    if ptr is None or ptr == 0:
                        # 兜底：尝试获取其 value() 的物理地址
                        ptr = param.value().data_ptr()
                except AttributeError:
                    raise RuntimeError(f"Failed to get data_ptr from parameter {param.name}. Please ensure you are using MindSpore 2.x.")

            # 致命错误防御：如果指针为 0，说明框架还没有为它分配显存！
            if ptr == 0 or ptr is None:
                raise RuntimeError(
                    f"\n[Fatal Error] Parameter '{param.name}' has an invalid physical address (0x0).\n"
                    f"-> Reason: MindSpore uses lazy memory allocation.\n"
                    f"-> Solution: You MUST call `register_tasks()` AFTER `model.build()` or after the first forward pass."
                )

            # 填充 C 数组
            npu_ptrs[i] = ptr
            nvme_offsets[i] = chunk["nvme_offset"]
            sizes[i] = chunk["size"]

        # 4. 调用我们在 npu_nvme.c 中写好的 C 接口下发至底层后台线程
        rc = lib.npu_nvme_register_tasks(
            self.ctx, npu_ptrs, nvme_offsets, sizes, num_items
        )
        
        if rc != 0:
            raise RuntimeError(f"[DirectCkpt] Failed to register tasks! C API returned {rc}")
            
        print(f"[DirectCkpt] Rank {self.rank_id} Successfully registered {num_items} tensor pointers to SPDK background thread.")

    def save(self, model: ms.nn.Cell, step: int, meta_path: str = "checkpoint_meta.pkl", async_save: bool = False, commit_meta: bool = True):
        if async_save:
            return self.save_async(model, step, meta_path, commit_meta)

        t_start = time.perf_counter()
        
        # -- T_Prep: framework traversal and pointer resolution --
        t_prep_start = time.perf_counter()
        params = self._prepare_params(model)
        t_prep_end = time.perf_counter()
        T_Prep = t_prep_end - t_prep_start

        # -- T_Layout: physical layout, bounds check, C-types assembly --
        base_offset_bytes = self._get_current_slot_base_offset(step)
        print(f"[DirectCkpt] Rank {self.rank_id} saving step {step} to offset {base_offset_bytes / 1024**3:.2f} GB ...", flush=True)

        current_offset = base_offset_bytes
        layout, dev_params, host_params = [], [], []

        for p in params:
            aligned_bytes = int(math.ceil(p["size"] / 4096.0)) * 4096
            
            # Bounds check: disk physical capacity
            if current_offset + aligned_bytes > self.total_bytes:
                raise MemoryError(f"Rank {self.rank_id} CRITICAL: DMA Write will exceed disk physical capacity! Offset: {(current_offset + aligned_bytes)/1024**3:.2f}GB > Total: {self.total_bytes/1024**3:.2f}GB")

            if (current_offset - base_offset_bytes) + aligned_bytes > self.slot_bytes:
                raise MemoryError(f"Rank {self.rank_id} OOM! Tensor {p['name']} exceeds slot size.")

            p_record = {**p, "offset": current_offset}
            layout.append(p_record)

            # Host-resident params go to host_params; device params go to dev_params
            if p.get("np_arr") is not None:
                host_params.append(p_record)
            else:
                dev_params.append(p_record)

            current_offset += aligned_bytes

        total_written = 0
        
        # 预先完成所有 Dev (显存) 的 C 数组组装，绝不占用硬件传输时间
        dev_chunks, dev_sz = build_chunks(dev_params, self.chunk_size)
        if dev_chunks:
            # Debug: write chunk-to-parameter mapping for stall diagnosis
            with open(f"task_mapping_rank_{self.rank_id}.txt", "w") as f:
                for i, chunk in enumerate(dev_chunks):
                    f.write(f"TaskIdx: {i} | Name: {chunk[3]} | Size: {chunk[2].value}\n")

            num_dev = len(dev_chunks)
            c_ptrs_dev = (ctypes.c_void_p * num_dev)()
            c_offs_dev = (ctypes.c_uint64 * num_dev)()
            c_sizes_dev = (ctypes.c_size_t * num_dev)()
            
            for i, (p, o, s, name) in enumerate(dev_chunks): 
                c_ptrs_dev[i], c_offs_dev[i], c_sizes_dev[i] = p, ctypes.c_uint64(o.value), s
            
        # 预先完成所有 Host (内存) 的 C 数组组装
        host_chunks, host_sz = build_chunks(host_params, self.chunk_size)
        if host_chunks:
            num_host = len(host_chunks)
            c_ptrs_host = (ctypes.c_void_p * num_host)()
            c_offs_host = (ctypes.c_uint64 * num_host)()
            c_sizes_host = (ctypes.c_size_t * num_host)()
            
            for i, (p, o, s, name) in enumerate(host_chunks):
                c_ptrs_host[i], c_offs_host[i], c_sizes_host[i] = p, ctypes.c_uint64(o.value), s

        t_layout_end = time.perf_counter()
        T_Layout = t_layout_end - t_prep_end

        # -- T_SPDK: hardware DMA write (no Python overhead) --

        # Ensure all pending device operations are complete before reading buffers
        if hasattr(ms, "runtime") and hasattr(ms.runtime, "synchronize"):
            ms.runtime.synchronize()  
        elif hasattr(ms.hal, "synchronize"):
            ms.hal.synchronize()      
        else:
            ops.functional.depend(model.trainable_params()[0], model.trainable_params()[0])

        # background I/O worker (runs in a separate thread)
        # Pass ctypes arrays by argument to prevent early garbage collection
        def background_io_worker(c_ptrs_d, c_offs_d, c_sizes_d, n_dev, d_sz, 
                                 c_ptrs_h, c_offs_h, c_sizes_h, n_host, h_sz):
            t_spdk_start = time.perf_counter()
            total_written = 0

            # 1. 执行 C 引擎耗时轮询 (在此期间 Python GIL 会被释放)
            if n_dev > 0:
                rc = lib.npu_nvme_write_batch(self.ctx, c_ptrs_d, c_offs_d, c_sizes_d, n_dev)
                if rc != 0: print(f"[Fatal] write_batch failed (rc={rc})")
                total_written += d_sz
                
            if n_host > 0:
                if hasattr(lib, "npu_nvme_write_batch_host"):
                    rc = lib.npu_nvme_write_batch_host(self.ctx, c_ptrs_h, c_offs_h, c_sizes_h, n_host)
                    if rc != 0: print(f"[Fatal] write_batch_host failed (rc={rc})")
                    total_written += h_sz

            t_spdk_end = time.perf_counter()
            T_SPDK = t_spdk_end - t_spdk_start

            # Write metadata ledger to NVMe
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
            
            print(f"\n======================================================")
            print(f"[Timeline][Rank {self.rank_id}] Step {step} | Background SPDK Flush ENDED at {time.perf_counter():.3f}s")
            print(f"[Breakdown][Rank {self.rank_id}] Prep: {T_Prep*1000:.2f}ms | Layout: {T_Layout*1000:.2f}ms | SPDK(H/W): {T_SPDK*1000:.2f}ms | Meta: {T_Meta*1000:.2f}ms")
            print(f"[DirectCkpt][Rank {self.rank_id}] Background Safe Write: {total_written/1024/1024:.2f} MB | BW: {bw:.2f} MB/s")
            print(f"======================================================\n", flush=True)

        # launch the background I/O thread
        num_dev_val = num_dev if dev_chunks else 0
        dev_sz_val = dev_sz if dev_chunks else 0
        num_host_val = num_host if host_chunks else 0
        host_sz_val = host_sz if host_chunks else 0

        # Wait for any previous background I/O before starting a new one
        self.wait_for_io_completion()

        self.io_thread = threading.Thread(
            target=background_io_worker,
            args=(
                c_ptrs_dev if dev_chunks else None, c_offs_dev if dev_chunks else None, c_sizes_dev if dev_chunks else None, num_dev_val, dev_sz_val,
                c_ptrs_host if host_chunks else None, c_offs_host if host_chunks else None, c_sizes_host if host_chunks else None, num_host_val, host_sz_val
            )
        )
        self.io_thread.start()

        # Return immediately; I/O proceeds in background thread
        t_return = time.perf_counter()
        print(f"[Timeline][Rank {self.rank_id}] Step {step} | Python save() dispatched to background thread. Layout cost: {T_Layout*1000:.2f}ms", flush=True)
        
        # Timing data is printed from the background thread; return values are for API compatibility only
        return 0, len(dev_chunks) + len(host_chunks), 0.0, 0.0, {"prep_time": T_Prep, "layout_time": T_Layout}

    def save_async(self, models, step: int, meta_path: str, commit_meta: bool = True):
        t_start = time.time()
        if self.async_thread and self.async_thread.is_alive():
            self.async_thread.join()
            
        t_wait = time.time()
        models_list = [models] if not isinstance(models, (list, tuple)) else models
        host_snapshot = []
        total_bytes_copied = 0
        
        for model in models_list:
            if model is None or not hasattr(model, "parameters_and_names"): continue
            for name, p in model.parameters_and_names():
                if not hasattr(p, "asnumpy"): continue
                np_arr = p.asnumpy()
                total_bytes_copied += np_arr.nbytes
                host_snapshot.append({
                    "name": name, "np_arr": np_arr, "ptr": np_arr.ctypes.data,
                    "size": np_arr.nbytes, "shape": list(p.shape), "dtype": str(np_arr.dtype.name)
                })
        
        t_snapshot = time.time()
        snapshot_time = max(t_snapshot - t_wait, 0.0001)
        snapshot_bw = total_bytes_copied / 1024 / 1024 / snapshot_time
        
        self.async_thread = threading.Thread(
            target=self._background_write_worker,
            args=(host_snapshot, meta_path, total_bytes_copied, step, commit_meta)
        )
        self.async_thread.start()
        
        return total_bytes_copied, len(host_snapshot), snapshot_time, snapshot_bw, {
            "prep_time": t_wait - t_start, "write_time": 0.0,
            "total_time": snapshot_time, "bw_pure": snapshot_bw, "bw_e2e": snapshot_bw
        }

    def _background_write_worker(self, snapshot_params, meta_path, total_size, step, commit_meta):
        t0 = time.time()
        try:
            base_offset_bytes = self._get_current_slot_base_offset(step)
            current_offset = base_offset_bytes
            layout = []
            
            for p in snapshot_params:
                aligned_bytes = int(math.ceil(p["size"] / 4096.0)) * 4096
                p["offset"] = current_offset
                layout.append(p)
                current_offset += aligned_bytes
                
            chunks, total = build_chunks(layout, self.chunk_size)
            
            num = len(chunks)
            c_ptrs = (ctypes.c_void_p * num)()
            c_offs = (ctypes.c_uint64 * num)()
            c_sizes = (ctypes.c_size_t * num)()
            for i, (p, o, s) in enumerate(chunks):
                c_ptrs[i] = p; c_offs[i] = ctypes.c_uint64(o.value); c_sizes[i] = s
                
            if hasattr(lib, "npu_nvme_write_batch_host"):
                rc = lib.npu_nvme_write_batch_host(self.ctx, c_ptrs, c_offs, c_sizes, num)
            else:
                rc = -1
            if rc != 0: return

            self.last_layout = layout

            if commit_meta:
                self._commit_metadata(step, layout)
            
            with open(meta_path, "wb") as f:
                pickle.dump(self.meta_dict, f)
                
        except Exception as e:
            print(f"[AsyncWorker] Exception: {e}", flush=True)

    def load(self, model: ms.nn.Cell, step: int = None, meta_path: str = "checkpoint_meta.pkl"):
        t_start = time.time()

        if step is not None:
            ckpt_key = f"step_{step}"
        else:
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.meta_dict = pickle.load(f)
            valid_keys = [k for k in self.meta_dict.get("checkpoints", {}).keys() if "complete" not in k]
            if not valid_keys:
                raise FileNotFoundError("No checkpoints found in Meta Dictionary!")
            ckpt_key = sorted(valid_keys, key=lambda x: int(x.split('_')[1]))[-1]

        if ckpt_key not in self.meta_dict.get("checkpoints", {}):
            raise FileNotFoundError(f"Checkpoint for {ckpt_key} not found!")

        meta_info = self.meta_dict["checkpoints"][ckpt_key]
        chunk_size = min(meta_info.get("chunk_size", self.chunk_size), self.chunk_size)

        t_rebuild = time.time()
        dev_chunks, host_chunks, buffers = rebuild_chunks_from_meta(model, meta_info["params"], chunk_size)
        t_rebuild_end = time.time()

        t0 = time.time()
        total_read = 0

        # Load NPU Data
        if dev_chunks:
            num = len(dev_chunks)
            c_ptrs = (ctypes.c_void_p * num)()
            c_offs = (ctypes.c_uint64 * num)()
            c_sizes = (ctypes.c_size_t * num)()
            for i, (p, o, s) in enumerate(dev_chunks):
                c_ptrs[i], c_offs[i], c_sizes[i] = p, ctypes.c_uint64(o.value), s

            rc = lib.npu_nvme_read_batch(self.ctx, c_ptrs, c_offs, c_sizes, num)
            if rc != 0: raise RuntimeError("read_batch failed")
            total_read += sum(c[2].value for c in dev_chunks)

        # Note: host_chunks are reconstructed from metadata via
        # rebuild_chunks_from_meta and already have valid np_arr data;
        # no separate NVMe read needed for host-resident small params.

        t1 = time.time()
        t_update = time.time()

        for buf in buffers:
            if buf.get("use_dev", False): continue
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
            "bw_pure": total_read / 1024 / 1024 / pure_read_time if pure_read_time > 0 else 0,
            "bw_e2e": bw_e2e
        }
        return total_read, len(dev_chunks) + len(host_chunks), total_time, bw_e2e, stats

    def wait_async_io(self):
        if hasattr(self, 'async_thread') and self.async_thread and self.async_thread.is_alive():
            self.async_thread.join()

    def commit_last_layout(self, step: int):
        if self.rank_id == 0 and hasattr(self, 'last_layout'):
            self._commit_metadata(step, self.last_layout)

    def close(self):
        if not getattr(self, '_closed', True) and hasattr(self, 'ctx') and self.ctx:
            print(f"[DirectCkpt] Rank {self.rank_id} safely tearing down NPUNVME context...", flush=True)
            self.cleanup()
            self._closed = True

    # -- Delta frame I/O (incremental checkpoint) --
    def delta_init(self, slot_size_mb: int = 256, slot_count: int = 128):
        """Initialise the delta ring-buffer layout on disk.

        Args:
            slot_size_mb: size of each delta slot in MB
            slot_count:   number of slots in the ring buffer
        """
        slot_bytes = slot_size_mb * 1024 * 1024
        if not hasattr(lib, "npu_nvme_delta_init"):
            raise RuntimeError("C library missing npu_nvme_delta_init — rebuild required.")
        rc = lib.npu_nvme_delta_init(self.ctx, ctypes.c_uint64(slot_bytes),
                                     ctypes.c_uint32(slot_count))
        if rc != 0:
            raise RuntimeError(f"Delta init failed (rc={rc})")
        self._delta_slot_size = slot_bytes
        self._delta_slot_count = slot_count
        self._delta_next_slot = 0
        self._delta_step_map = {}
        self._delta_frame_sizes = []
        # Ensure delta chain storage in meta_dict
        if "delta_chain" not in self.meta_dict:
            self.meta_dict["delta_chain"] = {}
        # Initialise _dump_meta_pkl so delta_save() can auto-save the pickle
        # even when called before the first FULL checkpoint commit.
        if not hasattr(self, '_dump_meta_pkl'):
            import pickle as _pickle
            os.makedirs(os.path.dirname(self._meta_pkl), exist_ok=True)
            def _dump_meta_pkl():
                with open(self._meta_pkl, "wb") as _f:
                    _pickle.dump(self.meta_dict, _f)
            self._dump_meta_pkl = _dump_meta_pkl
        print(f"[DirectCkpt] Delta area initialized: {slot_count} slots × {slot_size_mb}MB", flush=True)

    def delta_save(self, step: int, block_patches: list, small_patches: list):
        """Serialise and write one delta frame to the NVMe delta ring.

        Args:
            step:          training step number
            block_patches: large-parameter blocks, each a dict
                           {layer_id, name, block_idx, int8_data, scale}
            small_patches: small parameters, same dict format
        Returns:
            slot_idx: index of the written slot
        """

        if not hasattr(self, '_delta_slot_count'):
            self.delta_init()

        # Wait for any background full-checkpoint I/O to finish.
        # Both full and delta I/O share the same SPDK qpair; concurrent writes would fail.
        self.wait_for_io_completion()
        # Also join the save_async thread if present
        self.wait_async_io()

        frame = pack_delta_frame(step, block_patches, small_patches)
        total_bytes = len(frame)

        if total_bytes > self._delta_slot_size:
            raise RuntimeError(f"Delta frame {total_bytes} bytes > slot {self._delta_slot_size}")

        slot_idx = self._delta_next_slot % self._delta_slot_count

        buf = ctypes.create_string_buffer(frame, total_bytes)
        rc = lib.npu_nvme_write_delta(self.ctx, ctypes.c_int(slot_idx),
                                       ctypes.c_void_p(ctypes.addressof(buf)),
                                       ctypes.c_uint32(total_bytes))
        if rc != 0:
            raise RuntimeError(f"Delta write failed at slot {slot_idx} (rc={rc})")

        self._delta_step_map[step] = slot_idx
        self._delta_next_slot += 1
        self._delta_frame_sizes.append(total_bytes)

        # Update meta in-memory (非阻塞 — full commit later)
        self.meta_dict["delta_chain"][f"step_{step}"] = {
            "type": "DELTA",
            "slot": slot_idx,
            "frame_size": total_bytes,
            "n_blocks": len(block_patches),
            "n_small": len(small_patches),
        }
        # Auto-save pickle after each delta write for crash-safe recovery
        if hasattr(self, '_dump_meta_pkl'):
            self._dump_meta_pkl()

        return slot_idx

    def delta_load_slot(self, slot_idx: int):
        """Read a single delta frame from NVMe.

        Returns (step_id, block_patches, small_patches)."""

        if not hasattr(self, '_delta_slot_size'):
            raise RuntimeError("Delta not initialized. Call delta_init() first.")

        buf = ctypes.create_string_buffer(self._delta_slot_size)
        rc = lib.npu_nvme_read_delta(self.ctx, ctypes.c_int(slot_idx),
                                      ctypes.c_void_p(ctypes.addressof(buf)),
                                      ctypes.c_uint32(self._delta_slot_size))
        if rc != 0:
            raise RuntimeError(f"Delta read failed at slot {slot_idx} (rc={rc})")

        # Read succeeded; parse header to get actual frame size
        magic, total_sz = struct.unpack_from("<I", buf.raw, 0)[0], struct.unpack_from("<I", buf.raw, 16)[0]
        if magic != DELTA_MAGIC:
            raise RuntimeError(f"Delta read at slot {slot_idx}: bad magic 0x{magic:08x}")
        frame = buf.raw[:total_sz]
        step_id, blocks, smalls = unpack_delta_frame(frame)
        return step_id, blocks, smalls

    def delta_load_chain(self, from_step: int, to_step: int):
        """Read all delta frames in the range [from_step+1, to_step].

        Returns a list of (step, blocks, smalls) tuples."""
        chain = []
        for s in range(from_step + 1, to_step + 1):
            key = f"step_{s}"
            if key not in self.meta_dict.get("delta_chain", {}):
                raise FileNotFoundError(f"Delta frame for step {s} not found in metadata")
            slot = self.meta_dict["delta_chain"][key]["slot"]
            sid, blocks, smalls = self.delta_load_slot(slot)
            if sid != s:
                print(f"[DirectCkpt] WARNING: delta slot {slot} step_id={sid} != expected {s}")
            chain.append((sid, blocks, smalls))
        return chain

    def _find_nearest_full(self, target_step: int):
        """Return the nearest FULL checkpoint step <= target_step."""

        def _scan():
            best = None
            for k, v in self.meta_dict.get("checkpoints", {}).items():
                if not k.startswith("step_"):
                    continue
                # meta entries without explicit 'type' field are legacy FULL checkpoints
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

        # Fallback: the in-memory dict may not have caught a commit from
        # the async background I/O thread.  Re-read metadata from NVMe.
        print("[DirectCkpt] Re-reading meta from NVMe to find FULL checkpoint...", flush=True)
        self._mount_filesystem()
        best = _scan()
        if best is not None:
            print(f"[DirectCkpt] Found FULL checkpoint step_{best} after disk re-read.", flush=True)
            return best

        raise FileNotFoundError(f"No FULL checkpoint found ≤ step {target_step}")

    def recover(self, model: "ms.nn.Cell", target_step: int):
        """Full incremental recovery: FULL ckpt + delta chain -> model weights.

        Steps:
          1. locate the nearest FULL checkpoint <= target_step
          2. npu_nvme_read_batch to restore weights to device
          3. if target_step > base_step: read delta chain, apply patches on CPU
          4. write reconstructed weights back to device
        Returns: dict with {base_step, n_deltas, total_time, ...}
        """
        t_start = time.perf_counter()

        base_step = self._find_nearest_full(target_step)
        print(f"[DirectCkpt] Recover target=step_{target_step}, base=FULL step_{base_step}", flush=True)

        # Load FULL checkpoint
        self.load(model, step=base_step)

        if base_step == target_step:
            dt = time.perf_counter() - t_start
            print(f"[DirectCkpt] Recovery done (FULL only, no deltas): {dt:.2f}s", flush=True)
            return {"base_step": base_step, "n_deltas": 0, "total_time": dt}

        # Read delta chain
        chain = self.delta_load_chain(base_step, target_step)
        print(f"[DirectCkpt] Loaded {len(chain)} delta frames (step {base_step+1}→{target_step})", flush=True)

        # Apply delta patches on CPU
        # Record each parameter's MindSpore dtype before pulling to host
        param_dtypes = {}
        host_weights = {}
        for name, p in model.parameters_and_names():
            param_dtypes[name] = p.dtype
            host_weights[name] = p.value().asnumpy().copy()

        # Apply delta chain sequentially
        for sid, blocks, smalls in chain:
            host_weights = apply_delta_patches(host_weights, blocks, smalls, self.chunk_size)

        # Write back to device (preserve original dtype per parameter)
        for name, p in model.parameters_and_names():
            if name in host_weights:
                ms_dtype = param_dtypes.get(name, ms.float16)
                np_dtype = np.dtype(ms.dtype_to_nptype(ms_dtype))
                p.set_data(Tensor(host_weights[name].astype(np_dtype), ms_dtype))

        dt = time.perf_counter() - t_start
        n_deltas = len(chain)
        print(f"[DirectCkpt] Recovery complete: {n_deltas} deltas applied, total {dt:.2f}s", flush=True)
        return {"base_step": base_step, "n_deltas": n_deltas, "total_time": dt}

    def __del__(self):
        self.close()