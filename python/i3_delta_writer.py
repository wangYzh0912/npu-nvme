#!/usr/bin/env python3
"""
I3 Delta Writer: SPDK incremental checkpoint I/O layer.
=========================================================
Python-side serialization + SPDK write/read for I3 delta frames.

Usage:
  from i3_delta_writer import I3DeltaWriter, I3DeltaRecovery

  writer = I3DeltaWriter(ctx, delta_slot_size=256*1024*1024, delta_slot_count=128)
  slot_idx = writer.write_frame(step=42, block_patches=..., small_patches=...)
  # ...
  recovery = I3DeltaRecovery(writer)
  w_rec = recovery.recover(target_step=42, full_checkpoint_path=...)
"""
import os, sys, time, json, math, struct, copy, ctypes
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LIB_PATH = os.path.join(REPO, "build_out/lib/libnpu_nvme.so")

DELTA_MAGIC = 0x414C5444   # "DLTA"
FRAME_HEADER_SIZE = 4096

# ═══════════════════════════════════════════════════════════════════
# Binary Frame Serialization
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# SPDK Writer
# ═══════════════════════════════════════════════════════════════════

class I3DeltaWriter:
    """SPDK-backed incremental checkpoint writer.

    Uses host-side buffer → npu_nvme_write_delta → NVMe delta ring.
    """
    def __init__(self, ctx, delta_slot_size=256*1024*1024, delta_slot_count=128):
        self.ctx = ctx
        self.slot_size = delta_slot_size
        self.slot_count = delta_slot_count

        # Load C library
        self.lib = ctypes.CDLL(_LIB_PATH)

        # npu_nvme_delta_init
        self.lib.npu_nvme_delta_init.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
        self.lib.npu_nvme_delta_init.restype = ctypes.c_int

        # npu_nvme_write_delta
        self.lib.npu_nvme_write_delta.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        self.lib.npu_nvme_write_delta.restype = ctypes.c_int

        # npu_nvme_read_delta
        self.lib.npu_nvme_read_delta.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        self.lib.npu_nvme_read_delta.restype = ctypes.c_int

        # npu_nvme_delta_get_area_offset
        self.lib.npu_nvme_delta_get_area_offset.argtypes = [ctypes.c_void_p]
        self.lib.npu_nvme_delta_get_area_offset.restype = ctypes.c_uint64

        # Init delta layout
        rc = self.lib.npu_nvme_delta_init(ctx, delta_slot_size, delta_slot_count)
        if rc != 0:
            raise RuntimeError(f"Delta init failed (rc={rc})")

        self.next_slot = 0
        self.step_map = {}   # step_id → slot_idx

        # Frame size stats
        self.frame_sizes = []

    @property
    def area_offset(self):
        return self.lib.npu_nvme_delta_get_area_offset(self.ctx)

    def write_frame(self, step_id, block_patches, small_patches):
        """Serialize and write one delta frame to next slot. Returns slot_idx."""
        frame = pack_delta_frame(step_id, block_patches, small_patches)
        total_bytes = len(frame)

        if total_bytes > self.slot_size:
            raise ValueError(f"Frame {total_bytes} bytes > slot {self.slot_size} bytes!")

        slot_idx = self.next_slot % self.slot_count

        buf = ctypes.create_string_buffer(frame, total_bytes)
        rc = self.lib.npu_nvme_write_delta(self.ctx, slot_idx,
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
        actual = self.lib.npu_nvme_read_delta(
            self.ctx, slot_idx,
            ctypes.c_void_p(ctypes.addressof(buf)),
            self.slot_size)
        if actual <= 0:
            raise RuntimeError(f"Delta read failed at slot {slot_idx} (rc={actual})")

        # Reconstruct full frame
        frame = buf.raw[:actual]
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


# ═══════════════════════════════════════════════════════════════════
# FileSystem Fallback Writer (bypasses SPDK when hugepages unavailable)
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Recovery
# ═══════════════════════════════════════════════════════════════════

def apply_delta_patches(init_weights, block_patches, small_patches, block_size):
    """Apply delta patches to a weight dictionary. Returns modified copy."""
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


print("[I3DeltaWriter] Module loaded.")
