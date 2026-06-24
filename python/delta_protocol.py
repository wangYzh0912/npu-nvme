"""I3 Delta frame binary protocol — serialization, deserialization, and patching.

Frame layout:
  Header (28 bytes, zero-padded to FRAME_HEADER_SIZE):
    magic(u32) + step_id(u32) + n_blocks(u32) + n_small(u32) + total_sz(u32) + checksum(u32)
  Block Records (variable):
    layer_id(i16) + name_len(u16) + name + block_idx(i32) + scale(f32) + data_len(i32) + data(i8[])
  Small Records (variable):
    layer_id(i16) + name_len(u16) + name + scale(f32) + data_len(i32) + data(i8[])
"""

import os
import struct

import numpy as np

from disk_layout import DELTA_MAGIC, FRAME_HEADER_SIZE


# -- Serialization -------------------------------------------------------------

def pack_delta_frame(step_id, block_patches, small_patches):
    """Serialize I3 delta to binary frame.  Returns bytes."""
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

    struct.pack_into(f"<I I I I I I", buf, 0, DELTA_MAGIC, step_id,
                     len(block_patches), len(small_patches), total_sz, checksum)

    frame = bytes(buf) + bytes(payload)
    return frame


# -- Deserialization -----------------------------------------------------------

def unpack_delta_frame(frame_bytes):
    """Deserialize binary frame back to (step_id, block_patches, small_patches)."""
    if len(frame_bytes) < FRAME_HEADER_SIZE:
        raise ValueError(f"Frame too short: {len(frame_bytes)} < {FRAME_HEADER_SIZE}")

    magic, step_id, n_blocks, n_small, total_sz, checksum = \
        struct.unpack_from("<I I I I I I", frame_bytes, 0)

    if magic != DELTA_MAGIC:
        raise ValueError(
            f"Invalid delta magic: 0x{magic:08x} (expected 0x{DELTA_MAGIC:08x})")

    pos = FRAME_HEADER_SIZE

    def _read_block(pos):
        lid, name_len = struct.unpack_from("<hH", frame_bytes, pos)
        pos += 4
        name = frame_bytes[pos:pos + name_len].decode('utf-8')
        pos += name_len
        bidx, scale = struct.unpack_from("<i f", frame_bytes, pos)
        pos += 8
        data_len = struct.unpack_from("<i", frame_bytes, pos)[0]
        pos += 4
        i8_data = np.frombuffer(frame_bytes[pos:pos + data_len], dtype=np.int8)
        pos += data_len
        return pos, {"layer_id": lid, "name": name, "block_idx": bidx,
                     "int8_data": i8_data, "scale": scale}

    def _read_small(pos):
        lid, name_len = struct.unpack_from("<hH", frame_bytes, pos)
        pos += 4
        name = frame_bytes[pos:pos + name_len].decode('utf-8')
        pos += name_len
        scale = struct.unpack_from("<f", frame_bytes, pos)[0]
        pos += 4
        data_len = struct.unpack_from("<i", frame_bytes, pos)[0]
        pos += 4
        i8_data = np.frombuffer(frame_bytes[pos:pos + data_len], dtype=np.int8)
        pos += data_len
        return pos, {"layer_id": lid, "name": name,
                     "int8_data": i8_data, "scale": scale}

    block_patches, small_patches = [], []
    for _ in range(n_blocks):
        pos, bp = _read_block(pos)
        block_patches.append(bp)
    for _ in range(n_small):
        pos, sp = _read_small(pos)
        small_patches.append(sp)

    return step_id, block_patches, small_patches


# -- Patch application ---------------------------------------------------------

def apply_delta_patches(init_weights, block_patches, small_patches, block_size):
    """Apply delta patches to a weight dictionary on CPU.

    Block patches write into the flattened weight at position
    bidx * block_size.  Small patches replace the entire weight.

    Returns a modified copy of init_weights.
    """
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
        wv[start:end] = fp32[:end - start]
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


# -- Filesystem-backed delta writer (no SPDK dependency) -----------------------

class FileDeltaWriter:
    """Filesystem-backed delta writer.

    Uses a ring of files under delta_dir.  No SPDK/NVMe dependency.
    """

    def __init__(self, delta_dir, delta_slot_count=128,
                 delta_slot_size=256 * 1024 * 1024):
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
            "total_mb": sum(self.frame_sizes) / (1024 * 1024),
            "avg_kb": (sum(self.frame_sizes) / max(len(self.frame_sizes), 1)) / 1024,
            "max_kb": max(self.frame_sizes) / 1024 if self.frame_sizes else 0,
            "slots_used": self.next_slot,
            "slot_capacity": self.slot_count,
            "backend": "file",
        }

    def close(self):
        pass
