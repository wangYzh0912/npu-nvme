"""I3 Delta frame binary protocol — serialization, deserialization, and patching.

Frame layout:
  Header (28 bytes, zero-padded to FRAME_HEADER_SIZE):
    magic(u32) + step_id(u32) + n_blocks(u32) + n_small(u32) + total_sz(u32) + checksum(u32)
  Block Records (variable):
    layer_id(i16) + name_len(u16) + name + block_idx(i32) + scale(f32) + data_len(i32) + data(i8[])
  Small Records (variable):
    layer_id(i16) + name_len(u16) + name + scale(f32) + data_len(i32) + data(i8[])
"""

import binascii
import os
import struct

import numpy as np

from disk_layout import DELTA_MAGIC, FRAME_HEADER_SIZE


# Version 3 is the S2/R0 replacement protocol.  Versions 1 and 2 below are
# retained for compatibility with the already validated additive frame tests.
S2_FRAME_VERSION = 3
S2_FRAME_FLAGS = 0x3  # replacement records + manifest digest present


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
    struct.pack_into("<HH", buf, 24, 1, 0)  # legacy INT8 encoding

    frame = bytes(buf) + bytes(payload)
    return frame


def pack_lossless_delta_frame(step_id, block_patches, small_patches,
                              base_generation=0, generation=0):
    """Serialize the R0 lossless FP16 Delta baseline.

    Records contain raw FP16 *deltas* and are applied additively.  The frame
    carries the FULL generation it is based on, so a stale frame cannot be
    silently applied to another checkpoint lineage.
    """
    buf = bytearray(FRAME_HEADER_SIZE)
    payload = bytearray()

    for patch in block_patches:
        name = patch["name"].encode("utf-8")
        data = np.asarray(patch.get("fp16_data", patch.get("data")),
                          dtype=np.dtype("<f2")).reshape(-1)
        payload += struct.pack("<hH", int(patch.get("layer_id", 0)), len(name))
        payload += name
        payload += struct.pack("<iII", int(patch.get("block_idx", 0)),
                               int(patch.get("element_offset", 0)), len(data))
        payload += struct.pack("<I", data.nbytes)
        payload += data.tobytes()

    for patch in small_patches:
        name = patch["name"].encode("utf-8")
        data = np.asarray(patch.get("fp16_data", patch.get("data")),
                          dtype=np.dtype("<f2")).reshape(-1)
        payload += struct.pack("<hH", int(patch.get("layer_id", 0)), len(name))
        payload += name
        payload += struct.pack("<I", len(data))
        payload += struct.pack("<I", data.nbytes)
        payload += data.tobytes()

    total_sz = FRAME_HEADER_SIZE + len(payload)
    checksum = binascii.crc32(payload) & 0xFFFFFFFF
    struct.pack_into("<IIIIII", buf, 0, DELTA_MAGIC, step_id,
                     len(block_patches), len(small_patches), total_sz, checksum)
    struct.pack_into("<HHQQ", buf, 24, 2, 1, base_generation, generation)
    return bytes(buf) + bytes(payload)


def pack_s2_replacement_frame(step_id, block_patches, small_patches,
                              base_generation=0, generation=0,
                              manifest_digest=""):
    """Pack an S2 frame containing native replacement values.

    Every block record carries its stable manifest block ID and its
    parameter-local element offset. Values are encoded in their native
    little-endian NumPy dtype; no scale or additive interpretation is used.
    """
    if step_id < 0 or base_generation < 0 or generation <= base_generation:
        raise ValueError("invalid S2 step or generation")
    digest = bytes.fromhex(manifest_digest) if manifest_digest else bytes(32)
    if len(digest) != 32:
        raise ValueError("manifest_digest must be a SHA-256 hex digest")
    payload = bytearray()

    def encode_value(value, dtype_name):
        dtype = np.dtype(dtype_name).newbyteorder("<")
        array = np.asarray(value, dtype=dtype).reshape(-1)
        return array, array.tobytes()

    for patch in block_patches:
        name = patch["name"].encode("utf-8")
        dtype_name = str(patch["dtype"])
        value, raw = encode_value(patch["value"], dtype_name)
        if value.size != int(patch["element_count"]):
            raise ValueError("S2 block value length mismatch")
        dtype = dtype_name.encode("ascii")
        # layer_id uses signed 32 bits.  The 12-byte record header is kept
        # stable while allowing the manifest's special negative IDs for
        # embeddings/layer norms/unknown layers.
        payload += struct.pack("<IihH", int(patch["block_id"]),
                               int(patch["layer_id"]), len(name), len(dtype))
        payload += name + dtype
        payload += struct.pack("<III", int(patch["block_idx"]),
                               int(patch["element_offset"]), int(value.size))
        payload += struct.pack("<I", len(raw)) + raw

    for patch in small_patches:
        name = patch["name"].encode("utf-8")
        dtype_name = str(patch["dtype"])
        value, raw = encode_value(patch["value"], dtype_name)
        if value.size != int(patch["element_count"]):
            raise ValueError("S2 small value length mismatch")
        dtype = dtype_name.encode("ascii")
        payload += struct.pack("<hH", int(patch["layer_id"]), len(name))
        payload += name
        payload += struct.pack("<H", len(dtype)) + dtype
        payload += struct.pack("<II", int(value.size), len(raw)) + raw

    total_sz = FRAME_HEADER_SIZE + len(payload)
    checksum = binascii.crc32(payload) & 0xFFFFFFFF
    if total_sz > 0xFFFFFFFF:
        raise ValueError("S2 frame too large")
    buf = bytearray(FRAME_HEADER_SIZE)
    struct.pack_into("<IIIIII", buf, 0, DELTA_MAGIC, int(step_id),
                     len(block_patches), len(small_patches), total_sz, checksum)
    struct.pack_into("<HHQQ", buf, 24, S2_FRAME_VERSION, S2_FRAME_FLAGS,
                     int(base_generation), int(generation))
    buf[48:80] = digest
    return bytes(buf) + bytes(payload)


# -- Deserialization -----------------------------------------------------------

def _unpack_lossless_delta_frame(frame_bytes, step_id, n_blocks, n_small,
                                 total_sz, checksum):
    payload = frame_bytes[FRAME_HEADER_SIZE:total_sz]
    if (binascii.crc32(payload) & 0xFFFFFFFF) != checksum:
        raise ValueError("Delta CRC mismatch")
    _version, flags, base_generation, generation = struct.unpack_from(
        "<HHQQ", frame_bytes, 24)
    pos = FRAME_HEADER_SIZE

    def require(count, label):
        nonlocal pos
        if count < 0 or pos + count > total_sz:
            raise ValueError(f"Truncated lossless delta {label}")

    blocks, smalls = [], []
    for _ in range(n_blocks):
        require(4, "block header")
        lid, name_len = struct.unpack_from("<hH", frame_bytes, pos)
        pos += 4
        require(name_len, "block name")
        name = frame_bytes[pos:pos + name_len].decode("utf-8")
        pos += name_len
        require(12, "block metadata")
        block_idx, element_offset, element_count = struct.unpack_from(
            "<iII", frame_bytes, pos)
        pos += 12
        require(4, "block byte length")
        data_len = struct.unpack_from("<I", frame_bytes, pos)[0]
        pos += 4
        if data_len != element_count * 2:
            raise ValueError("lossless block byte length mismatch")
        require(data_len, "block data")
        data = np.frombuffer(frame_bytes[pos:pos + data_len],
                             dtype=np.dtype("<f2")).copy()
        pos += data_len
        blocks.append({"layer_id": lid, "name": name,
                       "block_idx": block_idx,
                       "element_offset": element_offset,
                       "element_count": element_count,
                       "fp16_data": data, "encoding": "fp16"})

    for _ in range(n_small):
        require(4, "small header")
        lid, name_len = struct.unpack_from("<hH", frame_bytes, pos)
        pos += 4
        require(name_len, "small name")
        name = frame_bytes[pos:pos + name_len].decode("utf-8")
        pos += name_len
        require(8, "small metadata")
        element_count, data_len = struct.unpack_from("<II", frame_bytes, pos)
        pos += 8
        if data_len != element_count * 2:
            raise ValueError("lossless small byte length mismatch")
        require(data_len, "small data")
        data = np.frombuffer(frame_bytes[pos:pos + data_len],
                             dtype=np.dtype("<f2")).copy()
        pos += data_len
        smalls.append({"layer_id": lid, "name": name,
                       "element_count": element_count,
                       "fp16_data": data, "encoding": "fp16"})

    if pos != total_sz:
        raise ValueError(f"Delta contains {total_sz - pos} unparsed bytes")
    info = {"version": _version, "flags": flags,
            "base_generation": base_generation, "generation": generation}
    return step_id, blocks, smalls, info


def unpack_delta_frame_with_meta(frame_bytes):
    """Deserialize a frame and return ``(step, blocks, smalls, info)``."""
    if len(frame_bytes) < FRAME_HEADER_SIZE:
        raise ValueError(f"Frame too short: {len(frame_bytes)} < {FRAME_HEADER_SIZE}")
    magic, step_id, n_blocks, n_small, total_sz, checksum = \
        struct.unpack_from("<I I I I I I", frame_bytes, 0)
    if magic != DELTA_MAGIC:
        raise ValueError(f"Invalid delta magic: 0x{magic:08x}")
    if total_sz < FRAME_HEADER_SIZE or total_sz > len(frame_bytes):
        raise ValueError(f"Invalid delta frame size: {total_sz}")
    version, flags = struct.unpack_from("<HH", frame_bytes, 24)
    if version == S2_FRAME_VERSION and flags & S2_FRAME_FLAGS == S2_FRAME_FLAGS:
        return unpack_s2_replacement_frame(frame_bytes)
    if version == 2 and flags & 1:
        return _unpack_lossless_delta_frame(
            frame_bytes, step_id, n_blocks, n_small, total_sz, checksum)
    return _unpack_delta_frame_int8(frame_bytes)


def unpack_s2_replacement_frame(frame_bytes):
    """Unpack and validate an S2 replacement frame."""
    if len(frame_bytes) < FRAME_HEADER_SIZE:
        raise ValueError("Frame too short")
    magic, step_id, n_blocks, n_small, total_sz, checksum = struct.unpack_from(
        "<IIIIII", frame_bytes, 0)
    if magic != DELTA_MAGIC or total_sz < FRAME_HEADER_SIZE or total_sz > len(frame_bytes):
        raise ValueError("invalid S2 frame header")
    version, flags, base_generation, generation = struct.unpack_from(
        "<HHQQ", frame_bytes, 24)
    if version != S2_FRAME_VERSION or flags & S2_FRAME_FLAGS != S2_FRAME_FLAGS:
        raise ValueError("not an S2 replacement frame")
    payload = frame_bytes[FRAME_HEADER_SIZE:total_sz]
    if (binascii.crc32(payload) & 0xFFFFFFFF) != checksum:
        raise ValueError("S2 CRC mismatch")
    manifest_digest = frame_bytes[48:80].hex()
    pos = FRAME_HEADER_SIZE

    def take(count, label):
        nonlocal pos
        if count < 0 or pos + count > total_sz:
            raise ValueError(f"truncated S2 {label}")
        value = frame_bytes[pos:pos + count]
        pos += count
        return value

    blocks, smalls = [], []
    for _ in range(n_blocks):
        block_id, layer_id, name_len, dtype_len = struct.unpack(
            "<IihH", take(12, "block header"))
        name = take(name_len, "block name").decode("utf-8")
        dtype = take(dtype_len, "block dtype").decode("ascii")
        block_idx, element_offset, element_count = struct.unpack(
            "<III", take(12, "block location"))
        data_len = struct.unpack("<I", take(4, "block length"))[0]
        raw = take(data_len, "block data")
        np_dtype = np.dtype(dtype).newbyteorder("<")
        if data_len != element_count * np_dtype.itemsize:
            raise ValueError("S2 block byte length mismatch")
        blocks.append({"block_id": block_id, "layer_id": layer_id, "name": name,
                       "block_idx": block_idx, "element_offset": element_offset,
                       "element_count": element_count, "dtype": dtype,
                       "value": np.frombuffer(raw, dtype=np_dtype).copy()})

    for _ in range(n_small):
        layer_id, name_len = struct.unpack("<hH", take(4, "small header"))
        name = take(name_len, "small name").decode("utf-8")
        dtype_len = struct.unpack("<H", take(2, "small dtype length"))[0]
        dtype = take(dtype_len, "small dtype").decode("ascii")
        element_count, data_len = struct.unpack("<II", take(8, "small length"))
        raw = take(data_len, "small data")
        np_dtype = np.dtype(dtype).newbyteorder("<")
        if data_len != element_count * np_dtype.itemsize:
            raise ValueError("S2 small byte length mismatch")
        smalls.append({"layer_id": layer_id, "name": name,
                       "element_count": element_count, "dtype": dtype,
                       "value": np.frombuffer(raw, dtype=np_dtype).copy()})
    if pos != total_sz:
        raise ValueError("S2 frame contains unparsed bytes")
    return step_id, blocks, smalls, {
        "version": version, "flags": flags,
        "base_generation": base_generation, "generation": generation,
        "manifest_digest": manifest_digest,
    }


def unpack_delta_frame(frame_bytes):
    """Deserialize binary frame back to (step_id, block_patches, small_patches)."""
    result = unpack_delta_frame_with_meta(frame_bytes)
    return result[:3]


def _unpack_delta_frame_int8(frame_bytes):
    if len(frame_bytes) < FRAME_HEADER_SIZE:
        raise ValueError(f"Frame too short: {len(frame_bytes)} < {FRAME_HEADER_SIZE}")

    magic, step_id, n_blocks, n_small, total_sz, checksum = \
        struct.unpack_from("<I I I I I I", frame_bytes, 0)

    if magic != DELTA_MAGIC:
        raise ValueError(
            f"Invalid delta magic: 0x{magic:08x} (expected 0x{DELTA_MAGIC:08x})")

    if total_sz < FRAME_HEADER_SIZE or total_sz > len(frame_bytes):
        raise ValueError(
            f"Invalid delta frame size: header={total_sz}, buffer={len(frame_bytes)}")

    payload = frame_bytes[FRAME_HEADER_SIZE:total_sz]
    actual_checksum = sum(payload) & 0xFFFFFFFF
    if actual_checksum != checksum:
        raise ValueError(
            f"Delta checksum mismatch: stored={checksum}, actual={actual_checksum}")

    # Even an empty-name record has a fixed-size prefix.  Reject impossible
    # counts before entering record loops over untrusted metadata.
    max_records = len(payload) // 12
    if n_blocks + n_small > max_records:
        raise ValueError(
            f"Delta record count exceeds payload capacity: "
            f"{n_blocks + n_small} > {max_records}")

    pos = FRAME_HEADER_SIZE

    def _read_block(pos):
        _require_at(pos, 4, "block header")
        lid, name_len = struct.unpack_from("<hH", frame_bytes, pos)
        pos += 4
        _require_at(pos, name_len, "block name")
        try:
            name = frame_bytes[pos:pos + name_len].decode('utf-8')
        except UnicodeDecodeError as error:
            raise ValueError("Invalid UTF-8 in delta block name") from error
        pos += name_len
        _require_at(pos, 12, "block metadata")
        bidx, scale = struct.unpack_from("<i f", frame_bytes, pos)
        pos += 8
        data_len = struct.unpack_from("<i", frame_bytes, pos)[0]
        pos += 4
        if data_len < 0:
            raise ValueError(f"Negative delta block data length: {data_len}")
        _require_at(pos, data_len, "block data")
        i8_data = np.frombuffer(frame_bytes[pos:pos + data_len], dtype=np.int8)
        pos += data_len
        return pos, {"layer_id": lid, "name": name, "block_idx": bidx,
                     "int8_data": i8_data, "scale": scale}

    def _read_small(pos):
        _require_at(pos, 4, "small-patch header")
        lid, name_len = struct.unpack_from("<hH", frame_bytes, pos)
        pos += 4
        _require_at(pos, name_len, "small-patch name")
        try:
            name = frame_bytes[pos:pos + name_len].decode('utf-8')
        except UnicodeDecodeError as error:
            raise ValueError("Invalid UTF-8 in delta small-patch name") from error
        pos += name_len
        _require_at(pos, 8, "small-patch metadata")
        scale = struct.unpack_from("<f", frame_bytes, pos)[0]
        pos += 4
        data_len = struct.unpack_from("<i", frame_bytes, pos)[0]
        pos += 4
        if data_len < 0:
            raise ValueError(f"Negative delta small-patch data length: {data_len}")
        _require_at(pos, data_len, "small-patch data")
        i8_data = np.frombuffer(frame_bytes[pos:pos + data_len], dtype=np.int8)
        pos += data_len
        return pos, {"layer_id": lid, "name": name,
                     "int8_data": i8_data, "scale": scale}

    def _require_at(offset, count, label):
        if count < 0 or offset + count > total_sz:
            raise ValueError(
                f"Truncated delta {label} at byte {offset}: need {count}, "
                f"frame size {total_sz}")

    block_patches, small_patches = [], []
    for _ in range(n_blocks):
        pos, bp = _read_block(pos)
        block_patches.append(bp)
    for _ in range(n_small):
        pos, sp = _read_small(pos)
        small_patches.append(sp)

    if pos != total_sz:
        raise ValueError(
            f"Delta frame contains {total_sz - pos} unparsed payload bytes")

    return step_id, block_patches, small_patches, {
        "version": 1, "flags": 0, "base_generation": 0, "generation": 0,
    }


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
        if bp.get("encoding") == "fp16" or "fp16_data" in bp:
            delta = np.asarray(bp["fp16_data"], dtype=np.float32)
            start = int(bp.get("element_offset",
                             bp.get("block_idx", 0) * block_size))
            end = min(start + len(delta), int(np.prod(w[name].shape)))
            flat = w[name].astype(np.float32).reshape(-1)
            flat[start:end] += delta[:end - start]
            w[name] = flat.reshape(w[name].shape)
            continue
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
        if sp.get("encoding") == "fp16" or "fp16_data" in sp:
            delta = np.asarray(sp["fp16_data"], dtype=np.float32)
            flat = w[name].astype(np.float32).reshape(-1)
            flat[:len(delta)] += delta[:len(flat)]
            w[name] = flat.reshape(w[name].shape)
            continue
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
