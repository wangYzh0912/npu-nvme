"""Archived-behavior FaF, weights/delta compatibility adapter for batch B.

Explicit collaborators only. This path is not the strict FULL contract.
"""
from __future__ import annotations
import ctypes
import math
import os
import time
import struct
import numpy as np
import mindspore as ms
from mindspore import Tensor
from npu_nvme.storage.layout import BLOCK_SIZE, UINT32_BYTES, DELTA_MAGIC, FRAME_HEADER_SIZE
from npu_nvme.storage.chunks import build_chunks_host, build_ctypes_arrays
from delta_protocol import (pack_delta_frame, pack_lossless_delta_frame,
    unpack_delta_frame_with_meta, apply_delta_patches)

class LegacyExperimental:
    def __init__(self, *, binding, context, acl, pointer_of, commit, capture,
                 rank_id, device_id, chunk_size, slot_bytes, total_bytes,
                 slot_offset, drain, load_weights, diagnostic_path):
        self.binding = binding
        self.ctx = context
        self.acl = acl
        self.pointer_of = pointer_of
        self.commit = commit
        self._prepare_params = capture.prepare
        self.rank_id = rank_id
        self.npu_device_id = device_id
        self.chunk_size = chunk_size
        self.slot_bytes = slot_bytes
        self.total_bytes = total_bytes
        self._get_current_slot_base_offset = slot_offset
        self.wait_for_io_completion = drain
        self.load = load_weights
        self._meta_pkl = diagnostic_path

    @property
    def layout(self):
        return self.commit.state.layout

    @property
    def meta_dict(self):
        return self.commit.state.meta_dict

    @property
    def metadata_generation(self):
        return self.commit.state.metadata_generation

    def _persist_metadata(self, generation=None):
        return self.commit.persist(generation)

    def _mount_filesystem(self):
        return self.commit.mount(self.total_bytes, self.rank_id)

    def wait_async_io(self):
        pass

    def set_probe_flag_ptr(self, flag_tensor: Tensor = None):
        if not hasattr(self.binding, "npu_nvme_set_probe_flag_ptr"):
            raise RuntimeError(
                "npu_nvme_set_probe_flag_ptr is not available in the C library.")

        ptr = self.pointer_of(flag_tensor) if flag_tensor is not None else 0
        rc = self.binding.npu_nvme_set_probe_flag_ptr(self.ctx, ctypes.c_void_p(ptr))
        if rc != 0:
            raise RuntimeError(
                f"Failed to set probe flag pointer. C API returned {rc}")

        if hasattr(self.binding, "npu_nvme_get_probe_flag_dev_ptr"):
            actual_ptr = self.binding.npu_nvme_get_probe_flag_dev_ptr(self.ctx)
            if actual_ptr:
                self.probe_flag_ptr = actual_ptr
                return
        self.probe_flag_ptr = ptr


    def read_probe_flag_dev(self) -> int:
        if self.acl is None:
            raise RuntimeError("self.acl not available for device read")
        if not hasattr(self, "probe_flag_ptr") or self.probe_flag_ptr == 0:
            raise RuntimeError("probe_flag_ptr is not set")

        ret = self.acl.aclrtSetDevice(self.npu_device_id)
        if ret != 0:
            raise RuntimeError(f"aclrtSetDevice failed, ret={ret}")

        host_buf = ctypes.create_string_buffer(UINT32_BYTES)
        ret = self.acl.aclrtMemcpy(
            ctypes.byref(host_buf), UINT32_BYTES,
            ctypes.c_void_p(self.probe_flag_ptr), UINT32_BYTES, 2)
        if ret != 0:
            raise RuntimeError(f"aclrtMemcpy device->host failed, ret={ret}")
        return int.from_bytes(host_buf.raw[:UINT32_BYTES],
                              byteorder="little", signed=False)


    def write_probe_flag_dev(self, value: int):
        if self.acl is None:
            raise RuntimeError("self.acl not available for device write")
        if not hasattr(self, "probe_flag_ptr") or self.probe_flag_ptr == 0:
            raise RuntimeError("probe_flag_ptr is not set")
        if value < 0:
            raise ValueError("probe flag value must be non-negative")

        ret = self.acl.aclrtSetDevice(self.npu_device_id)
        if ret != 0:
            raise RuntimeError(f"aclrtSetDevice failed, ret={ret}")

        host_buf = ctypes.create_string_buffer(
            int(value).to_bytes(UINT32_BYTES, byteorder="little", signed=False))
        ret = self.acl.aclrtMemcpy(
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
        if not hasattr(self.binding, "npu_nvme_set_probe_flag_value"):
            raise RuntimeError(
                "npu_nvme_set_probe_flag_value is not available in the C library.")
        if value < 0:
            raise ValueError("probe flag value must be non-negative")
        rc = self.binding.npu_nvme_set_probe_flag_value(
            self.ctx, ctypes.c_uint32(value))
        if rc != 0:
            raise RuntimeError(
                f"npu_nvme_set_probe_flag_value failed with rc={rc}")


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
                base = self.pointer_of(param)
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

        rc = self.binding.npu_nvme_register_tasks(
            self.ctx, npu_ptrs, nvme_offsets, sizes, num_items)
        if rc != 0:
            raise RuntimeError(
                f"[DirectCkpt] Failed to register tasks! C API returned {rc}")

        print(f"[DirectCkpt] Rank {self.rank_id} Successfully registered "
              f"{num_items} tensor pointers to SPDK background thread.")


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
        self._require_incremental_enabled()
        if not hasattr(self, '_delta_slot_count'):
            self.delta_init()

        from delta_cell import DeltaTrainCell
        if not isinstance(delta_cell, DeltaTrainCell):
            raise TypeError("build_layout_for_delta expects a DeltaTrainCell")

        # Store the delta block size for correct recovery later.
        # Must match the block_size used by DeltaTrainCell (default 524288).
        self.delta_block_size = delta_cell.bs

        slot_offset = (self.binding.npu_nvme_delta_get_area_offset(self.ctx)
                       + self._delta_next_slot % self._delta_slot_count
                       * self._delta_slot_size)

        # Use self.pointer_of for HBM addresses
        quant_ptr = self.pointer_of(delta_cell.delta_quant_buf)
        scale_ptr = self.pointer_of(delta_cell.delta_scale_buf)
        idx_ptr = self.pointer_of(delta_cell.delta_idx_buf)
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
        self._require_incremental_enabled()
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

        rc = self.binding.npu_nvme_register_tasks(
            self.ctx, npu_ptrs, nvme_offsets, sizes, num_items)
        if rc != 0:
            raise RuntimeError(
                f"register_delta_tasks: C API returned {rc}")

        # Wire step counter
        dev_step = self.pointer_of(delta_cell.step_counter)
        rc = self.binding.npu_nvme_set_step_ptr(
            self.ctx, ctypes.c_void_p(dev_step), ckpt_interval)
        if rc != 0:
            raise RuntimeError(f"set_step_ptr failed: {rc}")

        # Wire probe flag
        dev_flag = self.pointer_of(getattr(delta_cell, 'flag',
                                        delta_cell.step_counter))
        if hasattr(delta_cell, 'flag'):
            dev_flag = self.pointer_of(delta_cell.flag)
            self.binding.npu_nvme_set_probe_flag_ptr(self.ctx, ctypes.c_void_p(dev_flag))
        else:
            self.binding.npu_nvme_set_probe_flag_ptr(self.ctx, ctypes.c_void_p(0))

        if dev_flag == 0 and hasattr(self.binding, "npu_nvme_get_probe_flag_dev_ptr"):
            dev_flag = self.binding.npu_nvme_get_probe_flag_dev_ptr(self.ctx)
        self.probe_flag_ptr = dev_flag

        print(f"[DirectCkpt] Rank {self.rank_id} Registered {num_items} delta "
              f"buffers to FaF listener. step_counter={hex(dev_step)} "
              f"flag={hex(dev_flag)} interval={ckpt_interval}",
              flush=True)
        return dev_flag, dev_step


    @staticmethod
    def _require_incremental_enabled():
        if os.environ.get("NPU_NVME_FULL_ONLY") == "1":
            raise RuntimeError(
                "incremental checkpoint entry points are disabled in FULL-only mode")


    def delta_init(self, slot_size_mb: int = 256, slot_count: int = 128):
        self._require_incremental_enabled()
        slot_bytes = slot_size_mb * 1024 * 1024
        if self.layout is None:
            raise RuntimeError("disk layout is not mounted")
        if (slot_bytes != self.layout.delta_slot_bytes or
                slot_count != self.layout.delta_slot_count):
            raise RuntimeError(
                "requested Delta geometry differs from formatted disk: "
                f"requested {slot_count}x{slot_bytes}, formatted "
                f"{self.layout.delta_slot_count}x{self.layout.delta_slot_bytes}")
        if not hasattr(self.binding, "npu_nvme_delta_init"):
            raise RuntimeError(
                "C library missing npu_nvme_delta_init — rebuild required.")
        rc = self.binding.npu_nvme_delta_init(
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
        self._require_incremental_enabled()
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

        if not hasattr(self.binding, "npu_nvme_write_batch_host"):
            raise RuntimeError("C library missing npu_nvme_write_batch_host")
        rc = self.binding.npu_nvme_write_batch_host(
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
        self._require_incremental_enabled()
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
        rc = self.binding.npu_nvme_write_batch_host(
            self.ctx, c_ptrs, c_offs, c_sizes, len(chunks))
        if rc != 0:
            raise RuntimeError(f"Host-SPDK frame write failed: {rc}")
        return {"offset": byte_offset, "bytes": len(frame),
                "aligned_bytes": aligned_size, "chunks": len(chunks),
                "c_io_us": int(self.binding.npu_nvme_get_last_io_us(self.ctx, 0))}


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
        rc = self.binding.npu_nvme_read_batch_host(
            self.ctx, c_ptrs, c_offs, c_sizes, len(chunks))
        if rc != 0:
            raise RuntimeError(f"Host-SPDK frame read failed: {rc}")
        return bytes(frame_buf.raw[:frame_size])


    def delta_load_slot(self, slot_idx: int, return_meta: bool = False):
        self._require_incremental_enabled()
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
        if not hasattr(self.binding, "npu_nvme_read_batch_host"):
            raise RuntimeError("C library missing npu_nvme_read_batch_host")
        rc = self.binding.npu_nvme_read_batch_host(
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
        rc = self.binding.npu_nvme_read_batch_host(
            self.ctx, d_ptrs, d_offs, d_sizes, len(d_chunks))
        if rc != 0:
            raise RuntimeError(
                f"Delta data read failed at slot {slot_idx} (rc={rc})")

        decoded = unpack_delta_frame_with_meta(data_buf.raw[:total_sz])
        if return_meta:
            return decoded
        return decoded[:3]


    def delta_load_chain(self, from_step: int, to_step: int):
        self._require_incremental_enabled()
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

