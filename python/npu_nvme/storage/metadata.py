"""V2 metadata transport; no checkpoint selection policy or framework objects."""
import ctypes
import json
from dataclasses import replace

from .format import pack_metadata, unpack_metadata, pack_superblock, unpack_superblock
from .layout import SUPERBLOCK_OFFSET, META_SLOT_A_OFFSET, META_SLOT_B_OFFSET, META_SLOT_BYTES


class MetadataIO:
    def __init__(self, binding, context=None):
        self.binding = binding
        self.ctx = context

    def mount(self, state, total_bytes, rank_id):
        sb_buf = ctypes.create_string_buffer(4096)
        rc = self.binding.npu_nvme_sync_meta_io(
            self.ctx, SUPERBLOCK_OFFSET, 4096, 1,
            ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc != 0:
            raise RuntimeError("Failed to read Superblock.")

        try:
            state.layout = unpack_superblock(sb_buf.raw)
        except ValueError as error:
            raise RuntimeError(
                f"Superblock verification failed: {error}. "
                "Run format_npu_disk.py with the V2 layout to initialize the disk.") from error
        if state.layout.total_bytes != total_bytes:
            raise RuntimeError("superblock capacity does not match NVMe capacity")
        sb_active_meta_slot = state.layout.active_meta_slot
        sb_generation = state.layout.generation
        state.active_meta_slot = sb_active_meta_slot
        state.metadata_generation = sb_generation
        state.stack_start_bytes = state.layout.full_base

        valid = []
        for slot, target_offset in enumerate((META_SLOT_A_OFFSET,
                                               META_SLOT_B_OFFSET)):
            meta_buf = ctypes.create_string_buffer(META_SLOT_BYTES)
            rc = self.binding.npu_nvme_sync_meta_io(
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
        state.metadata_generation = generation
        state.active_meta_slot = slot
        state.meta_dict = payload
        state.layout = replace(state.layout, generation=generation,
                              active_meta_slot=slot)

        print(f"[Rank {rank_id}] FileSystem Mounted. "
              f"V{2} generation={generation} "
              f"Active Slot: {'A' if state.active_meta_slot == 0 else 'B'}")

    def persist(self, state, generation=None):
        """Persist the current metadata dict using the A/B commit protocol."""
        if state.layout is None:
            raise RuntimeError("cannot persist metadata before mounting layout")
        if generation is None:
            generation = state.metadata_generation + 1
        next_slot = 1 if state.active_meta_slot == 0 else 0
        target_offset = (META_SLOT_B_OFFSET if next_slot == 1
                         else META_SLOT_A_OFFSET)
        meta_buf = ctypes.create_string_buffer(
            pack_metadata(state.meta_dict, generation), META_SLOT_BYTES)
        rc = self.binding.npu_nvme_sync_meta_io(
            self.ctx, target_offset, META_SLOT_BYTES, 0,
            ctypes.c_void_p(ctypes.addressof(meta_buf)))
        if rc != 0:
            raise RuntimeError(f"metadata replica write failed (rc={rc})")
        self.flush_nvme()
        new_layout = replace(state.layout, generation=generation,
                             active_meta_slot=next_slot)
        sb_buf = ctypes.create_string_buffer(pack_superblock(new_layout), 4096)
        rc = self.binding.npu_nvme_sync_meta_io(
            self.ctx, SUPERBLOCK_OFFSET, 4096, 0,
            ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc != 0:
            raise RuntimeError(f"superblock commit failed (rc={rc})")
        self.flush_nvme()
        state.active_meta_slot = next_slot
        state.metadata_generation = generation
        state.layout = new_layout

    def flush_nvme(self):
        """Wait for the namespace persistence barrier used by R0 commits."""
        if not hasattr(self.binding, "npu_nvme_flush"):
            raise RuntimeError("C library lacks npu_nvme_flush; rebuild required")
        rc = self.binding.npu_nvme_flush(self.ctx)
        if rc != 0:
            raise RuntimeError(f"NVMe flush failed (rc={rc})")
