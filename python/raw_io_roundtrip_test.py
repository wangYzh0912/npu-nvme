"""Hardware roundtrip test for raw NPU-SSD transfers."""

import ctypes
import json
import os
import struct
import sys

from c_bindings import lib, NPUNVMEContext
from disk_layout import (
    BLOCK_SIZE,
    HEAP_START_OFFSET,
    SUPERBLOCK_OFFSET,
    SUPERBLOCK_HEADER_BYTES,
    META_SLOT_A_OFFSET,
    META_SLOT_B_OFFSET,
    META_SLOT_BYTES,
    MAGIC_NUMBER,
)
from raw_io import RawIO


def _buf_from_bytes(data: bytes):
    buf = ctypes.create_string_buffer(data, len(data))
    return buf, ctypes.addressof(buf)


def main():
    pci_addr = sys.argv[1] if len(sys.argv) > 1 else "0000:83:00.0"
    npu_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ctx = ctypes.POINTER(NPUNVMEContext)()
    rc = lib.npu_nvme_init(ctypes.byref(ctx), pci_addr.encode(), npu_id, 4,
                           4 * 1024 * 1024, False, b".")
    if rc != 0:
        raise RuntimeError(f"npu_nvme_init failed: {rc}")

    try:
        total_bytes = lib.npu_nvme_get_total_blocks(ctx)
        if total_bytes <= 0:
            raise RuntimeError("device capacity is invalid")

        # Verify checkpoint metadata when the disk has been formatted.
        sb = ctypes.create_string_buffer(4096)
        if lib.npu_nvme_sync_meta_io(ctx, SUPERBLOCK_OFFSET, 4096, 1,
                                     ctypes.c_void_p(ctypes.addressof(sb))) != 0:
            raise RuntimeError("failed to read superblock")
        magic, active_slot, _, stack_start = struct.unpack(
            "<8s I Q Q", sb.raw[:SUPERBLOCK_HEADER_BYTES])
        metadata_available = magic == MAGIC_NUMBER

        raw = RawIO(ctx)
        raw_offset = HEAP_START_OFFSET + 1024 * 1024 * 1024
        seed = b"RAW-IO-ROUNDTRIP-" + bytes(range(32))
        payload = (seed * ((BLOCK_SIZE + len(seed) - 1) // len(seed)))[:BLOCK_SIZE]
        if raw_offset + len(payload) > total_bytes:
            raise RuntimeError("device capacity is too small for raw IO test")
        wbuf = ctypes.create_string_buffer(payload, len(payload))
        rbuf = ctypes.create_string_buffer(len(payload))
        rc = raw.write_host([ctypes.addressof(wbuf)], [raw_offset], [len(payload)])
        if rc != 0:
            raise RuntimeError(f"raw write failed: {rc}")
        rc = raw.read_host([ctypes.addressof(rbuf)], [raw_offset], [len(payload)])
        if rc != 0:
            raise RuntimeError(f"raw read failed: {rc}")
        if rbuf.raw[:len(payload)] != payload:
            raise RuntimeError("raw payload roundtrip mismatch")

        if metadata_available:
            # Re-read checkpoint metadata to ensure raw IO did not corrupt it.
            sb2 = ctypes.create_string_buffer(4096)
            if lib.npu_nvme_sync_meta_io(
                    ctx, SUPERBLOCK_OFFSET, 4096, 1,
                    ctypes.c_void_p(ctypes.addressof(sb2))) != 0:
                raise RuntimeError("failed to reread superblock")
            magic2, active_slot2, total2, stack2 = struct.unpack(
                "<8s I Q Q", sb2.raw[:SUPERBLOCK_HEADER_BYTES])
            if (magic2, active_slot2, total2, stack2) != (
                    magic, active_slot, total_bytes, stack_start):
                raise RuntimeError("superblock changed after raw IO")

            meta = ctypes.create_string_buffer(META_SLOT_BYTES)
            for off in (META_SLOT_A_OFFSET, META_SLOT_B_OFFSET):
                if lib.npu_nvme_sync_meta_io(
                        ctx, off, META_SLOT_BYTES, 1,
                        ctypes.c_void_p(ctypes.addressof(meta))) != 0:
                    raise RuntimeError(f"failed to reread meta slot at {off}")

        print(json.dumps({
            "status": "ok",
            "pci_addr": pci_addr,
            "npu_id": npu_id,
            "raw_offset": raw_offset,
            "payload_len": len(payload),
            "metadata_checked": metadata_available,
        }, ensure_ascii=False))
    finally:
        lib.npu_nvme_cleanup(ctx)


if __name__ == "__main__":
    main()
