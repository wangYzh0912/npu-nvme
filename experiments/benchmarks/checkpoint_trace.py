#!/usr/bin/env python3
"""Paired byte-stream checkpoint trace for filesystem and SPDK backends.

The payload is generated once per run and written/read with the same chunk
size.  The filesystem backend uses pwrite+fdatasync; the SPDK backend uses the
existing Host batch ABI.  This is intentionally model-independent: it
isolates persistence and control-plane cost before the MindFormers model
matrix is run.
"""

import argparse
import ctypes
import hashlib
import json
import mmap
import os
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

ALIGNMENT = 4096


def now():
    clock = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
    return time.clock_gettime_ns(clock)


def stats(values):
    values = [float(v) for v in values]
    values.sort()
    if not values:
        return {"n": 0}
    return {"n": len(values), "median": values[len(values) // 2],
            "min": values[0], "max": values[-1],
            "mean": sum(values) / len(values)}


def make_payload(size):
    period = bytes((3 + 17 * i) % 256 for i in range(4096))
    return period * (size // len(period)) + period[:size % len(period)]


def aligned_buffer(size, payload):
    buf = mmap.mmap(-1, size, access=mmap.ACCESS_WRITE)
    if payload != b"\x00":
        repeats, remainder = divmod(size, len(payload))
        buf[:] = payload * repeats + payload[:remainder]
    return buf


def chunks(size, chunk_size):
    for offset in range(0, size, chunk_size):
        yield offset, min(chunk_size, size - offset)


def hash_buffer(buf, size):
    digest = hashlib.sha256()
    for offset, length in chunks(size, 16 * 1024 * 1024):
        digest.update(buf[offset:offset + length])
    return digest.hexdigest()


def fs_sample(args, payload, index, warmup, output_dir):
    path = Path(args.root) / f"checkpoint_trace_{index:04d}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    source = aligned_buffer(args.size, payload)
    dest = aligned_buffer(args.size, b"\x00")
    source_view = memoryview(source)
    dest_view = memoryview(dest)
    events = [{"name": "checkpoint_trigger", "ts_ns": now()}]
    t0 = now()
    flags = os.O_CREAT | os.O_RDWR | os.O_TRUNC
    if args.direct:
        flags |= os.O_DIRECT
    fd = os.open(path, flags, 0o600)
    try:
        if args.direct:
            # fallocate is performed by a buffered descriptor; the data
            # descriptor is reopened with O_DIRECT for the timed path.
            os.close(fd)
            prep_fd = os.open(path, os.O_RDWR)
            os.posix_fallocate(prep_fd, 0, args.size)
            os.close(prep_fd)
            fd = os.open(path, os.O_RDWR | os.O_DIRECT)
        else:
            os.posix_fallocate(fd, 0, args.size)
        events.append({"name": "parameter_pack_end", "ts_ns": now()})
        write_enter = now()
        events.append({"name": "fs_write_enter", "ts_ns": write_enter})
        for offset, length in chunks(args.size, args.chunk_size):
            os.pwrite(fd, source_view[offset:offset + length], offset)
        events.append({"name": "fs_write_syscalls_end", "ts_ns": now()})
        os.fdatasync(fd)
        sync_end = now()
        events.append({"name": "fs_fdatasync_end", "ts_ns": sync_end})
        write_ms = (sync_end - write_enter) / 1e6
        read_enter = now()
        events.append({"name": "fs_read_enter", "ts_ns": read_enter})
        read_fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if args.direct else 0))
        try:
            for offset, length in chunks(args.size, args.chunk_size):
                if args.direct:
                    os.preadv(read_fd, [dest_view[offset:offset + length]], offset)
                else:
                    data = os.pread(read_fd, length, offset)
                    dest_view[offset:offset + length] = data
        finally:
            os.close(read_fd)
        read_end = now()
        events.append({"name": "fs_read_end", "ts_ns": read_end})
    finally:
        os.close(fd)
    verify_enter = now()
    expected = hash_buffer(source, args.size)
    actual = hash_buffer(dest, args.size)
    verify_end = now()
    if expected != actual:
        raise AssertionError("filesystem byte-stream readback mismatch")
    end = now()
    source_view.release()
    dest_view.release()
    source.close()
    dest.close()
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return {
        "index": index, "warmup": warmup, "backend": "filesystem",
        "bytes": args.size, "chunk_size": args.chunk_size, "status": "pass",
        "events": events, "sha256": actual,
        "timeline_ms": {"write_persist": write_ms,
                         "read": (read_end - read_enter) / 1e6,
                         "verify": (verify_end - verify_enter) / 1e6,
                         "end_to_end": (end - t0) / 1e6},
    }


def spdk_sample(args, payload, index, warmup, ckpt):
    from c_bindings import lib

    source = aligned_buffer(args.size, payload)
    dest = aligned_buffer(args.size, b"\x00")
    source_base = ctypes.addressof(ctypes.c_char.from_buffer(source))
    dest_base = ctypes.addressof(ctypes.c_char.from_buffer(dest))
    count = (args.size + args.chunk_size - 1) // args.chunk_size
    ptrs = (ctypes.c_void_p * count)()
    read_ptrs = (ctypes.c_void_p * count)()
    offsets = (ctypes.c_uint64 * count)()
    sizes = (ctypes.c_size_t * count)()
    for i, (inner, length) in enumerate(chunks(args.size, args.chunk_size)):
        ptrs[i] = source_base + inner
        read_ptrs[i] = dest_base + inner
        offsets[i] = args.offset + inner
        sizes[i] = length
    events = [{"name": "checkpoint_trigger", "ts_ns": now()}]
    t0 = now()
    events.append({"name": "parameter_pack_end", "ts_ns": now()})
    marshal_end = now()
    events.append({"name": "ctypes_arrays_end", "ts_ns": marshal_end,
                   "chunks": count})
    write_enter = now()
    events.append({"name": "spdk_write_enter", "ts_ns": write_enter})
    rc = lib.npu_nvme_write_batch_host(
        ckpt.ctx, ptrs, offsets, sizes, count)
    write_end = now()
    events.append({"name": "spdk_write_return", "ts_ns": write_end,
                   "rc": rc, "c_io_us": ckpt.get_last_io_us(False)})
    if rc != 0:
        raise RuntimeError(f"SPDK host write failed: {rc}")
    profile_dir = Path(ckpt.profiling_dir)
    write_profile = profile_dir / "time_write.csv"
    if write_profile.exists():
        write_profile.replace(profile_dir / f"time_write_{index:04d}.csv")
    read_enter = now()
    events.append({"name": "spdk_read_enter", "ts_ns": read_enter})
    rc = lib.npu_nvme_read_batch_host(
        ckpt.ctx, read_ptrs, offsets, sizes, count)
    read_end = now()
    events.append({"name": "spdk_read_return", "ts_ns": read_end,
                   "rc": rc, "c_io_us": ckpt.get_last_io_us(True)})
    if rc != 0:
        raise RuntimeError(f"SPDK host read failed: {rc}")
    read_profile = profile_dir / "time_read.csv"
    if read_profile.exists():
        read_profile.replace(profile_dir / f"time_read_{index:04d}.csv")
    verify_enter = now()
    expected = hash_buffer(source, args.size)
    actual = hash_buffer(dest, args.size)
    verify_end = now()
    if expected != actual:
        raise AssertionError("SPDK byte-stream readback mismatch")
    end = now()
    source.close()
    dest.close()
    return {
        "index": index, "warmup": warmup, "backend": "spdk_host",
        "bytes": args.size, "chunk_size": args.chunk_size, "status": "pass",
        "events": events, "sha256": actual,
        "timeline_ms": {"marshal": (marshal_end - t0) / 1e6,
                         "write_persist": (write_end - write_enter) / 1e6,
                         "read": (read_end - read_enter) / 1e6,
                         "verify": (verify_end - verify_enter) / 1e6,
                         "end_to_end": (end - t0) / 1e6},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("filesystem", "spdk"), required=True)
    parser.add_argument("--root", default="/mnt/npu_nvme83_fs/checkpoint_trace")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=183)
    parser.add_argument("--size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--offset", type=int, default=64 * 1024 * 1024 * 1024)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--direct", action="store_true",
                        help="use O_DIRECT for filesystem write/read")
    args = parser.parse_args()
    if any(value % ALIGNMENT for value in (args.size, args.chunk_size, args.offset)):
        raise ValueError("size, chunk-size and offset must be 4 KiB aligned")
    if args.backend == "spdk":
        from experiments.benchmarks.io_matrix import check_npu_free
        check_npu_free(args.npu)
        from direct_checkpoint import DirectCheckpoint
        ckpt = DirectCheckpoint(
            nvme_addr=args.pci, npu_device_id=args.npu, pipeline_depth=4,
            requested_chunk_size=args.chunk_size, rank_id=0, world_size=1,
            keep_last_n=3, slot_size_gb=10, spdk_shm_id=args.shm_id,
            profiling_dir=str(Path(args.output).parent / "profiling"),
            enable_profiling=True)
    else:
        ckpt = None
    payload = make_payload(4096)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"python": sys.version, "platform": platform.platform(),
                   "clock": "CLOCK_MONOTONIC_RAW-equivalent Python monotonic_ns"})
    output.write_text(json.dumps({"config": config}, indent=2) + "\n")
    samples = []
    try:
        for index in range(args.warmups + args.repetitions):
            sample = (fs_sample(args, payload, index, index < args.warmups, output.parent)
                      if args.backend == "filesystem" else
                      spdk_sample(args, payload, index, index < args.warmups, ckpt))
            if not sample["warmup"]:
                samples.append(sample)
            with output.open("a") as stream:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
    finally:
        if ckpt is not None:
            ckpt.cleanup()
    summary = {name: stats([s["timeline_ms"][name] for s in samples])
               for name in samples[0]["timeline_ms"]} if samples else {"n": 0}
    with output.open("a") as stream:
        stream.write(json.dumps({"summary": summary, "samples": len(samples),
                                 "status": "pass"}) + "\n")
    print(json.dumps({"backend": args.backend, "samples": len(samples),
                      "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
