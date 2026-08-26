#!/usr/bin/env python3
"""Safe GPT-2 XL path measurements for E2--E5.

This runner intentionally does not call ``DirectCheckpoint.save``.  The
normal checkpoint layout owns the live correctness ledger on 83.0.0.  Model
benchmark payloads are written only to ``--safe-offset`` (64 GiB by default),
which is after the current FULL region and before the Delta area.

The filesystem side is confined to ``/models/npu_nvme_exp/<run_id>`` and is
removed after each sample.  The result files remain on the repository's
normal filesystem.

Supported paths:
  p1_fs       MindSpore model checkpoint -> 84.0.0 filesystem
  p2_host_fs  HBM -> Host DRAM -> 84.0.0 filesystem, then H2D restore
  p4_spdk     HBM pointers -> SPDK raw batch -> HBM, then hash verification
  p0_train    training-only control
  p5_async    safe D2D snapshot -> asynchronous SPDK raw batch during train
"""

import argparse
import ctypes
import hashlib
import json
import mmap
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from experiments.benchmarks.io_matrix import (  # noqa: E402
    ALIGNMENT, FS_ROOT, SAFE_OFFSET, ResultWriter, check_npu_free,
    environment_snapshot, stats, usage_snapshot,
)
from experiments.baselines import two_phase_common as tpc  # noqa: E402
from experiments.common import (  # noqa: E402
    init_env, make_causal_lm_checkpoint_model, make_causal_lm_training,
    setup_faf_checkpointing, warmup_checkpoint_model, warmup_model,
)
from chunk_helpers import build_chunks, build_ctypes_arrays  # noqa: E402


def round_up(value, alignment=ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def total_param_bytes(model):
    return tpc.get_total_param_bytes(model)


def assign_safe_offsets(param_descs, base_offset):
    """Copy descriptors and assign 4 KiB-spaced raw-device offsets."""
    safe = []
    cursor = base_offset
    for descriptor in param_descs:
        item = dict(descriptor)
        item["offset"] = cursor
        safe.append(item)
        cursor += round_up(item["size"])
    return safe, cursor


def check_safe_range(ckpt, start, end):
    if start < SAFE_OFFSET:
        raise RuntimeError(f"unsafe model benchmark offset: {start}")
    if end > ckpt.total_bytes:
        raise RuntimeError(
            f"model benchmark exceeds 83.0.0: end={end}, capacity={ckpt.total_bytes}")


def descriptor_offset_map(param_descs):
    offsets = {}
    cursor = 0
    for descriptor in param_descs:
        offsets[descriptor["name"]] = cursor
        cursor += descriptor["size"]
    return offsets, cursor


def hash_host_segments(host_buf, param_descs, offset_map):
    result = {}
    base = host_buf.ctypes.data
    for descriptor in param_descs:
        begin = offset_map[descriptor["name"]]
        view = memoryview(host_buf)[begin:begin + descriptor["size"]]
        result[descriptor["name"]] = hashlib.sha256(view).hexdigest()
    return result


def hash_device_params(param_descs, device_id):
    """Hash HBM parameters through a bounded host scratch buffer."""
    tpc._ensure_acl_device(device_id)
    acl = tpc.acl_lib
    max_size = max(descriptor["size"] for descriptor in param_descs)
    scratch = np.empty(max_size, dtype=np.uint8)
    hashes = {}
    timings = []
    for descriptor in param_descs:
        start = time.perf_counter_ns()
        ret = acl.aclrtMemcpy(
            ctypes.c_void_p(scratch.ctypes.data), descriptor["size"],
            ctypes.c_void_p(descriptor["ptr"]), descriptor["size"],
            tpc.ACL_MEMCPY_DEVICE_TO_HOST)
        if ret != 0:
            raise RuntimeError(f"hash D2H failed for {descriptor['name']}: {ret}")
        hashes[descriptor["name"]] = hashlib.sha256(
            memoryview(scratch)[:descriptor["size"]]).hexdigest()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return hashes, {"mean_ms": float(np.mean(timings)),
                    "p95_ms": float(np.percentile(timings, 95)),
                    "n": len(timings)}


def restore_host_timed(filepath, param_descs, offset_map, device_id):
    """Restore a raw host file with first-parameter and full-restore times."""
    t0 = time.perf_counter_ns()
    with open(filepath, "rb") as stream:
        # ACCESS_COPY makes the mapping writable from Python's buffer API
        # while preserving the file (no dirty pages are flushed back).
        file_map = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
        try:
            base = ctypes.addressof(ctypes.c_char.from_buffer(file_map))
            first_ms = None
            t_restore = time.perf_counter_ns()
            for index, descriptor in enumerate(param_descs):
                src = ctypes.c_void_p(base + offset_map[descriptor["name"]])
                ret = tpc.acl_lib.aclrtMemcpy(
                    ctypes.c_void_p(descriptor["ptr"]), descriptor["size"],
                    src, descriptor["size"], tpc.ACL_MEMCPY_HOST_TO_DEVICE)
                if ret != 0:
                    raise RuntimeError(
                        f"restore H2D failed for {descriptor['name']}: {ret}")
                if index == 0:
                    first_ms = (time.perf_counter_ns() - t_restore) / 1e6
            tpc.ms.hal.synchronize()
            full_ms = (time.perf_counter_ns() - t_restore) / 1e6
        finally:
            file_map.close()
    return {"first_param_ms": first_ms, "full_restore_ms": full_ms,
            "file_open_ms": (time.perf_counter_ns() - t0) / 1e6 - full_ms}


def build_batch_arrays(param_descs, chunk_size):
    chunks, total = build_chunks(param_descs, chunk_size)
    return chunks, total, build_ctypes_arrays(chunks)


def direct_batch(ckpt, param_descs, chunk_size, read=False,
                 submit_mode="batch"):
    chunks, total, arrays = build_batch_arrays(param_descs, chunk_size)
    start = time.perf_counter_ns()
    c_io_us = 0
    if submit_mode == "batch":
        ptrs, offsets, sizes = arrays
        if read:
            ret = ckpt._lib_read_batch(ptrs, offsets, sizes, len(chunks))
        else:
            ret = ckpt._lib_write_batch(ptrs, offsets, sizes, len(chunks))
        c_io_us = ckpt.get_last_io_us(read)
    elif submit_mode == "scalar":
        ret = 0
        for chunk in chunks:
            one = build_ctypes_arrays([chunk])
            if read:
                ret = ckpt._lib_read_batch(*one, 1)
            else:
                ret = ckpt._lib_write_batch(*one, 1)
            if ret != 0:
                break
            c_io_us += ckpt.get_last_io_us(read)
    else:
        raise ValueError(f"unsupported submit mode: {submit_mode}")
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    if ret != 0:
        operation = "read" if read else "write"
        raise RuntimeError(f"raw SPDK {operation} returned {ret}")
    return {"chunks": len(chunks), "bytes": total, "elapsed_ms": elapsed_ms,
            "c_io_us": c_io_us}


def make_spdk_context(args, writer):
    from direct_checkpoint import DirectCheckpoint

    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu,
        pipeline_depth=args.pipeline_depth,
        requested_chunk_size=args.chunk_size,
        rank_id=0, world_size=1, keep_last_n=3, slot_size_gb=10,
        spdk_shm_id=args.shm_id,
        profiling_dir=str(writer.run_dir / "profiling"),
        enable_profiling=args.profiling,
    )
    # The runner uses the same ctypes ABI, but the helper methods keep the
    # context private in DirectCheckpoint.  Bind the calls once here.
    ckpt._lib_write_batch = lambda ptrs, offsets, sizes, count: (
        __import__("c_bindings").lib.npu_nvme_write_batch(
            ckpt.ctx, ptrs, offsets, sizes, count))
    ckpt._lib_read_batch = lambda ptrs, offsets, sizes, count: (
        __import__("c_bindings").lib.npu_nvme_read_batch(
            ckpt.ctx, ptrs, offsets, sizes, count))
    ckpt._lib_write_batch_host = lambda ptrs, offsets, sizes, count: (
        __import__("c_bindings").lib.npu_nvme_write_batch_host(
            ckpt.ctx, ptrs, offsets, sizes, count))
    ckpt._lib_read_batch_host = lambda ptrs, offsets, sizes, count: (
        __import__("c_bindings").lib.npu_nvme_read_batch_host(
            ckpt.ctx, ptrs, offsets, sizes, count))
    return ckpt


def p1_fs_sample(model, param_descs, writer, root, index, warmup):
    path = root / f"p1_model_{index:02d}.ckpt"
    expected, hash_stats = hash_device_params(param_descs, writer.config["npu"])
    events = [{"name": "checkpoint_trigger", "monotonic_ns": time.monotonic_ns()}]
    start = time.perf_counter_ns()
    save_enter = time.monotonic_ns()
    ms = __import__("mindspore")
    ms.save_checkpoint(model, str(path))
    save_ms = (time.perf_counter_ns() - start) / 1e6
    events.append({"name": "save_return", "monotonic_ns": time.monotonic_ns()})
    size = path.stat().st_size
    load_start = time.perf_counter_ns()
    payload = ms.load_checkpoint(str(path))
    load_dict_ms = (time.perf_counter_ns() - load_start) / 1e6
    first_start = time.perf_counter_ns()
    ms.load_param_into_net(model, payload)
    ms.hal.synchronize()
    restore_ms = (time.perf_counter_ns() - first_start) / 1e6
    events.append({"name": "restore_end", "monotonic_ns": time.monotonic_ns()})
    actual, _ = hash_device_params(param_descs, writer.config["npu"])
    if actual != expected:
        raise AssertionError("P1 model checkpoint hash mismatch")
    end = time.perf_counter_ns()
    return {
        "run_id": writer.run_id, "request_id": f"{writer.run_id}/request_{index:02d}",
        "checkpoint_id": f"checkpoint_{index:02d}", "warmup": warmup,
        "path": "P1_kernel_fs", "bytes": total_param_bytes(model),
        "file_bytes": size, "status": "pass", "hashes": len(expected),
        "events": events,
        "timeline_us": {"save": save_ms * 1000,
                         "load_dictionary": load_dict_ms * 1000,
                         "restore": restore_ms * 1000,
                         "end_to_end": (end - start) / 1000},
        "first_param_ms": load_dict_ms + restore_ms,
        "full_model_ms": load_dict_ms + restore_ms,
        "hash_ms": hash_stats["mean_ms"],
    }


def p2_host_fs_sample(model, param_descs, writer, root, index, warmup):
    offset_map, total = descriptor_offset_map(param_descs)
    host_buf = tpc.allocate_host_buffer(total)
    expected_hashes = None
    path = root / f"p2_raw_{index:02d}.bin"
    events = [{"name": "checkpoint_trigger", "monotonic_ns": time.monotonic_ns()}]
    snapshot = tpc.snapshot_d2h(param_descs, host_buf, offset_map,
                                writer.config["npu"])
    expected_hashes = hash_host_segments(host_buf, param_descs, offset_map)
    persist_start = time.perf_counter_ns()
    persist_ms = tpc.persist_to_file(host_buf, str(path))
    events.append({"name": "host_snapshot_end", "monotonic_ns": time.monotonic_ns(),
                   "snapshot": snapshot})
    events.append({"name": "persist_return", "monotonic_ns": time.monotonic_ns()})
    restore = restore_host_timed(str(path), param_descs, offset_map,
                                 writer.config["npu"])
    actual_hashes, hash_stats = hash_device_params(
        param_descs, writer.config["npu"])
    if actual_hashes != expected_hashes:
        raise AssertionError("P2 HBM-Host-FS restore hash mismatch")
    end = time.perf_counter_ns()
    return {
        "run_id": writer.run_id, "request_id": f"{writer.run_id}/request_{index:02d}",
        "checkpoint_id": f"checkpoint_{index:02d}", "warmup": warmup,
        "path": "P2_HBM_Host_FS", "bytes": total, "file_bytes": path.stat().st_size,
        "status": "pass", "hashes": len(expected_hashes), "events": events,
        "timeline_us": {"snapshot": snapshot["total_ms"] * 1000,
                         "persist": persist_ms * 1000,
                         "restore": restore["full_restore_ms"] * 1000,
                         "end_to_end": (end - persist_start) / 1000},
        "first_param_ms": restore["first_param_ms"],
        "full_model_ms": restore["full_restore_ms"],
        "snapshot": snapshot, "restore": restore,
        "hash_ms": hash_stats["mean_ms"],
    }


def p4_spdk_sample(model, param_descs, safe_descs, ckpt, writer, index, warmup):
    expected, hash_stats = hash_device_params(param_descs, writer.config["npu"])
    events = [{"name": "checkpoint_trigger", "monotonic_ns": time.monotonic_ns()}]
    start = time.perf_counter_ns()
    write = direct_batch(ckpt, safe_descs, writer.config["chunk_size"], read=False,
                         submit_mode=writer.config["submit_mode"])
    events.append({"name": "spdk_write_return", "monotonic_ns": time.monotonic_ns(),
                   "c_io_us": write["c_io_us"]})

    first_desc = safe_descs[0:1]
    first = direct_batch(ckpt, first_desc, writer.config["chunk_size"], read=True,
                         submit_mode=writer.config["submit_mode"])
    first_ms = first["elapsed_ms"]
    read = direct_batch(ckpt, safe_descs, writer.config["chunk_size"], read=True,
                        submit_mode=writer.config["submit_mode"])
    events.append({"name": "spdk_read_return", "monotonic_ns": time.monotonic_ns(),
                   "c_io_us": read["c_io_us"]})
    actual, _ = hash_device_params(param_descs, writer.config["npu"])
    if actual != expected:
        raise AssertionError("P4 HBM-SPDK-HBM restore hash mismatch")
    end = time.perf_counter_ns()
    return {
        "run_id": writer.run_id, "request_id": f"{writer.run_id}/request_{index:02d}",
        "checkpoint_id": f"checkpoint_{index:02d}", "warmup": warmup,
        "path": "P4_HBM_SPDK", "bytes": total_param_bytes(model),
        "status": "pass", "hashes": len(expected), "events": events,
        "timeline_us": {"write": write["elapsed_ms"] * 1000,
                         "read": read["elapsed_ms"] * 1000,
                         "end_to_end": (end - start) / 1000},
        "first_param_ms": first_ms,
        "full_model_ms": read["elapsed_ms"],
        "write": write, "read": read, "hash_ms": hash_stats["mean_ms"],
    }


def p3_host_spdk_sample(model, param_descs, safe_descs, ckpt, writer, index, warmup):
    """HBM -> Host DRAM -> SPDK raw path, with a host-side read-back."""
    offset_map, total = descriptor_offset_map(param_descs)
    host_buf = tpc.allocate_host_buffer(total)
    destination = tpc.allocate_host_buffer(total)
    snapshot = tpc.snapshot_d2h(param_descs, host_buf, offset_map,
                                writer.config["npu"])
    expected = hash_host_segments(host_buf, param_descs, offset_map)
    host_desc = [{"ptr": host_buf.ctypes.data, "size": total,
                  "offset": writer.config["safe_offset"], "name": "host_model"}]
    read_desc = [{"ptr": destination.ctypes.data, "size": total,
                  "offset": writer.config["safe_offset"], "name": "host_model"}]
    start = time.perf_counter_ns()
    write_chunks, _, write_arrays = build_batch_arrays(host_desc, writer.config["chunk_size"])
    ret = ckpt._lib_write_batch_host(*write_arrays, len(write_chunks))
    if ret != 0:
        raise RuntimeError(f"P3 Host->SPDK returned {ret}")
    write_ms = (time.perf_counter_ns() - start) / 1e6
    read_start = time.perf_counter_ns()
    read_chunks, _, read_arrays = build_batch_arrays(read_desc, writer.config["chunk_size"])
    ret = ckpt._lib_read_batch_host(*read_arrays, len(read_chunks))
    if ret != 0:
        raise RuntimeError(f"P3 SPDK->Host returned {ret}")
    read_ms = (time.perf_counter_ns() - read_start) / 1e6
    actual = hash_host_segments(destination, param_descs, offset_map)
    if actual != expected:
        raise AssertionError("P3 Host-SPDK-Host restore hash mismatch")
    end = time.perf_counter_ns()
    return {
        "run_id": writer.run_id, "request_id": f"{writer.run_id}/request_{index:02d}",
        "checkpoint_id": f"checkpoint_{index:02d}", "warmup": warmup,
        "path": "P3_Host_SPDK", "bytes": total, "status": "pass",
        "hashes": len(expected), "events": [{"name": "snapshot_end",
                                               "monotonic_ns": time.monotonic_ns()}],
        "timeline_us": {"snapshot": snapshot["total_ms"] * 1000,
                         "write": write_ms * 1000, "read": read_ms * 1000,
                         "end_to_end": (end - start) / 1000},
        "first_param_ms": None, "full_model_ms": read_ms,
        "snapshot": snapshot, "write_ms": write_ms, "read_ms": read_ms,
    }


class SafeAsyncSpdk:
    """One-owner async SPDK engine backed by D2D snapshot buffers."""

    def __init__(self, model, param_descs, ckpt, base_offset, chunk_size, device_id):
        self.device_id = device_id
        self.ckpt = ckpt
        self.param_descs = param_descs
        # The shadow buffer is a raw DMA image, so its logical offsets must
        # carry the same 4 KiB padding as the NVMe offsets.  The ordinary
        # host snapshot map is byte-packed and is not suitable here.
        self.shadow_descs = []
        self.offset_map = {}
        cursor = 0
        for descriptor in param_descs:
            cursor = round_up(cursor)
            item = dict(descriptor)
            item["offset"] = cursor
            self.shadow_descs.append(item)
            self.offset_map[item["name"]] = cursor
            cursor += descriptor["size"]
        self.total = round_up(cursor)
        self.flat_chunks = tpc.build_flat_hbm_buffer(
            model, self.shadow_descs, self.offset_map, self.total, device_id)
        self.write_chunks = []
        for ptr, start, end in self.flat_chunks:
            self.write_chunks.append({"ptr": ptr, "size": end - start,
                                      "offset": base_offset + start,
                                      "name": f"shadow@{start}"})
        self._thread = None
        self._error = None
        self.events = []

    def checkpoint(self, step):
        self.wait()
        t_wait = time.perf_counter_ns()
        tpc._ensure_acl_device(self.device_id)
        t_sync = time.perf_counter_ns()
        tpc.ms.hal.synchronize()
        tpc.d2d_to_chunks(self.flat_chunks, self.param_descs,
                          self.offset_map, self.device_id)
        snapshot_ms = (time.perf_counter_ns() - t_sync) / 1e6
        chunks, total = build_chunks(self.write_chunks, 4 * 1024 * 1024)
        arrays = build_ctypes_arrays(chunks)
        ptrs, offsets, sizes = arrays
        event = {"step": step,
                 "trigger_wait_ms": (t_sync - t_wait) / 1e6,
                 "snapshot_ms": snapshot_ms, "bytes": total,
                 "snapshot_end_ns": time.monotonic_ns(),
                 "persist_ms": None}
        self.events.append(event)

        def worker():
            try:
                start = time.perf_counter_ns()
                from c_bindings import lib
                ret = lib.npu_nvme_write_batch(
                    self.ckpt.ctx, ptrs, offsets, sizes, len(chunks))
                if ret != 0:
                    raise RuntimeError(f"P5 raw write returned {ret}")
                event["persist_ms"] = (time.perf_counter_ns() - start) / 1e6
                event["persist_end_ns"] = time.monotonic_ns()
            except BaseException as error:
                self._error = error

        self._thread = threading.Thread(target=worker, name="safe-spdk-writer")
        self._thread.start()
        return event

    def wait(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error = self._error
            self._error = None
            raise RuntimeError("P5 async write failed") from error

    def cleanup(self):
        self.wait()
        tpc.free_flat_hbm_chunks(self.flat_chunks)


def p0_train(args, model, ds, opt, writer):
    from direct_checkpoint import ProbeTrainOneStepCell
    cell = ProbeTrainOneStepCell(model, opt, enable_probe=False, ckpt_interval=9999)
    iterator = ds.create_tuple_iterator()
    step_times = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = ds.create_tuple_iterator()
            batch = next(iterator)
        start = time.perf_counter_ns()
        loss = cell(*batch)
        t_ms = (time.perf_counter_ns() - start) / 1e6
        step_times.append(t_ms)
    writer.add_sample({"run_id": writer.run_id, "request_id": f"{writer.run_id}/train",
                       "checkpoint_id": "train", "warmup": False,
                       "path": "P0_train", "bytes": 0, "status": "pass",
                       "steps": args.steps, "loss_last": str(loss),
                       "timeline_us": {"step": [v * 1000 for v in step_times],
                                        "end_to_end": sum(step_times) * 1000},
                       "events": []})
    return {"step_ms": stats(step_times), "steps": args.steps}


def p5_async(args, model, ds, opt, param_descs, ckpt, writer):
    from direct_checkpoint import ProbeTrainOneStepCell
    engine = SafeAsyncSpdk(model, param_descs, ckpt, args.safe_offset,
                           args.chunk_size, args.npu)
    cell = ProbeTrainOneStepCell(model, opt, enable_probe=False, ckpt_interval=9999)
    iterator = ds.create_tuple_iterator()
    step_times = []
    checkpoint_count = 0
    try:
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = ds.create_tuple_iterator()
                batch = next(iterator)
            start = time.perf_counter_ns()
            _ = cell(*batch)
            step_times.append((time.perf_counter_ns() - start) / 1e6)
            if step % args.ckpt_every == 0:
                event = engine.checkpoint(step)
                checkpoint_count += 1
                writer.add_sample({
                    "run_id": writer.run_id,
                    "request_id": f"{writer.run_id}/step_{step}",
                    "checkpoint_id": f"checkpoint_{step:04d}",
                    "warmup": checkpoint_count <= args.warmups,
                    "path": "P5_async_safe_spdk", "bytes": event["bytes"],
                    "status": "pass", "events": [event],
                    "timeline_us": {"snapshot": event["snapshot_ms"] * 1000,
                                     "persist": (event["persist_ms"] or 0) * 1000,
                                     "end_to_end": event["snapshot_ms"] * 1000},
                    "blocking_ms": event["trigger_wait_ms"],
                    "step_ms": step_times[-1],
                })
        engine.wait()
    finally:
        engine.cleanup()
    return {"step_ms": stats(step_times), "steps": args.steps,
            "checkpoints": checkpoint_count, "events": engine.events}


def p5_faf(args, model, ds, opt, ckpt, writer):
    """True graph-counter + Reactor-poller trigger path for A7.

    The first probe-cell invocation is a compile/warmup only.  The counter is
    reset before registration, so formal checkpoints are associated with the
    actual training steps below.  We wait for the device completion flag at
    each checkpoint boundary to make skipped/failed FaF writes observable.
    """
    import mindspore as ms
    from direct_checkpoint import ProbeTrainOneStepCell

    cell = ProbeTrainOneStepCell(model, opt, enable_probe=True,
                                 ckpt_interval=args.ckpt_every)
    iterator = ds.create_tuple_iterator()
    first_batch = next(iterator)
    _ = cell(*first_batch)
    ms.hal.synchronize()
    # Keep the compiled device Parameters in place. Replacing them with a
    # fresh host Tensor makes get_dev_ptr() return NULL and invalidates the
    # C poller's step registration. The one warmup increment is included in
    # the counter base below.
    counter_base = int(cell.step_counter.asnumpy().reshape(-1)[0])
    setup_faf_checkpointing(ckpt, model, cell, args.ckpt_every)

    step_times = []
    checkpoint_count = 0
    try:
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = ds.create_tuple_iterator()
                batch = next(iterator)
            start = time.perf_counter_ns()
            _ = cell(*batch)
            ms.hal.synchronize()
            step_ms = (time.perf_counter_ns() - start) / 1e6
            step_times.append(step_ms)
            observed_counter = counter_base + step
            if observed_counter % args.ckpt_every != 0:
                continue

            checkpoint_count += 1
            wait_start = time.perf_counter_ns()
            deadline = time.monotonic() + max(30.0, args.io_timeout_s)
            observed = 0
            while time.monotonic() < deadline:
                observed = ckpt.read_probe_flag_dev()
                if observed >= observed_counter:
                    break
                time.sleep(0.001)
            if observed < observed_counter:
                raise TimeoutError(
                    f"FaF completion flag did not reach step {observed_counter}: "
                    f"{observed}")
            wait_ms = (time.perf_counter_ns() - wait_start) / 1e6
            writer.add_sample({
                "run_id": writer.run_id,
                "request_id": f"{writer.run_id}/step_{step}",
                "checkpoint_id": f"checkpoint_{observed_counter:04d}",
                "warmup": checkpoint_count <= args.warmups,
                "path": "P5_faf_graph_counter_reactor",
                "bytes": writer.config["parameter_bytes"],
                "status": "pass",
                "events": [{"name": "graph_counter_trigger", "step": observed_counter,
                             "monotonic_ns": time.monotonic_ns()},
                            {"name": "reactor_persisted", "step": observed,
                             "monotonic_ns": time.monotonic_ns()}],
                "timeline_us": {"end_to_end": wait_ms * 1000},
                "blocking_ms": wait_ms,
                "step_ms": step_ms,
            })
    finally:
        # The C layer owns the registered task list until DirectCheckpoint
        # cleanup; no Python-side task array is reused here.
        pass
    return {"step_ms": stats(step_times), "steps": args.steps,
            "checkpoints": checkpoint_count,
            "blocking_ms": stats([s["blocking_ms"] for s in writer.samples])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("E2", "E3", "E4", "E5"), required=True)
    parser.add_argument("--model", choices=("gpt2", "gpt2_xl", "gpt2_13b",
                                             "llama2_7b", "glm4_9b",
                                             "llama2_13b"),
                        default="gpt2_xl")
    parser.add_argument("--path", choices=("p0_train", "p1_fs", "p2_host_fs",
                                             "p3_host_spdk", "p4_spdk",
                                             "p5_async", "p5_faf"), required=True)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=83)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--safe-offset", type=int, default=SAFE_OFFSET)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=1025,
                        help="training sequence length (use 128 for the 13B scale lane)")
    parser.add_argument("--ckpt-every", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--io-timeout-s", type=float, default=120.0)
    parser.add_argument("--profiling", action="store_true")
    parser.add_argument("--submit-mode", choices=("batch", "scalar"),
                        default="batch",
                        help="P4 submission granularity for A3")
    parser.add_argument("--fs-root", default=str(FS_ROOT),
                        help="filesystem root for P1/P2; may be a same-device mount")
    parser.add_argument("--checkpoint-only", action="store_true",
                        help="build the real model without optimizer/training state")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.chunk_size % ALIGNMENT or args.safe_offset % ALIGNMENT:
        raise ValueError("chunk-size and safe-offset must be 4 KiB aligned")
    if args.checkpoint_only and args.path in ("p0_train", "p5_async", "p5_faf"):
        raise ValueError("checkpoint-only mode supports P1/P2/P3/P4 only")
    if args.path in ("p5_async", "p5_faf") and args.repetitions != args.steps // args.ckpt_every:
        args.repetitions = args.steps // args.ckpt_every
    writer = ResultWriter(args.experiment, args)
    writer.config.update({"path": args.path, "model": args.model,
                          "parameter_precision": "model-native",
                          "submit_mode": args.submit_mode,
                          "seq_len": args.seq_len,
                          "fs_root": args.fs_root,
                          "safe_region": [args.safe_offset, args.safe_offset],
                          "formal_repetitions": args.repetitions})
    writer.write_json("config.json", writer.config)
    npu_info = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, npu_info))
    model = ds = opt = ckpt = None
    root = Path(args.fs_root) / writer.run_id
    try:
        init_env(device_id=args.npu)
        if args.checkpoint_only:
            model, cfg = make_causal_lm_checkpoint_model(args.model, seq_len=128)
            ds = opt = None
            warmup_checkpoint_model(model, cfg, seq_len=128)
        else:
            model, ds, opt = make_causal_lm_training(
                model_name=args.model,
                total_steps=max(args.steps, args.ckpt_every), device_id=args.npu,
                seq_len=args.seq_len)
            warmup_model(model, opt, ds)
        param_descs = tpc.get_param_descriptors(model)
        if not param_descs:
            raise RuntimeError(f"no warm-allocated {args.model} parameters")
        total = total_param_bytes(model)
        safe_descs, safe_end = assign_safe_offsets(param_descs, args.safe_offset)
        writer.config.update({"parameter_count": len(param_descs),
                              "parameter_bytes": total,
                              "safe_region": [args.safe_offset, safe_end]})
        writer.write_json("config.json", writer.config)
        if args.path == "p0_train":
            summary = p0_train(args, model, ds, opt, writer)
        else:
            root.mkdir(parents=True, exist_ok=False)
            if args.path == "p1_fs":
                for index in range(args.warmups + args.repetitions):
                    sample = p1_fs_sample(model, param_descs, writer, root, index,
                                          index < args.warmups)
                    if index >= args.warmups:
                        writer.add_sample(sample)
                summary = {"save_ms": stats([s["timeline_us"]["save"] / 1000
                                              for s in writer.samples]),
                           "restore_ms": stats([s["full_model_ms"]
                                                for s in writer.samples])}
            elif args.path == "p2_host_fs":
                for index in range(args.warmups + args.repetitions):
                    sample = p2_host_fs_sample(model, param_descs, writer, root, index,
                                               index < args.warmups)
                    if index >= args.warmups:
                        writer.add_sample(sample)
                summary = {"snapshot_ms": stats([s["snapshot"]["total_ms"]
                                                  for s in writer.samples]),
                           "persist_ms": stats([s["timeline_us"]["persist"] / 1000
                                                for s in writer.samples]),
                           "restore_ms": stats([s["full_model_ms"]
                                                for s in writer.samples])}
            else:
                check_npu_free(args.npu)
                ckpt = make_spdk_context(args, writer)
                check_safe_range(ckpt, args.safe_offset, safe_end)
                if args.path == "p3_host_spdk":
                    for index in range(args.warmups + args.repetitions):
                        sample = p3_host_spdk_sample(model, param_descs, safe_descs,
                                                     ckpt, writer, index,
                                                     index < args.warmups)
                        if index >= args.warmups:
                            writer.add_sample(sample)
                    summary = {"write_ms": stats([s["write_ms"]
                                                  for s in writer.samples]),
                               "read_ms": stats([s["read_ms"]
                                                 for s in writer.samples]),
                               "snapshot_ms": stats([s["snapshot"]["total_ms"]
                                                     for s in writer.samples])}
                elif args.path == "p4_spdk":
                    for index in range(args.warmups + args.repetitions):
                        sample = p4_spdk_sample(model, param_descs, safe_descs,
                                                ckpt, writer, index, index < args.warmups)
                        if index >= args.warmups:
                            writer.add_sample(sample)
                    summary = {"write_ms": stats([s["write"]["elapsed_ms"]
                                                  for s in writer.samples]),
                               "read_ms": stats([s["read"]["elapsed_ms"]
                                                 for s in writer.samples]),
                               "first_param_ms": stats([s["first_param_ms"]
                                                        for s in writer.samples])}
                elif args.path == "p5_async":
                    summary = p5_async(args, model, ds, opt, param_descs, ckpt, writer)
                else:
                    summary = p5_faf(args, model, ds, opt, ckpt, writer)
        expected_samples = args.repetitions if args.path != "p0_train" else 1
        status = "pass" if not writer.failed and len(writer.samples) == expected_samples else "fail"
    except BaseException as error:
        writer.add_failure({"error": repr(error)})
        summary = {"error": repr(error)}
        status = "fail"
    finally:
        if ckpt is not None:
            ckpt.cleanup()
        if root.exists():
            shutil.rmtree(root)
    result = writer.finalize(summary, status=status)
    print(json.dumps({"run_id": writer.run_id, "status": result["status"],
                      "path": args.path, "samples": result["samples"],
                      "failed_samples": result["failed_samples"],
                      "summary": summary}, indent=2, sort_keys=True), flush=True)
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
