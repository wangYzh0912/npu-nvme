#!/usr/bin/env python3
"""A9 real HBM snapshot-slot lifecycle benchmark.

The model is trained on one NPU while a single SPDK owner drains frozen HBM
slots in the background.  Slot payloads are copied D2D before being queued;
the worker writes and reads each payload through 83.0.0 and compares the
read-back digest with the digest of the frozen HBM slot.
"""

import argparse
import ctypes
import hashlib
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from experiments.baselines import two_phase_common as tpc  # noqa: E402
from experiments.benchmarks.io_matrix import (  # noqa: E402
    ALIGNMENT, ResultWriter, SAFE_OFFSET, check_npu_free, environment_snapshot,
    stats,
)
from experiments.common import (  # noqa: E402
    init_env, make_causal_lm_training, training_numeric_health, warmup_model,
)
from frame_lifecycle import FrameBufferPool  # noqa: E402
from chunk_helpers import build_chunks, build_chunks_host, build_ctypes_arrays  # noqa: E402


CHUNK_SIZE = 4 * 1024 * 1024
BASE_OFFSET = SAFE_OFFSET + 8 * 1024**3


def round_up(value, alignment=ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def parameter_offsets(param_descs):
    offsets = {}
    cursor = 0
    for descriptor in param_descs:
        cursor = round_up(cursor)
        offsets[descriptor["name"]] = cursor
        cursor += descriptor["size"]
    return offsets, round_up(cursor)


def hash_hbm_chunks(chunks, device_id):
    tpc._ensure_acl_device(device_id)
    max_size = max(end - start for _ptr, start, end in chunks)
    scratch = np.empty(max_size, dtype=np.uint8)
    digest = hashlib.sha256()
    for ptr, start, end in chunks:
        size = end - start
        ret = tpc.acl_lib.aclrtMemcpy(
            ctypes.c_void_p(scratch.ctypes.data), size,
            ctypes.c_void_p(ptr), size, tpc.ACL_MEMCPY_DEVICE_TO_HOST)
        tpc._check_acl_ret(ret, f"A9 HBM hash {start}")
        digest.update(memoryview(scratch)[:size])
    return digest.hexdigest()


def readback_hash(ckpt, write_chunks):
    digest = hashlib.sha256()
    lib = __import__("c_bindings").lib
    for _ptr, offset_value, size_value, _name in write_chunks:
        offset = int(offset_value.value)
        size = int(size_value.value)
        buffer = ctypes.create_string_buffer(round_up(size))
        chunks, _ = build_chunks_host(ctypes.addressof(buffer), offset, size,
                                      ckpt.chunk_size)
        ptrs, offsets, sizes = build_ctypes_arrays(chunks)
        ret = lib.npu_nvme_read_batch_host(
            ckpt.ctx, ptrs, offsets, sizes, len(chunks))
        if ret != 0:
            raise RuntimeError(f"A9 readback failed at {start}: {ret}")
        digest.update(buffer.raw[:size])
    return digest.hexdigest()


class HbmSlotRunner:
    def __init__(self, args, model, param_descs, ckpt, writer, slot_count):
        self.args = args
        self.model = model
        self.param_descs = param_descs
        self.ckpt = ckpt
        self.writer = writer
        self.slot_count = slot_count
        self.offset_map, self.total = parameter_offsets(param_descs)
        self.pool = FrameBufferPool(slot_count)
        self.pending = queue.Queue()
        self.results = []
        self.errors = []
        self.stop = False
        self.slots = []
        for slot_id in range(slot_count):
            flat = tpc.build_flat_hbm_buffer(
                model, param_descs, self.offset_map, self.total, args.npu)
            base = BASE_OFFSET + slot_id * round_up(self.total)
            write_chunks = []
            for ptr, start, end in flat:
                write_chunks.append({"ptr": ptr, "size": end - start,
                                     "offset": base + start,
                                     "name": f"slot{slot_id}@{start}"})
            chunks, _ = build_chunks(write_chunks, CHUNK_SIZE)
            self.slots.append({"slot_id": slot_id, "flat": flat,
                               "write_chunks": chunks})
        self.capture_bytes = sum(
            end - start for _ptr, start, end in self.slots[0]["flat"])
        end = BASE_OFFSET + slot_count * round_up(self.total)
        if end > ckpt.total_bytes:
            raise RuntimeError(f"A9 safe range exceeds 83.0.0: {end}")
        self.worker = threading.Thread(target=self._drain, name="a9-spdk-owner",
                                       daemon=True)

    def _drain(self):
        lib = __import__("c_bindings").lib
        while True:
            item = self.pending.get()
            if item is None:
                self.pending.task_done()
                return
            slot_id, sample = item
            try:
                descriptor = self.pool.begin_hbm_write(slot_id)
                if (descriptor.generation != sample["generation"] or
                        descriptor.step_id != sample["step"] or
                        descriptor.checksum != sample["frozen_sha256"] or
                        descriptor.valid_bytes != self.capture_bytes):
                    raise AssertionError("HBM frame descriptor ownership mismatch")
                start = time.monotonic_ns()
                if self.args.io_delay_ms > 0:
                    time.sleep(self.args.io_delay_ms / 1000.0)
                chunks = self.slots[slot_id]["write_chunks"]
                ptrs, offsets, sizes = build_ctypes_arrays(chunks)
                ret = lib.npu_nvme_write_batch(
                    self.ckpt.ctx, ptrs, offsets, sizes, len(chunks))
                if ret != 0:
                    raise RuntimeError(f"A9 write failed: {ret}")
                c_write = int(lib.npu_nvme_get_last_io_us(self.ckpt.ctx, 0))
                actual = readback_hash(self.ckpt, chunks)
                if actual != sample["frozen_sha256"]:
                    raise AssertionError(
                        f"slot {slot_id} frozen/readback hash mismatch")
                end = time.monotonic_ns()
                self.pool.ack(slot_id, actual)
                sample.update({
                    "status": "pass", "persisted_ns": end,
                    "c_write_us": c_write,
                    "readback_sha256": actual,
                    "timeline_us": {
                        "snapshot": sample["snapshot_ms"] * 1000,
                        "io": (end - start) / 1000,
                        "end_to_end": (end - sample["capture_ns"]) / 1000,
                    },
                    "events": sample["events"] + [
                        {"name": "io_start", "monotonic_ns": start},
                        {"name": "persisted", "monotonic_ns": end},
                        {"name": "slot_free", "monotonic_ns": end},
                    ],
                })
                self.results.append(sample)
            except BaseException as error:
                try:
                    self.pool.fail(slot_id, error)
                except RuntimeError:
                    pass
                self.errors.append({"slot_id": slot_id, "error": repr(error)})
            finally:
                self.pending.task_done()

    def capture(self, step, checkpoint_index):
        wait_start = time.monotonic_ns()
        if self.errors:
            raise RuntimeError(self.errors[-1]["error"])
        generation = checkpoint_index + 1
        slot_id = self.pool.acquire(
            generation, step, timeout=self.args.io_timeout_s)
        request_id = f"{self.writer.run_id}/step_{step}/slot_{slot_id}"
        wait_ms = (time.monotonic_ns() - wait_start) / 1e6
        slot = self.slots[slot_id]
        capture_start = time.monotonic_ns()
        tpc._ensure_acl_device(self.args.npu)
        tpc.ms.hal.synchronize()
        t0 = time.perf_counter_ns()
        tpc.d2d_to_chunks(slot["flat"], self.param_descs,
                          self.offset_map, self.args.npu)
        frozen_sha256 = hash_hbm_chunks(slot["flat"], self.args.npu)
        snapshot_ms = (time.perf_counter_ns() - t0) / 1e6
        capture_end = time.monotonic_ns()
        sample = {
            "run_id": self.writer.run_id,
            "request_id": request_id,
            "checkpoint_id": f"checkpoint_{step:04d}",
            "warmup": checkpoint_index < self.args.warmups,
            "path": "A9_HBM_snapshot_slot_lifecycle",
            "slot_id": slot_id, "slot_count": self.slot_count,
            "step": step, "generation": generation,
            "bytes": self.capture_bytes,
            "logical_layout_bytes": self.total,
            "frozen_sha256": frozen_sha256,
            "snapshot_ms": snapshot_ms, "capture_ns": capture_start,
            "slot_wait_ms": wait_ms,
            "events": [
                {"name": "snapshot_start", "monotonic_ns": capture_start},
                {"name": "snapshot_ready", "monotonic_ns": capture_end},
            ],
        }
        self.pool.publish_hbm(
            slot_id,
            [(ptr, end - start) for ptr, start, end in slot["flat"]],
            self.capture_bytes, frozen_sha256, event_token=str(capture_end))
        self.pending.put((slot_id, sample))

    def finish(self):
        self.pending.join()
        self.pending.put(None)
        self.worker.join(timeout=max(30, self.args.io_timeout_s))
        if self.worker.is_alive():
            raise TimeoutError("A9 SPDK owner did not stop")
        if self.errors:
            raise RuntimeError(self.errors[0]["error"])

    def cleanup(self):
        for slot in self.slots:
            tpc.free_flat_hbm_chunks(slot["flat"])


def run_slot_count(args, slot_count):
    writer = ResultWriter("A9_HBM", args)
    writer.config.update({"slot_count": slot_count, "model": "gpt2_xl",
                          "path": "A9_HBM_snapshot_slot_lifecycle",
                          "scope": "real MindSpore HBM slots",
                          "chunk_size": CHUNK_SIZE})
    writer.write_json("config.json", writer.config)
    npu_info = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, npu_info))
    ckpt = None
    runner = None
    try:
        init_env(device_id=args.npu)
        model, dataset, optimizer = make_causal_lm_training(
            "gpt2_xl", total_steps=args.steps + 2, device_id=args.npu,
            seq_len=1025)
        warmup_model(model, optimizer, dataset)
        initial_health = training_numeric_health(model, optimizer)
        if initial_health["nonfinite_arrays"]:
            raise FloatingPointError(
                f"A9 post-warmup state is non-finite: "
                f"{initial_health['nonfinite'][:3]}")
        param_descs = tpc.get_param_descriptors(model)
        ckpt = __import__("direct_checkpoint").DirectCheckpoint(
            nvme_addr=args.pci, npu_device_id=args.npu,
            pipeline_depth=args.pipeline_depth, requested_chunk_size=CHUNK_SIZE,
            rank_id=0, world_size=1, keep_last_n=3, slot_size_gb=10,
            spdk_shm_id=args.shm_id + slot_count)
        runner = HbmSlotRunner(args, model, param_descs, ckpt, writer, slot_count)
        runner.worker.start()
        cell = __import__("direct_checkpoint").ProbeTrainOneStepCell(
            model, optimizer, enable_probe=False, ckpt_interval=9999)
        iterator = dataset.create_tuple_iterator()
        step_times = []
        checkpoint_index = 0
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = dataset.create_tuple_iterator()
                batch = next(iterator)
            start = time.perf_counter_ns()
            loss = cell(*batch)
            tpc.ms.hal.synchronize()
            loss_value = float(np.asarray(loss.asnumpy()).reshape(()))
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    f"A9 non-finite loss at step {step}: {loss_value}")
            step_times.append((time.perf_counter_ns() - start) / 1e6)
            if step % args.ckpt_every == 0:
                runner.capture(step, checkpoint_index)
                checkpoint_index += 1
        runner.finish()
        final_health = training_numeric_health(model, optimizer)
        if final_health["nonfinite_arrays"]:
            raise FloatingPointError(
                f"A9 final state is non-finite: "
                f"{final_health['nonfinite'][:3]}")
        for sample in sorted(runner.results, key=lambda item: item["step"]):
            if not sample["warmup"]:
                writer.add_sample(sample)
        summary = {
            "steps": args.steps,
            "checkpoints": checkpoint_index,
            "formal_samples": len(writer.samples),
            "step_ms": stats(step_times),
            "snapshot_ms": stats([s["snapshot_ms"] for s in runner.results]),
            "slot_wait_ms": stats([s["slot_wait_ms"] for s in runner.results]),
            "end_to_end_ms": stats([
                s["timeline_us"]["end_to_end"] / 1000
                for s in runner.results]),
            "hbm_bytes_per_slot": runner.total,
            "captured_segment_bytes_per_slot": runner.capture_bytes,
            "numeric_health": {"initial": initial_health,
                               "final": final_health},
        }
        result = writer.finalize(summary, status="pass")
        print({"slot_count": slot_count, "status": result["status"],
               "summary": summary}, flush=True)
    except BaseException as error:
        writer.add_failure({"error": repr(error), "slot_count": slot_count})
        writer.finalize({"slot_count": slot_count}, status="fail")
        raise
    finally:
        if runner is not None:
            runner.cleanup()
        if ckpt is not None:
            ckpt.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=1200)
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=125)
    parser.add_argument("--ckpt-every", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--io-timeout-s", type=float, default=120.0)
    parser.add_argument("--io-delay-ms", type=float, default=0.0,
                        help="controlled delay before each SPDK write")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if any(slot <= 0 for slot in args.slots) or args.steps <= 0:
        raise ValueError("invalid A9 dimensions")
    for slot_count in args.slots:
        run_slot_count(args, slot_count)


if __name__ == "__main__":
    main()
