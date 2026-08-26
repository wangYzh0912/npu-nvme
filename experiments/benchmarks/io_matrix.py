#!/usr/bin/env python3
"""Unified experiment runner for E1 and the I/O-path ablations.

The runner deliberately keeps the raw 83.0.0 experiments in the V2 gap after
the FULL region.  It never calls DirectCheckpoint.save(), because that API is
reserved for correctness checkpoints and would overwrite the live gate
ledger.  The 84.0.0 filesystem path is confined to /models/npu_nvme_exp.

Initial modes:
  E1       Host SPDK sweep (size/depth/read/write)
  A1       filesystem-vs-SPDK Host path
  A3       batch item count sweep
  A4       pipeline depth sweep
  A5       chunk size sweep
  A10      CPU/NUMA label is recorded; run under numactl for each topology

The same result schema is used by the later model and ACL modes.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
CHUNK_SIZE = 4 * 1024 * 1024
SAFE_OFFSET = 64 * 1024 * 1024 * 1024
ALIGNMENT = 4096
FS_ROOT = Path("/models/npu_nvme_exp")


def round_up(value, alignment=ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def run_command(argv, timeout=30):
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                check=False, timeout=timeout)
        return {"argv": list(argv), "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": list(argv), "returncode": -1, "stdout": "",
                "stderr": repr(error)}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "<not-installed>"


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError as error:
        return f"<unavailable: {error}>"


def check_npu_free(npu_id):
    info = run_command(["npu-smi", "info"])
    if info["returncode"] != 0:
        raise RuntimeError(f"npu-smi failed: {info['stderr']}")
    lines = info["stdout"].splitlines()
    # Only inspect the process table.  A chip summary row can contain the
    # same text in an unrelated field (for example NPU0's AICore utilization
    # can be ``7``), so matching the whole npu-smi output by ``| <id>`` gives
    # false positives when another NPU is busy.
    process_header = next(
        (index for index, line in enumerate(lines)
         if "Process id" in line and "Process name" in line),
        None)
    process_lines = lines[process_header + 1:] if process_header is not None else []
    own_pid = str(os.getpid())
    process_row = re.compile(rf"^\|\s*{npu_id}\s+0\s+\d+\s+\|")
    occupied = [line for line in process_lines
                if process_row.search(line) and own_pid not in line]
    if occupied:
        raise RuntimeError(f"NPU {npu_id} appears occupied: {occupied}")
    return info


def usage_snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    status = read_text("/proc/self/status")
    fields = {}
    for key in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
        prefix = key + ":"
        fields[key] = next((int(line.split()[-1]) for line in status.splitlines()
                            if line.startswith(prefix)), None)
    return {"user_cpu_s": usage.ru_utime, "system_cpu_s": usage.ru_stime,
            **fields}


def stats(values):
    values = [float(value) for value in values]
    if not values:
        return {"n": 0}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    # t(0.975, 9) is the mandated 10-sample CI used by the gate scripts.
    margin = 2.262 * stdev / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "stdev": stdev,
            "ci95": [mean - margin, mean + margin],
            "min": min(values), "max": max(values)}


class ResultWriter:
    def __init__(self, experiment, args):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment}_{stamp}_{uuid.uuid4().hex[:8]}"
        root = Path(args.output_root or REPO_ROOT / "experiments" / "output" /
                    "e1-e5-a1-a10")
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.samples_path = self.run_dir / "samples.jsonl"
        self.timeline_path = self.run_dir / "timeline.jsonl"
        self.samples = []
        self.failed = []
        self.config = vars(args).copy()
        self.config.update({"experiment": experiment, "run_id": self.run_id})
        self.write_json("config.json", self.config)

    def write_json(self, name, value):
        with (self.run_dir / name).open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=str)

    def add_sample(self, sample):
        self.samples.append(sample)
        with self.samples_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample, sort_keys=True, default=str) + "\n")
        with self.timeline_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"run_id": self.run_id,
                                     "request_id": sample["request_id"],
                                     "events": sample["events"]},
                                    sort_keys=True) + "\n")

    def add_failure(self, failure):
        self.failed.append(failure)
        with (self.run_dir / "failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(failure, sort_keys=True, default=str) + "\n")

    def finalize(self, summary, status="pass"):
        result = {"status": status, "run_id": self.run_id,
                  "config": self.config, "samples": len(self.samples),
                  "failed_samples": len(self.failed), "summary": summary,
                  "sample_policy": "failed samples excluded from statistics",
                  "paths": {"result": "result.json", "environment": "environment.json",
                            "samples": "samples.jsonl", "timeline": "timeline.jsonl"}}
        self.write_json("result.json", result)
        return result


def environment_snapshot(args, npu_info=None):
    pci = args.pci
    pci_path = Path("/sys/bus/pci/devices") / pci
    spdk = REPO_ROOT / "third_party" / "spdk"
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": {
            "path": str(REPO_ROOT),
            "commit": run_command(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])["stdout"].strip(),
            "branch": run_command(["git", "-C", str(REPO_ROOT), "branch", "--show-current"])["stdout"].strip(),
            "status": run_command(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])["stdout"],
            "spdk_commit": run_command(["git", "-C", str(spdk), "rev-parse", "HEAD"])["stdout"].strip(),
        },
        "hardware": {
            "target_pci": pci,
            "target_numa_node": read_text(pci_path / "numa_node"),
            "target_driver": run_command(["readlink", "-f", str(pci_path / "driver")]),
            "target_pci_info": run_command(["lspci", "-s", pci, "-nn"]),
            "filesystem_pci": "0000:84:00.0",
            "filesystem_mount": run_command(["findmnt", "-T", "/models"]),
            "filesystem_device": run_command(["findmnt", "-n", "-o", "SOURCE", "-T", "/models"]),
            "cpu": run_command(["lscpu"]),
            "numa": run_command(["numactl", "-H"]),
            "kernel": platform.uname()._asdict(),
            "npu_smi_before_init": npu_info,
        },
        "software": {
            "python": platform.python_version(),
            "mindspore": package_version("mindspore"),
            "numpy": package_version("numpy"),
            "cann": read_text("/usr/local/Ascend/ascend-toolkit/latest/version.info"),
            "compiler": run_command(["cc", "--version"]),
        },
    }


def make_payload(size, item_id=0, seed=17):
    # The pattern has period 256 for the formal integer seeds.  Constructing
    # the period once keeps the 1 GiB E2 synthetic control CPU-bound only by
    # allocation/copy, rather than by a Python byte-at-a-time generator.
    period = bytes((index * seed + item_id * 29 + 3) % 256
                   for index in range(256))
    return period * (size // len(period)) + period[:size % len(period)]


def host_arrays(buffers, offsets):
    count = len(buffers)
    ptrs = (ctypes.c_void_p * count)()
    offs = (ctypes.c_uint64 * count)()
    sizes = (ctypes.c_size_t * count)()
    for index, (buffer, offset) in enumerate(zip(buffers, offsets)):
        ptrs[index] = ctypes.addressof(buffer)
        offs[index] = offset
        sizes[index] = len(buffer)
    return ptrs, offs, sizes


def host_chunk_arrays(buffers, offsets, chunk_size):
    """Split large host objects into pointers accepted by the DMA limit."""
    ptr_values = []
    offset_values = []
    size_values = []
    for buffer, base_offset in zip(buffers, offsets):
        for inner in range(0, len(buffer), chunk_size):
            ptr_values.append(ctypes.addressof(buffer) + inner)
            offset_values.append(base_offset + inner)
            size_values.append(min(chunk_size, len(buffer) - inner))
    count = len(ptr_values)
    ptrs = (ctypes.c_void_p * count)()
    offs = (ctypes.c_uint64 * count)()
    sizes = (ctypes.c_size_t * count)()
    for index, (ptr, offset, size) in enumerate(
            zip(ptr_values, offset_values, size_values)):
        ptrs[index] = ptr
        offs[index] = offset
        sizes[index] = size
    return ptrs, offs, sizes


def host_spdk_sample(ckpt, writer, item_bytes, items, index, warmup, base_offset):
    from c_bindings import lib

    total = item_bytes * items
    offset = base_offset + index * round_up(total)
    payloads = [make_payload(item_bytes, item_id) for item_id in range(items)]
    sources = [ctypes.create_string_buffer(payload, len(payload))
               for payload in payloads]
    events = []
    start = time.monotonic_ns()
    events.append({"name": "checkpoint_trigger", "monotonic_ns": start})
    before = usage_snapshot()
    events.append({"name": "snapshot_end", "monotonic_ns": time.monotonic_ns(),
                   "snapshot": "host_buffer_generation"})
    item_offsets = [offset + item_id * item_bytes for item_id in range(items)]
    ptrs, offs, sizes = host_chunk_arrays(sources, item_offsets, ckpt.chunk_size)
    write_enter = time.monotonic_ns()
    events.append({"name": "write_api_enter", "monotonic_ns": write_enter})
    rc = lib.npu_nvme_write_batch_host(
        ckpt.ctx, ptrs, offs, sizes, len(sizes))
    write_return = time.monotonic_ns()
    events.append({"name": "write_api_return", "monotonic_ns": write_return,
                   "rc": rc})
    if rc != 0:
        raise RuntimeError(f"write_batch_host returned {rc}")
    c_write = ckpt.get_last_io_us(False)

    destinations = [ctypes.create_string_buffer(item_bytes)
                    for _ in range(items)]
    read_ptrs, read_offs, read_sizes = host_chunk_arrays(
        destinations, item_offsets, ckpt.chunk_size)
    read_enter = time.monotonic_ns()
    events.append({"name": "read_api_enter", "monotonic_ns": read_enter})
    rc = lib.npu_nvme_read_batch_host(
        ckpt.ctx, read_ptrs, read_offs, read_sizes, len(read_sizes))
    read_return = time.monotonic_ns()
    events.append({"name": "read_api_return", "monotonic_ns": read_return,
                   "rc": rc})
    if rc != 0:
        raise RuntimeError(f"read_batch_host returned {rc}")
    c_read = ckpt.get_last_io_us(True)

    verify_start = time.monotonic_ns()
    digests = []
    for expected, actual in zip(payloads, destinations):
        value = bytes(actual.raw[:item_bytes])
        if value != expected:
            raise AssertionError("Host round-trip content mismatch")
        digests.append(hashlib.sha256(value).hexdigest())
    verify_end = time.monotonic_ns()
    events.append({"name": "verify_end", "monotonic_ns": verify_end})
    end = time.monotonic_ns()
    events.append({"name": "checkpoint_end", "monotonic_ns": end})
    after = usage_snapshot()
    return {
        "run_id": writer.run_id, "checkpoint_id": f"checkpoint_{index:04d}",
        "request_id": f"{writer.run_id}/request_{index:04d}",
        "warmup": warmup, "path": "host_spdk", "offset": offset,
        "bytes": total, "items": items, "item_bytes": item_bytes,
        "sha256": digests, "status": "pass", "events": events,
        "timeline_us": {"end_to_end": (end - start) / 1000,
                         "write_api": (write_return - write_enter) / 1000,
                         "read_api": (read_return - read_enter) / 1000,
                         "verify": (verify_end - verify_start) / 1000},
        "c_layer_us": {"write": c_write, "read": c_read},
        "cpu": {"before": before, "after": after},
    }


def run_host_spdk(args, writer, item_bytes, items, pipeline_depth):
    check = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, check))
    sys.path.insert(0, str(REPO_ROOT / "python"))
    from direct_checkpoint import DirectCheckpoint

    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu,
        pipeline_depth=pipeline_depth,
        requested_chunk_size=min(item_bytes, CHUNK_SIZE),
        rank_id=0, world_size=1, keep_last_n=3, slot_size_gb=10,
        spdk_shm_id=args.shm_id,
        profiling_dir=str(writer.run_dir / "profiling"))
    try:
        total = args.warmups + args.repetitions
        for index in range(total):
            try:
                sample = host_spdk_sample(
                    ckpt, writer, item_bytes, items, index, index < args.warmups,
                    args.offset)
                if index >= args.warmups:
                    writer.add_sample(sample)
            except BaseException as error:
                writer.add_failure({"index": index, "warmup": index < args.warmups,
                                    "error": repr(error)})
                if index < args.warmups:
                    raise
    finally:
        ckpt.cleanup()


def fs_roundtrip_sample(path, payloads, index, warmup, writer):
    start = time.monotonic_ns()
    events = [{"name": "checkpoint_trigger", "monotonic_ns": start}]
    payload = b"".join(payloads)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb", buffering=1024 * 1024) as stream:
        write_enter = time.monotonic_ns()
        events.append({"name": "write_api_enter", "monotonic_ns": write_enter})
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        write_end = time.monotonic_ns()
    events.append({"name": "write_api_return", "monotonic_ns": write_end})
    with path.open("rb", buffering=1024 * 1024) as stream:
        read_enter = time.monotonic_ns()
        events.append({"name": "read_api_enter", "monotonic_ns": read_enter})
        actual = stream.read()
        read_end = time.monotonic_ns()
    events.append({"name": "read_api_return", "monotonic_ns": read_end})
    if actual != payload:
        raise AssertionError("filesystem round-trip content mismatch")
    end = time.monotonic_ns()
    events.append({"name": "checkpoint_end", "monotonic_ns": end})
    return {"run_id": writer.run_id, "checkpoint_id": f"checkpoint_{index:04d}",
            "request_id": f"{writer.run_id}/request_{index:04d}",
            "warmup": warmup, "path": "filesystem", "bytes": len(payload),
            "items": len(payloads), "item_bytes": len(payloads[0]),
            "sha256": hashlib.sha256(actual).hexdigest(), "status": "pass",
            "events": events,
            "timeline_us": {"end_to_end": (end - start) / 1000,
                             "write_api": (write_end - write_enter) / 1000,
                             "read_api": (read_end - read_enter) / 1000}}


def run_filesystem(args, writer, item_bytes, items):
    root = FS_ROOT / writer.run_id
    root.mkdir(parents=True, exist_ok=False)
    writer.write_json("environment.json", environment_snapshot(args))
    try:
        for index in range(args.warmups + args.repetitions):
            payloads = [make_payload(item_bytes, item_id)
                        for item_id in range(items)]
            path = root / f"sample_{index:04d}.bin"
            try:
                sample = fs_roundtrip_sample(
                    path, payloads, index, index < args.warmups, writer)
                if index >= args.warmups:
                    writer.add_sample(sample)
            except BaseException as error:
                writer.add_failure({"index": index, "warmup": index < args.warmups,
                                    "error": repr(error)})
                if index < args.warmups:
                    raise
    finally:
        shutil.rmtree(root)


def summarize_samples(samples):
    if not samples:
        return {"n": 0}
    return {
        "end_to_end_us": stats([sample["timeline_us"]["end_to_end"] for sample in samples]),
        "write_api_us": stats([sample["timeline_us"]["write_api"] for sample in samples]),
        "read_api_us": stats([sample["timeline_us"]["read_api"] for sample in samples]),
        "effective_mib_per_s": stats([
            sample["bytes"] / (sample["timeline_us"]["end_to_end"] / 1e6) / (1024 ** 2)
            for sample in samples]),
        "c_write_us": stats([sample["c_layer_us"]["write"] for sample in samples
                              if "c_layer_us" in sample]),
        "c_read_us": stats([sample["c_layer_us"]["read"] for sample in samples
                             if "c_layer_us" in sample]),
    }


def matrix(experiment, selected_size=None, selected_depths=None):
    sizes = [4 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024,
             4 * 1024 * 1024, 16 * 1024 * 1024]
    if selected_size is not None:
        sizes = [selected_size]
    depths = tuple(selected_depths or (1, 4, 8, 16))
    if experiment == "E1":
        return [("host_spdk", size, 1, depth) for size in sizes
                for depth in depths]
    if experiment == "E2":
        # E2's synthetic control is intentionally placed by the caller in a
        # separate raw region (128 GiB in the formal run) so it cannot overlap
        # the GPT-2 XL and metadata safety regions used by the other runs.
        return [("filesystem", 1024 * 1024 * 1024, 1, 1),
                ("host_spdk", 1024 * 1024 * 1024, 1, 4)]
    if experiment == "A1":
        return [("filesystem", 4 * 1024 * 1024, 1, 1),
                ("host_spdk", 4 * 1024 * 1024, 1, 4)]
    if experiment == "A3":
        return [("host_spdk", 1024 * 1024, items, 4)
                for items in (1, 4, 16, 64)]
    if experiment == "A4":
        return [("host_spdk", 4 * 1024 * 1024, 1, depth)
                for depth in (selected_depths or (1, 2, 4, 8, 16))]
    if experiment == "A5":
        # A5 explicitly reports the requested chunk-size range; keep the
        # 64 KiB point even though E1 also includes the 4 KiB latency point.
        return [("host_spdk", size, 1, 4)
                for size in (64 * 1024, 256 * 1024, 1024 * 1024,
                             4 * 1024 * 1024, 16 * 1024 * 1024)]
    if experiment == "A6":
        # Controlled single-owner comparison: depth=1 is the synchronous
        # control, depth=4 enables the request ring/FSM overlap.
        return [("host_spdk", 4 * 1024 * 1024, 1, depth)
                for depth in (1, 4)]
    if experiment == "A9":
        # Slot-pressure sweep using the same single-owner request path.  The
        # depth is the bounded in-flight slot count; no unsafe multi-owner
        # SPDK contexts are created for this ablation.
        return [("host_spdk", 4 * 1024 * 1024, 1, depth)
                for depth in (1, 2, 4, 8)]
    if experiment == "A10":
        return [("host_spdk", 4 * 1024 * 1024, 1, 4)]
    raise ValueError(f"unsupported initial experiment {experiment}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("E1", "E2", "A1", "A3", "A4", "A5",
                                                   "A6", "A9", "A10"),
                        required=True)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=83)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--offset", type=int, default=SAFE_OFFSET)
    parser.add_argument("--item-bytes", type=int, default=None,
                        help="restrict a matrix to one item size")
    parser.add_argument("--depths", type=int, nargs="+", default=None,
                        help="restrict E1/A4 to selected pipeline depths")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.offset % ALIGNMENT:
        raise ValueError("offset must be 4 KiB aligned")
    if args.repetitions < 10:
        raise ValueError("formal repetitions must be at least 10")

    for path, item_bytes, items, depth in matrix(
            args.experiment, args.item_bytes, args.depths):
        if item_bytes % ALIGNMENT:
            raise ValueError("item size must be 4 KiB aligned")
        writer = ResultWriter(args.experiment, args)
        writer.config.update({"path": path, "item_bytes": item_bytes,
                              "items": items, "pipeline_depth": depth,
                              "cpu_affinity": os.sched_getaffinity(0)})
        writer.write_json("config.json", writer.config)
        if path == "filesystem":
            run_filesystem(args, writer, item_bytes, items)
        else:
            run_host_spdk(args, writer, item_bytes, items, depth)
        status = "pass" if not writer.failed and len(writer.samples) == args.repetitions else "fail"
        result = writer.finalize(summarize_samples(writer.samples), status=status)
        print(json.dumps({"run_id": writer.run_id, "status": result["status"],
                          "path": path, "item_bytes": item_bytes,
                          "items": items, "pipeline_depth": depth,
                          "summary": result["summary"]},
                         indent=2, sort_keys=True), flush=True)
        if status != "pass":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
