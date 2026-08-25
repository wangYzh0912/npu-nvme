#!/usr/bin/env python3
"""G5 structured performance evidence for the Host -> SPDK -> NVMe path.

This is an evidence gate, not a cross-device benchmark.  It uses the safe
unallocated gap on the formatted 83.0.0 layout, records one run/checkpoint/
request identity per repetition, and keeps Python/API/C timing boundaries
separate.  The target NPU is checked with ``npu-smi`` before SPDK/ACL setup.
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
import statistics
import subprocess
import sys
import time
import uuid


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON = sys.executable
CHUNK_SIZE = 4 * 1024 * 1024
SAFE_OFFSET = 64 * 1024 * 1024 * 1024


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                check=False, timeout=30)
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "returncode": -1, "stdout": "",
                "stderr": repr(error)}


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return stream.read().strip()
    except OSError as error:
        return f"<unavailable: {error}>"


def npu_check(npu_id):
    result = command(["npu-smi", "info"])
    if result["returncode"] != 0:
        raise RuntimeError(f"npu-smi failed: {result['stderr']}")
    # The process table uses: | <id> 0 <pid> <name> |.  Header rows contain
    # the chip name and therefore do not match this expression.
    busy = re.search(rf"\|\s*{npu_id}\s+0\s+\d+\s+\|", result["stdout"])
    if busy:
        raise RuntimeError(f"target NPU {npu_id} is occupied: {busy.group(0)}")
    return result


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "<not-installed>"


def environment_snapshot(args, npu_info):
    pci_path = f"/sys/bus/pci/devices/{args.pci}"
    repo_commit = command(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"])
    spdk_dir = os.path.join(REPO_ROOT, "third_party", "spdk")
    spdk_commit = command(["git", "-C", spdk_dir, "rev-parse", "HEAD"])
    dirty = command(["git", "-C", REPO_ROOT, "status", "--porcelain"])
    submodules = command(["git", "-C", REPO_ROOT, "submodule", "status"])
    return {
        "run_id": args.run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": {
            "path": REPO_ROOT,
            "commit": repo_commit["stdout"].strip(),
            "dirty": bool(dirty["stdout"].strip()),
            "status_porcelain": dirty["stdout"],
            "submodules": submodules["stdout"],
        },
        "spdk": {"path": spdk_dir, "commit": spdk_commit["stdout"].strip()},
        "hardware": {
            "target_pci": args.pci,
            "target_npu": args.npu,
            "target_numa_node": read_text(os.path.join(pci_path, "numa_node")),
            "target_driver": command(["readlink", "-f",
                                       os.path.join(pci_path, "driver")]),
            "target_pci_info": command(["lspci", "-s", args.pci, "-nn"]),
            "protected_pci": "0000:84:00.0",
            "protected_driver": command(["readlink", "-f",
                                           "/sys/bus/pci/devices/0000:84:00.0/driver"]),
            "models_mount": command(["findmnt", "-T", "/models"]),
            "nvme_list": command(["nvme", "list"]),
            "npu_smi_before_init": npu_info,
            "cpu": command(["lscpu"]),
            "numa": command(["numactl", "-H"]),
            "kernel": platform.uname()._asdict(),
        },
        "software": {
            "python": platform.python_version(),
            "mindspore": package_version("mindspore"),
            "numpy": package_version("numpy"),
            "cann_version_info": read_text(
                "/usr/local/Ascend/ascend-toolkit/latest/version.info"),
            "compiler": command(["cc", "--version"]),
        },
        "configuration": {
            "path": "Host buffer -> SPDK -> raw NVMe",
            "chunk_size": CHUNK_SIZE,
            "pipeline_depth": args.pipeline_depth,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "payload_bytes": args.payload_bytes,
            "safe_offset": args.offset,
            "safe_gap": "between V2 FULL end and Delta base",
            "numa_binding": "record-only; no implicit rebinding",
        },
    }


def event(events, name):
    stamp = time.monotonic_ns()
    events.append({"name": name, "monotonic_ns": stamp})
    return stamp


def usage_snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    status = read_text("/proc/self/status")
    voluntary = re.search(r"^voluntary_ctxt_switches:\s+(\d+)$", status, re.M)
    involuntary = re.search(r"^nonvoluntary_ctxt_switches:\s+(\d+)$", status, re.M)
    return {
        "user_cpu_s": usage.ru_utime,
        "system_cpu_s": usage.ru_stime,
        "voluntary_context_switches": int(voluntary.group(1)) if voluntary else None,
        "involuntary_context_switches": int(involuntary.group(1)) if involuntary else None,
    }


def arrays_for(buffer, offset, size):
    ptrs = (ctypes.c_void_p * 1)(ctypes.addressof(buffer))
    offsets = (ctypes.c_uint64 * 1)(offset)
    sizes = (ctypes.c_size_t * 1)(size)
    return ptrs, offsets, sizes


def one_request(ckpt, request_id, offset, payload, warmup):
    from c_bindings import lib

    size = len(payload)
    events = []
    start = event(events, "checkpoint_trigger")
    before = usage_snapshot()
    event(events, "snapshot_start")
    source = ctypes.create_string_buffer(payload, size)
    expected = hashlib.sha256(payload).hexdigest()
    event(events, "snapshot_end")

    write_ptrs, write_offsets, write_sizes = arrays_for(source, offset, size)
    event(events, "write_api_enter")
    write_enter = events[-1]["monotonic_ns"]
    rc = lib.npu_nvme_write_batch_host(
        ckpt.ctx, write_ptrs, write_offsets, write_sizes, 1)
    write_return = event(events, "write_api_return")
    if rc != 0:
        raise RuntimeError(f"request {request_id} write failed: {rc}")
    write_c_us = ckpt.get_last_io_us(False)

    destination = ctypes.create_string_buffer(size)
    read_ptrs, read_offsets, read_sizes = arrays_for(destination, offset, size)
    event(events, "read_api_enter")
    read_enter = events[-1]["monotonic_ns"]
    rc = lib.npu_nvme_read_batch_host(
        ckpt.ctx, read_ptrs, read_offsets, read_sizes, 1)
    read_return = event(events, "read_api_return")
    if rc != 0:
        raise RuntimeError(f"request {request_id} read failed: {rc}")
    read_c_us = ckpt.get_last_io_us(True)

    verify_start = event(events, "verify_start")
    actual = bytes(destination.raw[:size])
    actual_digest = hashlib.sha256(actual).hexdigest()
    if actual != payload or actual_digest != expected:
        raise AssertionError(f"request {request_id} checksum mismatch")
    verify_end = event(events, "verify_end")
    end = event(events, "checkpoint_end")
    after = usage_snapshot()
    return {
        "run_id": request_id.split("/", 1)[0],
        "checkpoint_id": request_id,
        "request_id": request_id,
        "warmup": warmup,
        "offset": offset,
        "bytes": size,
        "sha256": expected,
        "events": events,
        "timeline_ns": {
            "end_to_end": end - start,
            "write_api": write_return - write_enter,
            "read_api": read_return - read_enter,
            "verify": verify_end - verify_start,
        },
        "c_layer_us": {"write": write_c_us, "read": read_c_us},
        "cpu": {
            "before": before,
            "after": after,
            "user_cpu_s": after["user_cpu_s"] - before["user_cpu_s"],
            "system_cpu_s": after["system_cpu_s"] - before["system_cpu_s"],
        },
        "status": "pass",
    }


def summarize(samples):
    def values(path):
        result = []
        for sample in samples:
            value = sample
            for key in path:
                value = value[key]
            result.append(float(value))
        return result

    def stats(items):
        mean = statistics.fmean(items)
        stdev = statistics.stdev(items) if len(items) > 1 else 0.0
        margin = 2.262 * stdev / (len(items) ** 0.5) if len(items) > 1 else 0.0
        return {"n": len(items), "mean": mean, "stdev": stdev,
                "ci95": [mean - margin, mean + margin],
                "min": min(items), "max": max(items)}

    return {
        "end_to_end_us": stats([value / 1000 for value in values(["timeline_ns", "end_to_end"])]),
        "write_api_us": stats([value / 1000 for value in values(["timeline_ns", "write_api"])]),
        "read_api_us": stats([value / 1000 for value in values(["timeline_ns", "read_api"])]),
        "c_write_us": stats(values(["c_layer_us", "write"])),
        "c_read_us": stats(values(["c_layer_us", "read"])),
        "effective_mib_per_s": stats([
            sample["bytes"] / (sample["timeline_ns"]["end_to_end"] / 1e9) / (1024 ** 2)
            for sample in samples]),
        "cpu_user_us": stats([sample["cpu"]["user_cpu_s"] * 1e6 for sample in samples]),
        "cpu_system_us": stats([sample["cpu"]["system_cpu_s"] * 1e6 for sample in samples]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--payload-bytes", type=int, default=CHUNK_SIZE)
    parser.add_argument("--offset", type=int, default=SAFE_OFFSET)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    if args.payload_bytes <= 0 or args.payload_bytes > CHUNK_SIZE:
        raise ValueError("payload-bytes must be in (0, 4 MiB]")
    if args.offset % 4096 or args.payload_bytes % 4096:
        raise ValueError("offset and payload-bytes must be 4 KiB aligned")
    args.run_id = time.strftime("g5_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = os.path.abspath(args.run_dir or os.path.join(
        REPO_ROOT, "experiments", "output", "gates", args.run_id))
    os.makedirs(run_dir, exist_ok=True)
    npu_info = npu_check(args.npu)
    with open(os.path.join(run_dir, "environment.json"), "w", encoding="utf-8") as stream:
        json.dump(environment_snapshot(args, npu_info), stream, indent=2, sort_keys=True, default=str)

    sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
    from direct_checkpoint import DirectCheckpoint

    payload = bytes((index * 17 + 3) % 256 for index in range(args.payload_bytes))
    samples = []
    warmup_samples = []
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu,
        pipeline_depth=args.pipeline_depth, requested_chunk_size=CHUNK_SIZE,
        rank_id=0, world_size=2, keep_last_n=3, slot_size_gb=10,
        spdk_shm_id=83, profiling_dir=os.path.join(run_dir, "profiling"))
    try:
        for index in range(args.warmups):
            warmup_samples.append(one_request(
                ckpt, f"{args.run_id}/warmup_{index:02d}",
                args.offset + index * CHUNK_SIZE, payload, True))
        for index in range(args.repetitions):
            sample = one_request(
                ckpt, f"{args.run_id}/request_{index:04d}",
                args.offset + (args.warmups + index) * CHUNK_SIZE, payload, False)
            samples.append(sample)
            print(f"[G5] request={index:02d} e2e={sample['timeline_ns']['end_to_end']/1000:.1f}us "
                  f"write_c={sample['c_layer_us']['write']}us "
                  f"read_c={sample['c_layer_us']['read']}us", flush=True)
    finally:
        ckpt.cleanup()

    result = {
        "status": "pass",
        "gate": "G5",
        "run_id": args.run_id,
        "config": vars(args),
        "warmups": warmup_samples,
        "samples": samples,
        "summary": summarize(samples),
        "correctness": {"samples_passed": len(samples), "failed_samples_excluded": 0},
        "timing_contract": {
            "clock": "time.monotonic_ns",
            "end_to_end": "checkpoint_trigger -> checkpoint_end",
            "api": "ctypes call enter -> return",
            "c_layer": "C request enqueue -> completion, reported by npu_nvme_get_last_io_us",
            "training_blocking": "not measured by this Host-path gate",
        },
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, default=str)
    print(json.dumps({"status": "pass", "run_id": args.run_id,
                      "summary": result["summary"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
