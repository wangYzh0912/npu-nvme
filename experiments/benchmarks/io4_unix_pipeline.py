#!/usr/bin/env python3
"""IO-4 B2/B3/B4 Unix-stream and single-Reactor decomposition."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import queue
import resource
import select
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from c_bindings import NPUNVMEContext, NPUNVMEStats, acl_lib, lib  # noqa: E402
from ppt_evidence import command, environment_snapshot  # noqa: E402

HEADER = struct.Struct("!IIQ")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def recv_exact(sock, size):
    parts = []
    while size:
        value = sock.recv(size)
        if not value:
            raise EOFError("producer socket closed")
        parts.append(value)
        size -= len(value)
    return b"".join(parts)


def send_nonblocking(sock, payload):
    view = memoryview(payload)
    blocked_ns = 0
    calls = 0
    while view:
        calls += 1
        try:
            sent = sock.send(view)
            view = view[sent:]
        except BlockingIOError:
            started = time.perf_counter_ns()
            select.select([], [sock], [])
            blocked_ns += time.perf_counter_ns() - started
    return blocked_ns, calls


def pattern(rank, size):
    value = np.arange(size, dtype=np.uint8)
    value[:] = (value * 131 + rank * 17 + 3) & 0xFF
    return value


def producer(args):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + 120
    while True:
        try:
            sock.connect(args.socket)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    sock.setblocking(False)
    base = pattern(args.rank_id, args.chunk)
    host = (ctypes.c_ubyte * args.chunk).from_buffer(base)
    device = ctypes.c_void_p()
    d2h_ns = 0
    if args.source == "hbm":
        if acl_lib is None:
            raise RuntimeError("ACL runtime unavailable")
        if acl_lib.aclrtSetDevice(args.npu) != 0:
            raise RuntimeError("aclrtSetDevice failed")
        if acl_lib.aclrtMalloc(ctypes.byref(device), args.chunk, 0) != 0:
            raise RuntimeError("aclrtMalloc failed")
        if acl_lib.aclrtMemcpy(device, args.chunk, host, args.chunk, 1) != 0:
            raise RuntimeError("H2D precondition failed")
    blocked_ns = 0
    send_call_ns = 0
    calls = 0
    digest = hashlib.sha256()
    sent_bytes = 0
    try:
        for sequence, offset in enumerate(range(0, args.payload, args.chunk)):
            size = min(args.chunk, args.payload - offset)
            if args.source == "hbm":
                started = time.perf_counter_ns()
                if acl_lib.aclrtMemcpy(host, args.chunk, device, args.chunk, 2) != 0:
                    raise RuntimeError("D2H failed")
                d2h_ns += time.perf_counter_ns() - started
            payload = memoryview(base)[:size]
            digest.update(payload)
            frame = HEADER.pack(args.rank_id, size, sequence)
            started = time.perf_counter_ns()
            blocked, count = send_nonblocking(sock, frame)
            more_blocked, more_count = send_nonblocking(sock, payload)
            send_call_ns += time.perf_counter_ns() - started
            blocked_ns += blocked + more_blocked
            calls += count + more_count
            sent_bytes += size
        send_nonblocking(sock, HEADER.pack(args.rank_id, 0, sent_bytes))
    finally:
        sock.close()
        if device:
            acl_lib.aclrtFree(device)
    print(json.dumps({"rank": args.rank_id, "bytes": sent_bytes,
                      "sha256": digest.hexdigest(), "d2h_ns": d2h_ns,
                      "socket_send_call_ns": send_call_ns,
                      "socket_send_blocked_ns": blocked_ns,
                      "send_syscalls": calls}, sort_keys=True), flush=True)


def reader(conn, output, errors):
    try:
        while True:
            started = time.perf_counter_ns()
            rank, size, sequence = HEADER.unpack(recv_exact(conn, HEADER.size))
            if size == 0:
                output.put(("done", rank, sequence, time.monotonic_ns(), b"", 0))
                return
            payload = recv_exact(conn, size)
            output.put(("data", rank, sequence, time.monotonic_ns(), payload,
                        time.perf_counter_ns() - started))
    except BaseException as error:
        errors.put(repr(error))
    finally:
        conn.close()


def write_host(ctx, payload, offset):
    buffer = ctypes.create_string_buffer(payload, len(payload))
    ptrs = (ctypes.c_void_p * 1)(ctypes.addressof(buffer))
    offsets = (ctypes.c_uint64 * 1)(offset)
    sizes = (ctypes.c_size_t * 1)(len(payload))
    rc = lib.npu_nvme_write_batch_host(ctx, ptrs, offsets, sizes, 1)
    if rc != 0:
        raise RuntimeError(f"SPDK write failed: {rc}")


def read_host(ctx, size, offset):
    buffer = ctypes.create_string_buffer(size)
    ptrs = (ctypes.c_void_p * 1)(ctypes.addressof(buffer))
    offsets = (ctypes.c_uint64 * 1)(offset)
    sizes = (ctypes.c_size_t * 1)(size)
    rc = lib.npu_nvme_read_batch_host(ctx, ptrs, offsets, sizes, 1)
    if rc != 0:
        raise RuntimeError(f"SPDK read failed: {rc}")
    return buffer.raw[:size]


def direct_host_spdk(args):
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = run_dir / "raw"
    raw.mkdir(exist_ok=True)
    context = ctypes.POINTER(NPUNVMEContext)()
    chunk = pattern(0, args.chunk)
    host = (ctypes.c_ubyte * args.chunk).from_buffer(chunk)
    descriptors = [(offset, min(args.chunk, args.payload - offset))
                   for offset in range(0, args.payload, args.chunk)]
    ptrs = (ctypes.c_void_p * len(descriptors))(
        *([ctypes.addressof(host)] * len(descriptors)))
    offsets = (ctypes.c_uint64 * len(descriptors))(
        *(args.offset + offset for offset, _ in descriptors))
    sizes = (ctypes.c_size_t * len(descriptors))(*(size for _, size in descriptors))
    try:
        rc = lib.npu_nvme_init(ctypes.byref(context), args.pci.encode(),
                               args.coordinator_npu, args.depth, args.chunk,
                               False, str(raw).encode())
        if rc != 0:
            raise RuntimeError(f"npu_nvme_init failed: {rc}")
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        started = time.perf_counter_ns()
        rc = lib.npu_nvme_write_batch_host(
            context, ptrs, offsets, sizes, len(descriptors))
        service_ns = time.perf_counter_ns() - started
        if rc != 0:
            raise RuntimeError(f"SPDK host batch failed: {rc}")
        flush_started = time.perf_counter_ns()
        if lib.npu_nvme_flush(context) != 0:
            raise RuntimeError("SPDK flush failed")
        flush_ns = time.perf_counter_ns() - flush_started
        elapsed_ns = time.perf_counter_ns() - started
        actual = hashlib.sha256()
        expected = hashlib.sha256()
        for offset, size in descriptors:
            expected.update(memoryview(chunk)[:size])
            actual.update(read_host(context, size, args.offset + offset))
        counters = NPUNVMEStats()
        if lib.npu_nvme_get_stats(context, ctypes.byref(counters)) != 0:
            raise RuntimeError("stats query failed")
        stats = {name: int(getattr(counters, name)) for name, _ in counters._fields_}
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        result = {
            "status": "pass" if actual.digest() == expected.digest() else "fail",
            "path": "B0", "source": "host", "sink": "spdk",
            "publishes_generation": False, "producers": 1,
            "payload_per_producer": args.payload, "chunk": args.chunk,
            "pipeline_depth": args.depth, "elapsed_ns": elapsed_ns,
            "spdk_write_ns": service_ns, "flush_ns": flush_ns,
            "throughput_bytes_per_second": args.payload / (elapsed_ns / 1e9),
            "byte_exact": actual.digest() == expected.digest(),
            "coordinator_cpu_seconds": (
                cpu_after.ru_utime - cpu_before.ru_utime +
                cpu_after.ru_stime - cpu_before.ru_stime),
            "spdk_stats": stats,
        }
        atomic_json(run_dir / "config.json", vars(args))
        atomic_json(run_dir / "environment.json", environment_snapshot(
            pci=args.pci, npu=str(args.coordinator_npu), repo_root=ROOT,
            npu_info=command(["npu-smi", "info"])))
        atomic_json(run_dir / "result.json", result)
        (run_dir / "events.jsonl").write_text(json.dumps({
            "run_id": run_dir.name, "rank": 0, "request_id": "b0-host-batch",
            "spdk_service_ns": service_ns, "flush_ns": flush_ns,
        }, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
        raise SystemExit(0 if result["status"] == "pass" else 1)
    finally:
        if context:
            lib.npu_nvme_cleanup(context)


def coordinator(args):
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = run_dir / "raw"
    raw.mkdir(exist_ok=True)
    socket_path = str(run_dir / "pipeline.sock")
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(args.producers)
    context = ctypes.POINTER(NPUNVMEContext)()
    if args.sink == "spdk":
        rc = lib.npu_nvme_init(ctypes.byref(context), args.pci.encode(),
                               args.coordinator_npu, args.depth, args.chunk,
                               False, str(raw).encode())
        if rc != 0:
            raise RuntimeError(f"npu_nvme_init failed: {rc}")
    producer_source = "hbm" if args.path == "B3" else "host"
    producer_argv = []
    processes = []
    devices = [int(item) for item in args.rank_devices.split(",")]
    started = time.perf_counter_ns()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    for rank in range(args.producers):
        argv = [sys.executable, str(Path(__file__).resolve()), "--child",
                "--socket", socket_path, "--rank-id", str(rank),
                "--npu", str(devices[rank]), "--source", producer_source,
                "--payload", str(args.payload), "--chunk", str(args.chunk)]
        producer_argv.append(argv)
        processes.append(subprocess.Popen(argv, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True))
    incoming = queue.Queue(maxsize=max(1, args.producers * args.depth))
    errors = queue.Queue()
    readers = []
    for _ in range(args.producers):
        conn, _ = server.accept()
        thread = threading.Thread(target=reader, args=(conn, incoming, errors), daemon=True)
        thread.start()
        readers.append(thread)
    digests = {rank: hashlib.sha256() for rank in range(args.producers)}
    events = []
    done = set()
    queue_wait_ns = 0
    receive_ns = 0
    spdk_write_ns = 0
    while len(done) < args.producers:
        kind, rank, sequence, enqueued_ns, payload, recv_ns = incoming.get(timeout=120)
        queue_wait_ns += time.monotonic_ns() - enqueued_ns
        receive_ns += recv_ns
        if kind == "done":
            done.add(rank)
            continue
        digests[rank].update(payload)
        service_ns = 0
        if args.sink_delay_ms:
            time.sleep(args.sink_delay_ms / 1000.0)
        if context:
            service_started = time.perf_counter_ns()
            write_host(context, payload, args.offset + rank * args.payload +
                       sequence * args.chunk)
            service_ns = time.perf_counter_ns() - service_started
            spdk_write_ns += service_ns
        events.append({"run_id": run_dir.name, "rank": rank,
                       "request_id": f"rank{rank}-chunk{sequence}",
                       "sequence": sequence, "bytes": len(payload),
                       "coordinator_queue_wait_ns": time.monotonic_ns() - enqueued_ns,
                       "coordinator_receive_ns": recv_ns,
                       "spdk_service_ns": service_ns})
    flush_ns = 0
    stats = {}
    if context:
        flush_started = time.perf_counter_ns()
        if lib.npu_nvme_flush(context) != 0:
            raise RuntimeError("SPDK flush failed")
        flush_ns = time.perf_counter_ns() - flush_started
        counters = NPUNVMEStats()
        if lib.npu_nvme_get_stats(context, ctypes.byref(counters)) != 0:
            raise RuntimeError("stats query failed")
        stats = {name: int(getattr(counters, name)) for name, _ in counters._fields_}
    elapsed_ns = time.perf_counter_ns() - started
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    producer_results = []
    for rank, process in enumerate(processes):
        stdout, stderr = process.communicate(timeout=120)
        (raw / f"producer_{rank}.stderr.log").write_text(stderr)
        if process.returncode != 0:
            raise RuntimeError(f"producer {rank} failed: {stderr}")
        producer_results.append(json.loads(stdout))
    for thread in readers:
        thread.join(timeout=5)
    if not errors.empty():
        raise RuntimeError(errors.get())
    byte_exact = all(digests[item["rank"]].hexdigest() == item["sha256"]
                     for item in producer_results)
    if context:
        for item in producer_results:
            digest = hashlib.sha256()
            for offset in range(0, args.payload, args.chunk):
                size = min(args.chunk, args.payload - offset)
                digest.update(read_host(context, size, args.offset +
                                        item["rank"] * args.payload + offset))
            byte_exact = byte_exact and digest.hexdigest() == item["sha256"]
    cpu_seconds = ((cpu_after.ru_utime - cpu_before.ru_utime) +
                   (cpu_after.ru_stime - cpu_before.ru_stime))
    result = {
        "status": "pass" if byte_exact else "fail", "path": args.path,
        "source": producer_source, "sink": args.sink,
        "publishes_generation": False, "producers": args.producers,
        "payload_per_producer": args.payload, "chunk": args.chunk,
        "pipeline_depth": args.depth, "elapsed_ns": elapsed_ns,
        "sink_delay_ms_per_chunk": args.sink_delay_ms,
        "throughput_bytes_per_second": args.payload * args.producers / (elapsed_ns / 1e9),
        "coordinator_receive_ns": receive_ns,
        "coordinator_queue_wait_ns": queue_wait_ns,
        "spdk_write_ns": spdk_write_ns, "flush_ns": flush_ns,
        "coordinator_cpu_seconds": cpu_seconds, "byte_exact": byte_exact,
        "producer_results": producer_results, "spdk_stats": stats,
    }
    atomic_json(run_dir / "config.json", vars(args))
    atomic_json(run_dir / "environment.json", environment_snapshot(
        pci=args.pci, npu=args.rank_devices, repo_root=ROOT,
        npu_info=command(["npu-smi", "info"])))
    atomic_json(run_dir / "result.json", result)
    with (run_dir / "events.jsonl").open("w") as stream:
        for event in events:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    if context:
        lib.npu_nvme_cleanup(context)
    server.close()
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    print(json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(0 if result["status"] == "pass" else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--path", choices=("B0", "B2", "B3", "B4"), default="B2")
    parser.add_argument("--source", choices=("host", "hbm"), default="host")
    parser.add_argument("--sink", choices=("memory", "spdk"), default="memory")
    parser.add_argument("--producers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--payload", type=int, default=256 * 1024**2)
    parser.add_argument("--chunk", type=int, default=4 * 1024**2)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--sink-delay-ms", type=float, default=0.0)
    parser.add_argument("--rank-devices", default="2,3,0,1")
    parser.add_argument("--coordinator-npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--offset", type=int, default=160 * 1024**3)
    parser.add_argument("--run-dir", default="/tmp/io4-unix-pipeline")
    parser.add_argument("--socket")
    parser.add_argument("--rank-id", type=int, default=0)
    parser.add_argument("--npu", type=int, default=2)
    args = parser.parse_args()
    if args.child:
        producer(args)
        return
    if args.path == "B0":
        direct_host_spdk(args)
        return
    if args.path == "B2":
        args.source, args.sink = "host", "memory"
    elif args.path == "B3":
        args.source, args.sink = "hbm", "memory"
    else:
        args.source, args.sink = "host", "spdk"
    if args.chunk <= 0 or args.chunk % 4096 or args.payload <= 0:
        raise ValueError("payload must be positive; chunk must be 4 KiB aligned")
    if len(args.rank_devices.split(",")) < args.producers:
        raise ValueError("not enough rank devices")
    coordinator(args)


if __name__ == "__main__":
    main()
