#!/usr/bin/env python3
"""Isolated HW1 fault/reopen gate. The orchestrator never opens ACL or SPDK."""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from experiment_evidence import command, environment_snapshot, sha256_file

ALIGN = 4096
OFFSET = 64 * 1024**3 + 512 * 1024**2
PAYLOAD = b"stage4-fault-lifecycle".ljust(ALIGN, b"\0")
FAULTS = {
    "acl_copy": ("NPU_NVME_TEST_FAIL_ACL_COPY", "async"),
    "event_query": ("NPU_NVME_TEST_FAIL_EVENT_QUERY", "async"),
    "event_record": ("NPU_NVME_TEST_FAIL_EVENT_RECORD", "async"),
    "nvme_submit": ("NPU_NVME_TEST_FAIL_NVME_SUBMIT", "host"),
    "nvme_completion": ("NPU_NVME_TEST_FAIL_NVME_COMPLETION", "host"),
    "metadata_write": ("NPU_NVME_TEST_FAIL_METADATA_WRITE", "metadata"),
    "flush": ("NPU_NVME_TEST_FAIL_FLUSH", "flush"),
    "timeout": ("NPU_NVME_TEST_META_DELAY_MS", "timeout"),
}
CRASHES = {"before_data_complete": 86, "before_metadata_commit": 87}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def manifest(directory):
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "evidence_manifest.json":
            records.append({"path": str(path.relative_to(directory)),
                            "size": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(directory / "evidence_manifest.json", {"artifacts": records})


def resource_snapshot():
    return {"meminfo": Path("/proc/meminfo").read_text(),
            "process_status": Path("/proc/self/status").read_text(),
            "fd_count": len(list(Path("/proc/self/fd").iterdir()))}


class Worker:
    def __init__(self, args, record):
        from npu_nvme.storage.bindings import (load_backend, NPUNVMEContext,
            NPUNVMERequest, NPUNVMEStats, NPUNVMERetainedSlot)
        self.args, self.record = args, record
        self.request_type, self.stats_type = NPUNVMERequest, NPUNVMEStats
        self.slot_type = NPUNVMERetainedSlot
        backend = load_backend(args.library)
        self.lib, self.acl = backend.lib, backend.acl_lib
        require(self.lib is not None and self.acl is not None, "backend unavailable")
        self.acl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
        self.acl.aclrtMalloc.restype = ctypes.c_int
        self.acl.aclrtFree.argtypes = [ctypes.c_void_p]
        self.acl.aclrtFree.restype = ctypes.c_int
        self.ctx = ctypes.POINTER(NPUNVMEContext)()
        self.host_buffers, self.device_buffers, self.requests = [], [], []

    def open(self):
        rc = self.lib.npu_nvme_init(ctypes.byref(self.ctx), self.args.pci.encode(),
            self.args.npu, self.args.depth, ALIGN, True, str(self.args.output).encode())
        self.record["init_rc"] = rc
        require(rc == 0 and bool(self.ctx), f"init failed: {rc}")
        require(self.lib.npu_nvme_get_total_blocks(self.ctx) >= OFFSET + 2 * ALIGN,
                "scratch extent exceeds device capacity")

    def buffer(self, payload=PAYLOAD):
        value = ctypes.create_string_buffer(payload, len(payload))
        self.host_buffers.append(value)  # pins survive every timeout until close succeeds
        return value

    def arrays(self, pointer, offset=OFFSET):
        return ((ctypes.c_void_p * 1)(pointer), (ctypes.c_uint64 * 1)(offset),
                (ctypes.c_size_t * 1)(ALIGN))

    def host_write(self):
        value = self.buffer()
        return self.lib.npu_nvme_write_batch_host(self.ctx,
            *self.arrays(ctypes.addressof(value)), 1)

    def device_source(self):
        source = ctypes.c_void_p()
        require(self.acl.aclrtSetDevice(self.args.npu) == 0, "set device failed")
        require(self.acl.aclrtMalloc(ctypes.byref(source), ALIGN, 0) == 0, "HBM allocation failed")
        self.device_buffers.append(source)
        require(self.acl.aclrtMemcpy(source, ALIGN, self.buffer(), ALIGN, 1) == 0, "H2D failed")
        return source.value

    def submit(self, pointer, host=False):
        request = ctypes.POINTER(self.request_type)()
        submit = self.lib.npu_nvme_submit_write_batch_host if host else self.lib.npu_nvme_submit_write_batch
        rc = submit(self.ctx, *self.arrays(pointer), 1, ctypes.byref(request))
        if rc == 0:
            self.requests.append(request)
        return rc, request

    def snapshot(self):
        stats = self.stats_type()
        require(self.lib.npu_nvme_get_stats(self.ctx, ctypes.byref(stats)) == 0, "stats unavailable")
        slots = (self.slot_type * self.args.depth)()
        count = ctypes.c_uint32()
        rc = self.lib.npu_nvme_get_retained_slots(self.ctx, slots, self.args.depth, ctypes.byref(count))
        require(rc == 0, f"retained slots unavailable: {rc}")
        return {"stats": {name: int(getattr(stats, name)) for name, _ in stats._fields_},
                "retained_slots": [{name: int(getattr(slot, name)) for name, _ in slot._fields_}
                                   for slot in slots[:count.value]]}

    def superblock(self):
        value = self.buffer()
        require(self.lib.npu_nvme_sync_meta_io(self.ctx, 0, ALIGN, 1, value) == 0,
                "superblock read failed")
        return hashlib.sha256(value.raw).hexdigest()

    def verify(self):
        self.record["superblock_sha256"] = self.superblock()
        require(self.host_write() == 0, "healthy write failed")
        require(self.lib.npu_nvme_flush(self.ctx) == 0, "healthy flush failed")
        value = self.buffer(b"\0" * ALIGN)
        rc = self.lib.npu_nvme_read_batch_host(self.ctx, *self.arrays(ctypes.addressof(value)), 1)
        require(rc == 0 and value.raw == PAYLOAD, "healthy readback mismatch")
        self.record["readback_sha256"] = hashlib.sha256(value.raw).hexdigest()

    def fault(self, name):
        env_name, operation = FAULTS[name]
        os.environ[env_name] = "500" if name == "timeout" else "1"
        self.record["fault"] = {env_name: os.environ[env_name]}
        if name == "timeout":
            require(self.lib.npu_nvme_set_io_timeout_ms(self.ctx, 50) == 0, "set I/O timeout failed")
        started = time.monotonic()
        if operation == "async":
            rc, request = self.submit(self.device_source())
            require(rc == 0, f"request admission failed before injection: {rc}")
            rc = self.lib.npu_nvme_wait_request(request, 1000)
        elif operation in ("metadata", "timeout"):
            rc = self.lib.npu_nvme_sync_meta_io(self.ctx, OFFSET + ALIGN, ALIGN, 0, self.buffer())
        elif operation == "flush":
            require(self.host_write() == 0, "pre-flush write failed")
            rc = self.lib.npu_nvme_flush(self.ctx)
        else:
            rc = self.host_write()
        self.record.update(operation_rc=rc, operation_seconds=time.monotonic() - started)
        require(rc < 0, f"fault {name} was not observed")
        if name == "timeout":
            require(rc == -errno.ETIMEDOUT and self.record["operation_seconds"] < self.args.timeout_bound,
                    "metadata timeout code/deadline mismatch")
            start = time.monotonic()
            short_rc = self.lib.npu_nvme_close(self.ctx, 50)
            self.record.update(short_close_rc=short_rc, short_close_seconds=time.monotonic() - start)
            require(short_rc == -errno.ETIMEDOUT, "expected pending work at original short close deadline")
        # Keep injection enabled until all pending work is drained.

    def backpressure(self):
        os.environ["NPU_NVME_TEST_NVME_SUBMIT_DELAY_MS"] = "100"
        self.record["fault"] = {"NPU_NVME_TEST_NVME_SUBMIT_DELAY_MS": "100"}
        source = self.device_source()
        busy = 0
        for _ in range(64):
            rc, _ = self.submit(source)
            if rc:
                busy = rc
                break
        self.record.update(busy_rc=busy, submitted=len(self.requests))
        require(busy == -errno.EBUSY, f"expected -EBUSY, got {busy}")
        codes = [self.lib.npu_nvme_wait_request(request, self.args.close_timeout_ms)
                 for request in self.requests]
        self.record["request_codes"] = codes
        require(all(code == 0 for code in codes), "accepted backpressure request failed")
        require(self.snapshot()["stats"]["request_ring_peak"] > 0, "ring peak missing")

    def crash(self, mode):
        self.record["superblock_sha256"] = self.superblock()
        if mode == "before_data_complete":
            # Host-only pre-submit window: do not kill a process with active HBM DMA.
            os.environ["NPU_NVME_TEST_NVME_SUBMIT_DELAY_MS"] = "5000"
            rc, _ = self.submit(ctypes.addressof(self.buffer()), host=True)
            require(rc == 0, "crash-window admission failed")
            snap = self.snapshot()
            require(snap["stats"]["nvme_submit_count"] == 0, "missed pre-submit window")
            self.record["crash_boundary"] = "host request accepted, before NVMe submit; no HBM DMA"
            self.record["before_crash"] = snap
        else:
            require(self.host_write() == 0, "crash-window write failed")
            require(self.lib.npu_nvme_flush(self.ctx) == 0, "crash-window flush failed")
            require(self.lib.npu_nvme_wait_quiescent(self.ctx, self.args.close_timeout_ms) == 0,
                    "crash-window data not quiescent")
            self.record["crash_boundary"] = "scratch data flushed; no metadata publish invoked"
        self.record.update(status="expected_crash", cleanup_safe=False,
                           scope="process-crash smoke only; no DMA-stop or power-loss proof")
        write_json(self.args.output / "worker.json", self.record)
        os._exit(CRASHES[mode])

    def close(self):
        if not self.ctx:
            return
        self.record["before_close"] = self.snapshot()
        start = time.monotonic()
        rc = self.lib.npu_nvme_close(self.ctx, self.args.close_timeout_ms)
        self.record.update(close_rc=rc, close_seconds=time.monotonic() - start)
        self.record["after_close"] = self.snapshot()
        require(rc == 0, f"close failed; retain all buffers/context: {rc}")
        require(self.lib.npu_nvme_wait_quiescent(self.ctx, 1) == 0, "closed context is not quiescent")
        snap = self.record["after_close"]
        require(not snap["retained_slots"] and all(snap["stats"][name] == 0 for name in
                ("dma_inflight", "nvme_outstanding", "request_ring_depth")), "resources still retained")
        require(snap["stats"]["dma_inflight_peak"] <= self.args.depth and
                snap["stats"]["nvme_outstanding_peak"] <= self.args.depth and
                snap["stats"]["request_ring_peak"] <= 16, "resource peak exceeds configured bounds")
        for request in self.requests:
            done = ctypes.c_int()
            rc = self.lib.npu_nvme_poll_request(request, ctypes.byref(done))
            require(done.value == 1, f"request not terminal after close: {rc}")
            self.lib.npu_nvme_release_request(request)
        self.requests.clear()
        for source in self.device_buffers:
            require(self.acl.aclrtFree(source) == 0, "HBM free failed")
        self.device_buffers.clear()
        self.lib.npu_nvme_cleanup(self.ctx)
        self.ctx = None
        self.host_buffers.clear()
        self.record["cleanup_safe"] = True


def worker_main(args):
    # Called only in a newly exec'd process; EAL observes this ID exactly once.
    for key in list(os.environ):
        if key.startswith("NPU_NVME_TEST_") or key == "NPU_NVME_IO_TIMEOUT_MS":
            os.environ.pop(key)
    os.environ["SPDK_SHM_ID"] = str(args.shm_id)
    record = {"status": "fail", "cleanup_safe": False, "pid": os.getpid(),
              "shm_id": args.shm_id, "case": args.worker_case or args.crash_child or "verify"}
    worker = None
    try:
        worker = Worker(args, record)
        worker.open()
        if args.crash_child:
            worker.crash(args.crash_child)
        elif args.verify:
            worker.verify()
        elif args.worker_case == "request_ring_busy":
            worker.backpressure()
        else:
            worker.fault(args.worker_case)
        record["status"] = "pass"
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception:
                record["cleanup_error"] = traceback.format_exc()
        if not record["cleanup_safe"]:
            record["status"] = "fail"
        record["resources"] = resource_snapshot()
        write_json(args.output / "worker.json", record)
    # On unsafe close do not run Python finalizers over retained caller buffers.
    if not record["cleanup_safe"]:
        os._exit(2)
    return 0 if record["status"] == "pass" else 1


def run_process(argv, directory, timeout, expected_rc=0):
    directory.mkdir(parents=True, exist_ok=False)
    result = {"argv": argv, "cwd": str(ROOT), "expected_rc": expected_rc, "status": "fail"}
    write_json(directory / "invocation.json", result)
    start = time.monotonic()
    try:
        with (directory / "stdout.txt").open("w") as out, (directory / "stderr.txt").open("w") as err:
            proc = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err, start_new_session=True)
            result["pid"] = proc.pid
            write_json(directory / "invocation.json", result)
            try:
                result["returncode"] = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                os.killpg(proc.pid, signal.SIGKILL)
                result["returncode"] = proc.wait()
        record = json.loads((directory / "worker.json").read_text())
        result["worker"] = record
        require(not result.get("timed_out"), "worker deadline expired; no device reopen")
        require(result["returncode"] == expected_rc, "unexpected worker exit code")
        if expected_rc == 0:
            require(record["status"] == "pass" and record["cleanup_safe"] is True,
                    "worker failed or cleanup unsafe; no device reopen")
        else:
            require(record["status"] == "expected_crash", "missing intentional crash marker")
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["seconds"] = time.monotonic() - start
        write_json(directory / "process.json", result)
    return result


def run_pair(launch, case, record):
    record["fault"] = launch(case, False)
    if record["fault"]["status"] != "pass":
        record["status"] = "fail"
        return False
    record["verify"] = launch(case, True)
    record["status"] = record["verify"]["status"]
    if case in CRASHES and record["status"] == "pass":
        before = record["fault"]["worker"]["superblock_sha256"]
        after = record["verify"]["worker"]["superblock_sha256"]
        record["metadata_unchanged"] = before == after
        if before != after:
            record["status"] = "fail"
    return record["status"] == "pass"


def orchestrate(args):
    # Do not accept an old directory: failures and partial runs are immutable.
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", vars(args))
    result = {"status": "fail", "execution_status": "aborted", "validation_status": "fail",
              "scope": "H02 existing fault/backpressure subset plus process-crash smoke",
              "cases": [], "resources_before": resource_snapshot()}
    try:
        require(args.pci == "0000:83:00.0", "only authorized 83:00.0 may be tested")
        region = json.loads((ROOT / "config/raw_test_region.json").read_text())
        require(region["pci_addr"] == args.pci and region["write_authorized"] is True,
                "raw authorization missing")
        require(region["offset"] == 0 and region["length"] == "device_capacity_bytes",
                "this scratch gate requires the registered whole-device extent")
        spdk_root = args.spdk_root.resolve()
        environment = environment_snapshot(pci=args.pci, npu=str(args.npu), repo_root=ROOT,
                                           npu_info=command(["npu-smi", "info"]))
        environment["repo"]["spdk_commit"] = command(["git", "-C", str(spdk_root), "rev-parse", "HEAD"])
        write_json(args.output / "environment.json", environment)
        (args.output / "executed_stage4_fault_lifecycle.py").write_bytes(Path(__file__).read_bytes())
        write_json(args.output / "provenance.json", {
            "commit": command(["git", "rev-parse", "HEAD"]),
            "dirty_diff": command(["git", "diff", "--no-ext-diff"]),
            "script_sha256": sha256_file(Path(__file__)),
            "library": str(args.library), "binary_sha256": sha256_file(args.library),
            "spdk_root": str(spdk_root),
            "spdk_commit": command(["git", "-C", str(spdk_root), "rev-parse", "HEAD"]),
            "dependencies": [{"path": str(p), "sha256": sha256_file(p)} for p in sorted(
                (spdk_root / "build/lib").glob("*.a"))],
            "cann_library": {"path": str(args.acl_library.resolve()),
                             "sha256": sha256_file(args.acl_library.resolve())}})
        sequence = 0
        def launch(case, verify):
            nonlocal sequence
            sequence += 1
            directory = args.output / f"{sequence:03d}-{case}-{'verify' if verify else 'fault'}"
            argv = [sys.executable, str(Path(__file__).resolve()), "--output", str(directory),
                    "--pci", args.pci, "--npu", str(args.npu), "--depth", str(args.depth),
                    "--shm-id", str(args.shm_id + sequence), "--library", str(args.library),
                    "--close-timeout-ms", str(args.close_timeout_ms),
                    "--timeout-bound", str(args.timeout_bound)]
            if verify:
                argv.append("--verify")
            else:
                argv.extend(["--crash-child" if case in CRASHES else "--worker-case", case])
            return run_process(argv, directory, args.worker_timeout,
                               CRASHES.get(case, 0) if not verify else 0)
        for case in [*args.cases, "request_ring_busy", *CRASHES]:
            record = {"case": case, "scope": "crash-smoke" if case in CRASHES else "H02-subset"}
            result["cases"].append(record)
            ok = run_pair(launch, case, record)
            write_json(args.output / "result.json", result)
            if not ok:
                result["execution_status"] = "completed"
                processes = [record.get("fault", {}), record.get("verify", {})]
                if any(p.get("timed_out") for p in processes):
                    result.update(execution_status="aborted", validation_status="invalid")
                raise AssertionError(f"{case} failed; remaining device tests stopped")
        result.update(status="pass", execution_status="completed", validation_status="pass")
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["resources_after"] = resource_snapshot()
        result["case_count"] = len(result["cases"])
        write_json(args.output / "result.json", result)
        manifest(args.output)
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--checkpoint-slots", type=int, default=1)
    parser.add_argument("--shm-id", type=int, default=18300)
    parser.add_argument("--output", type=Path, default=Path("/tmp/stage4-fault"))
    parser.add_argument("--timeout-bound", type=float, default=0.20)
    parser.add_argument("--close-timeout-ms", type=int, default=5000)
    parser.add_argument("--worker-timeout", type=float, default=60)
    parser.add_argument("--library", type=Path, default=ROOT / "build_out/lib/libnpu_nvme.so")
    parser.add_argument("--spdk-root", type=Path, default=Path("/home/user7/npu-nvme/third_party/spdk"))
    parser.add_argument("--acl-library", type=Path,
                        default=Path("/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--crash-child", choices=CRASHES)
    mode.add_argument("--worker-case", choices=[*FAULTS, "request_ring_busy"])
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=FAULTS, default=list(FAULTS))
    args = parser.parse_args()
    args.output, args.library = args.output.resolve(), args.library.resolve()
    require(args.pci == "0000:83:00.0", "protected/undeclared PCI address")
    require(args.close_timeout_ms > 0 and args.worker_timeout > 0 and args.timeout_bound > 0,
            "deadlines must be positive")
    if args.worker_case or args.crash_child or args.verify:
        args.output.mkdir(parents=True, exist_ok=True)
        return worker_main(args)
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
