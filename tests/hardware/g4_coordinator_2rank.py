#!/usr/bin/env python3
"""G4 two-rank prepare/commit gate.

Rank 0 and rank 1 run small real MindSpore Ascend workloads on NPU1/NPU2 and
send frozen model/optimizer/RNG/data state to a coordinator over Unix sockets.
Only the coordinator initializes SPDK and owns 83.0.0.  A successful step is
published to V2 metadata only after both ranks acknowledge PREPARED.  A second
run deliberately drops rank 1 before commit; that step must not appear in the
metadata ledger.
"""

import argparse
import ctypes
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, REPO_ROOT)

from direct_checkpoint import DirectCheckpoint  # noqa: E402
from c_bindings import lib  # noqa: E402


FIELDS = ("weight", "optimizer", "rng", "data_cursor")


def multirank_slot_base(ckpt, rank_id, step, keep_last_n=3):
    """Use the declared multi-rank area rather than the single-rank FULL ring."""
    base = ckpt.layout.full_end + 1024 * 1024 * 1024
    slot_index = rank_id * keep_last_n + (step % keep_last_n)
    end = base + 2 * keep_last_n * ckpt.slot_bytes
    if end > ckpt.layout.delta_base:
        raise MemoryError("G4 multi-rank area overlaps Delta partition")
    return base + slot_index * ckpt.slot_bytes


def send_msg(sock, message):
    raw = json.dumps(message, sort_keys=True).encode("utf-8")
    sock.sendall(struct.pack("!I", len(raw)) + raw)


def recv_exact(sock, count):
    chunks = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_msg(sock):
    header = recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack("!I", header)[0]
    if length > 4 * 1024 * 1024:
        raise RuntimeError("coordinator message is too large")
    payload = recv_exact(sock, length)
    return None if payload is None else json.loads(payload.decode("utf-8"))


def rank_state(rank_id, step, device_id):
    import mindspore as ms
    from mindspore import context

    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target="Ascend", device_id=device_id)
    # These operations execute on the assigned NPU; asnumpy freezes the
    # resulting state before the rank sends PREPARE.
    weight = ms.Tensor(np.arange(16, dtype=np.float16))
    weight = weight + ms.Tensor(
        np.full(16, rank_id + step * 0.01, dtype=np.float16))
    optimizer = ms.Tensor(np.full(16, 0.5, dtype=np.float32))
    optimizer = optimizer + ms.Tensor(
        np.full(16, step * 0.001, dtype=np.float32))
    fields = {
        "weight": weight.asnumpy().tobytes().hex(),
        "optimizer": optimizer.asnumpy().tobytes().hex(),
        "rng": struct.pack("<Q", 0xA5000000 + rank_id * 1000 + step).hex(),
        "data_cursor": struct.pack("<Q", rank_id * 100000 + step).hex(),
    }
    digest = hashlib.sha256(
        b"".join(bytes.fromhex(fields[name]) for name in FIELDS)).hexdigest()
    return fields, digest


def rank_worker(args):
    fields, digest = rank_state(args.rank_id, args.step, args.npu_id)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.socket)
    try:
        send_msg(sock, {"type": "PREPARE", "rank": args.rank_id,
                        "step": args.step, "fields": fields, "digest": digest})
        if args.fail_after_prepare:
            return 0
        message = recv_msg(sock)
        if not message or message.get("type") != "PREPARED_OK":
            return 2
        send_msg(sock, {"type": "COMMIT_READY", "rank": args.rank_id,
                        "step": args.step})
        message = recv_msg(sock)
        if not message:
            return 3
        if message.get("type") == "ABORT":
            return 0
        if message.get("type") != "COMMIT":
            return 4
        send_msg(sock, {"type": "DONE", "rank": args.rank_id,
                        "step": args.step, "digest": digest})
        return 0
    finally:
        sock.close()


def write_rank_state(ckpt, rank_id, step, fields):
    base = multirank_slot_base(ckpt, rank_id, step, ckpt.keep_last_n)
    buffers = []
    ptrs = []
    offsets = []
    sizes = []
    layout = {}
    current = base
    for name in FIELDS:
        payload = bytes.fromhex(fields[name])
        buffer = ctypes.create_string_buffer(payload, len(payload))
        buffers.append(buffer)
        ptrs.append(ctypes.addressof(buffer))
        offsets.append(current)
        sizes.append(len(payload))
        layout[name] = {"offset": current, "size": len(payload),
                        "dtype": "uint8", "shape": [len(payload)]}
        current += (len(payload) + 4095) & ~4095
    ptr_array = (ctypes.c_void_p * len(ptrs))(*ptrs)
    offset_array = (ctypes.c_uint64 * len(offsets))(*offsets)
    size_array = (ctypes.c_size_t * len(sizes))(*sizes)
    rc = lib.npu_nvme_write_batch_host(
        ckpt.ctx, ptr_array, offset_array, size_array, len(ptrs))
    if rc != 0:
        raise RuntimeError(f"coordinator rank {rank_id} write failed: {rc}")
    return layout, current - base


def read_rank_state(ckpt, rank_layout):
    buffers = []
    ptrs = []
    offsets = []
    sizes = []
    for name in FIELDS:
        record = rank_layout[name]
        buffer = ctypes.create_string_buffer(record["size"])
        buffers.append(buffer)
        ptrs.append(ctypes.addressof(buffer))
        offsets.append(record["offset"])
        sizes.append(record["size"])
    ptr_array = (ctypes.c_void_p * len(ptrs))(*ptrs)
    offset_array = (ctypes.c_uint64 * len(offsets))(*offsets)
    size_array = (ctypes.c_size_t * len(sizes))(*sizes)
    rc = lib.npu_nvme_read_batch_host(
        ckpt.ctx, ptr_array, offset_array, size_array, len(ptrs))
    if rc != 0:
        raise RuntimeError(f"coordinator read failed: {rc}")
    return {name: buffers[index].raw.hex() for index, name in enumerate(FIELDS)}


def coordinator_phase(args, run_dir):
    # Linux limits AF_UNIX addresses to 108 bytes; campaign run directories
    # encode the full experiment key, so keep the rendezvous path short.
    socket_path = "/tmp/npuio-g4-" + hashlib.sha256(
        run_dir.encode("utf-8")).hexdigest()[:16] + ".sock"
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(2)
    def collect_and_commit(ckpt, step, fail_after_prepare, fault_mode=None):
        connections = {}
        prepared = {}
        for _ in range(2):
            conn, _ = server.accept()
            conn.settimeout(5)
            message = recv_msg(conn)
            if message and message.get("type") == "PREPARE":
                connections[message["rank"]] = conn
                prepared[message["rank"]] = message
            else:
                conn.close()
        if set(prepared) != {0, 1}:
            for conn in connections.values():
                conn.close()
            return False
        for conn in connections.values():
            try:
                send_msg(conn, {"type": "PREPARED_OK", "step": step})
            except OSError:
                pass

        ready = set()
        failed = False
        for rank_id, conn in connections.items():
            try:
                message = recv_msg(conn)
            except (OSError, socket.timeout):
                message = None
            if not message or message.get("type") != "COMMIT_READY":
                failed = True
            else:
                ready.add(rank_id)
        if failed or ready != {0, 1}:
            for conn in connections.values():
                try:
                    send_msg(conn, {"type": "ABORT", "step": step})
                except OSError:
                    pass
                conn.close()
            return False

        rank_records = {}
        for rank_id in (0, 1):
            layout, written = write_rank_state(
                ckpt, rank_id, step, prepared[rank_id]["fields"])
            rank_records[str(rank_id)] = {
                "rank_id": rank_id,
                "digest": prepared[rank_id]["digest"],
                "fields": layout,
                "written_bytes": written,
                "optimizer_state": True,
                "rng_state": True,
                "data_cursor": True,
            }

        if fault_mode == "coordinator_precommit" and step == 3:
            # Simulate coordinator loss after all rank data is durable but
            # before metadata publication.  The previous committed record must
            # remain the only visible generation after restart.
            os._exit(86)

        ckpt.meta_dict["checkpoints"][f"step_{step}"] = {
            "type": "MULTI_FULL",
            "generation": ckpt.metadata_generation + 1,
            "world_size": 2,
            "coordinator": "single-owner-spdk",
            "ranks": rank_records,
        }
        ckpt._persist_metadata(ckpt.metadata_generation + 1)

        if fault_mode == "after_commit" and step == 3:
            # Metadata is durable; source processes are deliberately lost
            # before COMMIT replies are delivered to model an abrupt crash
            # immediately after global commit.
            os._exit(87)

        for conn in connections.values():
            send_msg(conn, {"type": "COMMIT", "step": step})
        for rank_id, conn in connections.items():
            message = recv_msg(conn)
            if not message or message.get("type") != "DONE":
                raise RuntimeError(f"rank {rank_id} did not acknowledge commit")
            conn.close()
        return True

    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.coordinator_npu,
        pipeline_depth=4, requested_chunk_size=4 * 1024 * 1024,
        rank_id=0, world_size=2, keep_last_n=3, slot_size_gb=10,
        spdk_shm_id=args.shm_id, profiling_dir=os.path.join(run_dir, "profiling"))
    ckpt._meta_pkl = os.path.join(run_dir, "checkpoint_meta.pkl")
    try:
        # The outer orchestrator starts ranks only after this marker.  Keeping
        # Ascend rank processes outside this SPDK-owning process avoids the
        # runtime's fork/exec interaction with an initialized DPDK reactor.
        with open(os.path.join(run_dir, "round_2.ready"), "w", encoding="utf-8") as stream:
            stream.write("ready\n")
        if not collect_and_commit(ckpt, 2, False):
            raise RuntimeError("successful two-rank prepare/commit unexpectedly aborted")
        with open(os.path.join(run_dir, "round_2.done"), "w", encoding="utf-8") as stream:
            stream.write("done\n")
        if args.fault_mode == "rank_partial":
            if collect_and_commit(ckpt, 3, True):
                raise RuntimeError("failed rank incorrectly published step_3")
        elif args.fault_mode in ("coordinator_precommit", "after_commit"):
            # The selected fault exits from collect_and_commit.  This branch is
            # only reached if the injection unexpectedly returned normally.
            if collect_and_commit(ckpt, 3, False, args.fault_mode):
                raise RuntimeError("coordinator fault unexpectedly returned")
        else:
            if collect_and_commit(ckpt, 3, True):
                raise RuntimeError("failed rank incorrectly published step_3")
        with open(os.path.join(run_dir, "round_3.done"), "w", encoding="utf-8") as stream:
            stream.write("done\n")
        if args.fault_mode != "after_commit" and "step_3" in ckpt.meta_dict.get("checkpoints", {}):
            raise RuntimeError("step_3 appeared after rank failure")
        if args.fault_mode == "after_commit" and "step_3" not in ckpt.meta_dict.get("checkpoints", {}):
            raise RuntimeError("step_3 missing after post-commit crash")
        with open(os.path.join(run_dir, "g4_manifest.json"), "w", encoding="utf-8") as stream:
            json.dump({"status": "pass", "step": 3 if args.fault_mode == "after_commit" else 2,
                       "world_size": 2, "fault_mode": args.fault_mode,
                       "failed_step": 3 if args.fault_mode != "after_commit" else None,
                       "coordinator_npu": args.coordinator_npu},
                      stream, indent=2, sort_keys=True)
        print("[G4/coordinator] PASS step_2 committed; step_3 aborted on rank1 failure",
              flush=True)
    finally:
        ckpt.cleanup()
        server.close()
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass


def launch_pair(args, run_dir, socket_path, step, fail_after_prepare):
    workers = []
    for rank_id, npu_id in ((0, 1), (1, 2)):
        command = [sys.executable, os.path.abspath(__file__), "--role", "rank",
                   "--rank-id", str(rank_id), "--npu-id", str(npu_id),
                   "--step", str(step), "--socket", socket_path]
        if fail_after_prepare and rank_id == 1:
            command.append("--fail-after-prepare")
        workers.append(subprocess.Popen(command, cwd=run_dir))
    return workers


def wait_for_file(path, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {path}")


def stop_workers(workers):
    for worker in workers:
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.terminate()
    for worker in workers:
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)


def verify_phase(args, run_dir):
    with open(os.path.join(run_dir, "g4_manifest.json"), "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.coordinator_npu,
        pipeline_depth=4, requested_chunk_size=4 * 1024 * 1024,
        rank_id=0, world_size=2, keep_last_n=3, slot_size_gb=10,
        spdk_shm_id=args.shm_id)
    try:
        record = ckpt.meta_dict["checkpoints"].get("step_2")
        if not record or record.get("type") != "MULTI_FULL" or record.get("world_size") != 2:
            raise AssertionError("step_2 multi-rank commit is missing after restart")
        if manifest.get("fault_mode") != "after_commit" and "step_3" in ckpt.meta_dict["checkpoints"]:
            raise AssertionError("aborted step_3 was published")
        if manifest.get("fault_mode") == "after_commit" and "step_3" not in ckpt.meta_dict["checkpoints"]:
            raise AssertionError("post-commit step_3 is missing after restart")
        for rank_id in (0, 1):
            rank_record = record["ranks"][str(rank_id)]
            fields = read_rank_state(ckpt, rank_record["fields"])
            digest = hashlib.sha256(
                b"".join(bytes.fromhex(fields[name]) for name in FIELDS)).hexdigest()
            if digest != rank_record["digest"]:
                raise AssertionError(f"rank {rank_id} persisted digest mismatch")
        print("[G4/verify] PASS restart metadata and both rank payloads verified",
              flush=True)
    finally:
        ckpt.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("orchestrate", "coordinator", "verify"),
                        default="orchestrate")
    parser.add_argument("--role", choices=("coordinator", "rank"), default=None)
    parser.add_argument("--rank-id", type=int, default=0)
    parser.add_argument("--npu-id", type=int, default=1)
    parser.add_argument("--coordinator-npu", type=int, default=7)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--socket", default="")
    parser.add_argument("--fail-after-prepare", action="store_true")
    parser.add_argument("--fault-mode", choices=("rank_partial", "coordinator_precommit",
                                                  "after_commit"),
                        default="rank_partial")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=83)
    parser.add_argument("--run-dir", default=os.path.join(
        REPO_ROOT, "experiments", "output", "gates", "g4"))
    args = parser.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    if args.role == "rank":
        raise SystemExit(rank_worker(args))
    if args.phase == "coordinator":
        coordinator_phase(args, run_dir)
        return
    if args.phase == "verify":
        verify_phase(args, run_dir)
        return

    coordinator_cmd = [sys.executable, os.path.abspath(__file__),
                       "--phase", "coordinator", "--pci", args.pci,
                       "--coordinator-npu", str(args.coordinator_npu),
                       "--shm-id", str(args.shm_id), "--run-dir", run_dir,
                       "--fault-mode", args.fault_mode]
    verify_cmd = [sys.executable, os.path.abspath(__file__), "--phase", "verify",
                  "--pci", args.pci, "--coordinator-npu", str(args.coordinator_npu),
                  "--shm-id", str(args.shm_id), "--run-dir", run_dir]
    for marker in ("round_2.ready", "round_2.done", "round_3.done",
                   "g4_manifest.json"):
        try:
            os.unlink(os.path.join(run_dir, marker))
        except FileNotFoundError:
            pass
    coordinator = subprocess.Popen(coordinator_cmd, cwd=run_dir)
    workers = []
    try:
        socket_path = "/tmp/npuio-g4-" + hashlib.sha256(
            run_dir.encode("utf-8")).hexdigest()[:16] + ".sock"
        wait_for_file(os.path.join(run_dir, "round_2.ready"))
        workers.extend(launch_pair(args, run_dir, socket_path, 2, False))
        wait_for_file(os.path.join(run_dir, "round_2.done"))
        stop_workers(workers)
        workers = []
        workers.extend(launch_pair(args, run_dir, socket_path, 3,
                                    args.fault_mode == "rank_partial"))
        if args.fault_mode == "rank_partial":
            wait_for_file(os.path.join(run_dir, "round_3.done"))
            stop_workers(workers)
            workers = []
            coordinator.wait(timeout=30)
            if coordinator.returncode != 0:
                raise subprocess.CalledProcessError(coordinator.returncode, coordinator_cmd)
        else:
            # Coordinator exits intentionally from the injected crash point.
            try:
                coordinator.wait(timeout=60)
            finally:
                stop_workers(workers)
                workers = []
            expected = 86 if args.fault_mode == "coordinator_precommit" else 87
            if coordinator.returncode != expected:
                raise subprocess.CalledProcessError(coordinator.returncode, coordinator_cmd)
        # The coordinator's intentional _exit path cannot write its manifest;
        # the parent records the injected scenario before running fresh verify.
        manifest_path = os.path.join(run_dir, "g4_manifest.json")
        if not os.path.exists(manifest_path):
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump({"status": "pass", "step": 3 if args.fault_mode == "after_commit" else 2,
                           "world_size": 2, "fault_mode": args.fault_mode,
                           "failed_step": 3 if args.fault_mode != "after_commit" else None,
                           "coordinator_npu": args.coordinator_npu},
                          stream, indent=2, sort_keys=True)
    finally:
        stop_workers(workers)
        if coordinator.poll() is None:
            coordinator.terminate()
            try:
                coordinator.wait(timeout=10)
            except subprocess.TimeoutExpired:
                coordinator.kill()
                coordinator.wait(timeout=5)
    subprocess.run(verify_cmd, cwd=run_dir, check=True)
    print("[G4] PASS", flush=True)


if __name__ == "__main__":
    main()
