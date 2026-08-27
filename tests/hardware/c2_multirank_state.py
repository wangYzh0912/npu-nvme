#!/usr/bin/env python3
"""C2: two real training ranks, single SPDK owner, and fresh restore.

Ranks train a MindFormers model with a real Adam optimizer, freeze their
local state into host chunks, and stream those chunks to one coordinator.
The coordinator is the only process that opens 83.0.0.  A multi-rank
generation is published only after both rank manifests and all payload
checksums have arrived.  After commit, the original ranks run one continuation
step and exit; fresh rank processes then load their own shard and compare that
continuation loss.
"""

import argparse
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "python"),
               str(REPO_ROOT / "tests" / "hardware")]

from c1_training_state_restart import (  # noqa: E402
    apply_control_state, batch_for_step, build_training, iter_unique_parameters,
    state_digest, train_range, write_json,
)
from direct_checkpoint import DirectCheckpoint  # noqa: E402
from chunk_helpers import build_chunks_host, build_ctypes_arrays  # noqa: E402
from c_bindings import lib  # noqa: E402
from training_state import (TRAINING_STATE_SCHEMA_VERSION, encode_control_value)


FRAME_LIMIT = 8 * 1024 * 1024
DATA_CHUNK = 4 * 1024 * 1024
RANKS = ((0, 1), (1, 2))


def shard_base(ckpt, rank_id, step, args):
    """Return a non-overlapping multi-rank area without reformatting 83.0.0.

    The shipped disk was formatted with three single-rank FULL slots.  The
    space between that FULL partition and the Delta tail is intentionally
    used for this correctness-only multi-rank lane.
    """
    base = ckpt.layout.full_end + 1024 * 1024 * 1024
    slot_index = rank_id * ckpt.keep_last_n + (step % ckpt.keep_last_n)
    offset = base + slot_index * ckpt.slot_bytes
    end = base + 2 * ckpt.keep_last_n * ckpt.slot_bytes
    if end > ckpt.layout.delta_base:
        raise MemoryError("multi-rank correctness area overlaps Delta partition")
    return offset


def recv_exact(sock, size):
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise EOFError("rank/coordinator socket closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def send_frame(sock, header, payload=b""):
    header = dict(header)
    header["payload_len"] = len(payload)
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    if len(raw) > FRAME_LIMIT:
        raise ValueError("control frame exceeds protocol limit")
    sock.sendall(struct.pack("!I", len(raw)) + raw + payload)


def recv_frame(sock):
    header_len = struct.unpack("!I", recv_exact(sock, 4))[0]
    if header_len > FRAME_LIMIT:
        raise ValueError("control frame exceeds protocol limit")
    header = json.loads(recv_exact(sock, header_len).decode())
    payload_len = int(header.pop("payload_len", 0))
    if payload_len < 0 or payload_len > DATA_CHUNK:
        raise ValueError("data frame exceeds chunk limit")
    return header, recv_exact(sock, payload_len)


def make_rank_payload(ms, model, optimizer, save_step, args):
    fields = []
    for name, parameter in iter_unique_parameters(model, optimizer):
        array = np.ascontiguousarray(parameter.value().asnumpy())
        raw = array.tobytes()
        fields.append({
            "name": name, "kind": "parameter", "shape": list(array.shape),
            "dtype": array.dtype.name, "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "payload": raw,
        })
    controls = {
        "global_step": np.asarray(optimizer.global_step.asnumpy()).copy(),
        "loss_scale": np.float32(args.loss_scale),
        "python_rng": __import__("random").getstate(),
        "numpy_rng": np.random.get_state(),
        "mindspore_seed": int(args.seed),
        "mindspore_rng": np.asarray(ms.get_rng_state().asnumpy()).copy(),
        "data_cursor": {"epoch": 0, "sample": int(save_step)},
    }
    for name in sorted(controls):
        array, codec = encode_control_value(controls[name])
        raw = array.tobytes()
        fields.append({
            "name": f"control/{name}", "kind": "control", "shape": [len(raw)],
            "dtype": "uint8", "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "codec": codec["codec"],
            "payload": raw,
        })
    digest = hashlib.sha256()
    manifest_fields = []
    for field in fields:
        digest.update(field["name"].encode())
        digest.update(field["payload"])
        manifest_fields.append({key: value for key, value in field.items()
                                if key != "payload"})
    return fields, {
        "components": ["model", "optimizer"],
        "control_names": sorted(controls),
        "checksum": "sha256", "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "fields": manifest_fields, "total_bytes": sum(f["size"] for f in fields),
        "digest": digest.hexdigest(),
    }


def write_chunk(ckpt, absolute_offset, payload):
    buffer = __import__("ctypes").create_string_buffer(payload, len(payload))
    chunks, _ = build_chunks_host(__import__("ctypes").addressof(buffer),
                                  absolute_offset, len(payload), DATA_CHUNK)
    ptrs, offsets, sizes = build_ctypes_arrays(chunks)
    rc = lib.npu_nvme_write_batch_host(
        ckpt.ctx, ptrs, offsets, sizes, len(chunks))
    if rc != 0:
        raise RuntimeError(f"coordinator host write failed: {rc}")


def rank_train(args, rank_id, npu_id, socket_path):
    ms, model, optimizer, cell = build_training(args)
    losses, _ = train_range(ms, cell, 1, args.save_step, args.seq_len)
    fields, manifest = make_rank_payload(ms, model, optimizer, args.save_step, args)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    try:
        send_frame(sock, {"type": "PREPARE", "rank": rank_id,
                          "step": args.save_step, "manifest": manifest})
        header, _ = recv_frame(sock)
        if header.get("type") != "PREPARED_OK":
            raise RuntimeError(f"rank {rank_id} prepare rejected: {header}")
        for field in fields:
            raw = field["payload"]
            for offset in range(0, len(raw), DATA_CHUNK):
                chunk = raw[offset:offset + DATA_CHUNK]
                send_frame(sock, {"type": "DATA", "rank": rank_id,
                                  "name": field["name"], "offset": offset}, chunk)
        send_frame(sock, {"type": "DATA_DONE", "rank": rank_id})
        header, _ = recv_frame(sock)
        if header.get("type") != "DATA_OK":
            raise RuntimeError(f"rank {rank_id} data rejected: {header}")
        send_frame(sock, {"type": "COMMIT_READY", "rank": rank_id})
        header, _ = recv_frame(sock)
        if header.get("type") != "COMMIT":
            raise RuntimeError(f"rank {rank_id} commit rejected: {header}")
        send_frame(sock, {"type": "DONE", "rank": rank_id})
    finally:
        sock.close()

    continuation, _ = train_range(
        ms, cell, args.save_step + 1, args.save_step + args.continue_steps,
        args.seq_len)
    write_json(Path(args.run_dir) / f"rank_{rank_id}_expected.json", {
        "rank": rank_id, "losses": continuation,
        "state": state_digest(model, optimizer), "initial_losses": losses,
    })


def rank_restore(args, rank_id, npu_id):
    ms, model, optimizer, cell = build_training(args)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=npu_id, pipeline_depth=args.pipeline_depth,
        requested_chunk_size=DATA_CHUNK, rank_id=rank_id, world_size=2,
        keep_last_n=3, slot_size_gb=args.slot_size_gb, spdk_shm_id=args.shm_id,
        profiling_dir=str(Path(args.run_dir) / f"profiling_restore_{rank_id}"))
    try:
        controls = ckpt.load_state(
            {"model": model, "optimizer": optimizer}, step=args.save_step)
        apply_control_state(ms, controls, args.save_step, args)
        losses, _ = train_range(
            ms, cell, args.save_step + 1,
            args.save_step + args.continue_steps, args.seq_len)
        expected = json.loads((Path(args.run_dir) /
                               f"rank_{rank_id}_expected.json").read_text())
        if not np.allclose(losses, expected["losses"], rtol=1e-4, atol=1e-5):
            raise AssertionError(
                f"rank {rank_id} continuation loss mismatch: {losses} != "
                f"{expected['losses']}")
        write_json(Path(args.run_dir) / f"rank_{rank_id}_result.json", {
            "status": "pass", "rank": rank_id, "losses": losses,
            "expected_losses": expected["losses"]
        })
    finally:
        ckpt.cleanup()


def coordinator(args):
    socket_path = str(Path(args.run_dir) / "c2.sock")
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(2)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.coordinator_npu,
        pipeline_depth=args.pipeline_depth, requested_chunk_size=DATA_CHUNK,
        rank_id=0, world_size=2, keep_last_n=3, slot_size_gb=args.slot_size_gb,
        spdk_shm_id=args.shm_id, profiling_dir=str(Path(args.run_dir) / "profiling_coord"))
    connections, manifests = {}, {}
    try:
        with open(Path(args.run_dir) / "coordinator.ready", "w") as stream:
            stream.write("ready\n")
        while len(manifests) < 2:
            conn, _ = server.accept()
            header, _ = recv_frame(conn)
            if header.get("type") != "PREPARE":
                conn.close()
                raise RuntimeError("first rank frame must be PREPARE")
            rank = int(header["rank"])
            if rank in manifests or int(header["step"]) != args.save_step:
                raise RuntimeError("duplicate rank or step mismatch")
            manifests[rank] = header["manifest"]
            connections[rank] = conn
        rank_records = {}
        for rank in sorted(manifests):
            manifest = manifests[rank]
            base = shard_base(ckpt, rank, args.save_step, args)
            cursor, records = 0, {}
            for item in manifest["fields"]:
                size = int(item["size"])
                records[item["name"]] = {key: item[key] for key in
                    ("shape", "dtype", "size", "sha256", "kind", "codec")
                    if key in item}
                records[item["name"]].update({"offset": base + cursor})
                cursor += (size + 4095) & ~4095
            rank_records[str(rank)] = {
                "components": manifest["components"],
                "control_names": manifest["control_names"],
                "checksum": manifest["checksum"], "params": records,
                "digest": manifest["digest"], "written_bytes": cursor,
            }
            send_frame(connections[rank], {"type": "PREPARED_OK"})

        for rank in sorted(connections):
            conn, manifest = connections[rank], manifests[rank]
            by_name = {item["name"]: item for item in manifest["fields"]}
            received = {name: bytearray() for name in by_name}
            while True:
                header, payload = recv_frame(conn)
                if header.get("type") == "DATA_DONE":
                    break
                if (header.get("type") != "DATA" or
                        int(header["rank"]) != rank or
                        header["name"] not in by_name):
                    raise RuntimeError(f"invalid data frame from rank {rank}")
                item = by_name[header["name"]]
                offset = int(header["offset"])
                if offset != len(received[header["name"]]):
                    raise RuntimeError("data chunks are missing or reordered")
                received[header["name"]].extend(payload)
                expected_size = int(item["size"])
                if len(received[header["name"]]) > expected_size:
                    raise RuntimeError("field payload exceeds manifest size")
            base = shard_base(ckpt, rank, args.save_step, args)
            cursor = 0
            for item in manifest["fields"]:
                payload = bytes(received[item["name"]])
                if len(payload) != int(item["size"]):
                    raise RuntimeError(f"field size mismatch: {item['name']}")
                if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                    raise RuntimeError(f"field checksum mismatch: {item['name']}")
                write_chunk(ckpt, base + cursor, payload)
                cursor += (len(payload) + 4095) & ~4095
            send_frame(conn, {"type": "DATA_OK"})

        ready = set()
        for rank, conn in connections.items():
            header, _ = recv_frame(conn)
            if header.get("type") != "COMMIT_READY":
                raise RuntimeError(f"rank {rank} did not reach COMMIT_READY")
            ready.add(rank)
        if ready != {0, 1}:
            raise RuntimeError("not all ranks reached COMMIT_READY")
        next_generation = ckpt.metadata_generation + 1
        record = {
            "type": "MULTI_TRAINING_STATE_FULL",
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "state_step": args.save_step, "generation": next_generation,
            "chunk_size": DATA_CHUNK, "world_size": 2, "ranks": rank_records,
        }
        ckpt.meta_dict["checkpoints"][f"step_{args.save_step}"] = record
        ckpt._persist_metadata(next_generation)
        for conn in connections.values():
            send_frame(conn, {"type": "COMMIT", "generation": next_generation})
        for rank, conn in connections.items():
            header, _ = recv_frame(conn)
            if header.get("type") != "DONE":
                raise RuntimeError(f"rank {rank} did not acknowledge commit")
            conn.close()
        write_json(Path(args.run_dir) / "coordinator.json", {
            "status": "pass", "generation": next_generation,
            "world_size": 2, "step": args.save_step,
            "ranks": {rank: {"fields": len(manifest["fields"]),
                              "bytes": manifest["total_bytes"]}
                      for rank, manifest in manifests.items()},
        })
    finally:
        for conn in connections.values():
            try:
                conn.close()
            except OSError:
                pass
        ckpt.cleanup()
        server.close()
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass


def wait_file(path, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(path).exists():
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {path}")


def child_args(args, phase, rank_id=None, npu_id=None):
    result = [sys.executable, str(Path(__file__).resolve()), "--phase", phase,
              "--run-dir", str(Path(args.run_dir).resolve()), "--pci", args.pci,
              "--model", args.model, "--seq-len", str(args.seq_len),
              "--seed", str(args.seed), "--save-step", str(args.save_step),
              "--continue-steps", str(args.continue_steps), "--loss-scale", str(args.loss_scale),
              "--dropout-rate", str(args.dropout_rate), "--pipeline-depth", str(args.pipeline_depth),
              "--slot-size-gb", str(args.slot_size_gb), "--shm-id", str(args.shm_id)]
    if phase in ("rank", "restore"):
        result += ["--rank-id", str(rank_id), "--npu", str(npu_id)]
    else:
        result += ["--coordinator-npu", str(args.coordinator_npu)]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("orchestrate", "coordinator", "rank", "restore"),
                        default="orchestrate")
    parser.add_argument("--run-dir", default=str(REPO_ROOT / "results" / "next-correctness" /
                                                   time.strftime("c2_2rank_%Y%m%d_%H%M%S")))
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-step", type=int, default=2)
    parser.add_argument("--continue-steps", type=int, default=1)
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--pipeline-depth", type=int, default=8)
    parser.add_argument("--slot-size-gb", type=int, default=10)
    parser.add_argument("--shm-id", type=int, default=94)
    parser.add_argument("--coordinator-npu", type=int, default=7)
    parser.add_argument("--rank-id", type=int, default=0)
    parser.add_argument("--npu", type=int, default=1)
    args = parser.parse_args()
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    if args.phase == "coordinator":
        coordinator(args)
        return
    if args.phase == "rank":
        rank_train(args, args.rank_id, args.npu,
                   str(Path(args.run_dir) / "c2.sock"))
        return
    if args.phase == "restore":
        rank_restore(args, args.rank_id, args.npu)
        return

    for marker in ("coordinator.ready", "coordinator.json"):
        try:
            os.unlink(Path(args.run_dir) / marker)
        except FileNotFoundError:
            pass
    coordinator_proc = subprocess.Popen(
        child_args(args, "coordinator"), cwd=args.run_dir)
    workers = []
    try:
        wait_file(Path(args.run_dir) / "coordinator.ready")
        for rank_id, npu_id in RANKS:
            workers.append(subprocess.Popen(
                child_args(args, "rank", rank_id, npu_id), cwd=args.run_dir))
        for worker in workers:
            if worker.wait() != 0:
                raise RuntimeError("C2 rank training process failed")
        if coordinator_proc.wait() != 0:
            raise RuntimeError("C2 coordinator failed")
        # SPDK requires a single primary process per shared-memory instance;
        # restore ranks therefore run one at a time.  The payload commit above
        # remains genuinely multi-rank and is owned by one coordinator.
        for rank_id, npu_id in RANKS:
            worker = subprocess.Popen(
                child_args(args, "restore", rank_id, npu_id), cwd=args.run_dir)
            if worker.wait() != 0:
                raise RuntimeError("C2 fresh restore process failed")
        write_json(Path(args.run_dir) / "result.json", {
            "status": "pass", "gate": "C2", "world_size": 2,
            "step": args.save_step, "model": args.model,
            "rank_results": [json.loads((Path(args.run_dir) /
                f"rank_{rank}_result.json").read_text()) for rank, _ in RANKS],
        })
        print("[C2] PASS two-rank training-state commit and fresh restore", flush=True)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        if coordinator_proc.poll() is None:
            coordinator_proc.terminate()


if __name__ == "__main__":
    main()
