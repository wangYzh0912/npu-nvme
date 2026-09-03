#!/usr/bin/env python3
"""C2: 2/4 training ranks, single SPDK owner, and fresh restore.

Ranks train a MindFormers model with a real Adam optimizer, freeze their
local state into host chunks, and stream those chunks to one coordinator.
The coordinator is the only process that opens 83.0.0.  A multi-rank
generation is published only after every rank manifest and payload checksum
has arrived.  After commit, the original ranks run one continuation
step and exit; fresh rank processes then load their own shard and compare that
continuation loss.
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
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "python"),
               str(REPO_ROOT / "tests" / "hardware")]

from c1_training_state_restart import (  # noqa: E402
    apply_control_state, batch_for_step, build_training as build_training_base,
    iter_unique_parameters,
    state_digest, train_range, write_json,
)
from direct_checkpoint import DirectCheckpoint  # noqa: E402
from chunk_helpers import build_chunks_host, build_ctypes_arrays  # noqa: E402
from c_bindings import lib  # noqa: E402
from training_state import (TRAINING_STATE_SCHEMA_VERSION, decode_control_value,
                            encode_control_value)
from ppt_evidence import command as shell_command, environment_snapshot  # noqa: E402


FRAME_LIMIT = 8 * 1024 * 1024
DATA_CHUNK = 4 * 1024 * 1024
DEFAULT_RANK_DEVICES = (1, 2, 3, 4)


def rank_mapping(args):
    configured = args.rank_devices
    if configured is None:
        configured = ",".join(str(item) for item in DEFAULT_RANK_DEVICES[:args.world_size])
    devices = tuple(int(item) for item in configured.split(",") if item)
    if len(devices) != args.world_size:
        raise ValueError("--rank-devices must contain world_size device ids")
    return tuple((rank, devices[rank]) for rank in range(args.world_size))


def effective_rank_id(args):
    if args.hccl:
        return int(os.getenv("RANK_ID", os.getenv("OMPI_COMM_WORLD_RANK", args.rank_id)))
    return args.rank_id


def effective_rank_device(args):
    if args.hccl:
        configured = os.getenv("ASCEND_DEVICE_ID", os.getenv("DEVICE_ID"))
        if configured is not None:
            return int(configured)
        rank_id = effective_rank_id(args)
        return dict(rank_mapping(args))[rank_id]
    return args.npu


def checkpoint_steps(args):
    if args.checkpoint_steps:
        steps = sorted(set(int(item) for item in args.checkpoint_steps))
    elif args.checkpoint_interval:
        steps = list(range(args.checkpoint_interval, args.total_steps + 1,
                           args.checkpoint_interval))
    else:
        steps = [args.save_step]
    if not steps or steps[0] <= 0:
        raise ValueError("checkpoint steps must be positive")
    if steps[-1] > args.total_steps:
        raise ValueError("checkpoint step exceeds --total-steps")
    return steps


def selected_restore_step(args):
    return int(args.restore_step or checkpoint_steps(args)[-1])


def rank_expected_path(args, rank_id):
    return Path(args.run_dir) / f"rank_{rank_id}_expected.json"


def rank_result_path(args, rank_id, step=None):
    if step is None and len(checkpoint_steps(args)) == 1:
        return Path(args.run_dir) / f"rank_{rank_id}_result.json"
    return Path(args.run_dir) / f"rank_{rank_id}_result_step_{step or selected_restore_step(args)}.json"


def build_training(args):
    """Build the rank model, optionally joining the real HCCL process group."""
    if not args.hccl:
        return build_training_base(args)
    import mindspore as ms
    from mindspore import context
    from mindspore.communication import get_group_size, init

    device_id = effective_rank_device(args)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        device_id=device_id)
    init()
    world = get_group_size()
    if world != args.world_size:
        raise RuntimeError(f"HCCL world size {world} != requested {args.world_size}")
    context.set_auto_parallel_context(
        device_num=world, parallel_mode="data_parallel",
        gradients_mean=True, full_batch=False)
    # Data-parallel ranks must start from identical parameter initialization;
    # rank identity only selects the device/process, never the model seed.
    ms.set_seed(args.seed)
    return build_training_base(args)


def shard_base(ckpt, rank_id, step, args):
    """Return a non-overlapping multi-rank area without reformatting 83.0.0.

    The shipped disk was formatted with three single-rank FULL slots.  The
    space between that FULL partition and the Delta tail is intentionally
    used for this correctness-only multi-rank lane.
    """
    base = ckpt.layout.full_end + 1024 * 1024 * 1024
    slot_index = rank_id * ckpt.keep_last_n + (step % ckpt.keep_last_n)
    offset = base + slot_index * ckpt.slot_bytes
    end = base + args.world_size * ckpt.keep_last_n * ckpt.slot_bytes
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


def append_event(args, event):
    value = {"run_id": Path(args.run_dir).name,
             "monotonic_ns": time.monotonic_ns(), **event}
    with (Path(args.run_dir) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def read_chunk(ckpt, absolute_offset, size):
    buffer = ctypes.create_string_buffer(size)
    ptrs = (ctypes.c_void_p * 1)(ctypes.addressof(buffer))
    offsets = (ctypes.c_uint64 * 1)(absolute_offset)
    sizes = (ctypes.c_size_t * 1)(size)
    rc = lib.npu_nvme_read_batch_host(ckpt.ctx, ptrs, offsets, sizes, 1)
    if rc != 0:
        raise RuntimeError(f"coordinator host read failed: {rc}")
    return ctypes.string_at(buffer, size)


def rank_train(args, rank_id, npu_id, socket_path):
    args.rank_id, args.npu = rank_id, npu_id
    ms, model, optimizer, cell = build_training(args)
    save_steps = checkpoint_steps(args)
    losses_by_step = {}
    checkpoint_states = {}
    all_losses = {}
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    try:
        next_step = 1
        for save_step in save_steps:
            losses, _ = train_range(ms, cell, next_step, save_step, args.seq_len)
            for offset, loss in enumerate(losses, start=next_step):
                all_losses[offset] = loss
            next_step = save_step + 1
            snapshot_started = time.perf_counter_ns()
            fields, manifest = make_rank_payload(ms, model, optimizer, save_step, args)
            snapshot_ns = time.perf_counter_ns() - snapshot_started
            checkpoint_states[str(save_step)] = state_digest(model, optimizer)
            prepare_ns = time.monotonic_ns()
            send_frame(sock, {"type": "PREPARE", "rank": rank_id,
                              "step": save_step, "manifest": manifest,
                              "prepare_monotonic_ns": prepare_ns})
            header, _ = recv_frame(sock)
            if header.get("type") != "PREPARED_OK":
                raise RuntimeError(f"rank {rank_id} prepare rejected: {header}")
            socket_send_ns = 0
            for field in fields:
                raw = field["payload"]
                for offset in range(0, len(raw), DATA_CHUNK):
                    chunk = raw[offset:offset + DATA_CHUNK]
                    send_started = time.perf_counter_ns()
                    send_frame(sock, {"type": "DATA", "rank": rank_id,
                                      "name": field["name"], "offset": offset}, chunk)
                    socket_send_ns += time.perf_counter_ns() - send_started
            send_frame(sock, {"type": "DATA_DONE", "rank": rank_id})
            header, _ = recv_frame(sock)
            if header.get("type") != "DATA_OK":
                raise RuntimeError(f"rank {rank_id} data rejected: {header}")
            send_frame(sock, {"type": "COMMIT_READY", "rank": rank_id})
            header, _ = recv_frame(sock)
            if header.get("type") != "COMMIT":
                raise RuntimeError(f"rank {rank_id} commit rejected: {header}")
            losses_by_step[str(save_step)] = {
                "generation": header["generation"],
                "snapshot_ns": snapshot_ns,
                "socket_send_ns": socket_send_ns,
                "persist_latency_ns": time.monotonic_ns() - prepare_ns,
                "payload_bytes": manifest["total_bytes"],
            }
            send_frame(sock, {"type": "DONE", "rank": rank_id})

        final_step = max(args.total_steps, save_steps[-1]) + args.continue_steps
        if next_step <= final_step:
            losses, _ = train_range(ms, cell, next_step, final_step, args.seq_len)
            for offset, loss in enumerate(losses, start=next_step):
                all_losses[offset] = loss
    finally:
        sock.close()
    for save_step in save_steps:
        losses_by_step[str(save_step)].update({
            "losses": [all_losses[step] for step in
                       range(save_step + 1, save_step + args.continue_steps + 1)],
            "checkpoint_state": checkpoint_states[str(save_step)],
        })
    latest = save_steps[-1]
    write_json(rank_expected_path(args, rank_id), {
        "rank": rank_id, "checkpoints": losses_by_step,
        "losses": losses_by_step[str(latest)]["losses"],
        "checkpoint_state": checkpoint_states[str(latest)],
        "continued_state": state_digest(model, optimizer),
        "initial_losses": [all_losses[step] for step in sorted(all_losses)
                           if step <= args.total_steps],
        "checkpoint_steps": save_steps,
    })


def rank_restore(args, rank_id, npu_id):
    restore_step = selected_restore_step(args)
    ms, model, optimizer, cell = build_training(args)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=npu_id, pipeline_depth=args.pipeline_depth,
        requested_chunk_size=DATA_CHUNK, rank_id=rank_id, world_size=args.world_size,
        keep_last_n=args.keep_last_n, slot_size_gb=args.slot_size_gb,
        spdk_shm_id=args.shm_id,
        profiling_dir=str(Path(args.run_dir) / f"profiling_restore_{rank_id}"))
    try:
        controls = ckpt.load_state(
            {"model": model, "optimizer": optimizer}, step=restore_step)
        apply_control_state(ms, optimizer, controls, restore_step, args)
        losses, _ = train_range(
            ms, cell, restore_step + 1,
            restore_step + args.continue_steps, args.seq_len)
        expected = json.loads(rank_expected_path(args, rank_id).read_text())
        checkpoint = expected.get("checkpoints", {}).get(
            str(restore_step), expected)
        if not np.allclose(losses, checkpoint["losses"], rtol=1e-4, atol=1e-5):
            raise AssertionError(
                f"rank {rank_id} continuation loss mismatch: {losses} != "
                f"{checkpoint['losses']}")
        write_json(rank_result_path(args, rank_id, restore_step), {
            "status": "pass", "rank": rank_id, "step": restore_step,
            "losses": losses, "expected_losses": checkpoint["losses"]
        })
    finally:
        ckpt.cleanup()


def rank_restore_hccl(args, rank_id, npu_id, socket_path):
    """Restore one shard through the single SPDK owner, then rejoin training."""
    args.rank_id, args.npu = rank_id, npu_id
    restore_step = selected_restore_step(args)
    ms, model, optimizer, cell = build_training(args)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    try:
        send_frame(sock, {"type": "RESTORE_REQUEST", "rank": rank_id,
                          "step": restore_step})
        header, _ = recv_frame(sock)
        if header.get("type") != "RESTORE_MANIFEST":
            raise RuntimeError(f"rank {rank_id} restore rejected: {header}")
        manifest = header["manifest"]
        saved = manifest.get("params", {})
        received = {name: bytearray() for name in saved}
        while True:
            frame, payload = recv_frame(sock)
            if frame.get("type") == "RESTORE_DONE":
                break
            name = frame.get("name")
            if (frame.get("type") != "RESTORE_DATA" or name not in saved or
                    int(frame.get("offset", -1)) != len(received[name])):
                raise RuntimeError(f"invalid restore frame for rank {rank_id}")
            received[name].extend(payload)
        send_frame(sock, {"type": "RESTORE_ACK", "rank": rank_id})
    finally:
        sock.close()

    targets = dict(iter_unique_parameters(model, optimizer))
    saved_parameters = {name for name in saved if not name.startswith("control/")}
    if set(targets) != saved_parameters:
        raise ValueError(f"rank {rank_id} parameter manifest mismatch")
    controls = {}
    from mindspore import ops
    for name, info in saved.items():
        payload = bytes(received[name])
        if len(payload) != int(info["size"]):
            raise ValueError(f"rank {rank_id} field size mismatch: {name}")
        if hashlib.sha256(payload).hexdigest() != info["sha256"]:
            raise ValueError(f"rank {rank_id} field checksum mismatch: {name}")
        if name.startswith("control/"):
            controls[name.split("/", 1)[1]] = decode_control_value(
                np.frombuffer(payload, dtype=np.uint8).copy(), info)
            continue
        parameter = targets[name]
        array = np.frombuffer(payload, dtype=np.dtype(info["dtype"])).copy()
        array = array.reshape(tuple(info["shape"]))
        if list(parameter.shape) != list(info["shape"]):
            singleton = array.size == 1 and int(np.prod(parameter.shape or (1,))) == 1
            if not singleton:
                raise ValueError(f"rank {rank_id} field shape mismatch: {name}")
            array = array.reshape(tuple(parameter.shape))
        ops.assign(parameter, ms.Tensor(array, dtype=parameter.dtype))
    ms.hal.synchronize()
    expected = json.loads(rank_expected_path(args, rank_id).read_text())
    checkpoint = expected.get("checkpoints", {}).get(str(restore_step), expected)
    restored_state = state_digest(model, optimizer)
    if restored_state != checkpoint["checkpoint_state"]:
        raise AssertionError(f"rank {rank_id} restored state digest mismatch")
    apply_control_state(ms, optimizer, controls, restore_step, args)
    losses, _ = train_range(
        ms, cell, restore_step + 1,
        restore_step + args.continue_steps, args.seq_len)
    if not np.allclose(losses, checkpoint["losses"], rtol=1e-4, atol=1e-5):
        raise AssertionError(
            f"rank {rank_id} HCCL continuation mismatch: {losses} != "
            f"{checkpoint['losses']}")
    continued_state = state_digest(model, optimizer)
    write_json(rank_result_path(args, rank_id, restore_step), {
        "status": "pass", "rank": rank_id, "step": restore_step,
        "generation": checkpoint.get("generation"), "losses": losses,
        "expected_losses": checkpoint["losses"],
        "checkpoint_state": restored_state,
        "loaded_state_byte_exact": True,
        "continued_state": continued_state,
        "source_continued_state": expected["continued_state"],
        "continued_state_byte_exact": continued_state == expected["continued_state"],
        "continuation_numeric_verified": True,
        "restore_transport": "single-spdk-owner-unix-stream",
        "continuation_context": "hccl",
    })


def restore_coordinator(args):
    """Read committed rank shards through one SPDK primary and stream them."""
    restore_step = selected_restore_step(args)
    socket_path = str(Path(args.run_dir) / "c2_restore.sock")
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(args.world_size)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.coordinator_npu,
        pipeline_depth=args.pipeline_depth, requested_chunk_size=DATA_CHUNK,
        rank_id=0, world_size=args.world_size, keep_last_n=args.keep_last_n,
        slot_size_gb=args.slot_size_gb, spdk_shm_id=args.shm_id,
        profiling_dir=str(Path(args.run_dir) / "profiling_restore_coord"))
    served = set()
    try:
        selected_step, record = ckpt._select_checkpoint_record(restore_step)
        if (selected_step != restore_step or
                record.get("type") != "MULTI_TRAINING_STATE_FULL" or
                int(record.get("world_size", -1)) != args.world_size):
            raise ValueError("committed multi-rank restore record is invalid")
        with open(Path(args.run_dir) / "restore_coordinator.ready", "w") as stream:
            stream.write("ready\n")
        while len(served) < args.world_size:
            conn, _ = server.accept()
            try:
                header, _ = recv_frame(conn)
                rank = int(header.get("rank", -1))
                if (header.get("type") != "RESTORE_REQUEST" or rank in served or
                        int(header.get("step", -1)) != restore_step):
                    raise RuntimeError("invalid or duplicate restore request")
                manifest = record.get("ranks", {}).get(str(rank))
                if manifest is None:
                    raise ValueError(f"checkpoint has no shard for rank {rank}")
                send_frame(conn, {"type": "RESTORE_MANIFEST", "rank": rank,
                                  "manifest": manifest})
                for name, info in manifest["params"].items():
                    size = int(info["size"])
                    for offset in range(0, size, DATA_CHUNK):
                        count = min(DATA_CHUNK, size - offset)
                        payload = read_chunk(ckpt, int(info["offset"]) + offset, count)
                        send_frame(conn, {"type": "RESTORE_DATA", "rank": rank,
                                          "name": name, "offset": offset}, payload)
                send_frame(conn, {"type": "RESTORE_DONE", "rank": rank})
                ack, _ = recv_frame(conn)
                if ack.get("type") != "RESTORE_ACK" or int(ack["rank"]) != rank:
                    raise RuntimeError(f"rank {rank} did not acknowledge restore")
                served.add(rank)
            finally:
                conn.close()
        output = (Path(args.run_dir) / "restore_coordinator.json" if
                  len(checkpoint_steps(args)) == 1 else
                  Path(args.run_dir) / f"restore_coordinator_step_{restore_step}.json")
        write_json(output, {
            "status": "pass", "step": restore_step,
            "generation": record.get("generation"), "served_ranks": sorted(served),
        })
    finally:
        ckpt.cleanup()
        server.close()
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass


def coordinator(args):
    socket_path = str(Path(args.run_dir) / "c2.sock")
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(args.world_size)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.coordinator_npu,
        pipeline_depth=args.pipeline_depth, requested_chunk_size=DATA_CHUNK,
        rank_id=0, world_size=args.world_size, keep_last_n=args.keep_last_n,
        slot_size_gb=args.slot_size_gb,
        spdk_shm_id=args.shm_id, profiling_dir=str(Path(args.run_dir) / "profiling_coord"))
    connections = {}
    committed = []
    try:
        with open(Path(args.run_dir) / "coordinator.ready", "w") as stream:
            stream.write("ready\n")
        for save_step in checkpoint_steps(args):
            generation_started = time.monotonic_ns()
            manifests = {}
            prepare_times = {}
            if not connections:
                while len(manifests) < args.world_size:
                    conn, _ = server.accept()
                    header, _ = recv_frame(conn)
                    rank = int(header.get("rank", -1))
                    if (header.get("type") != "PREPARE" or rank in manifests or
                            int(header.get("step", -1)) != save_step):
                        conn.close()
                        raise RuntimeError("invalid first PREPARE frame")
                    manifests[rank] = header["manifest"]
                    prepare_times[rank] = int(header.get("prepare_monotonic_ns", 0))
                    connections[rank] = conn
            else:
                for rank, conn in connections.items():
                    header, _ = recv_frame(conn)
                    if (header.get("type") != "PREPARE" or
                            int(header.get("rank", -1)) != rank or
                            int(header.get("step", -1)) != save_step):
                        raise RuntimeError("rank/checkpoint step mismatch")
                    manifests[rank] = header["manifest"]
                    prepare_times[rank] = int(header.get("prepare_monotonic_ns", 0))

            rank_records = {}
            for rank in sorted(manifests):
                manifest = manifests[rank]
                base = shard_base(ckpt, rank, save_step, args)
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

            receive_ns = 0
            spdk_write_ns = 0
            received_bytes = 0
            for rank in sorted(connections):
                conn, manifest = connections[rank], manifests[rank]
                by_name = {item["name"]: item for item in manifest["fields"]}
                received = {name: bytearray() for name in by_name}
                receive_started = time.perf_counter_ns()
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
                    received_bytes += len(payload)
                    if len(received[header["name"]]) > int(item["size"]):
                        raise RuntimeError("field payload exceeds manifest size")
                receive_ns += time.perf_counter_ns() - receive_started
                base = shard_base(ckpt, rank, save_step, args)
                cursor = 0
                write_started = time.perf_counter_ns()
                for item in manifest["fields"]:
                    payload = bytes(received[item["name"]])
                    if len(payload) != int(item["size"]):
                        raise RuntimeError(f"field size mismatch: {item['name']}")
                    if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                        raise RuntimeError(f"field checksum mismatch: {item['name']}")
                    write_chunk(ckpt, base + cursor, payload)
                    cursor += (len(payload) + 4095) & ~4095
                spdk_write_ns += time.perf_counter_ns() - write_started
                send_frame(conn, {"type": "DATA_OK"})

            for rank, conn in connections.items():
                header, _ = recv_frame(conn)
                if (header.get("type") != "COMMIT_READY" or
                        int(header.get("rank", -1)) != rank):
                    raise RuntimeError(f"rank {rank} did not reach COMMIT_READY")
            next_generation = ckpt.metadata_generation + 1
            record = {
                "type": "MULTI_TRAINING_STATE_FULL",
                "schema_version": TRAINING_STATE_SCHEMA_VERSION,
                "state_step": save_step, "generation": next_generation,
                "chunk_size": DATA_CHUNK, "world_size": args.world_size,
                "ranks": rank_records,
            }
            slot = save_step % args.keep_last_n
            for key, prior in list(ckpt.meta_dict["checkpoints"].items()):
                if (prior.get("type") == "MULTI_TRAINING_STATE_FULL" and
                        int(prior.get("state_step", -1)) % args.keep_last_n == slot):
                    del ckpt.meta_dict["checkpoints"][key]
            ckpt.meta_dict["checkpoints"][f"step_{save_step}"] = record
            ckpt._persist_metadata(next_generation)
            valid_prepare = [value for value in prepare_times.values() if value]
            committed.append({
                "step": save_step, "generation": next_generation,
                "received_bytes": received_bytes, "socket_receive_ns": receive_ns,
                "spdk_write_ns": spdk_write_ns,
                "rank_prepare_skew_ns": (max(valid_prepare) - min(valid_prepare)
                                         if valid_prepare else None),
                "global_commit_latency_ns": time.monotonic_ns() - generation_started,
            })
            append_event(args, {"event": "GLOBAL_COMMIT", "rank": None,
                                **committed[-1]})
            for conn in connections.values():
                send_frame(conn, {"type": "COMMIT", "generation": next_generation})
            for rank, conn in connections.items():
                header, _ = recv_frame(conn)
                if (header.get("type") != "DONE" or
                        int(header.get("rank", -1)) != rank):
                    raise RuntimeError(f"rank {rank} did not acknowledge commit")

        for conn in connections.values():
            conn.close()
        write_json(Path(args.run_dir) / "coordinator.json", {
            "status": "pass", "generation": committed[-1]["generation"],
            "world_size": args.world_size, "step": committed[-1]["step"],
            "committed": committed, "keep_last_n": args.keep_last_n,
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


def wait_file(path, timeout=180, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(path).exists():
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"child exited before creating {path}: rc={process.returncode}")
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {path}")


def child_args(args, phase, rank_id=None, npu_id=None, include_hccl=None):
    result = [sys.executable, str(Path(__file__).resolve()), "--phase", phase,
              "--run-dir", str(Path(args.run_dir).resolve()), "--pci", args.pci,
              "--model", args.model, "--seq-len", str(args.seq_len),
              "--seed", str(args.seed), "--save-step", str(args.save_step),
              "--total-steps", str(args.total_steps),
              "--checkpoint-interval", str(args.checkpoint_interval),
              "--keep-last-n", str(args.keep_last_n),
              "--restore-step", str(args.restore_step or 0),
              "--continue-steps", str(args.continue_steps), "--loss-scale", str(args.loss_scale),
              "--dropout-rate", str(args.dropout_rate), "--pipeline-depth", str(args.pipeline_depth),
              "--slot-size-gb", str(args.slot_size_gb), "--shm-id", str(args.shm_id)]
    result += ["--master-port", str(args.master_port),
               "--world-size", str(args.world_size), "--rank-devices",
               ",".join(str(device) for _, device in rank_mapping(args))]
    if args.checkpoint_steps:
        result += ["--checkpoint-steps", *[str(item) for item in args.checkpoint_steps]]
    if include_hccl is None:
        include_hccl = args.hccl
    if include_hccl:
        result.append("--hccl")
    if phase in ("rank", "restore", "restore_hccl"):
        result += ["--rank-id", str(rank_id), "--npu", str(npu_id)]
    else:
        result += ["--coordinator-npu", str(args.coordinator_npu)]
    return result


def hccl_command(args, phase, log_dir, master_port):
    first_rank, first_device = rank_mapping(args)[0]
    task = child_args(args, phase, first_rank, first_device,
                      include_hccl=True)[1:]
    return ["msrun", f"--worker_num={args.world_size}",
            f"--local_worker_num={args.world_size}",
            "--master_addr=127.0.0.1", f"--master_port={master_port}",
            "--join=True", f"--log_dir={log_dir}", *task]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("orchestrate", "coordinator", "rank",
                                             "restore", "restore_coordinator",
                                             "restore_hccl"),
                        default="orchestrate")
    parser.add_argument("--run-dir", default=str(REPO_ROOT / "results" / "next-correctness" /
                                                   time.strftime("c2_2rank_%Y%m%d_%H%M%S")))
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-step", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=None)
    parser.add_argument("--keep-last-n", type=int, default=3)
    parser.add_argument("--restore-step", type=int, default=None)
    parser.add_argument("--restore-retained", action="store_true")
    parser.add_argument("--continue-steps", type=int, default=1)
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--pipeline-depth", type=int, default=8)
    parser.add_argument("--slot-size-gb", type=int, default=10)
    parser.add_argument("--shm-id", type=int, default=94)
    parser.add_argument("--master-port", type=int, default=8127)
    parser.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--rank-devices", default=None)
    parser.add_argument("--hccl", action="store_true")
    parser.add_argument("--coordinator-npu", type=int, default=7)
    parser.add_argument("--rank-id", type=int, default=0)
    parser.add_argument("--npu", type=int, default=1)
    args = parser.parse_args()
    if args.total_steps is None:
        args.total_steps = args.save_step
    if args.keep_last_n < 1:
        raise ValueError("--keep-last-n must be positive")
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    if args.phase == "orchestrate":
        write_json(Path(args.run_dir) / "config.json", {
            **vars(args), "checkpoint_steps_resolved": checkpoint_steps(args),
            "scope": "FULL-only", "reactor_count": 1,
        })
        write_json(Path(args.run_dir) / "environment.json", environment_snapshot(
            pci=args.pci, npu=args.rank_devices or "auto", repo_root=REPO_ROOT,
            npu_info=shell_command(["npu-smi", "info"])))
        write_json(Path(args.run_dir) / "commit.json", {
            "repo": shell_command(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
            "branch": shell_command(["git", "-C", str(REPO_ROOT), "branch", "--show-current"]),
            "dirty": shell_command(["git", "-C", str(REPO_ROOT), "status", "--porcelain"]),
        })
        (Path(args.run_dir) / "events.jsonl").write_text("")
    if args.phase == "coordinator":
        coordinator(args)
        return
    if args.phase == "restore_coordinator":
        restore_coordinator(args)
        return
    if args.phase == "rank":
        rank_train(args, effective_rank_id(args), effective_rank_device(args),
                   str(Path(args.run_dir) / "c2.sock"))
        return
    if args.phase == "restore":
        rank_restore(args, effective_rank_id(args), effective_rank_device(args))
        return
    if args.phase == "restore_hccl":
        rank_restore_hccl(
            args, effective_rank_id(args), effective_rank_device(args),
            str(Path(args.run_dir) / "c2_restore.sock"))
        return

    for marker in ("coordinator.ready", "coordinator.json",
                   "restore_coordinator.ready", "restore_coordinator.json"):
        try:
            os.unlink(Path(args.run_dir) / marker)
        except FileNotFoundError:
            pass
    coordinator_proc = subprocess.Popen(
        child_args(args, "coordinator"), cwd=args.run_dir)
    source_coordinator_pid = coordinator_proc.pid
    workers = []
    restore_coordinator_proc = None
    try:
        wait_file(Path(args.run_dir) / "coordinator.ready", process=coordinator_proc)
        if args.hccl:
            command = hccl_command(
                args, "rank", "source_hccl_logs", args.master_port)
            workers.append(subprocess.Popen(command, cwd=args.run_dir))
            if workers[0].wait() != 0:
                raise RuntimeError("C2 HCCL rank training process failed")
        else:
            for rank_id, npu_id in rank_mapping(args):
                workers.append(subprocess.Popen(
                    child_args(args, "rank", rank_id, npu_id), cwd=args.run_dir))
            for worker in workers:
                if worker.wait() != 0:
                    raise RuntimeError("C2 rank training process failed")
        if coordinator_proc.wait() != 0:
            raise RuntimeError("C2 coordinator failed")
        source_coordinator_exit_code = coordinator_proc.returncode
        source_training_exit_code = workers[0].returncode if args.hccl else 0
        append_event(args, {"event": "SOURCE_PROCESSES_EXITED", "rank": None,
                            "source_training_exit_code": source_training_exit_code,
                            "source_coordinator_exit_code": source_coordinator_exit_code})
        committed = json.loads((Path(args.run_dir) / "coordinator.json").read_text())["committed"]
        retained = [item["step"] for item in committed[-args.keep_last_n:]]
        restore_steps = retained if args.restore_retained else [selected_restore_step(args)]
        rank_results = []
        original_restore_step = args.restore_step
        for restore_index, restore_step in enumerate(restore_steps):
            args.restore_step = restore_step
            for marker in ("restore_coordinator.ready", "restore_coordinator.json"):
                try:
                    os.unlink(Path(args.run_dir) / marker)
                except FileNotFoundError:
                    pass
            if args.hccl:
                restore_coordinator_proc = subprocess.Popen(
                    child_args(args, "restore_coordinator", include_hccl=False),
                    cwd=args.run_dir)
                wait_file(Path(args.run_dir) / "restore_coordinator.ready",
                          process=restore_coordinator_proc)
                worker = subprocess.Popen(
                    hccl_command(args, "restore_hccl",
                                 f"restore_hccl_logs_step_{restore_step}",
                                 args.master_port + 1 + restore_index), cwd=args.run_dir)
                workers.append(worker)
                if worker.wait() != 0:
                    raise RuntimeError("C2 HCCL fresh restore process failed")
                if restore_coordinator_proc.wait() != 0:
                    raise RuntimeError("C2 restore coordinator failed")
            else:
                # A standalone restore opens the SPDK primary itself and must run
                # one rank at a time.
                for rank_id, npu_id in rank_mapping(args):
                    worker = subprocess.Popen(
                        child_args(args, "restore", rank_id, npu_id,
                                   include_hccl=False), cwd=args.run_dir)
                    if worker.wait() != 0:
                        raise RuntimeError("C2 fresh restore process failed")
            rank_results.extend(json.loads(rank_result_path(args, rank, restore_step).read_text())
                                for rank, _ in rank_mapping(args))
        args.restore_step = original_restore_step
        restore_step = restore_steps[-1]
        write_json(Path(args.run_dir) / "result.json", {
            "status": "pass", "gate": "C2", "world_size": args.world_size,
            "step": restore_step, "model": args.model,
            "checkpoint_steps": checkpoint_steps(args),
            "restored_steps": restore_steps,
            "source_training_exit_code": source_training_exit_code,
            "source_coordinator_exit_code": source_coordinator_exit_code,
            "source_coordinator_pid": source_coordinator_pid,
            "source_training": "hccl" if args.hccl else "standalone",
            "restore_transport": ("single-spdk-owner-unix-stream" if args.hccl
                                  else "standalone-spdk-primary"),
            "continuation_context": "hccl" if args.hccl else "standalone",
            "rank_results": rank_results,
        })
        write_json(Path(args.run_dir) / "restore.json", {
            "status": "pass", "fresh_process": True,
            "source_training_exit_code": source_training_exit_code,
            "source_coordinator_exit_code": source_coordinator_exit_code,
            "restored_steps": restore_steps, "rank_results": rank_results,
            "all_rank_states_byte_exact": all(
                item.get("loaded_state_byte_exact") is True for item in rank_results),
            "all_continuations_numeric_verified": all(
                item.get("continuation_numeric_verified", True) for item in rank_results),
        })
        write_json(Path(args.run_dir) / "checkpoint_gate.json", {
            "status": "pass", "gate": "C2", "world_size": args.world_size,
            "generation": json.loads((Path(args.run_dir) / "coordinator.json").read_text())["generation"],
            "step": restore_step,
            "restored_steps": restore_steps,
            "restore_transport": ("single-spdk-owner-unix-stream" if args.hccl
                                  else "standalone-spdk-primary"),
            "continuation_context": "hccl" if args.hccl else "standalone",
            "ranks": [{"rank": rank, "persisted": True, "fresh_restore": True,
                       "continuation_verified": True,
                       "continuation_context": ("hccl" if args.hccl
                                                else "standalone")}
                      for rank, _ in rank_mapping(args)],
        })
        print(f"[C2] PASS {args.world_size}-rank training-state commit and fresh restore", flush=True)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        if coordinator_proc.poll() is None:
            coordinator_proc.terminate()
        if (restore_coordinator_proc is not None and
                restore_coordinator_proc.poll() is None):
            restore_coordinator_proc.terminate()


if __name__ == "__main__":
    main()
