#!/usr/bin/env python3
"""I6 raw-ring wrap, restart, A/B metadata, and fault-matrix gate."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "python")]

from delta_protocol import unpack_s2_replacement_frame  # noqa: E402
from direct_checkpoint import DirectCheckpoint  # noqa: E402
from experiments.benchmarks.io_matrix import (  # noqa: E402
    ResultWriter, SAFE_OFFSET, check_npu_free, environment_snapshot,
)
from raw_ring import (KIND_DELTA, KIND_FULL, pack_ring_metadata,  # noqa: E402
                      pack_ring_slot, select_ab_metadata,
                      select_recovery_chain, unpack_ring_slot)
from s2_delta import S2DeltaOracle  # noqa: E402


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = np.asarray(state[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def make_chain(steps, full_every):
    initial = {
        "backbone.blocks.0.weight": np.zeros(33, dtype=np.float32),
        "backbone.blocks.1.weight": np.ones(19, dtype=np.float16),
        "backbone.layernorm.bias": np.zeros(5, dtype=np.float32),
    }
    current = {name: value.copy() for name, value in initial.items()}
    delta_oracle = S2DeltaOracle(initial, block_size=8, small_threshold=4)
    records = []
    for step in range(1, steps + 1):
        current = {name: value.copy() for name, value in current.items()}
        current["backbone.blocks.0.weight"][step % 33] += np.float32(step * 0.25)
        current["backbone.blocks.1.weight"][step % 19] += np.float16(0.5)
        current["backbone.layernorm.bias"][step % 5] += np.float32(0.125)
        delta_oracle.set_current(current)
        delta_frame = delta_oracle.observe(step, generation=step)
        delta_oracle.ack(delta_frame)
        kind = KIND_FULL if step == 1 or step % full_every == 0 else KIND_DELTA
        if kind == KIND_FULL:
            full_oracle = S2DeltaOracle(initial, block_size=8,
                                        small_threshold=4)
            full_oracle.set_current(current)
            frame = full_oracle.observe(step, generation=step)
        else:
            frame = delta_frame
        records.append((step, kind, frame))
    return initial, current, records, delta_oracle.manifest_digest


def validate_semantic_chain(records, manifest_digest):
    previous_generation = 0
    for index, record in enumerate(records):
        step, _blocks, _smalls, info = unpack_s2_replacement_frame(
            record["frame"])
        if step != record["step_id"]:
            raise ValueError("S2/envelope step mismatch")
        if info["generation"] != record["slot_generation"]:
            raise ValueError("S2/envelope generation mismatch")
        if info["manifest_digest"] != manifest_digest:
            raise ValueError("S2 manifest digest mismatch")
        expected_base = 0 if index == 0 else previous_generation
        if info["base_generation"] != expected_base:
            raise ValueError("S2 base generation mismatch")
        previous_generation = info["generation"]


def expect_failure(name, callback, failures):
    try:
        callback()
    except (ValueError, RuntimeError) as error:
        failures[name] = str(error)
    else:
        raise AssertionError(f"fault case was accepted: {name}")


def fault_matrix(raw_slots, chain, manifest_digest, meta_a, meta_b):
    failures = {}
    good = next(raw for raw in raw_slots
                if unpack_ring_slot(raw)["slot_generation"] == chain[-1]["slot_generation"])
    payload = bytearray(good)
    payload[80] ^= 1
    expect_failure("outer_payload_crc", lambda: unpack_ring_slot(payload), failures)
    header = bytearray(good)
    header[8] ^= 1
    expect_failure("envelope_crc", lambda: unpack_ring_slot(header), failures)
    expect_failure("torn_payload", lambda: unpack_ring_slot(good[:70]), failures)

    missing_generation = [raw for raw in raw_slots
                          if unpack_ring_slot(raw)["slot_generation"] !=
                          chain[-2]["slot_generation"]]
    expect_failure("missing_generation",
                   lambda: select_recovery_chain(missing_generation), failures)
    expect_failure("duplicate_generation",
                   lambda: select_recovery_chain(raw_slots + [good]), failures)

    target = chain[-1]
    inner = bytearray(target["frame"])
    inner[4:8] = int(target["step_id"] + 2).to_bytes(4, "little")
    reordered = pack_ring_slot(inner, target["slot_generation"],
                               target["step_id"] + 2, target["kind"],
                               len(good))
    other = [raw for raw in raw_slots
             if unpack_ring_slot(raw)["slot_generation"] !=
             target["slot_generation"]]
    expect_failure("reordered_step",
                   lambda: select_recovery_chain(other + [reordered]), failures)

    wrong_base_inner = bytearray(target["frame"])
    wrong_base_inner[28:36] = (123456).to_bytes(8, "little")
    wrong_base = pack_ring_slot(wrong_base_inner, target["slot_generation"],
                                target["step_id"], target["kind"], len(good))
    wrong_base_chain = select_recovery_chain(other + [wrong_base])
    expect_failure("base_generation",
                   lambda: validate_semantic_chain(wrong_base_chain,
                                                   manifest_digest), failures)

    wrong_manifest_inner = bytearray(target["frame"])
    wrong_manifest_inner[48] ^= 1
    wrong_manifest = pack_ring_slot(
        wrong_manifest_inner, target["slot_generation"], target["step_id"],
        target["kind"], len(good))
    wrong_manifest_chain = select_recovery_chain(other + [wrong_manifest])
    expect_failure("manifest",
                   lambda: validate_semantic_chain(wrong_manifest_chain,
                                                   manifest_digest), failures)

    inner_crc = bytearray(target["frame"])
    inner_crc[-1] ^= 1
    wrapped_inner_crc = pack_ring_slot(
        inner_crc, target["slot_generation"], target["step_id"],
        target["kind"], len(good))
    expect_failure("inner_frame_crc",
                   lambda: validate_semantic_chain(
                       select_recovery_chain(other + [wrapped_inner_crc]),
                       manifest_digest), failures)

    torn_meta = bytearray(meta_b)
    torn_meta[24] ^= 1
    selected_name, _selected = select_ab_metadata(meta_a, torn_meta)
    if selected_name != "A":
        raise AssertionError("A/B metadata did not fall back to valid A")
    failures["metadata_b_torn_fallback"] = "selected A"
    expect_failure("metadata_both_torn",
                   lambda: select_ab_metadata(torn_meta, torn_meta), failures)
    return failures


def verify(args, manifest):
    os.environ["SPDK_SHM_ID"] = str(args.shm_id + 1)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu, pipeline_depth=4,
        requested_chunk_size=4 * 1024**2, spdk_shm_id=args.shm_id + 1)
    try:
        meta_a = ckpt.read_host_frame(manifest["metadata_a_offset"], 64)
        meta_b = ckpt.read_host_frame(manifest["metadata_b_offset"], 64)
        selected_copy, metadata = select_ab_metadata(meta_a, meta_b)
        raw_slots = [ckpt.read_host_frame(
            manifest["offset"] + slot * manifest["slot_size"],
            manifest["slot_size"]) for slot in range(manifest["ring_slots"])]
    finally:
        ckpt.cleanup()
    chain = select_recovery_chain(raw_slots)
    validate_semantic_chain(chain, manifest["manifest_digest"])
    initial, expected, _records, digest = make_chain(
        manifest["steps"], manifest["full_every"])
    if digest != manifest["manifest_digest"]:
        raise AssertionError("reconstructed manifest differs")
    oracle = S2DeltaOracle(initial, block_size=8, small_threshold=4)
    recovered = oracle.recover(initial, [item["frame"] for item in chain])
    if state_digest(recovered["state"]) != state_digest(expected):
        raise AssertionError("raw-ring recovered state is not byte exact")
    faults = fault_matrix(raw_slots, chain, manifest["manifest_digest"],
                          meta_a, meta_b)
    return {"metadata_copy": selected_copy, "metadata": metadata,
            "chain_steps": [item["step_id"] for item in chain],
            "wraps": manifest["steps"] / manifest["ring_slots"],
            "byte_exact": True, "faults": faults,
            "fault_count": len(faults)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("orchestrate", "verify"),
                        default="orchestrate")
    parser.add_argument("--manifest")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=4)
    parser.add_argument("--shm-id", type=int, default=1900)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ring-slots", type=int, default=16)
    parser.add_argument("--slot-size", type=int, default=2 * 1024**2)
    parser.add_argument("--full-every", type=int, default=8)
    parser.add_argument("--offset", type=int,
                        default=SAFE_OFFSET + 512 * 1024**2)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.phase == "verify":
        manifest = json.loads(Path(args.manifest).read_text())
        print(json.dumps(verify(args, manifest), sort_keys=True), flush=True)
        return
    if (args.steps < args.ring_slots or args.ring_slots <= args.full_every or
            args.slot_size % 4096 or args.offset % 4096):
        raise ValueError("invalid I6 ring dimensions")
    writer = ResultWriter("I6_RAW_RING", args)
    npu_info = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, npu_info))
    initial, expected, records, digest = make_chain(args.steps, args.full_every)
    del initial
    manifest = {"pci": args.pci, "npu": args.npu, "shm_id": args.shm_id,
                "steps": args.steps, "ring_slots": args.ring_slots,
                "slot_size": args.slot_size, "full_every": args.full_every,
                "offset": args.offset,
                "metadata_a_offset": args.offset - 8192,
                "metadata_b_offset": args.offset - 4096,
                "manifest_digest": digest,
                "expected_digest": state_digest(expected)}
    manifest_path = writer.run_dir / "ring_manifest.json"
    writer.write_json("ring_manifest.json", manifest)
    os.environ["SPDK_SHM_ID"] = str(args.shm_id)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu, pipeline_depth=4,
        requested_chunk_size=4 * 1024**2, spdk_shm_id=args.shm_id)
    try:
        for step, kind, frame in records:
            slot = (step - 1) % args.ring_slots
            raw = pack_ring_slot(frame, step, step, kind, args.slot_size)
            ckpt.write_host_frame(raw, args.offset + slot * args.slot_size)
        latest_full = args.steps - (args.steps % args.full_every)
        ckpt.write_host_frame(pack_ring_metadata(
            args.steps - 1, args.steps - 1,
            max(0, args.steps - 1 - args.ring_slots), latest_full),
            manifest["metadata_a_offset"])
        ckpt.write_host_frame(pack_ring_metadata(
            args.steps, args.steps, max(0, args.steps - args.ring_slots),
            latest_full), manifest["metadata_b_offset"])
    finally:
        ckpt.cleanup()
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", "verify",
               "--manifest", str(manifest_path), "--pci", args.pci,
               "--npu", str(args.npu), "--shm-id", str(args.shm_id)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               check=False)
    if completed.returncode:
        raise RuntimeError(f"I6 restart child failed: {completed.stderr}")
    lines = [line for line in completed.stdout.splitlines()
             if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError("I6 restart child emitted no JSON")
    summary = json.loads(lines[-1])
    sample = {"run_id": writer.run_id,
              "request_id": writer.run_id + "/restart",
              "checkpoint_id": "raw_ring_restart_fault_matrix",
              "status": "pass", "summary": summary,
              "events": [], "timeline_us": {"end_to_end": 0}}
    writer.add_sample(sample)
    result = writer.finalize(summary, status="pass")
    print(json.dumps({"status": result["status"],
                      "run_id": writer.run_id,
                      "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
