#!/usr/bin/env python3
"""G1 correctness gate: GPT-2 XL FULL save, process restart, and load.

Phase ``save`` writes one FULL checkpoint to the already formatted 83.0.0
layout and records a SHA-256 digest for every parameter.  It then starts a
fresh Python process for phase ``load``; the second process mounts metadata
from NVMe, loads the checkpoint, and compares every parameter digest.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, REPO_ROOT)


def parameter_hashes(model):
    result = {}
    for name, parameter in model.parameters_and_names():
        array = parameter.value().asnumpy()
        result[name] = {
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
        }
    return result


def build_model(device_id):
    from experiments.common import init_env, make_gpt2xl_training, warmup_model

    init_env(device_id=device_id)
    model, dataset, optimizer = make_gpt2xl_training(
        total_steps=2, device_id=device_id)
    warmup_model(model, optimizer, dataset)
    return model


def save_phase(args, manifest_path):
    from direct_checkpoint import DirectCheckpoint

    model = build_model(args.npu)
    expected = parameter_hashes(model)
    run_dir = os.path.dirname(manifest_path)
    os.makedirs(run_dir, exist_ok=True)

    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu, pipeline_depth=8,
        requested_chunk_size=4 * 1024 * 1024, rank_id=0, world_size=1,
        keep_last_n=3, slot_size_gb=10, spdk_shm_id=args.shm_id,
        profiling_dir=os.path.join(run_dir, "profiling"))
    ckpt._meta_pkl = os.path.join(run_dir, "checkpoint_meta.pkl")
    try:
        start = time.perf_counter()
        handle = ckpt.save(
            model, step=args.step,
            meta_path=os.path.join(run_dir, "checkpoint_meta.pkl"))
        if handle.status != handle.DISPATCHED:
            raise RuntimeError(f"save did not return DISPATCHED: {handle.status}")
        handle.wait()
        persist_seconds = time.perf_counter() - start
        if handle.status != handle.PERSISTED:
            raise RuntimeError(f"save did not reach PERSISTED: {handle.status}")
        manifest = {
            "gate": "G1",
            "pci": args.pci,
            "npu": args.npu,
            "step": args.step,
            "persist_seconds": persist_seconds,
            "generation": handle.generation,
            "parameter_count": len(expected),
            "parameters": expected,
        }
        with open(manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        print(f"[G1/save] PERSISTED generation={handle.generation} "
              f"parameters={len(expected)} time={persist_seconds:.3f}s", flush=True)
    finally:
        ckpt.cleanup()


def load_phase(args, manifest_path):
    from direct_checkpoint import DirectCheckpoint

    with open(manifest_path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    model = build_model(args.npu)
    run_dir = os.path.dirname(manifest_path)
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu, pipeline_depth=8,
        requested_chunk_size=4 * 1024 * 1024, rank_id=0, world_size=1,
        keep_last_n=3, slot_size_gb=10, spdk_shm_id=args.shm_id,
        profiling_dir=os.path.join(run_dir, "profiling_load"))
    ckpt._meta_pkl = os.path.join(run_dir, "checkpoint_meta_load.pkl")
    try:
        ckpt.load(model, step=manifest["step"])
        if hasattr(__import__("mindspore").hal, "synchronize"):
            __import__("mindspore").hal.synchronize()
        actual = parameter_hashes(model)
        expected = manifest["parameters"]
        if set(actual) != set(expected):
            raise AssertionError("parameter name set changed across restart")
        mismatches = [
            name for name in expected
            if actual[name]["sha256"] != expected[name]["sha256"]
            or actual[name]["shape"] != expected[name]["shape"]
            or actual[name]["nbytes"] != expected[name]["nbytes"]
        ]
        if mismatches:
            raise AssertionError(
                f"{len(mismatches)} parameter digests differ; first={mismatches[0]}")
        print(f"[G1/load] PASS restart load; verified {len(actual)} parameters",
              flush=True)
    finally:
        ckpt.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("orchestrate", "save", "load"),
                        default="orchestrate")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--shm-id", type=int, default=83)
    parser.add_argument(
        "--run-dir",
        default=os.path.join(REPO_ROOT, "experiments", "output", "gates",
                             time.strftime("g1_%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    manifest_path = os.path.join(run_dir, "g1_manifest.json")
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    if args.phase == "load":
        load_phase(args, manifest_path)
        return

    if args.phase == "save":
        save_phase(args, manifest_path)
        return

    # The orchestrator itself never initializes MindSpore/ACL.  The save
    # child must exit before the load child is created, so the device runtime
    # and all HBM allocations are released at the OS process boundary.
    save_child = [
        sys.executable, os.path.abspath(__file__), "--phase", "save",
        "--pci", args.pci, "--npu", str(args.npu), "--step", str(args.step),
        "--shm-id", str(args.shm_id), "--run-dir", run_dir,
    ]
    subprocess.run(save_child, cwd=run_dir, check=True)
    load_child = [
        sys.executable, os.path.abspath(__file__), "--phase", "load",
        "--pci", args.pci, "--npu", str(args.npu), "--step", str(args.step),
        "--shm-id", str(args.shm_id), "--run-dir", run_dir,
    ]
    subprocess.run(load_child, cwd=run_dir, check=True)
    print("[G1] PASS FULL save -> exited save process -> fresh load process -> "
          "per-parameter verification", flush=True)


if __name__ == "__main__":
    main()
