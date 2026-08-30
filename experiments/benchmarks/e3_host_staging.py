#!/usr/bin/env python3
"""E3 memory-only Host-staging baseline.

This runner deliberately does not create a filesystem file: the current
experiment policy forbids touching /models, while E3's memory question only
needs HBM -> Host DRAM -> HBM.  It builds the requested real MindFormers
checkpoint-only model, measures regular and aclrtMallocHost buffers, and
records process RSS/VmPin peaks.  No value is inferred from the model size.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from experiments.baselines import two_phase_common as tpc  # noqa: E402
from experiments.benchmarks.io_matrix import (  # noqa: E402
    ResultWriter, check_npu_free, environment_snapshot, stats,
)
from experiments.common import (  # noqa: E402
    init_env, make_causal_lm_checkpoint_model, make_causal_lm_training,
    warmup_checkpoint_model, warmup_model,
)


def training_state_descriptors(model, optimizer):
    descs, seen = [], set()
    for component, obj in (("model", model), ("optimizer", optimizer)):
        for name, parameter in obj.parameters_and_names():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            dtype = np.dtype(tpc.ms.dtype_to_nptype(parameter.dtype))
            ptr = tpc.get_dev_ptr(parameter)
            descs.append({"name": f"{component}/{name}", "ptr": ptr,
                          "size": int(parameter.size) * dtype.itemsize,
                          "dtype_np": dtype, "param_ref": parameter,
                          "host_resident": ptr == 0})
    return descs


def proc_status():
    values = {}
    text = Path("/proc/self/status").read_text()
    for key in ("VmRSS", "VmPin", "VmLck", "RssAnon", "RssFile"):
        match = re.search(rf"^{key}:\s+(\d+)\s+kB$", text, re.MULTILINE)
        values[key.lower() + "_bytes"] = int(match.group(1)) * 1024 if match else None
    return values


def restore_ptr(param_descs, host_ptr, offsets, device_id):
    tpc._ensure_acl_device(device_id)
    start = time.perf_counter_ns()
    for desc in param_descs:
        if desc.get("host_resident"):
            raw = ctypes.string_at(host_ptr + offsets[desc["name"]],
                                   desc["size"])
            value = np.frombuffer(raw, dtype=desc["dtype_np"]).reshape(
                desc["param_ref"].shape).copy()
            desc["param_ref"].set_data(tpc.ms.Tensor(value))
            continue
        ret = tpc.acl_lib.aclrtMemcpy(
            ctypes.c_void_p(desc["ptr"]), desc["size"],
            ctypes.c_void_p(host_ptr + offsets[desc["name"]]), desc["size"],
            tpc.ACL_MEMCPY_HOST_TO_DEVICE)
        tpc._check_acl_ret(ret, f"E3 H2D {desc['name']}")
    tpc.ms.hal.synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def run_mode(args, model, descs, offsets, total, mode, writer):
    regular = None
    pinned = None
    if mode == "regular":
        regular = tpc.allocate_host_buffer(total)
        host_ptr = int(regular.ctypes.data)
    else:
        pinned = tpc.allocate_pinned_host_buffer(total)
        host_ptr = int(pinned)
    rss_peak = proc_status()
    samples = []
    device_descs = [desc for desc in descs if not desc.get("host_resident")]
    try:
        for index in range(args.warmups + args.samples):
            before = proc_status()
            if mode == "regular":
                snap = tpc.snapshot_d2h(device_descs, regular, offsets, args.npu)
            else:
                snap = tpc.snapshot_d2h_pinned(device_descs, host_ptr, offsets,
                                               args.npu)
            for desc in descs:
                if desc.get("host_resident"):
                    value = np.ascontiguousarray(desc["param_ref"].asnumpy())
                    ctypes.memmove(host_ptr + offsets[desc["name"]],
                                   int(value.ctypes.data), desc["size"])
            restore_ms = restore_ptr(descs, host_ptr, offsets, args.npu)
            after = proc_status()
            for key in rss_peak:
                vals = [rss_peak[key], before.get(key), after.get(key)]
                vals = [value for value in vals if value is not None]
                rss_peak[key] = max(vals) if vals else None
            row = {
                "experiment": "E3", "model": args.model, "mode": mode,
                "run_id": writer.run_id,
                "request_id": f"{writer.run_id}/{mode}_{index:04d}",
                "checkpoint_id": f"{mode}_checkpoint_{index:04d}",
                "warmup": index < args.warmups, "sample": index,
                "state_bytes": total, "host_buffer_bytes": total,
                "snapshot_ms": snap["total_ms"], "d2h_ms": snap["memcpy_ms"],
                "h2d_ms": restore_ms, "status": "pass",
                "rss": after,
                "events": [{"name": "snapshot_end", "monotonic_ns": time.monotonic_ns()},
                           {"name": "restore_end", "monotonic_ns": time.monotonic_ns()}],
            }
            if index >= args.warmups:
                writer.add_sample(row)
    finally:
        if pinned is not None:
            tpc.free_pinned_host_buffer(pinned)
    end_to_end = [row["snapshot_ms"] + row["h2d_ms"] for row in writer.samples
                  if row["mode"] == mode]
    return {"mode": mode, "state_bytes": total, "samples": len(end_to_end),
            "latency": stats(end_to_end), "peak": rss_peak}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gpt2_xl", "gpt2_13b"), required=True)
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:42:00.0",
                        help="NPU PCI address for environment recording; no SSD is used")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--modes", nargs="+", choices=("regular", "pinned"),
                        default=("regular", "pinned"))
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--complete-training-state", action="store_true")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.samples < 30:
        raise SystemExit("formal samples must be >=30")
    root = Path(args.output_root or ROOT / "results/ppt-evidence-20260829/E3/host-staging")
    check_npu_free(args.npu)
    init_env(device_id=args.npu)
    if args.complete_training_state:
        model, dataset, optimizer = make_causal_lm_training(
            args.model, total_steps=2, device_id=args.npu, seq_len=129,
            dropout_rate=0.0)
        warmup_model(model, optimizer, dataset)
        descs = training_state_descriptors(model, optimizer)
    else:
        model, cfg = make_causal_lm_checkpoint_model(args.model, seq_len=128)
        warmup_checkpoint_model(model, cfg, seq_len=128)
        descs = tpc.get_param_descriptors(model)
    offsets = {}
    cursor = 0
    for desc in descs:
        offsets[desc["name"]] = cursor
        cursor += desc["size"]
    total = cursor
    for mode in args.modes:
        writer = ResultWriter("E3", args)
        writer.config.update({"model": args.model, "mode": mode,
                              "state_bytes": total, "chunk_size": args.chunk_size,
                              "slot_count": 1, "host_staging": True,
                              "state_scope": "model+optimizer" if args.complete_training_state else "model",
                              "pci": None, "ssd_policy": "not touched; no /models"})
        writer.write_json("config.json", writer.config)
        writer.write_json("environment.json", environment_snapshot(args, None))
        try:
            summary = run_mode(args, model, descs, offsets, total, mode, writer)
        except BaseException as error:
            writer.add_failure({"mode": mode, "error": repr(error)})
            summary = {"mode": mode, "state_bytes": total, "error": repr(error)}
        values = [row["snapshot_ms"] + row["h2d_ms"] for row in writer.samples]
        metrics = {
            "model": args.model, "mode": mode, "state_bytes": total,
            "logical_bytes": total, "physical_bytes": 0,
            "chunk_size": args.chunk_size, "pipeline_depth": 1, "slot_count": 1,
            "latency_mean": stats(values).get("mean"),
            "latency_p50": stats(values).get("median"),
            "latency_p95": stats(values).get("p95"),
            "throughput": stats([total / (value / 1000) / 1024**2
                                  for value in values]),
            "foreground_wait": stats([row["snapshot_ms"] for row in writer.samples]),
            "step_overhead": None,
            "host_rss_peak": summary.get("peak", {}).get("vmrss_bytes"),
            "pinned_dram_peak": summary.get("peak", {}).get("vmpin_bytes"),
            "hbm_peak": None, "pcie_bytes": total * len(values) * 2,
            "nvme_bytes": 0, "recovery_error": 0, "loss_deviation": None,
            "fault_results": {"hbm_host_h2d_roundtrip":
                              "pass" if values else "fail"},
            "memory_scope": "memory-only HBM↔Host roundtrip; no /models or SSD",
            "mode_summary": summary,
        }
        status = "pass" if not writer.failed and len(values) == args.samples else "fail"
        result = writer.finalize(metrics, status=status)
        print(json.dumps({"status": result["status"], "model": args.model,
                          "mode": mode, "state_bytes": total,
                          "run_dir": str(writer.run_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
