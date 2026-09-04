#!/usr/bin/env python3
"""INC-2 graph-resident auxiliary-load experiment.

The old ``p6_aux_injection.py`` deliberately uses a Host thread and explicit
stream synchronization, which is useful as a prototype but invalid evidence
for graph-edge load.  This runner puts every auxiliary operation in the same
MindSpore graph as the training update.  The auxiliary checksum is threaded
through ``Depend`` so the compiler cannot delete the edge or move the update
ahead of it.

Formal timing records host submission latency.  The device is synchronized
once after the formal batch, outside the measured step loop; therefore the
result never presents host submission time as a device idle-window proof.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import Parameter, Tensor, ops

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from experiments.common import (  # noqa: E402
    init_env,
    make_causal_lm_training,
    warmup_model,
)
from experiments.benchmarks.io_matrix import (  # noqa: E402
    check_npu_free, environment_snapshot, stats,
)
from direct_checkpoint import ProbeTrainOneStepCell  # noqa: E402


MODES = ("baseline", "marker_only", "compute_only", "memory_scan_only",
         "incremental_chain")


class GraphAuxiliaryTrainCell(ProbeTrainOneStepCell):
    """Training cell with a graph-only auxiliary edge before optimizer update."""

    def __init__(self, network, optimizer, mode, compute_repeats=4,
                 block_size=65536, top_k=128):
        # Do not enable the legacy checkpoint probe here: INC-2 has no I/O.
        super().__init__(network, optimizer, enable_probe=False,
                         ckpt_interval=999999)
        self.mode = mode
        self.compute_repeats = int(compute_repeats)
        self.block_size = int(block_size)
        self.top_k = int(top_k)
        self.depend = ops.Depend()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.abs = ops.Abs()
        self.square = ops.Square()
        self.sqrt = ops.Sqrt()
        self.cast = ops.Cast()

        # A modest fixed HBM buffer isolates compute-only from model-scale
        # memory traffic.  It is a Parameter solely to force device residency.
        compute_values = np.linspace(-1.0, 1.0, 262144,
                                     dtype=np.float32).reshape(1024, 256)
        self.compute_buffer = Parameter(Tensor(compute_values),
                                        requires_grad=False,
                                        name="inc2_compute_buffer")
        self.marker = Parameter(Tensor([0], ms.int32), requires_grad=False,
                                name="inc2_step_marker")
        params = network.trainable_params()
        if not params:
            raise ValueError("training network has no trainable parameters")
        self.scan_parameter = params[0]
        flat_size = int(np.prod(self.scan_parameter.shape))
        if flat_size < self.block_size:
            raise ValueError("first trainable parameter is smaller than block size")
        block_count = flat_size // self.block_size
        self.block_count = block_count
        self.top_k_effective = min(max(1, self.top_k), block_count)
        # The reference is device-resident and updated only after the complete
        # chain has been computed.  It is not a checkpoint/reference protocol.
        self.reference = Parameter(
            Tensor(np.asarray(self.scan_parameter.asnumpy()).copy()),
            requires_grad=False, name="inc2_reference")
        self.zero = Tensor(np.asarray(0.0, dtype=np.float32))
        self.one = Tensor(np.asarray(1.0, dtype=np.float32))
        self.marker_one = Tensor([1], ms.int32)
        self.compute_scale = Tensor(1.0001, ms.float32)
        self.compute_bias = Tensor(0.0001, ms.float32)

    def _auxiliary(self):
        if self.mode == "baseline":
            return self.zero
        if self.mode == "marker_only":
            # A named device-side state update gives the marker-only graph a
            # real dependency edge without scanning model memory.
            marker = ops.assign_add(self.marker, self.marker_one)
            return ops.reshape(self.cast(marker, ms.float32), ())
        if self.mode == "compute_only":
            value = self.compute_buffer
            for _ in range(self.compute_repeats):
                value = ops.tanh(value * self.compute_scale)
                value = value + self.compute_bias
            return self.reduce_sum(value)

        flat = ops.reshape(self.scan_parameter, (-1,))
        if self.mode == "memory_scan_only":
            # Read the model-scale HBM buffer and do only a scalar reduction.
            return self.reduce_sum(self.abs(self.cast(flat, ms.float32)))

        # incremental_chain: diff -> block norm -> Top-K -> cast/checksum.
        reference = ops.reshape(self.reference, (-1,))
        diff = flat - reference
        trimmed = diff[:self.block_count * self.block_size]
        blocks = ops.reshape(trimmed, (self.block_count, self.block_size))
        norms = self.sqrt(self.reduce_sum(self.square(blocks), 1))
        values, _indices = ops.top_k(
            norms, self.top_k_effective, sorted=False)
        encoded = self.cast(values, ms.float16)
        checksum = self.reduce_sum(self.cast(encoded, ms.float32))
        update_reference = ops.assign(self.reference, self.scan_parameter)
        return self.depend(checksum, update_reference)

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        auxiliary = self._auxiliary()
        # Depend on the gradient tuple, so optimizer kernels cannot run before
        # the auxiliary edge.  No Host synchronization is inserted here.
        grads = self.depend(grads, auxiliary)
        opt_res = self.optimizer(grads)
        return self.depend(loss, opt_res)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                      default=str) + "\n", encoding="utf-8")


def next_batch(iterator, dataset):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = dataset.create_tuple_iterator()
        return next(iterator), iterator


def run(args):
    if args.mode not in MODES:
        raise ValueError(f"unsupported mode: {args.mode}")
    if args.warmups < 0 or args.steps <= 0:
        raise ValueError("warmups must be >= 0 and steps must be positive")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = (f"INC2_{args.model}_{args.mode}_seed{args.seed}_"
              f"{stamp}_{os.getpid()}")
    root = Path(args.output_root or ROOT /
                "results/incremental-observation-20260904/INC2_graph_edge_load")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    config = {
        **vars(args),
        "run_id": run_id,
        "experiment": "INC2_graph_edge_load",
        "formal_policy": "20 warmup + 50 formal; no step-internal host synchronization",
        "auxiliary_location": "MindSpore graph before optimizer via Depend",
        "invalid_prototype_excluded": "experiments/benchmarks/p6_aux_injection.py",
    }
    write_json(run_dir / "config.json", config)
    npu_info = check_npu_free(args.npu)
    write_json(run_dir / "environment.json", environment_snapshot(args, npu_info))

    def event(value):
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"run_id": run_id, **value},
                                    sort_keys=True, default=str) + "\n")

    start_ns = time.monotonic_ns()
    records = []
    losses = []
    status = "pass"
    error = None
    try:
        init_env(device_id=args.npu, seed=args.seed)
        model, dataset, optimizer = make_causal_lm_training(
            args.model, total_steps=args.warmups + args.steps + 3,
            device_id=args.npu, seq_len=args.seq_len,
            dropout_rate=0.0, train_mr=args.train_mr)
        cell = GraphAuxiliaryTrainCell(
            model, optimizer, args.mode,
            compute_repeats=args.compute_repeats,
            block_size=args.block_size, top_k=args.top_k)
        # One excluded compile/materialisation step.
        warmup_model(model, optimizer, dataset, cell=cell)
        iterator = dataset.create_tuple_iterator()
        total = args.warmups + args.steps
        for index in range(total):
            batch, iterator = next_batch(iterator, dataset)
            submit_start = time.perf_counter_ns()
            loss = cell(*batch)
            submit_end = time.perf_counter_ns()
            losses.append(loss)
            formal = index >= args.warmups
            if formal:
                step = index - args.warmups + 1
                sample = {
                    "run_id": run_id,
                    "request_id": f"{run_id}/step_{step:04d}",
                    "step": step,
                    "mode": args.mode,
                    "host_submit_start_ns": submit_start,
                    "host_submit_end_ns": submit_end,
                    "host_submit_ms": (submit_end - submit_start) / 1e6,
                    "device_synchronized": False,
                }
                records.append(sample)
                event({"event": "step_submitted", **sample})
        # This is intentionally outside the formal step loop.  It establishes
        # completion before reading losses and writing the result.
        sync_start = time.perf_counter_ns()
        ms.hal.synchronize()
        sync_end = time.perf_counter_ns()
        for index, value in enumerate(losses):
            loss_value = float(np.asarray(value.asnumpy()).reshape(()))
            if not np.isfinite(loss_value):
                raise FloatingPointError(f"non-finite loss at loop index {index}: {loss_value}")
        for index, sample in enumerate(records):
            sample["loss"] = float(np.asarray(losses[args.warmups + index]
                                               .asnumpy()).reshape(()))
        synchronized_ms = (sync_end - sync_start) / 1e6
        summary = {
            "model": args.model,
            "seed": args.seed,
            "mode": args.mode,
            "warmups": args.warmups,
            "formal_steps": args.steps,
            "host_submit_ms": stats([item["host_submit_ms"] for item in records]),
            "post_loop_device_sync_ms": synchronized_ms,
            "wallclock_ms": (time.monotonic_ns() - start_ns) / 1e6,
            "loss_first": records[0]["loss"],
            "loss_last": records[-1]["loss"],
            "graph_edge": {
                "mode": args.mode,
                "checksum_participates_in_depend": args.mode != "baseline",
                "optimizer_precedes_auxiliary": False,
                "host_worker": False,
                "step_internal_ms_hal_synchronize": False,
                "block_size_elements": args.block_size,
                "top_k_blocks": cell.top_k_effective,
            },
            "device_timing_limit":
                "No per-step device duration is claimed; synchronize was once after formal loop.",
        }
    except BaseException as exc:
        status = "fail"
        error = repr(exc)
        summary = {"model": args.model, "seed": args.seed,
                   "mode": args.mode, "formal_steps_completed": len(records)}
        event({"event": "run_failed", "error": error})
    result = {
        "status": status,
        "run_id": run_id,
        "config": config,
        "samples": len(records),
        "summary": summary,
        "paths": {"config": "config.json", "environment": "environment.json",
                  "events": "events.jsonl", "result": "result.json"},
    }
    if error:
        result["error"] = error
    write_json(run_dir / "result.json", result)
    print(json.dumps({"run_id": run_id, "status": status,
                      "samples": len(records), "error": error},
                     sort_keys=True), flush=True)
    return 0 if status == "pass" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model", choices=("gpt2", "gpt2_xl"), default="gpt2")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--npu", type=int, default=2)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--train-mr", default=None)
    parser.add_argument("--compute-repeats", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
