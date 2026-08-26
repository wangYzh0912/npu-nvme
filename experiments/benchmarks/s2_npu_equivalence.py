#!/usr/bin/env python3
"""I2 CPU/NPU equivalence gate for block scoring and quantization.

This gate deliberately uses a deterministic block tensor rather than a model
snapshot.  I1 owns the model-trajectory question; I2 isolates the graph
operators and the parameter-local valid-length semantics needed by R0/R1.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from experiments.benchmarks.io_matrix import (  # noqa: E402
    ResultWriter, check_npu_free, environment_snapshot,
)


def cpu_reference(current, reference, valid_lengths, top_k):
    diff = current.astype(np.float32) - reference.astype(np.float32)
    masked = np.zeros_like(diff)
    for index, length in enumerate(valid_lengths):
        masked[index, :int(length)] = diff[index, :int(length)]
    norms = np.sqrt(np.sum(masked * masked, axis=1, dtype=np.float32))
    order = np.argsort(-norms, kind="stable")[:top_k]
    selected = current[order].copy()
    maxima = np.max(np.abs(masked), axis=1)
    scales = np.where(maxima > 0, maxima / 127.0, 1.0).astype(np.float32)
    quantized = np.clip(np.rint(masked / scales[:, None]), -127, 127).astype(np.int8)
    return {
        "norms": norms,
        "top_values": norms[order],
        "top_indices": order.astype(np.int32),
        "selected_values": selected,
        "scales": scales,
        "quantized": quantized,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", type=int, default=5)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=257)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.blocks <= 0 or args.block_size <= 0 or not 0 < args.top_k <= args.blocks:
        raise ValueError("invalid I2 dimensions")

    writer = ResultWriter("I2_NPU_EQUIV", args)
    writer.config.update({
        "scope": "synthetic deterministic block operator gate",
        "valid_length_semantics": "parameter-local; padded elements excluded",
        "checks": ["norm", "top_k_values", "top_k_indices", "selected_values",
                   "per_block_scale", "int8_quantization"],
    })
    writer.write_json("config.json", writer.config)
    npu_info = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, npu_info))

    import mindspore as ms
    from mindspore import nn, ops

    rng = np.random.default_rng(20260826)
    current = rng.normal(0, 0.25, (args.blocks, args.block_size)).astype(np.float32)
    reference = current.copy()
    # Distinct magnitudes avoid ambiguous TopK ties; the final block also
    # exercises a non-full tail and the zero block exercises zero-scale logic.
    for index in range(args.blocks):
        length = 1 + ((index * 37) % args.block_size)
        if index == args.blocks - 1:
            length = args.block_size - 3
        if index == 2:
            current[index, :length] = reference[index, :length]
        else:
            current[index, :length] += np.float32((index + 1) * 0.013)
    valid_lengths = np.array(
        [1 + ((index * 37) % args.block_size) for index in range(args.blocks)],
        dtype=np.int32)
    valid_lengths[-1] = args.block_size - 3
    expected = cpu_reference(current, reference, valid_lengths, args.top_k)

    class BlockOps(nn.Cell):
        def __init__(self, top_k):
            super().__init__()
            self.top_k = top_k
            self.abs = ops.Abs()
            self.cast = ops.Cast()
            self.expand = ops.ExpandDims()
            self.less = ops.Less()
            self.reduce_sum = ops.ReduceSum(keep_dims=False)
            self.reduce_max = ops.ReduceMax(keep_dims=False)
            self.sqrt = ops.Sqrt()
            self.topk = ops.TopK(sorted=True)
            self.gather = ops.Gather()
            self.round = ops.Round()
            self.minimum = ops.Minimum()
            self.maximum = ops.Maximum()
            self.select = ops.Select()
            self.positions = ms.Tensor(np.arange(args.block_size, dtype=np.int32))

        def construct(self, current_tensor, reference_tensor, lengths):
            diff = current_tensor - reference_tensor
            valid = self.less(self.positions, self.expand(lengths, 1))
            mask = self.cast(valid, ms.float32)
            masked = diff * mask
            norms = self.sqrt(self.reduce_sum(masked * masked, 1))
            top_values, top_indices = self.topk(norms, self.top_k)
            selected = self.gather(current_tensor, top_indices, 0)
            maxima = self.reduce_max(self.abs(masked), 1)
            positive = maxima > ms.Tensor(0.0, ms.float32)
            scales = self.select(positive, maxima / ms.Tensor(127.0, ms.float32),
                                 ms.Tensor(1.0, ms.float32))
            quantized = self.cast(
                self.maximum(self.minimum(self.round(masked / self.expand(scales, 1)),
                                          ms.Tensor(127.0, ms.float32)),
                             ms.Tensor(-127.0, ms.float32)), ms.int8)
            return norms, top_values, top_indices, selected, scales, quantized

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=args.npu)
    cell = BlockOps(args.top_k)
    start = time.perf_counter_ns()
    outputs = cell(ms.Tensor(current), ms.Tensor(reference), ms.Tensor(valid_lengths))
    ms.hal.synchronize()
    elapsed_us = (time.perf_counter_ns() - start) / 1000.0
    actual = [value.asnumpy() for value in outputs]

    checks = {
        "norms": np.allclose(actual[0], expected["norms"], rtol=2e-5, atol=2e-5),
        "top_values": np.allclose(actual[1], expected["top_values"], rtol=2e-5, atol=2e-5),
        "top_indices": np.array_equal(actual[2], expected["top_indices"]),
        "selected_values": np.array_equal(actual[3], expected["selected_values"]),
        "scales": np.allclose(actual[4], expected["scales"], rtol=2e-5, atol=2e-5),
        "quantized": np.array_equal(actual[5], expected["quantized"]),
    }
    mismatches = {}
    for name, passed in checks.items():
        if not passed:
            mismatches[name] = {
                "expected": np.asarray(expected[name]).tolist(),
                "actual": np.asarray(actual[list(checks).index(name)]).tolist(),
            }
    status = "pass" if all(checks.values()) else "fail"
    sample = {
        "run_id": writer.run_id,
        "request_id": writer.run_id + "/graph_0001",
        "checkpoint_id": "i2_operator_gate",
        "status": status,
        "npu": args.npu,
        "blocks": args.blocks,
        "block_size": args.block_size,
        "top_k": args.top_k,
        "valid_lengths": valid_lengths.tolist(),
        "checks": checks,
        "elapsed_us": elapsed_us,
        "events": [{"name": "graph_complete", "monotonic_ns": time.monotonic_ns()}],
        "timeline_us": {"graph": elapsed_us},
    }
    if mismatches:
        sample["mismatches"] = mismatches
    writer.add_sample(sample)
    result = writer.finalize({"checks": checks, "elapsed_us": elapsed_us}, status=status)
    print(json.dumps({"status": result["status"], "run_id": writer.run_id,
                      "summary": result["summary"]}, indent=2), flush=True)
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
