#!/usr/bin/env python3
"""I1 real MindSpore trajectory collector for the S2 oracle.

The collector streams one snapshot at a time and records block statistics
instead of retaining every training state.  GPT-2 124M is the exact
weight+Adam state lane; larger models can use the same collector with
``--no-optimizer`` for a scale-only observation lane.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from experiments.benchmarks.io_matrix import (  # noqa: E402
    ResultWriter, check_npu_free, environment_snapshot,
)
from experiments.common import (  # noqa: E402
    init_env, make_causal_lm_training, warmup_model,
)
from direct_checkpoint import ProbeTrainOneStepCell  # noqa: E402
from delta_protocol import FRAME_HEADER_SIZE  # noqa: E402
from s2_delta import build_block_manifest, score_manifest_blocks  # noqa: E402


def snapshot_state(model, optimizer, include_optimizer):
    state = {}
    for parameter in model.get_parameters():
        state[f"model/{parameter.name}"] = parameter.asnumpy()
    if include_optimizer:
        for prefix, parameters in (
                ("optimizer/m", optimizer.moments1),
                ("optimizer/v", optimizer.moments2)):
            for parameter in parameters:
                state[f"{prefix}/{parameter.name}"] = parameter.asnumpy()
        state["optimizer/global_step"] = optimizer.global_step.asnumpy()
    return state


def finite_diagnostics(state):
    """Return a compact numeric-health report without retaining tensors."""
    bad = []
    finite_arrays = 0
    for name, value in state.items():
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.inexact):
            continue
        finite_arrays += 1
        nonfinite = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        if nonfinite:
            bad.append({
                "name": name,
                "dtype": array.dtype.name,
                "elements": int(array.size),
                "nonfinite": nonfinite,
                "nan": int(np.isnan(array).sum()),
                "inf": int(np.isinf(array).sum()),
            })
    return {"arrays": len(state), "floating_arrays": finite_arrays,
            "nonfinite_arrays": len(bad), "nonfinite": bad}


def block_scores(current, reference, manifest, workers, score_dtype):
    return score_manifest_blocks(current, reference, manifest,
                                 score_dtype=np.dtype(score_dtype),
                                 workers=workers,
                                 compute_current_l2=False)


def aggregate_block_scores(scores, source_manifest, target_manifest):
    """Aggregate aligned fine blocks into a coarser manifest.

    The norm of a coarse block is the square root of the sum of the fine
    block squared norms.  This reuses one FP32 state traversal for the 64K
    and 256K requested views while retaining the same per-block fields.
    """
    source_size = int(source_manifest["block_size"])
    target_size = int(target_manifest["block_size"])
    if target_size <= source_size or target_size % source_size:
        raise ValueError("target block size must be an aligned multiple")
    by_name = {}
    for item in scores:
        by_name.setdefault(item["name"], []).append(item)
    output = []
    for target in target_manifest["blocks"]:
        begin = int(target["element_offset"])
        end = begin + int(target["element_count"])
        fine = [item for item in by_name.get(target["name"], [])
                if int(item["element_offset"]) >= begin and
                int(item["element_offset"]) + int(item["element_count"]) <= end]
        covered = sum(int(item["element_count"]) for item in fine)
        if covered != int(target["element_count"]):
            raise ValueError("target block is not covered by source manifest")
        score_sq = sum(float(item["score"]) ** 2 for item in fine)
        current_l2_values = [item["current_l2"] for item in fine]
        current_l2_sq = (sum(float(value) ** 2
                             for value in current_l2_values)
                         if all(value is not None
                                for value in current_l2_values) else None)
        output.append({
            **target,
            "score": float(np.sqrt(score_sq)),
            "nonzero": sum(int(item["nonzero"]) for item in fine),
            "max_abs": max((float(item["max_abs"]) for item in fine),
                           default=0.0),
            "current_l2": (float(np.sqrt(current_l2_sq))
                           if current_l2_sq is not None else None),
            "relative_l2": (float(np.sqrt(score_sq)) /
                            max(float(np.sqrt(current_l2_sq)), 1e-30)
                            if current_l2_sq is not None else None),
        })
    output.sort(key=lambda item: (-item["score"], int(item["block_id"])))
    return output


def coverage(scores, budgets, include_ids=True):
    total = sum(item["score"] ** 2 for item in scores)
    changed = [item for item in scores if item["nonzero"]]
    changed_bytes = sum(int(item["element_count"]) *
                        np.dtype(item["dtype"]).itemsize for item in changed)
    output = {}
    for budget in budgets:
        count = max(1, int(np.ceil(len(scores) * budget))) if scores else 0
        selected = scores[:count]
        energy = sum(item["score"] ** 2 for item in selected)
        logical_bytes = sum(int(item["element_count"]) *
                            np.dtype(item["dtype"]).itemsize
                            for item in selected)
        physical_bytes = sum(
            ((int(item["element_count"]) * np.dtype(item["dtype"]).itemsize
              + 4095) // 4096) * 4096 for item in selected)
        record = {
            "count": len(selected),
            "energy_fraction": (energy / total) if total else 1.0,
            "logical_bytes": logical_bytes,
            "physical_bytes_aligned_4k": physical_bytes,
            "changed_block_count": len(changed),
            "changed_logical_bytes": changed_bytes,
        }
        if include_ids:
            record["block_ids"] = [item["block_id"] for item in selected]
        output[str(int(budget * 100))] = record
    return output


def state_category(name):
    if name.startswith("model/"):
        return "model"
    if name.startswith("optimizer/m/"):
        return "adam_m"
    if name.startswith("optimizer/v/"):
        return "adam_v"
    if name == "optimizer/global_step":
        return "optimizer_other"
    return "small"


def category_metrics(scores, budgets):
    grouped = {}
    for item in scores:
        grouped.setdefault(state_category(item["name"]), []).append(item)
    output = {}
    for category, records in sorted(grouped.items()):
        records.sort(key=lambda item: (-item["score"], item["block_id"]))
        relative_l2 = [item["relative_l2"] for item in records
                       if item["relative_l2"] is not None]
        output[category] = {
            "blocks": len(records),
            "changed_blocks": sum(bool(item["nonzero"]) for item in records),
            "l2": float(np.sqrt(sum(item["score"] ** 2 for item in records))),
            "max_abs": max((item["max_abs"] for item in records), default=0.0),
            "relative_l2_median": (float(np.median(relative_l2))
                                   if relative_l2 else None),
            "coverage": coverage(records, budgets, include_ids=False),
        }
    return output


def small_state_metrics(current, reference, manifest):
    by_category = {}
    for item in manifest["small"]:
        name = item["name"]
        left = np.asarray(current[name])
        right = np.asarray(reference[name])
        changed = not np.array_equal(left, right, equal_nan=True)
        category = state_category(name)
        record = by_category.setdefault(
            category, {"fields": 0, "changed_fields": 0,
                       "bytes": 0, "changed_bytes": 0})
        record["fields"] += 1
        record["bytes"] += int(left.nbytes)
        if changed:
            record["changed_fields"] += 1
            record["changed_bytes"] += int(left.nbytes)
    return by_category


def tensor_change_metrics(current, scores, small_metrics):
    """Derive tensor changes from the scoring pass already paid for.

    Re-reading both 9.8 GB states with ``array_equal`` added two redundant
    full-state scans per block size.  A blocked tensor changed exactly when
    one of its manifest blocks is nonzero; small tensors are already checked
    by ``small_state_metrics``.
    """
    changed_names = {item["name"] for item in scores if item["nonzero"]}
    changed_small_fields = sum(
        int(record["changed_fields"]) for record in small_metrics.values())
    changed_small_bytes = sum(
        int(record["changed_bytes"]) for record in small_metrics.values())
    total_tensors = len(current)
    total_bytes = sum(int(np.asarray(value).nbytes)
                      for value in current.values())
    changed_tensors = len(changed_names) + changed_small_fields
    changed_block_tensor_bytes = sum(
        int(np.asarray(current[name]).nbytes) for name in changed_names)
    changed_bytes = changed_block_tensor_bytes + changed_small_bytes
    return {
        "tensor_count": total_tensors,
        "changed_tensor_count": changed_tensors,
        "changed_tensor_fraction": (changed_tensors / total_tensors
                                     if total_tensors else 0.0),
        "total_bytes": total_bytes,
        "changed_bytes": changed_bytes,
        "changed_byte_fraction": (changed_bytes / total_bytes
                                   if total_bytes else 0.0),
    }


def reference_metrics(current, reference, manifest, scores, budgets):
    changed = [item for item in scores if item["nonzero"]]
    logical_bytes = sum(int(item["element_count"]) *
                        np.dtype(item["dtype"]).itemsize for item in changed)
    physical_bytes = sum(
        ((int(item["element_count"]) * np.dtype(item["dtype"]).itemsize
          + 4095) // 4096) * 4096 for item in changed)
    small = small_state_metrics(current, reference, manifest)
    return {
        "l2": float(np.sqrt(sum(item["score"] ** 2 for item in scores))),
        "changed_block_count": len(changed),
        "changed_logical_bytes": logical_bytes,
        "changed_physical_bytes_aligned_4k": physical_bytes,
        "tensor_changes": tensor_change_metrics(current, scores, small),
        "coverage": coverage(scores, budgets, include_ids=False),
        "categories": category_metrics(scores, budgets),
        "small": small,
    }


def advance_topk_reference(current, reference, manifest, ranked_scores, top_k):
    """Advance a sampled Top-K reference without materializing a huge frame.

    The returned byte count is exactly the v3 native-replacement frame size.
    This keeps trajectory observation semantically equivalent to
    ``S2DeltaOracle.observe()+ack()`` while avoiding a multi-GiB temporary
    frame and a second full-state scoring pass.
    """
    selected = [item for item in ranked_scores if item["nonzero"]][:top_k]
    frame_bytes = FRAME_HEADER_SIZE
    for item in selected:
        name = item["name"]
        start = int(item["element_offset"])
        count = int(item["element_count"])
        source = np.asarray(current[name]).reshape(-1)
        target = np.asarray(reference[name]).reshape(-1)
        target[start:start + count] = source[start:start + count]
        frame_bytes += (28 + len(name.encode("utf-8")) +
                        len(str(item["dtype"]).encode("ascii")) +
                        count * np.dtype(item["dtype"]).itemsize)
    changed_small = 0
    for item in manifest["small"]:
        name = item["name"]
        source = np.asarray(current[name])
        if np.array_equal(source, reference[name], equal_nan=True):
            continue
        reference[name] = np.array(source, copy=True)
        changed_small += 1
        frame_bytes += (14 + len(name.encode("utf-8")) +
                        len(str(item["dtype"]).encode("ascii")) +
                        int(source.nbytes))
    return {"selected": selected, "changed_small": changed_small,
            "frame_bytes": int(frame_bytes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gpt2", "gpt2_xl"), default="gpt2")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=1025)
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--block-sizes", default=None,
                        help="comma-separated block sizes; overrides --block-size")
    parser.add_argument("--small-threshold", type=int, default=10000)
    parser.add_argument("--top-k-percent", type=float, default=10.0)
    parser.add_argument("--top-k-percents", default="5,10,20",
                        help="comma-separated Top-K percentages for coverage")
    parser.add_argument("--max-age", type=int, default=50,
                        help="age in training steps after which a block is overdue")
    parser.add_argument("--score-dtype", choices=("float32", "float64"),
                        default="float32",
                        help="temporary diff dtype for XL observation")
    parser.add_argument("--score-workers", type=int, default=8,
                        help="parallel tensor scoring workers")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--sample-windows", default=None,
                        help="inclusive ranges such as 1-30,236-265,471-500")
    parser.add_argument("--no-optimizer", action="store_true")
    parser.add_argument("--numeric-only", action="store_true",
                        help="check loss/state health without S2 scoring/frames")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-mr", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.steps <= 0 or args.sample_every <= 0 or not 0 < args.top_k_percent <= 100:
        raise ValueError("invalid trajectory dimensions")
    if args.score_workers <= 0:
        raise ValueError("score workers must be positive")
    budgets = tuple(float(item.strip()) / 100.0
                    for item in args.top_k_percents.split(",")
                    if item.strip())
    if not budgets or any(not 0 < value <= 1 for value in budgets):
        raise ValueError("top-k percentages must be in (0, 100]")
    block_sizes = ([int(item) for item in args.block_sizes.split(",")]
                   if args.block_sizes else [args.block_size])
    if not block_sizes or any(value <= 0 for value in block_sizes):
        raise ValueError("block sizes must be positive")
    block_sizes = tuple(dict.fromkeys(block_sizes))
    sample_steps = None
    capture_only_steps = set()
    if args.sample_windows:
        sample_steps = set()
        for item in args.sample_windows.split(","):
            begin_text, end_text = item.split("-", 1)
            begin, end = int(begin_text), int(end_text)
            if begin <= 0 or end < begin or end > args.steps:
                raise ValueError("invalid sample window")
            sample_steps.update(range(begin, end + 1))
            if begin > 1:
                capture_only_steps.add(begin - 1)

    writer = ResultWriter("I1_REAL", args)
    writer.config.update({
        "model": args.model,
        "state": "weights_only" if args.no_optimizer else "weights+adam_m_v+global_step",
        "oracle": "S2 persisted-reference replacement",
        "block_sizes": list(block_sizes),
        "state_categories": ["model", "adam_m", "adam_v", "optimizer_other",
                             "small", "control"],
        "reference_semantics": ["adjacent", "persisted"],
        "top_k_percents": [value * 100 for value in budgets],
        "max_age_steps": args.max_age,
        "score_dtype": args.score_dtype,
        "score_workers": args.score_workers,
        "score_full_state_passes_per_sample": 1 + len(block_sizes),
        "adjacent_coarse_aggregation": "aligned_from_smallest_block_size",
        "persisted_scoring": "direct_per_block_size_independent_ack_reference",
        "score_accumulation": ("float32_block_float64_global"
                               if args.score_dtype == "float32"
                               else "float64_block_float64_global"),
        "relative_l2_auxiliary": "not_collected_to_avoid_redundant_full_state_scan",
        "sample_windows": args.sample_windows,
    })
    writer.write_json("config.json", writer.config)
    npu_info = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, npu_info))

    init_env(device_id=args.npu, seed=args.seed)
    model, dataset, optimizer = make_causal_lm_training(
        args.model, total_steps=args.steps + 2, device_id=args.npu,
        seq_len=args.seq_len, train_mr=args.train_mr)
    warmup_model(model, optimizer, dataset)
    cell = ProbeTrainOneStepCell(model, optimizer, enable_probe=False,
                                 ckpt_interval=9999)
    iterator = dataset.create_tuple_iterator()

    initial = snapshot_state(model, optimizer, not args.no_optimizer)
    initial_health = finite_diagnostics(initial)
    writer.write_json("numeric_gate.json", {
        "stage": "post_warmup_initial_state",
        **initial_health,
    })
    if initial_health["nonfinite_arrays"]:
        failure = {
            "error": "post-warmup model state contains non-finite values",
            "stage": "post_warmup_initial_state",
            "numeric_gate": initial_health,
        }
        writer.add_failure(failure)
        result = writer.finalize({"numeric_gate": initial_health}, status="fail")
        print(json.dumps({"status": result["status"],
                          "run_id": writer.run_id,
                          "summary": result["summary"]}, indent=2), flush=True)
        return
    if args.numeric_only:
        samples = []
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = dataset.create_tuple_iterator()
                batch = next(iterator)
            loss = cell(*batch)
            import mindspore as ms
            ms.hal.synchronize()
            loss_value = float(np.asarray(loss.asnumpy()).reshape(()))
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite training loss at step {step}: {loss_value}")
            if step % args.sample_every:
                continue
            health = finite_diagnostics(snapshot_state(
                model, optimizer, not args.no_optimizer))
            if health["nonfinite_arrays"]:
                raise FloatingPointError(
                    f"non-finite training state at step {step}: "
                    f"{health['nonfinite'][:3]}")
            sample = {
                "run_id": writer.run_id,
                "request_id": f"{writer.run_id}/step_{step:04d}",
                "checkpoint_id": f"numeric_{step:04d}",
                "step": step,
                "status": "pass",
                "loss": loss_value,
                "numeric_health": health,
                "events": [{"name": "numeric_health_pass",
                            "monotonic_ns": time.monotonic_ns()}],
                "timeline_us": {"end_to_end": 0},
            }
            writer.add_sample(sample)
            samples.append(sample)
        result = writer.finalize({
            "steps": args.steps,
            "samples": len(samples),
            "seed": args.seed,
            "numeric_health": "all sampled weights/optimizer states finite",
            "losses": [sample["loss"] for sample in samples],
        }, status="pass")
        print(json.dumps({"status": result["status"],
                          "run_id": writer.run_id,
                          "summary": result["summary"]}, indent=2), flush=True)
        return
    manifests = {
        block_size: build_block_manifest(
            initial, block_size=block_size,
            small_threshold=args.small_threshold)
        for block_size in block_sizes}
    previous = {name: np.array(value, copy=True)
                for name, value in initial.items()}
    persisted = {
        block_size: {name: np.array(value, copy=True)
                     for name, value in initial.items()}
        for block_size in block_sizes
    }
    generations = {block_size: 0 for block_size in block_sizes}
    last_selected = {block_size: set() for block_size in block_sizes}
    last_acked = {
        block_size: {
            int(item["block_id"]): 0
            for item in manifests[block_size]["blocks"]}
        for block_size in block_sizes
    }
    trajectory = []
    try:
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = dataset.create_tuple_iterator()
                batch = next(iterator)
            loss = cell(*batch)
            import mindspore as ms
            ms.hal.synchronize()
            loss_value = float(np.asarray(loss.asnumpy()).reshape(()))
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite training loss at step {step}: {loss_value}")
            should_record = (step in sample_steps if sample_steps is not None
                             else step % args.sample_every == 0)
            should_capture = should_record or step in capture_only_steps
            if not should_capture:
                continue
            current = snapshot_state(model, optimizer, not args.no_optimizer)
            health = finite_diagnostics(current)
            if health["nonfinite_arrays"]:
                raise FloatingPointError(
                    f"non-finite training state at step {step}: "
                    f"{health['nonfinite'][:3]}")
            if not should_record:
                previous = {name: np.array(value, copy=True)
                            for name, value in current.items()}
                del current
                continue
            block_results = {}
            scoring_started_ns = time.monotonic_ns()
            score_futures = {}
            base_block_size = min(block_sizes)
            with ThreadPoolExecutor(
                    max_workers=2 * len(block_sizes),
                    thread_name_prefix="s2-reference") as executor:
                base_manifest = manifests[base_block_size]
                score_futures[(base_block_size, "adjacent")] = executor.submit(
                    block_scores, current, previous, base_manifest,
                    args.score_workers, args.score_dtype)
                for block_size in block_sizes:
                    manifest = manifests[block_size]
                    # Each block size advances a different Top-K ACK
                    # reference, so persisted scores cannot be aggregated
                    # from another block-size policy.
                    score_futures[(block_size, "persisted")] = executor.submit(
                        block_scores, current, persisted[block_size], manifest,
                        args.score_workers, args.score_dtype)
                    if block_size == base_block_size:
                        continue
                    # Aligned coarser views are derived from the base pass;
                    # arbitrary block sizes retain a direct scoring fallback.
                    if (block_size > base_block_size and
                            block_size % base_block_size == 0):
                        continue
                    score_futures[(block_size, "adjacent")] = executor.submit(
                        block_scores, current, previous, manifest,
                        args.score_workers, args.score_dtype)
            for block_size in block_sizes:
                manifest = manifests[block_size]
                adjacent = score_futures[(base_block_size, "adjacent")].result()
                persisted_scores = score_futures[
                    (block_size, "persisted")].result()
                if block_size != base_block_size:
                    if (block_size > base_block_size and
                            block_size % base_block_size == 0):
                        adjacent = aggregate_block_scores(
                            adjacent, base_manifest, manifest)
                    else:
                        adjacent = score_futures[(block_size, "adjacent")].result()
                # Metrics must observe the last acknowledged reference.  ACK
                # mutates that reference and therefore happens strictly after
                # all persisted-reference statistics are materialized.
                adjacent_metrics = reference_metrics(
                    current, previous, manifest, adjacent, budgets)
                persisted_metrics = reference_metrics(
                    current, persisted[block_size], manifest,
                    persisted_scores, budgets)
                top_k = max(1, int(len(persisted_scores) *
                                   args.top_k_percent / 100))
                advance = advance_topk_reference(
                    current, persisted[block_size], manifest,
                    persisted_scores, top_k)
                selected = {item["block_id"] for item in advance["selected"]}
                for block_id in selected:
                    last_acked[block_size][block_id] = step
                ages = [step - last_acked[block_size][item["block_id"]]
                        for item in manifest["blocks"]]
                overdue_ids = [item["block_id"] for item in manifest["blocks"]
                               if step - last_acked[block_size][item["block_id"]]
                               > args.max_age]
                overdue_bytes = sum(
                    int(item["element_count"]) * np.dtype(item["dtype"]).itemsize
                    for item in manifest["blocks"]
                    if item["block_id"] in overdue_ids)
                generations[block_size] += 1
                previous_selected = last_selected[block_size]
                block_results[str(block_size)] = {
                    "block_count": len(adjacent),
                    "frame_bytes_top_k": advance["frame_bytes"],
                    "reference_semantics": ["adjacent", "persisted"],
                    "references": {
                        "adjacent": adjacent_metrics,
                        "persisted": persisted_metrics,
                    },
                    # Retain the old scalar aliases for consumers of the
                    # earlier trajectory schema.
                    "adjacent_l2": adjacent_metrics["l2"],
                    "persisted_l2": persisted_metrics["l2"],
                    "selected_count": len(selected),
                    "selected_jaccard": (
                        len(selected & previous_selected) /
                        len(selected | previous_selected)
                        if selected | previous_selected else 1.0),
                    "age": {"max": max(ages, default=0),
                            "mean": float(np.mean(ages)) if ages else 0.0,
                            "overdue_count": len(overdue_ids),
                            "overdue_bytes": overdue_bytes,
                            "permanently_unselected_count": sum(
                                value == 0 for value in last_acked[block_size].values()),
                            "permanently_unselected_bytes": sum(
                                int(item["element_count"]) *
                                np.dtype(item["dtype"]).itemsize
                                for item in manifest["blocks"]
                                if last_acked[block_size][item["block_id"]] == 0)},
                    "coverage": adjacent_metrics["coverage"],
                    "categories": adjacent_metrics["categories"],
                    "small": adjacent_metrics["small"],
                }
                last_selected[block_size] = selected
            primary = block_results[str(block_sizes[0])]
            scoring_elapsed_ns = time.monotonic_ns() - scoring_started_ns
            sample = {
                "run_id": writer.run_id,
                "request_id": f"{writer.run_id}/step_{step:04d}",
                "checkpoint_id": f"trajectory_{step:04d}",
                "step": step,
                "status": "pass",
                "loss": loss_value,
                "numeric_health": health,
                "state_arrays": len(current),
                "block_sizes": block_results,
                "block_count": primary["block_count"],
                "frame_bytes_top_k": primary["frame_bytes_top_k"],
                "adjacent_l2": primary["adjacent_l2"],
                "persisted_l2": primary["persisted_l2"],
                "selected_count": primary["selected_count"],
                "selected_jaccard": primary["selected_jaccard"],
                "age": primary["age"],
                "reference_semantics": primary["reference_semantics"],
                "references": primary["references"],
                "coverage": primary["coverage"],
                "timeline_us": {
                    "scoring": scoring_elapsed_ns / 1000.0,
                    "end_to_end": scoring_elapsed_ns / 1000.0,
                },
                "events": [{"name": "trajectory_sample",
                            "monotonic_ns": time.monotonic_ns()}],
            }
            writer.add_sample(sample)
            trajectory.append(sample)
            previous = {name: np.array(value, copy=True)
                        for name, value in current.items()}
            del current
        result = writer.finalize({
            "steps": args.steps,
            "samples": len(trajectory),
            "manifest_digests": {
                str(block_size): manifests[block_size]["digest"]
                for block_size in block_sizes},
            "final_generation": {
                str(block_size): generations[block_size]
                for block_size in block_sizes},
            "final_frame_bytes": trajectory[-1]["frame_bytes_top_k"] if trajectory else 0,
        }, status="pass")
        print(json.dumps({"status": result["status"], "run_id": writer.run_id,
                          "summary": result["summary"]}, indent=2), flush=True)
    except BaseException as error:
        writer.add_failure({"error": repr(error)})
        writer.finalize({}, status="fail")
        raise


if __name__ == "__main__":
    main()
