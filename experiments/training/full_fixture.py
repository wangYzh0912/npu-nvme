"""Shared deterministic training and fresh-restart oracle."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
from npu_nvme.framework.training_state import capture_training_controls

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")

def build_training(args, initialized=False):
    import mindspore as ms
    if args.deterministic is not None:
        ms.set_context(deterministic=args.deterministic)
    from experiments.common import init_env, make_causal_lm_training
    from npu_nvme.framework.cells import TrainOneStepCell

    if not initialized:
        init_env(device_id=args.npu, seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model, _dataset, optimizer = make_causal_lm_training(
        args.model, total_steps=1, device_id=args.npu, seq_len=args.seq_len,
        dropout_rate=args.dropout_rate, require_dataset=False)
    cell = TrainOneStepCell(
        model, optimizer)
    # A fresh GRAPH_MODE model still owns lazy initializers.  Loading before
    # the first compiled step appears to work byte-for-byte, but graph
    # materialisation can subsequently overwrite that state.  Compile and
    # allocate with the same excluded step in every process; the restore
    # process loads the checkpoint only after this barrier.
    warmup_start = time.perf_counter()
    warmup_loss = cell(*batch_for_step(ms, 0, args.seq_len))
    ms.hal.synchronize()
    warmup_value = float(np.asarray(warmup_loss.asnumpy()).reshape(()))
    if not np.isfinite(warmup_value):
        raise FloatingPointError(f"non-finite excluded warmup: {warmup_value}")
    print(f"[C1] excluded_step=0 loss={warmup_value:.9g} "
          f"time={time.perf_counter() - warmup_start:.3f}s", flush=True)
    return ms, model, optimizer, cell

def batch_for_step(ms, step, seq_len, vocab_size=50257):
    # GPT-2 consumes seq_len tokens and internally shifts to seq_len - 1.
    start = (step * 104729) % vocab_size
    ids = (np.arange(seq_len, dtype=np.int32) + start) % vocab_size
    mask = np.ones(seq_len, dtype=np.int32)
    return ms.Tensor(ids[None, :]), ms.Tensor(mask[None, :])

def train_range(ms, cell, begin, end, seq_len):
    losses = []
    times = []
    for step in range(begin, end + 1):
        batch = batch_for_step(ms, step, seq_len)
        start = time.perf_counter()
        loss = cell(*batch)
        ms.hal.synchronize()
        times.append(time.perf_counter() - start)
        value = float(np.asarray(loss.asnumpy()).reshape(()))
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite loss at step {step}: {value}")
        losses.append(value)
        print(f"[C1] step={step} loss={value:.9g} time={times[-1]:.3f}s",
              flush=True)
    return losses, times

def iter_unique_parameters(model, optimizer):
    seen = set()
    for component, obj in (("model", model), ("optimizer", optimizer)):
        for name, parameter in obj.parameters_and_names():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield f"{component}/{name}", parameter

def state_digest(model, optimizer):
    digest = hashlib.sha256()
    fields = 0
    total_bytes = 0
    for name, parameter in iter_unique_parameters(model, optimizer):
        array = np.ascontiguousarray(parameter.value().asnumpy())
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        fields += 1
        total_bytes += int(array.nbytes)
    return {"sha256": digest.hexdigest(), "fields": fields,
            "bytes": total_bytes}

def write_state_oracle(root, model, optimizer):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (name, parameter) in enumerate(
            iter_unique_parameters(model, optimizer)):
        array = np.ascontiguousarray(parameter.value().asnumpy())
        filename = f"{index:04d}.npy"
        np.save(root / filename, array, allow_pickle=False)
        records.append({"name": name, "file": filename,
                        "shape": list(array.shape), "dtype": array.dtype.str,
                        "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
    write_json(root / "manifest.json", {"fields": records})

def compare_state_oracle(root, model, optimizer, rtol, atol):
    root = Path(root)
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8"))["fields"]
    actual_fields = list(iter_unique_parameters(model, optimizer))
    if len(actual_fields) != len(manifest):
        raise AssertionError("final state field count changed")
    byte_exact = 0
    max_abs = 0.0
    max_rel = 0.0
    mismatches = []
    for record, (name, parameter) in zip(manifest, actual_fields):
        if name != record["name"]:
            raise AssertionError(
                f"final state order/name changed: {name} != {record['name']}")
        expected = np.load(root / record["file"], mmap_mode="r",
                           allow_pickle=False)
        actual = np.ascontiguousarray(parameter.value().asnumpy())
        if list(actual.shape) != record["shape"] or actual.dtype.str != record["dtype"]:
            mismatches.append({"name": name, "reason": "shape_or_dtype"})
            continue
        if hashlib.sha256(actual.tobytes()).hexdigest() == record["sha256"]:
            byte_exact += 1
            continue
        if np.issubdtype(actual.dtype, np.inexact):
            expected_float = np.asarray(expected, dtype=np.float64)
            actual_float = actual.astype(np.float64)
            difference = np.abs(actual_float - expected_float)
            field_max_abs = float(difference.max(initial=0.0))
            denominator = np.maximum(np.abs(expected_float), 1e-12)
            field_max_rel = float((difference / denominator).max(initial=0.0))
            max_abs = max(max_abs, field_max_abs)
            max_rel = max(max_rel, field_max_rel)
            if not np.allclose(actual, expected, rtol=rtol, atol=atol,
                               equal_nan=False):
                mismatches.append({"name": name, "reason": "not_allclose",
                                   "max_abs": field_max_abs,
                                   "max_rel": field_max_rel})
        elif not np.array_equal(actual, expected):
            mismatches.append({"name": name, "reason": "integer_mismatch"})
    return {"fields": len(manifest), "byte_exact_fields": byte_exact,
            "allclose": not mismatches, "max_abs": max_abs,
            "max_rel": max_rel, "mismatches": mismatches[:20],
            "rtol": rtol, "atol": atol}

def control_state(ms, optimizer, cursor, args):
    return capture_training_controls(
        ms, optimizer, {"epoch": 0, "sample": int(cursor)},
        args.loss_scale, args.seed)
