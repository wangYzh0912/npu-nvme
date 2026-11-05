"""Normalize follow-up AIC/AIV probe data without guessing missing evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


def _records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            yield row


def record_kv_tensor(path: str | Path, tensor: Any, *, run_id: str, layer: int,
                     cache_kind: str, phase: str, source: str,
                     cache_layout: str | None = None, block_size: int | None = None,
                     block_table_hash: str | None = None) -> dict[str, Any]:
    """Append tensor metadata only; never reads tensor values or synchronizes."""
    record = {
        "run_id": run_id, "layer": layer, "cache_kind": cache_kind,
        "phase": phase, "cache_layout": cache_layout,
        "shape": list(tensor.shape), "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()), "dtype": str(tensor.dtype),
        "block_size": block_size, "block_table_hash": block_table_hash,
        "tensor_name": getattr(tensor, "_aic_aiv_probe_name", None),
        "source": source,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return record


def normalize_kv_layout(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize one JSON object per observed KV tensor/view."""
    result = []
    required = ("run_id", "layer", "cache_kind", "phase", "shape", "stride", "dtype")
    for index, row in enumerate(rows, 1):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"KV row {index}: missing {', '.join(missing)}")
        shape, stride = row["shape"], row["stride"]
        if (not isinstance(shape, list) or not isinstance(stride, list)
                or len(shape) != len(stride) or not shape
                or any(not isinstance(x, int) or x < 0 for x in shape)
                or any(not isinstance(x, int) or x < 0 for x in stride)):
            raise ValueError(f"KV row {index}: shape/stride must be equal-length nonnegative integer arrays")
        expected, contiguous = 1, True
        for size, actual_stride in zip(reversed(shape), reversed(stride)):
            if size != 1 and actual_stride != expected:
                contiguous = False
            expected *= size
        normalized = {
            "run_id": str(row["run_id"]), "layer": row["layer"],
            "cache_kind": str(row["cache_kind"]), "phase": row.get("phase"),
            "cache_layout": row.get("cache_layout"), "shape": shape,
            "stride": stride, "storage_offset": row.get("storage_offset"),
            "dtype": str(row["dtype"]), "contiguous": contiguous,
            "block_size": row.get("block_size"),
            "block_table_hash": row.get("block_table_hash"),
            "tensor_name": row.get("tensor_name"),
            "source": row.get("source", "unspecified"),
        }
        result.append(normalized)
    return result


def normalize_hbm_csv(path: str | Path) -> list[dict[str, Any]]:
    """Import explicitly-unitized HBM CSV; reject ambiguous units/columns."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        time_col = "timestamp_ns" if "timestamp_ns" in fields else "timestamp_us" if "timestamp_us" in fields else None
        read_col = "read_bytes_per_s" if "read_bytes_per_s" in fields else None
        write_col = "write_bytes_per_s" if "write_bytes_per_s" in fields else None
        if not time_col or not read_col or not write_col:
            raise ValueError("HBM CSV requires timestamp_ns or timestamp_us, read_bytes_per_s, write_bytes_per_s")
        rows = []
        previous = {}
        for line_no, row in enumerate(reader, 2):
            try:
                scaled = Decimal(row[time_col]) * (1000 if time_col == "timestamp_us" else 1)
                if not scaled.is_finite() or scaled != scaled.to_integral_value():
                    raise ValueError("timestamp is not an integer number of nanoseconds")
                timestamp = int(scaled)
                read_bw, write_bw = float(row[read_col]), float(row[write_col])
            except (TypeError, ValueError, InvalidOperation) as exc:
                raise ValueError(f"HBM row {line_no}: invalid numeric field") from exc
            if timestamp < 0 or not all(math.isfinite(x) and x >= 0 for x in (read_bw, write_bw)):
                raise ValueError(f"HBM row {line_no}: non-finite or negative time or bandwidth")
            device = row.get("device_id") or None
            if device in previous and timestamp <= previous[device]:
                raise ValueError(f"HBM row {line_no}: timestamps must increase")
            previous[device] = timestamp
            rows.append({"timestamp_ns": timestamp, "read_bytes_per_s": read_bw,
                         "write_bytes_per_s": write_bw,
                         "device_id": row.get("device_id") or None,
                         "source": str(path)})
    return rows


def classify_windows(windows: Iterable[dict[str, Any]],
                     tasks: Iterable[dict[str, Any]],
                     dependencies: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify pending AIC work using time-scoped, complete dependency evidence.

    Dependency keys include run/device/stream/task and task_start_ns, so reused
    hardware task IDs cannot borrow a readiness observation from another task.
    Legacy records without those keys remain unknown rather than proving ready.
    """
    tasks = list(tasks)
    dependencies = list(dependencies)
    identity_fields = ('run_id', 'device_id', 'stream_id', 'task_id')

    def identity(row, *, observation=False):
        start_key = 'task_start_ns' if observation else 'start_ns'
        if any(key not in row for key in (*identity_fields, start_key)):
            return None
        return tuple(str(row[key]) for key in identity_fields) + (int(row[start_key]),)

    observations = defaultdict(list)
    for dep in dependencies:
        if 'task_id' not in dep or 'state' not in dep:
            raise ValueError('dependency rows require task_id and state')
        key = identity(dep, observation=True)
        if key is None or 'timestamp_ns' not in dep:
            continue
        if dep['state'] not in ('ready', 'blocked', 'waiting_dependency', 'unknown'):
            raise ValueError('unsupported dependency state')
        timestamp = int(dep['timestamp_ns'])
        if timestamp < 0 or timestamp > key[-1]:
            raise ValueError('dependency timestamp must precede task start')
        if not dep.get('source'):
            raise ValueError('timed dependency evidence requires a source')
        observations[key].append(dep)
    for values in observations.values():
        values.sort(key=lambda row: int(row['timestamp_ns']))
        if len({int(row['timestamp_ns']) for row in values}) != len(values):
            raise ValueError('ambiguous simultaneous dependency states')

    result = []
    for window in windows:
        start, end = int(window['window_start_ns']), int(window['window_end_ns'])
        if end <= start:
            raise ValueError('window end must be after start')
        aic, aiv = window.get('aic_activity_pct'), window.get('aiv_activity_pct')
        overlapping = [t for t in tasks if int(t['start_ns']) < end and int(t['end_ns']) > start]
        pending, ready = [], []
        blocked = []
        for task in tasks:
            if any(field not in window for field in ('run_id', 'device_id')):
                continue
            if task.get('engine') != 'aic' or int(task['start_ns']) <= start:
                continue
            if any(field in window and str(window[field]) != str(task.get(field))
                   for field in ('run_id', 'device_id')):
                continue
            key = identity(task)
            states = observations.get(key, []) if key is not None else []
            task_evidence = []
            for index, state in enumerate(states):
                left = max(start, int(state['timestamp_ns']))
                right = min(end, int(task['start_ns']),
                            int(states[index + 1]['timestamp_ns']) if index + 1 < len(states) else end)
                if left >= right:
                    continue
                task_evidence.append(state)
                if state['state'] == 'ready' and state.get('all_dependencies_observed') is True:
                    ready.append(dict(task_identity=list(key), begin_ns=left, end_ns=right,
                                      source=state['source']))
            if task_evidence:
                pending.append(task)
                if all(s['state'] in ('blocked', 'waiting_dependency') for s in task_evidence):
                    blocked.append(task)
        if not window.get('valid', True):
            reason = 'invalid_measurement'
        elif aic is None or aiv is None:
            reason = 'missing_pmu'
        elif aiv < 50 or aic > 10:
            reason = 'outside_target_condition'
        elif ready:
            reason = 'aic_ready_candidate'
        elif pending and len(blocked) == len(pending):
            reason = 'observed_candidates_dependency_blocked'
        else:
            reason = 'dependency_unknown'
        result.append(dict(window_start_ns=start, window_end_ns=end,
            aic_activity_pct=aic, aiv_activity_pct=aiv,
            overlapping_task_count=len(overlapping),
            candidate_aic_task_count=len(pending), classification=reason,
            ready_intervals=ready if reason == 'aic_ready_candidate' else [],
            confidence='dependency_evidence' if reason in
                ('aic_ready_candidate', 'observed_candidates_dependency_blocked') else
                'measured' if reason in ('invalid_measurement', 'missing_pmu', 'outside_target_condition') else 'incomplete'))
    return result


def compare_variants(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compare paired variant run metrics; report screen results, not causality."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for row in rows:
        for key in ("variant", "replicate", "latency_ms", "output_sha256", "valid"):
            if key not in row:
                raise ValueError(f"comparison row missing {key}")
        if type(row['valid']) is not bool:
            raise ValueError("valid must be a boolean")
        latency = row['latency_ms']
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
            raise ValueError("latency_ms must be finite and positive")
        case = str(row.get('case', 'default'))
        key = (case, str(row['variant']), str(row['replicate']))
        if key in seen:
            raise ValueError(f"duplicate experiment identifier: {key}")
        seen.add(key)
        groups[(case, str(row["variant"]))].append(row)
    if (not any(variant == 'baseline' for _, variant in groups)
            or not any(variant != 'baseline' for _, variant in groups)):
        raise ValueError("need baseline and at least one candidate variant")
    result = []
    baseline_count = 0
    for (case, name), variants in sorted(groups.items()):
        if name == "baseline":
            baseline_count += len(variants)
            continue
        baseline = {str(r['replicate']): r for r in groups.get((case, 'baseline'), [])}
        paired = [(baseline[str(r["replicate"])], r) for r in variants if str(r["replicate"]) in baseline]
        valid = [(b, r) for b, r in paired if b["valid"] and r["valid"] and b["output_sha256"] == r["output_sha256"]]
        if len(valid) != len(variants) or len(valid) != len(baseline) or not valid:
            verdict = "reject_incomplete_or_incorrect"
        else:
            improvements = [(b["latency_ms"] - r["latency_ms"]) / b["latency_ms"] * 100 for b, r in valid]
            verdict = "retain_for_confirmation" if all(x > 0 for x in improvements) else "no_consistent_gain"
        result.append({"case": case, "variant": name, "paired_valid_replicates": len(valid),
                       "latency_improvement_pct": [round((b["latency_ms"] - r["latency_ms"]) / b["latency_ms"] * 100, 3) for b, r in valid],
                       "verdict": verdict})
    return {"baseline_replicates": baseline_count, "results": result,
            "interpretation": "screening only; requires end-to-end confirmation and resource-interference review"}


def _write_json(path: str, rows: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    kv = sub.add_parser("kv-layout")
    kv.add_argument("--input", required=True, help="JSONL tensor metadata emitted by runtime probe")
    kv.add_argument("--output", required=True)
    hbm = sub.add_parser("hbm-csv")
    hbm.add_argument("--input", required=True)
    hbm.add_argument("--output", required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("--input", required=True, help="JSONL one row per variant replicate")
    comp.add_argument("--output", required=True)
    dep = sub.add_parser("classify-windows")
    dep.add_argument("--windows", required=True, help="PMU CSV with window_start_ns/window_end_ns and engine activity")
    dep.add_argument("--tasks", required=True, help="JSONL device tasks with task_id, engine, start_ns, end_ns")
    dep.add_argument("--dependencies", required=True, help="JSONL runtime dependency observations")
    dep.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "kv-layout":
        output = normalize_kv_layout(_records(args.input))
    elif args.command == "hbm-csv":
        output = normalize_hbm_csv(args.input)
    elif args.command == "compare":
        output = compare_variants(_records(args.input))
    else:
        with Path(args.windows).open(newline="") as stream:
            windows = list(csv.DictReader(stream))
        for window in windows:
            for key in ("window_start_ns", "window_end_ns"):
                window[key] = int(window[key])
            for key in ("aic_activity_pct", "aiv_activity_pct"):
                window[key] = float(window[key]) if window.get(key, "") else None
            window["valid"] = window.get("valid", "true").lower() in ("true", "1", "yes")
        output = classify_windows(windows, _records(args.tasks), _records(args.dependencies))
    _write_json(args.output, output)


if __name__ == "__main__":
    main()
