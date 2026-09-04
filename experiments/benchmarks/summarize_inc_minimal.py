#!/usr/bin/env python3
"""Validate and summarize the 2026-09-04 minimal INC observation matrix."""

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def distribution(values):
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def ensure_non_restore_artifacts(run_dir, events=None):
    """Make the non-applicable restore state and event stream explicit."""
    restore = run_dir / "restore.json"
    if not restore.exists():
        restore.write_text(json.dumps({
            "status": "not_applicable",
            "reason": "observation-only experiment; no checkpoint published",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    events_path = run_dir / "events.jsonl"
    if not events_path.exists() or (events and events_path.stat().st_size == 0):
        with events_path.open("w", encoding="utf-8") as stream:
            for event in events or []:
                stream.write(json.dumps(event, sort_keys=True) + "\n")


def command_output(command):
    return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False).stdout.strip()


def ensure_inc1_profile_artifacts(run_dir, result):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps({
            "experiment": "INC1_real_training_pmu",
            "model": result["model"],
            "seed": result["seed"],
            "npu": result["device"],
            "metric_group": result["metric_group"],
            "warmups": result["warmups"],
            "steps": result["steps"],
            "seq_len": result["seq_len"],
            "train_mr": result["train_mr"],
            "source": "reconstructed from immutable result.json",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    environment = run_dir / "environment.json"
    if not environment.exists():
        environment.write_text(json.dumps({
            "snapshot_semantics": "post-run reconstructed; not run-time evidence",
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "repo_commit": command_output(["git", "rev-parse", "HEAD"]),
            "repo_branch": command_output(["git", "branch", "--show-current"]),
            "repo_status": command_output(["git", "status", "--porcelain"]),
            "device": result["device"],
            "profile_command": result["profile_command"],
            "export_command": result["export_command"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    child_path = run_dir / "child_result.json"
    child = read_json(child_path) if child_path.exists() else {"samples": []}
    events = [{
        "run_id": run_dir.name,
        "step": sample["step"],
        "monotonic_ns": sample["monotonic_ns"],
        "device_marker_value": sample["device_marker_value"],
        "loss": sample["loss"],
        "step_ms": sample["step_ms"],
    } for sample in child.get("samples", [])]
    ensure_non_restore_artifacts(run_dir, events)


def collect_inc1(root):
    groups = [root / "INC1_real_training_pmu" / "gpt2_seed41_corrected",
              root / "INC1_real_training_pmu" / "gpt2_xl_seed41"]
    output = []
    for group in groups:
        for path in sorted(group.glob("*/result.json")):
            result = read_json(path)
            if result.get("status") != "pass":
                continue
            run_dir = path.parent
            ensure_inc1_profile_artifacts(run_dir, result)
            output.append({
                "run_dir": str(run_dir),
                "model": result["model"],
                "metric_group": result["metric_group"],
                "status": result["status"],
                "warmups": result["warmups"],
                "steps": result["steps"],
                "step_stats_ms": result["step_stats_ms"],
                "hbm": result["hbm"],
                "pmu_total_time_us": result["pmu_total_time_us"],
                "marker_sequence_valid": result["device_marker"]["sequence_valid"],
                "common_clock_alignment": result["device_marker"]["common_clock_alignment"],
                "idle_window_inference_allowed": result["device_marker"]["idle_window_inference_allowed"],
                "marker_task_matches": result["device_marker"]["task_timeline"]["count"],
            })
    return output


def collect_inc2(root):
    base = root / "INC2_graph_edge_load"
    rows = []
    for path in sorted(base.glob("formal*/*/result.json")):
        result = read_json(path)
        run_dir = path.parent
        ensure_non_restore_artifacts(run_dir)
        summary = result["summary"]
        rows.append({
            "run_dir": str(run_dir),
            "status": result["status"],
            "seed": summary["seed"],
            "mode": summary["mode"],
            "wallclock_ms": summary["wallclock_ms"],
            "host_submit_ms": summary["host_submit_ms"],
            "graph_edge": summary["graph_edge"],
        })
    paired = []
    by_key = {(row["seed"], row["mode"]): row for row in rows}
    for seed in (41, 42, 43):
        baseline = by_key.get((seed, "baseline"))
        chain = by_key.get((seed, "incremental_chain"))
        if baseline and chain:
            overhead = ((chain["wallclock_ms"] / baseline["wallclock_ms"])
                        - 1.0) * 100.0
            paired.append({"seed": seed, "overhead_percent": overhead})
    values = [row["overhead_percent"] for row in paired]
    if len(values) == 3:
        mean = statistics.fmean(values)
        margin = 4.3026527299 * statistics.stdev(values) / math.sqrt(3)
        interval = [mean - margin, mean + margin]
    else:
        mean = statistics.fmean(values) if values else None
        interval = None
    return {
        "runs": rows,
        "paired_incremental_chain": paired,
        "mean_overhead_percent": mean,
        "student_t_95ci_percent": interval,
        "equivalent_within_plus_minus_3_percent": bool(
            interval and interval[0] >= -3.0 and interval[1] <= 3.0),
    }


def summarize_reference(samples, block_size, reference):
    output = {"coverage": {}, "categories": {}}
    records = [sample["block_sizes"][block_size]["references"][reference]
               for sample in samples]
    for budget in ("5", "10", "20"):
        output["coverage"][budget] = {
            "energy_fraction": distribution(
                row["coverage"][budget]["energy_fraction"] for row in records),
            "logical_bytes": distribution(
                row["coverage"][budget]["logical_bytes"] for row in records),
            "physical_bytes_aligned_4k": distribution(
                row["coverage"][budget]["physical_bytes_aligned_4k"]
                for row in records),
        }
    categories = sorted({category for row in records
                         for category in row["categories"]})
    for category in categories:
        category_rows = [row["categories"][category] for row in records
                         if category in row["categories"]]
        output["categories"][category] = {
            "l2": distribution(row["l2"] for row in category_rows),
            "changed_blocks": distribution(
                row["changed_blocks"] for row in category_rows),
            "coverage_20_energy_fraction": distribution(
                row["coverage"]["20"]["energy_fraction"]
                for row in category_rows),
        }
    output["changed_tensor_fraction"] = distribution(
        row["tensor_changes"]["changed_tensor_fraction"] for row in records)
    output["changed_byte_fraction"] = distribution(
        row["tensor_changes"]["changed_byte_fraction"] for row in records)
    return output


def collect_inc3(root):
    base = root / "INC3_tensor_change_coverage"
    runs = []
    expected_steps = list(range(1, 21)) + list(range(51, 71)) + list(range(101, 121))
    for seed in (41, 42, 43):
        candidates = sorted(base.glob(
            f"XL_seed{seed}_formal_final_v2/*/result.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"seed {seed}: expected one final result, got {len(candidates)}")
        path = candidates[0]
        result = read_json(path)
        run_dir = path.parent
        samples = read_jsonl(run_dir / "samples.jsonl")
        events = [{"run_id": sample["run_id"],
                   "request_id": sample["request_id"],
                   "events": sample["events"]} for sample in samples]
        ensure_non_restore_artifacts(run_dir, events)
        steps = [sample["step"] for sample in samples]
        nonfinite = sum(sample["numeric_health"]["nonfinite_arrays"]
                        for sample in samples)
        if (result.get("status") != "pass" or len(samples) != 60 or
                steps != expected_steps or nonfinite):
            raise RuntimeError(f"seed {seed}: INC3 acceptance gate failed")
        block_summary = {}
        for block_size in ("65536", "262144"):
            block_summary[block_size] = {
                "adjacent": summarize_reference(samples, block_size, "adjacent"),
                "persisted": summarize_reference(samples, block_size, "persisted"),
                "selected_jaccard": distribution(
                    sample["block_sizes"][block_size]["selected_jaccard"]
                    for sample in samples[1:]),
                "max_block_age": max(
                    sample["block_sizes"][block_size]["age"]["max"]
                    for sample in samples),
                "max_overdue_blocks": max(
                    sample["block_sizes"][block_size]["age"]["overdue_count"]
                    for sample in samples),
                "max_permanently_unselected_blocks": max(
                    sample["block_sizes"][block_size]["age"]["permanently_unselected_count"]
                    for sample in samples),
            }
        runs.append({
            "run_dir": str(run_dir),
            "status": result["status"],
            "seed": seed,
            "steps": steps,
            "samples": len(samples),
            "state_bytes": samples[0]["references"]["adjacent"]["tensor_changes"]["total_bytes"],
            "numeric_nonfinite_arrays": nonfinite,
            "scoring_seconds": distribution(
                sample["timeline_us"]["scoring"] / 1e6 for sample in samples),
            "blocks": block_summary,
        })
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "results/incremental-observation-20260904"))
    args = parser.parse_args()
    summary = {
        "schema": "incremental-observation-minimal-v1",
        "inc1": collect_inc1(args.root),
        "inc2": collect_inc2(args.root),
        "inc3": collect_inc3(args.root),
    }
    path = args.root / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({"status": "pass", "summary": str(path),
                      "inc1_runs": len(summary["inc1"]),
                      "inc2_runs": len(summary["inc2"]["runs"]),
                      "inc3_runs": len(summary["inc3"])}, indent=2))


if __name__ == "__main__":
    main()
