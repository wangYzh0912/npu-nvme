#!/usr/bin/env python3
"""Aggregate formal S2 trajectories and exact policy replays.

The decision is deliberately conservative: a candidate qualifies only when
every supplied seed satisfies write, age, and recovery-loss gates.  Category
errors are reported separately so a large model-weight denominator cannot
hide optimizer-state drift.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def trajectory_summary(run_dirs, expected_steps=100):
    runs = []
    aggregate = defaultdict(lambda: defaultdict(list))
    categories = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for run_dir in map(Path, run_dirs):
        config = load_json(run_dir / "config.json")
        samples = load_jsonl(run_dir / "samples.jsonl")
        steps = [int(row["step"]) for row in samples]
        complete = (len(samples) == expected_steps and
                    steps == list(range(1, expected_steps + 1)))
        run = {"run_dir": str(run_dir), "seed": config.get("seed"),
               "samples": len(samples), "complete": complete,
               "block_sizes": {}}
        for block_size in map(str, config["block_sizes"]):
            rows = [sample["block_sizes"][block_size] for sample in samples]
            block = {
                "coverage": {},
                "selected_jaccard": distribution(
                    [row["selected_jaccard"] for row in rows]),
                "max_age": max((row["age"]["max"] for row in rows),
                               default=0),
                "categories": {},
            }
            for fraction in ("1", "5", "10", "20", "50"):
                values = [row["coverage"][fraction]["energy_fraction"]
                          for row in rows]
                block["coverage"][fraction] = distribution(values)
                aggregate[block_size][f"coverage_{fraction}"].extend(values)
            aggregate[block_size]["jaccard"].extend(
                row["selected_jaccard"] for row in rows)
            for category in sorted({name for row in rows
                                    for name in row["categories"]}):
                category_rows = [row["categories"][category] for row in rows
                                 if category in row["categories"]]
                total_energy = [max(row["adjacent_l2"] ** 2, 1e-30)
                                for row in rows if category in row["categories"]]
                energy_share = [record["l2"] ** 2 / total
                                for record, total in zip(category_rows,
                                                         total_energy)]
                record = {
                    "energy_share": distribution(energy_share),
                    "relative_l2_median": distribution(
                        [item["relative_l2_median"]
                         for item in category_rows]),
                    "coverage": {},
                }
                categories[block_size][category]["energy_share"].extend(
                    energy_share)
                for fraction in ("1", "5", "10", "20"):
                    values = [item["coverage"][fraction]["energy_fraction"]
                              for item in category_rows]
                    record["coverage"][fraction] = distribution(values)
                    categories[block_size][category][
                        f"coverage_{fraction}"].extend(values)
                block["categories"][category] = record
            run["block_sizes"][block_size] = block
        runs.append(run)
    overall = {}
    for block_size, metrics in sorted(aggregate.items(), key=lambda item: int(item[0])):
        overall[block_size] = {
            key: distribution(values) for key, values in metrics.items()}
        overall[block_size]["categories"] = {
            category: {key: distribution(values)
                       for key, values in metrics_by_name.items()}
            for category, metrics_by_name in categories[block_size].items()}
    return {"expected_steps": expected_steps, "complete": bool(runs) and all(
        run["complete"] for run in runs), "runs": runs, "overall": overall}


def _candidate_key(row):
    return json.dumps(row["config"], sort_keys=True, separators=(",", ":"))


def policy_summary(result_paths, expected_seeds=3):
    grouped = defaultdict(list)
    for path in map(Path, result_paths):
        result = load_json(path)
        if result.get("status") != "PASS":
            raise ValueError(f"policy result did not pass: {path}")
        for row in result["rows"]:
            grouped[_candidate_key(row)].append((result["seed"], row))
    candidates = []
    for key, seed_rows in grouped.items():
        config = json.loads(key)
        rows = [row for _seed, row in seed_rows]
        seeds = sorted({int(seed) for seed, _row in seed_rows})
        step_errors = [value for row in rows for value in row["errors"]]
        category_final = defaultdict(list)
        category_max = defaultdict(list)
        for row in rows:
            for category, value in row[
                    "final_category_relative_l2_error"].items():
                category_final[category].append(value)
            for category, value in row.get(
                    "max_category_relative_l2_error", {}).items():
                category_max[category].append(value)
        max_age = max((row["max_block_age"] for row in rows), default=0)
        age_ok = bool(config.get("max_age")) and max_age < config["max_age"]
        record = {
            "candidate_id": rows[0]["candidate_id"], "config": config,
            "seeds": seeds, "seed_count": len(seeds),
            "write_ratio": distribution([row["write_ratio"] for row in rows]),
            "step_nrmse": distribution(step_errors),
            "final_nrmse": distribution(
                [row["final_relative_l2_error"] for row in rows]),
            "recovery_loss_relative_error": distribution(
                [row["recovery_loss_relative_error"] for row in rows]),
            "max_block_age": max_age,
            "final_category_nrmse": {
                category: distribution(values)
                for category, values in category_final.items()},
            "max_category_nrmse": {
                category: distribution(values)
                for category, values in category_max.items()},
        }
        record["eligible_go"] = (
            len(seeds) >= expected_seeds and
            record["write_ratio"]["max"] < 0.20 and
            record["step_nrmse"]["median"] <= 1e-2 and
            record["recovery_loss_relative_error"]["max"] <= 0.01 and
            age_ok)
        candidates.append(record)
    candidates.sort(key=lambda row: (row["write_ratio"].get("median", 1.0),
                                     row["step_nrmse"].get("median", 1.0)))
    eligible = [row for row in candidates if row["eligible_go"]]
    finalists = {}
    if eligible:
        finalists["lowest_write"] = min(
            eligible, key=lambda row: row["write_ratio"]["median"])[
                "candidate_id"]
        finalists["lowest_error"] = min(
            eligible, key=lambda row: row["step_nrmse"]["median"])[
                "candidate_id"]
        write_min = min(row["write_ratio"]["median"] for row in eligible)
        error_min = min(row["step_nrmse"]["median"] for row in eligible)
        finalists["pareto"] = min(
            eligible, key=lambda row:
            row["write_ratio"]["median"] / max(write_min, 1e-30) +
            row["step_nrmse"]["median"] / max(error_min, 1e-30))[
                "candidate_id"]
    return {"expected_seeds": expected_seeds,
            "complete": bool(candidates) and all(
                row["seed_count"] >= expected_seeds for row in candidates),
            "candidates": candidates, "eligible_count": len(eligible),
            "finalists": finalists}


def make_decision(trajectories, policies):
    complete = trajectories["complete"] and policies.get("complete", False)
    eligible = policies["eligible_count"]
    if not complete:
        status = "INCOMPLETE"
    elif eligible:
        status = "GO_R2"
    else:
        status = "PIVOT"
    return {"status": status,
            "reason": ("formal trajectory or policy set is incomplete" if not complete
                       else "at least one exact R2 candidate passed all gates"
                       if eligible else
                       "no exact R2 candidate passed all first-round gates")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", action="append", required=True,
                        help="completed I1_REAL run directory; repeat per seed")
    parser.add_argument("--policy", action="append", required=True,
                        help="exact policy result.json; repeat per seed")
    parser.add_argument("--expected-steps", type=int, default=100)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trajectories = trajectory_summary(args.trajectory, args.expected_steps)
    policies = policy_summary(args.policy, args.expected_seeds)
    result = {"status": "PASS", "experiment": "S2_INCREMENTAL_DECISION",
              "trajectory": trajectories, "policy": policies,
              "decision": make_decision(trajectories, policies)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
