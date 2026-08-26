#!/usr/bin/env python3
"""Aggregate WP1 result JSON without hand-copying measurements."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from validate_timeline import validate


def result_files(root):
    return sorted(Path(root).rglob("result.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/output/wp1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    timeline_errors = {}
    for path in result_files(root):
        data = json.loads(path.read_text(encoding="utf-8"))
        errors = validate(path)
        if errors:
            timeline_errors[str(path)] = errors
        rows.append({
            "path": str(path),
            "experiment": data.get("config", {}).get("experiment",
                         path.parent.parent.name),
            "model": data.get("config", {}).get("model"),
            "path_type": data.get("config", {}).get("path"),
            "status": data.get("status"),
            "samples": data.get("samples", 0),
            "failed_samples": data.get("failed_samples", 0),
            "summary": data.get("summary", {}),
        })
    passed = sum(row["status"] == "pass" for row in rows)
    summary = {
        "root": str(root),
        "result_count": len(rows),
        "pass_count": passed,
        "fail_count": len(rows) - passed,
        "timeline_error_count": len(timeline_errors),
        "timeline_errors": timeline_errors,
        "formal_results_are_pass_only": True,
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["path", "experiment", "model", "path_type", "status",
                  "samples", "failed_samples"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(json.dumps({"output": str(output), "csv": str(csv_path),
                      "results": len(rows), "passed": passed,
                      "timeline_errors": len(timeline_errors)}, indent=2))
    raise SystemExit(0 if not timeline_errors else 1)


if __name__ == "__main__":
    main()
