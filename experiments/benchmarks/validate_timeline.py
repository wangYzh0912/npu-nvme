#!/usr/bin/env python3
"""Validate WP1 event timelines and reject non-monotonic measurements."""

import argparse
import json
from pathlib import Path


def validate_sample(sample):
    events = sample.get("events", [])
    timestamps = [event.get("monotonic_ns") for event in events
                  if event.get("monotonic_ns") is not None]
    errors = []
    if timestamps != sorted(timestamps):
        errors.append("event timestamps are not monotonic")
    if len(timestamps) != len(set(timestamps)):
        errors.append("event timestamps contain duplicates")
    for key, value in sample.get("timeline_us", {}).items():
        if isinstance(value, (int, float)) and value < 0:
            errors.append(f"negative duration: {key}")
    return errors


def validate(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if path.name == "result.json":
        samples_path = path.parent / data.get("paths", {}).get("samples", "samples.jsonl")
    else:
        samples_path = path
    if samples_path.exists() and samples_path.suffix == ".jsonl":
        for line_number, line in enumerate(samples_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            errors.extend(f"line {line_number}: {error}"
                         for error in validate_sample(json.loads(line)))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    errors = validate(args.path)
    print(json.dumps({"path": args.path, "status": "pass" if not errors else "fail",
                      "errors": errors}, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
