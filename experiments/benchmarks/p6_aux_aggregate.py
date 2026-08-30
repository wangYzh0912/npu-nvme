#!/usr/bin/env python3
"""Aggregate isolated P6 auxiliary-task injections without inventing values."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for result_path in sorted(args.root.rglob("result.json")):
        result = json.loads(result_path.read_text())
        config_path = result_path.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        records.append({
            "run_id": result.get("run_id"),
            "mode": config.get("mode"),
            "auxiliary": config.get("auxiliary"),
            "model": config.get("model"),
            "seed": config.get("seed"),
            "status": result.get("status"),
            "samples": result.get("samples"),
            "latency_mean_ms": (result.get("latency_mean") or {}).get("mean"),
            "latency_p95_ms": (result.get("latency_mean") or {}).get("p95"),
            "foreground_wait_mean_ms": (result.get("foreground_wait") or {}).get("mean"),
            "failure_count": result.get("failed_samples"),
            "source": str(result_path.parent),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "experiment": "P6_auxiliary_injection",
        "records": records,
        "note": "isolated process per mode/task; one failed TopK kernel is retained as evidence",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
