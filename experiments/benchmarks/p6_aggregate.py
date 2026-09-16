#!/usr/bin/env python3
"""Collect compact Vector/Cube/HBM summaries from completed P6/E8 runs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for result_path in sorted(args.source.rglob("result.json")):
        result = json.loads(result_path.read_text())
        if (result.get("model") not in ("gpt2", "gpt2_xl") or
                result.get("seed") not in (41, 42, 43) or
                result.get("status") != "pass"):
            continue
        timeline_path = result_path.parent / "p6_timeline.json"
        timeline = json.loads(timeline_path.read_text()) if timeline_path.exists() else {}
        summary = timeline.get("summary", {})
        records.append({
            "source_run": str(result_path.parent),
            "model": result.get("model"),
            "seed": result["seed"],
            "metric_group": result.get("metric_group"),
            "step_stats_ms": result.get("step_stats_ms"),
            "pmu_by_core": result.get("pmu_by_core"),
            "hbm": result.get("hbm"),
            "timeline_summary": summary,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({record.get("model") for record in records if record.get("model")})
    args.output.write_text(json.dumps({
        "experiment": "P6",
        "model": models[0] if len(models) == 1 else None,
        "models": models,
        "seeds": sorted({record["seed"] for record in records}),
        "source": str(args.source),
        "records": records,
        "note": "completed real msprof CSV exports; compact timeline summaries only",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
