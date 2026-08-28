#!/usr/bin/env python3
"""Backfill percentile fields from retained E1 fio JSON without rerunning I/O."""

import argparse
import json
from pathlib import Path

from e1_async_fs import native_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("results/ppt-evidence-20260829/E1"))
    args = parser.parse_args()
    updated = 0
    for result_path in args.root.glob("*/result.json"):
        raw_path = result_path.parent / "raw/fio_formal.json"
        if not raw_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = json.loads((result_path.parent / "config.json").read_text(
            encoding="utf-8"))
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        job = (data.get("jobs") or [{}])[0]
        section = job.get(config["operation"], {})
        latency = native_stats(section)
        result["latency_mean"] = latency.get("mean_ns")
        result["latency_p50"] = latency.get("p50_ns")
        result["latency_p95"] = latency.get("p95_ns")
        result["native_fio_latency"] = latency
        result["percentile_note"] = latency.get("percentile_source")
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        updated += 1
    print(json.dumps({"updated": updated}, sort_keys=True))


if __name__ == "__main__":
    main()
