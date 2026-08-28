#!/usr/bin/env python3
"""Create a conservative, chart-ready summary of E3 evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E3 = ROOT / "results/ppt-evidence-20260829/E3"
OUT = E3 / "summary.json"


def main():
    rows = []
    for path in sorted(E3.glob("E3_*/result.json")):
        result = json.loads(path.read_text())
        config = json.loads((path.parent / "config.json").read_text())
        rows.append({
            "run_id": result.get("run_id"),
            "status": result.get("status"),
            "measurement_kind": config.get("measurement_kind"),
            "model": result.get("model"),
            "slot_count": result.get("slot_count"),
            "samples": result.get("samples", 0),
            "state_bytes": result.get("state_bytes"),
            "hbm_slot_bytes": result.get("hbm_slot_bytes",
                                           result.get("state_bytes")),
            "hbm_peak": result.get("hbm_peak"),
            "host_rss_peak": result.get("host_rss_peak"),
            "pinned_dram_peak": result.get("pinned_dram_peak"),
            "latency_mean_ms": ((result.get("latency_mean") or {}).get("mean")
                                 if isinstance(result.get("latency_mean"), dict)
                                 else None),
            "foreground_wait_mean_ms": ((result.get("foreground_wait") or {}).get("mean")
                                         if isinstance(result.get("foreground_wait"), dict)
                                         else None),
            "source_a9_run": result.get("source_a9_run"),
        })

    groups = defaultdict(list)
    for row in rows:
        if row["measurement_kind"] and row["measurement_kind"].startswith("historical"):
            delay = "slow_5s" if "5s" in row["measurement_kind"] else "normal"
            groups[delay].append(row)
    formal = [row for row in rows if row["status"] == "pass"]
    historical = [row for row in rows if row["status"] == "historical_pass_insufficient_samples"]
    summary = {
        "experiment": "E3",
        "status": "partial",
        "scope": "real GPT-2 XL HBM snapshot slots through single-owner SPDK on 0000:83:00.0",
        "device_policy": "83.0.0 raw only; 84.0.0 and /models untouched",
        "matrix": {
            "historical_runs": len(historical),
            "failed_attempts": sum(1 for row in rows if row["status"] == "fail"),
            "normal_slot_counts": sorted({row["slot_count"] for row in groups["normal"]}),
            "slow_slot_counts": sorted({row["slot_count"] for row in groups["slow_5s"]}),
            "formal_samples_per_historical_run": sorted({row["samples"] for row in historical}),
        },
        "evidence": {
            "hbm_slot_capacity_observed_bytes": 3280687104,
            "hbm_slot_capacity_interpretation": "one immutable HBM snapshot slot; run-level allocation, not full HBM peak",
            "host_staging_rss": "not measured in this partial batch",
            "pinned_dram": "not measured; VmPin was unavailable in the historical runs",
            "readback_correctness": "all six historical A9 configurations matched frozen HBM SHA-256 on 83.0.0",
            "slow_disk_wait": "historical 5s-delay data shows 1-slot wait 7038.4 ms, 2-slot 3114.4 ms, 4-slot 0.051 ms",
        },
        "gate": {
            "30_formal_samples": False,
            "host_staging_vs_slot_memory": False,
            "gpt2_13b": False,
            "interpretation": "E3 remains partial; do not claim bounded host memory or 13B scaling until new measurements complete",
        },
        "runs": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "status": summary["status"],
                      "historical": len(historical),
                      "failed_attempts": summary["matrix"]["failed_attempts"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
