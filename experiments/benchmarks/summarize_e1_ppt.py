#!/usr/bin/env python3
"""Build a compact, chart-ready E1 summary from completed evidence bundles."""

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ppt_evidence import stats  # noqa: E402


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=ROOT / "results/ppt-evidence-20260829/E1")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.root / "summary.json"
    rows = []
    ignored = []
    for result_path in sorted(args.root.glob("*/result.json")):
        run_dir = result_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            ignored.append({"path": str(result_path), "reason": "missing config"})
            continue
        result = load(result_path)
        config = load(config_path)
        if result.get("status") != "pass" or result.get("samples") != 30:
            ignored.append({"path": str(result_path), "reason": "not 30 formal PASS"})
            continue
        if (run_dir / "raw/fio_formal.json").exists():
            raw = load(run_dir / "raw/fio_formal.json")
            job = (raw.get("jobs") or [{}])[0]
            section = job.get(config["operation"], {})
            rows.append({
                "path": str(result_path), "backend": config["mode"],
                "operation": config["operation"], "size_bytes": config["size_bytes"],
                "queue_depth": config["queue_depth"], "samples": 30,
                "latency_ns": result.get("native_fio_latency"),
                "io_bytes": section.get("io_bytes"),
                "persistence": config["persistence"],
                "pci": config["filesystem_pci"],
                "percentile_scope": result.get("percentile_note"),
            })
        elif config.get("mode") == "spdk_async_qpair":
            rows.append({
                "path": str(result_path), "backend": "spdk_async_qpair",
                "operation": "write_read", "size_bytes": config["state_bytes"],
                "queue_depth": config["pipeline_depth"], "samples": 30,
                "end_to_end_us": result.get("end_to_end_us"),
                "write_api_us": result.get("write_api_us"),
                "read_api_us": result.get("read_api_us"),
                "persistence": config["persistence"], "pci": config["target_pci"],
                "submission": "async qpair; blocking durable wait",
            })
        else:
            ignored.append({"path": str(result_path), "reason": "unknown backend"})

    groups = {}
    for row in rows:
        key = (row["backend"], row["operation"], row["size_bytes"],
               row["queue_depth"])
        group = groups.setdefault("|".join(map(str, key)), {
            "backend": row["backend"], "operation": row["operation"],
            "size_bytes": row["size_bytes"], "queue_depth": row["queue_depth"],
            "runs": [],
        })
        group["runs"].append(row)
    for group in groups.values():
        group["run_count"] = len(group["runs"])
        if group["backend"] == "spdk_async_qpair":
            group["end_to_end_us"] = group["runs"][0]["end_to_end_us"]
            group["write_api_us"] = group["runs"][0]["write_api_us"]
            group["read_api_us"] = group["runs"][0]["read_api_us"]
        else:
            group["latency_ns"] = group["runs"][0]["latency_ns"]
        group.pop("runs")

    summary = {
        "experiment": "E1", "status": "pass" if len(rows) == 50 else "incomplete",
        "protocol": {
            "filesystem_pci": "0000:84:00.0", "spdk_pci": "0000:83:00.0",
            "same_disk": False, "cross_disk_calibration_required": True,
            "filesystem": "XFS + io_uring + fsync",
            "spdk": "single-owner asynchronous qpair + flush/metadata commit",
            "formal_samples_per_configuration": 30,
            "warmups_per_configuration": 10,
        },
        "counts": {"filesystem": sum(r["backend"] in ("buffered", "odirect") for r in rows),
                   "spdk": sum(r["backend"] == "spdk_async_qpair" for r in rows),
                   "selected_runs": len(rows), "ignored_runs": len(ignored)},
        "groups": list(groups.values()),
        "ignored": {"count": len(ignored),
                     "reasons": sorted({item["reason"] for item in ignored})},
        "interpretation": [
            "FS and SPDK are cross-disk path measurements, not strict same-device comparisons.",
            "FS p50/p95 use fio completion latency when fio exposes no persistence percentile; mean/stdev are fio total latency.",
            "SPDK entries include both write and read in each formal sample and wait for durable completion.",
        ],
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "selected": len(rows),
                      "ignored": len(ignored), "status": summary["status"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
