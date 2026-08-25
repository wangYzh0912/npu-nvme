#!/usr/bin/env python3
"""Collect already executed path results into explicit plan manifests."""

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


SOURCES = {
    "A2": {
        "question": "HBM->Host->NVMe versus HBM->DMA->NVMe",
        "paths": [
            "experiments/output/e2_p2/E2_20260825_152851_4c30e23c/result.json",
            "experiments/output/e2_p3/E2_20260825_153145_aefbee7c/result.json",
            "experiments/output/e2_p4/E2_20260825_153412_f079efbd/result.json",
        ],
    },
    "A7": {
        "question": "training-only control versus safe asynchronous snapshot trigger",
        "paths": [
            "experiments/output/e2_p0/E2_20260825_155113_e283134c/result.json",
            "experiments/output/e2_p5_v2/E2_20260825_154703_5ee03404/result.json",
        ],
    },
    "E4": {
        "question": "object granularity, chunk size, pipeline depth, and synthetic scale",
        "paths": [
            "experiments/output/e1_retry/*/result.json",
            "experiments/output/e1_4k/*/result.json",
            "experiments/output/e1_memory/*/result.json",
            "experiments/output/a3/A3_*/result.json",
            "experiments/output/a4/A4_*/result.json",
            "experiments/output/a5/A5_*/result.json",
        ],
    },
    "E5": {
        "question": "P1/P2/P4 first-parameter and full-model restore",
        "paths": [
            "experiments/output/e2_p1/E2_20260825_153935_d3ea7199/result.json",
            "experiments/output/e2_p2/E2_20260825_152851_4c30e23c/result.json",
            "experiments/output/e2_p4/E2_20260825_153412_f079efbd/result.json",
        ],
    },
}


def expand(pattern):
    paths = sorted(REPO_ROOT.glob(pattern))
    return [path for path in paths if path.name == "result.json"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=sorted(SOURCES))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    source_paths = []
    for pattern in SOURCES[args.experiment]["paths"]:
        if "*" in pattern:
            source_paths.extend(expand(pattern))
        else:
            source_paths.append(REPO_ROOT / pattern)
    if not source_paths:
        raise RuntimeError("no source result files")
    sources = []
    statuses = []
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        statuses.append(data.get("status"))
        sources.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "status": data.get("status"),
            "config": data.get("config", {}),
            "samples": data.get("samples", 0),
            "failed_samples": data.get("failed_samples", 0),
            "summary": data.get("summary", {}),
        })
    output_root = Path(args.output_root or REPO_ROOT / "experiments" / "output" /
                       args.experiment.lower())
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": args.experiment,
        "status": "pass" if all(status == "pass" for status in statuses) else "fail",
        "question": SOURCES[args.experiment]["question"],
        "source_count": len(sources),
        "source_results_are_read_only": True,
        "sources": sources,
    }
    path = output_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"path": str(path), "status": manifest["status"],
                      "source_count": len(sources)}, indent=2), flush=True)
    if manifest["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
