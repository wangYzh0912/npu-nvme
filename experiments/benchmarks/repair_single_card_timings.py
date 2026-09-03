#!/usr/bin/env python3
"""Rebuild single-card latency fields from persisted state-machine events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from experiments.benchmarks.longrun_utils import atomic_json  # noqa: E402
from experiments.benchmarks.run_single_card_full import request_timing  # noqa: E402


def repair_run(run_dir: Path) -> bool:
    source_path = run_dir / "source.json"
    result_path = run_dir / "result.json"
    if not source_path.exists() or not result_path.exists():
        return False
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = source.get("checkpoints", [])
    if not records:
        return False
    for record in records:
        record.update(request_timing(record["request"]))
    atomic_json(source_path, source)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["checkpoint_latency_seconds"] = [
        record["persist_seconds"] for record in records
    ]
    result["checkpoint_state_machine_seconds"] = [
        record["state_machine_seconds"] for record in records
    ]
    result["latency_semantics"] = {
        "checkpoint_latency_seconds": "api_enter_to_persisted_event",
        "checkpoint_state_machine_seconds": "created_to_persisted_event",
        "foreground_wait_seconds": "host_wait_inside_finish_one",
    }
    atomic_json(result_path, result)

    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps({
                "run_id": run_dir.name,
                "status": "pass",
                "step": record["step"],
                "request_id": record["request"]["request_id"],
                "generation": record["request"]["metadata_generation"],
                "persist_seconds": record["persist_seconds"],
                "state_machine_seconds": record["state_machine_seconds"],
                "foreground_wait_seconds": record.get(
                    "foreground_wait_seconds", 0.0),
                "checksum": record["request"]["checksum"],
                "runtime_stats": record.get("runtime_stats", {}),
            }, sort_keys=True) + "\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repaired = []
    for source_path in sorted(args.root.rglob("source.json")):
        if repair_run(source_path.parent):
            repaired.append(str(source_path.parent))
    print(json.dumps({"status": "pass", "repaired": len(repaired),
                      "runs": repaired}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
