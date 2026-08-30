#!/usr/bin/env python3
"""Aggregate the latest successful P3 run for every tested configuration."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = {}
    for path in args.root.rglob("result.json"):
        result = json.loads(path.read_text())
        config_path = path.with_name("config.json")
        config = (json.loads(config_path.read_text())
                  if config_path.exists() else {})
        for field in ("seed", "chunk_size", "pipeline_depth", "delay_ms"):
            if result.get(field) is None and config.get(field) is not None:
                result[field] = config[field]
        mode = result.get("mode")
        if (result.get("model") not in ("gpt2", "gpt2_xl") or
                mode not in ("serial", "queue", "async") or
                result.get("status") != "pass"):
            continue
        key = (result.get("seed"), result.get("chunk_size"),
               result.get("pipeline_depth"), result.get("delay_ms"), mode)
        run_id = str(result.get("run_id", path.parent.name))
        previous = selected.get(key)
        if previous is None or run_id > previous[0]:
            selected[key] = (run_id, result)

    groups = {}
    for key, (_, result) in selected.items():
        groups.setdefault(key[:4], {})[key[4]] = result

    output = []
    for key, modes in sorted(groups.items(), key=lambda item: str(item[0])):
        serial = modes.get("serial", {}).get("latency_mean")
        queue = modes.get("queue", {}).get("latency_mean")
        async_latency = modes.get("async", {}).get("latency_mean")
        output.append({
            "seed": key[0],
            "chunk_size": key[1],
            "depth": key[2],
            "delay_ms": key[3],
            "serial_ms": serial,
            "queue_ms": queue,
            "async_ms": async_latency,
            "queue_speedup_vs_serial": (serial / queue
                                         if serial and queue else None),
            "async_speedup": (serial / async_latency
                              if serial and async_latency else None),
            "async_speedup_vs_queue": (queue / async_latency
                                       if queue and async_latency else None),
            "overlap_rate": modes.get("async", {}).get("overlap_rate"),
            "run_ids": {mode: result.get("run_id")
                        for mode, result in modes.items()},
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"groups": output}, indent=2,
                                      sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
