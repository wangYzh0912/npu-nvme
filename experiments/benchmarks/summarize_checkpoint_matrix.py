#!/usr/bin/env python3
"""Summarize the checkpoint trace and MindFormers model matrix."""

import argparse
import json
from pathlib import Path


def load_jsonl(path):
    text = path.read_text()
    decoder = json.JSONDecoder()
    rows = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        value, end = decoder.raw_decode(text, offset)
        rows.append(value)
        offset = end
    return rows


def model_result(root):
    result = {}
    for path in sorted(root.glob("model_*/*/result.json")):
        data = json.loads(path.read_text())
        name = path.parts[-3]
        result[name] = {"path": str(path), "status": data.get("status"),
                        "summary": data.get("summary", {})}
        samples = path.parent / "samples.jsonl"
        if samples.exists():
            rows = load_jsonl(samples)
            formal = [row for row in rows if not row.get("warmup")]
            if formal:
                result[name]["bytes"] = formal[0].get("bytes")
                result[name]["parameter_count"] = formal[0].get("hashes")
    return result


def trace_result(root, filename):
    rows = load_jsonl(root / filename)
    summary = next((row["summary"] for row in reversed(rows)
                    if "summary" in row), None)
    return {"summary": summary, "path": str(root / filename)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/output/wp1/current")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    report = {
        "models": model_result(root),
        "same_device": {
            "filesystem_buffered": trace_result(root, "same_device_fs_256m_v2.jsonl"),
            "filesystem_odirect": trace_result(root, "same_device_fs_odirect_256m_v2.jsonl"),
            "spdk_host": trace_result(root, "same_device_spdk_256m_v2.jsonl"),
        },
        "external_profile": str(root / "external_profile"),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
