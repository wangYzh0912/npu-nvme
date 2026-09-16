#!/usr/bin/env python3
"""Rebuild the E8 summary from per-run result.json evidence.

The XL PMU matrix was resumed after a disk-full failure, so the second
invocation replaced the top-level summary with only its resumed seeds.  This
utility deliberately treats the per-run result files as the source of truth
and excludes empty/failed result files from the successful matrix.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    results = []
    for path in sorted(args.output_dir.glob("E8_gpt2_xl_seed*_*/result.json")):
        if path.stat().st_size == 0:
            continue
        item = json.loads(path.read_text())
        if item.get("status") == "pass":
            results.append(item)
    if len(results) != 6:
        raise SystemExit(f"expected 6 successful XL runs, found {len(results)}")
    (args.output_dir / "E8_real_summary.json").write_text(
        json.dumps({"results": results}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output_dir / "E8_real_summary.json"),
                      "results": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
