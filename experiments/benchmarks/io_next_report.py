#!/usr/bin/env python3
"""Validate IO-next evidence and emit a compact hash-indexed report."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    failures = []
    runs = []
    for result_path in sorted(root.rglob("result.json")):
        if result_path == root / "result.json":
            continue
        result = json.loads(result_path.read_text())
        run_dir = result_path.parent
        if result.get("status") not in ("pass", "unsupported", "planned"):
            failures.append({"run": str(run_dir.relative_to(root)),
                             "reason": "result status is not pass"})
        checkpoint_run = ("persisted" in result and
                          result.get("status") != "unsupported" and
                          result.get("mode") != "none")
        required = ["config.json", "environment.json", "result.json"]
        if checkpoint_run and result.get("mode") != "none":
            required.extend(("events.jsonl", "restore.json"))
            if not (result.get("persisted") is True and
                    result.get("restore_verified") is True and
                    result.get("loaded_state_byte_exact") is True and
                    result.get("loss_allclose") is True):
                failures.append({"run": str(run_dir.relative_to(root)),
                                 "reason": "FULL persistence/restore gate incomplete"})
        missing = [name for name in required if not (run_dir / name).exists()]
        if missing:
            failures.append({"run": str(run_dir.relative_to(root)),
                             "reason": "missing evidence", "files": missing})
        files = {}
        for name in required:
            path = run_dir / name
            if path.exists():
                files[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        runs.append({"run": str(run_dir.relative_to(root)),
                     "status": result.get("status"), "files": files})
    report = {"status": "pass" if runs and not failures else "fail",
              "root": str(root), "runs": runs, "failures": failures,
              "run_count": len(runs)}
    output = (args.output or root / "evidence_index.json").resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"status": report["status"], "runs": len(runs),
                      "failures": len(failures), "output": str(output)}, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
