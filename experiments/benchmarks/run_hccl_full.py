#!/usr/bin/env python3
"""Unified HCCL launcher for the stage 6 FULL-training gates.

This wrapper owns launcher/environment reproducibility.  The child training
script is responsible for rank-local FULL snapshot and global commit hooks;
the wrapper never treats a launcher exit as a checkpoint pass by itself.
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="experiments/benchmarks/gpt2_13b_dist.py")
    parser.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", default="/tmp/hccl-full")
    parser.add_argument("--rank-table", default=None)
    parser.add_argument("--master-port", type=int, default=8118)
    parser.add_argument("--checkpoint-summary", default="checkpoint_gate.json",
                        help="JSON summary emitted by the child FULL checkpoint runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("child_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    script = str(Path(args.script).resolve())
    command = ["msrun", f"--worker_num={args.world_size}",
               f"--local_worker_num={args.world_size}",
               "--master_addr=127.0.0.1", f"--master_port={args.master_port}",
               sys.executable, script, "--steps", str(args.steps),
               "--output", str(output)] + list(args.child_args)
    env = os.environ.copy()
    if args.rank_table:
        env["RANK_TABLE_FILE"] = str(Path(args.rank_table).resolve())
    env.setdefault("HCCL_CONNECT_TIMEOUT", "120")
    snapshot = {
        "status": "planned" if args.dry_run else "running",
        "world_size": args.world_size, "script": script,
        "command": command, "command_shell": shlex.join(command),
        "rank_table": env.get("RANK_TABLE_FILE"),
        "created_unix": time.time(),
        "config_digest": hashlib.sha256(json.dumps(command, sort_keys=True).encode()).hexdigest(),
        "checkpoint_summary": str(output / args.checkpoint_summary),
    }
    (output / "launcher.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps(snapshot, sort_keys=True), flush=True)
    if args.dry_run:
        return
    completed = subprocess.run(command, env=env, cwd=str(Path(__file__).resolve().parents[2]),
                               check=False)
    checkpoint_path = output / args.checkpoint_summary
    checkpoint_ok = False
    checkpoint_error = None
    if completed.returncode == 0:
        try:
            summary = json.loads(checkpoint_path.read_text())
            ranks = summary.get("ranks", [])
            checkpoint_ok = (
                summary.get("status") == "pass" and len(ranks) == args.world_size and
                all(item.get("persisted") is True and
                    item.get("fresh_restore") is True and
                    item.get("continuation_verified") is True
                    for item in ranks))
            if not checkpoint_ok:
                checkpoint_error = "checkpoint gate fields are incomplete"
        except (OSError, ValueError) as error:
            checkpoint_error = f"missing or invalid checkpoint summary: {error}"
    snapshot.update({"status": "pass" if completed.returncode == 0 and checkpoint_ok else "fail",
                     "exit_code": completed.returncode, "finished_unix": time.time(),
                     "checkpoint_gate": checkpoint_ok,
                     "checkpoint_error": checkpoint_error})
    (output / "launcher.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
