#!/usr/bin/env python3
"""Run the resumable long-duration FULL validation campaign for stages 2-4."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from experiments.benchmarks.longrun_utils import (atomic_json, checked_stdout,
                                                   open_campaign, update_entry)
from ppt_evidence import command, environment_snapshot


RUNNERS = {
    "stage2": ROOT / "experiments" / "benchmarks" / "s2_async_data_plane.py",
    "stage3": ROOT / "experiments" / "benchmarks" / "s3_training_io_matrix.py",
    "stage4": ROOT / "experiments" / "benchmarks" / "s4_control_matrix.py",
}


def stage_command(stage, args, output):
    common = ["--npu", str(args.npu), "--pci", args.pci,
              "--output-root", str(output), "--fail-fast"]
    if args.resume:
        common.append("--resume")
    if stage == "stage2":
        return [sys.executable, str(RUNNERS[stage]), *common,
                "--shm-id", str(args.shm_id), "--timeout", str(args.timeout)]
    if stage == "stage3":
        return [sys.executable, str(RUNNERS[stage]), *common,
                "--model", "gpt2_xl", "--modes", "none", "serial", "queue",
                "async", "--seeds", "41", "42", "43", "--intervals", "10",
                "20", "50", "--chunks", str(4 * 1024**2), "--depths", "4",
                "--total-steps", "110", "--shm-id", str(args.shm_id + 5000),
                "--timeout", str(args.timeout)]
    return [sys.executable, str(RUNNERS[stage]), *common,
            "--dma-slots", "1", "2", "4", "--checkpoint-slots", "1", "2", "4",
            "--delays-ms", "0", "100", "1000", "5000", "--intervals", "1", "5",
            "10", "--shm-id", str(args.shm_id + 10000),
            "--timeout", str(args.timeout)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", choices=tuple(RUNNERS),
                        default=list(RUNNERS))
    parser.add_argument("--output-root")
    parser.add_argument("--index-path")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=30000)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        root = Path(args.output_root or "/tmp/npu-nvme-longrun-DRYRUN")
        for stage in args.stages:
            print(json.dumps({"stage": stage,
                              "command": stage_command(stage, args,
                                                         root / stage)},
                             sort_keys=True))
        return 0

    commit_id = checked_stdout(
        command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]), "git commit")
    status_result = command(["git", "-C", str(ROOT), "status", "--porcelain"])
    if status_result.get("returncode") != 0:
        parser.error(f"git status failed: {status_result.get('stderr', '').strip()}")
    dirty = status_result.get("stdout", "").strip()
    if dirty and not args.allow_dirty:
        parser.error("formal validation requires a clean worktree")
    root = Path(args.output_root or
                f"/tmp/npu-nvme-longrun-{commit_id[:12]}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {"stages": args.stages, "npu": args.npu, "pci": args.pci,
              "shm_id": args.shm_id, "timeout": args.timeout}
    campaign_path = root / "campaign.json"
    campaign = open_campaign(campaign_path, commit_id, config, resume=args.resume)
    atomic_json(root / "environment.json", environment_snapshot(
        pci=args.pci, npu=str(args.npu), repo_root=ROOT,
        npu_info=command(["npu-smi", "info"])))
    atomic_json(root / "commit.json", {
        "repo": commit_id,
        "branch": command(["git", "-C", str(ROOT), "branch", "--show-current"]),
        "status": dirty,
        "spdk": command(["git", "-C", str(ROOT / "third_party" / "spdk"),
                         "rev-parse", "HEAD"]),
    })

    stage_results = {}
    for stage in args.stages:
        stage_root = root / stage
        summary_path = stage_root / "summary.json"
        entry = campaign.get("entries", {}).get(stage, {})
        if entry.get("status") == "pass" and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "pass":
                stage_results[stage] = summary
                continue
        command_line = stage_command(stage, args, stage_root)
        update_entry(campaign_path, campaign, stage, "running",
                     output=str(stage_root), command=command_line)
        log_path = root / f"{stage}.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.run(command_line, stdout=log, stderr=subprocess.STDOUT)
        summary = (json.loads(summary_path.read_text(encoding="utf-8"))
                   if summary_path.exists() else {})
        if process.returncode != 0 or summary.get("status") != "pass":
            update_entry(campaign_path, campaign, stage, "fail",
                         returncode=process.returncode, summary=str(summary_path))
            atomic_json(root / "index.json", {
                "status": "fail", "commit": commit_id,
                "failed_stage": stage, "stages": stage_results,
            })
            return process.returncode or 1
        stage_results[stage] = summary
        update_entry(campaign_path, campaign, stage, "pass", returncode=0,
                     summary=str(summary_path))

    index = {"schema_version": 1, "status": "pass", "commit": commit_id,
             "output_root": str(root), "stages": {
                 name: {"status": result.get("status"),
                        "summary": str(root / name / "summary.json")}
                 for name, result in stage_results.items()}}
    atomic_json(root / "index.json", index)
    if args.index_path:
        atomic_json(Path(args.index_path), index)
    print(json.dumps(index, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
