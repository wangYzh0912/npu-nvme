#!/usr/bin/env python3
"""Run a bounded trend-only P1/P3/P4/P5/P6 experiment round.

The runner keeps raw profiler and per-sample files under ``/tmp`` and copies
only compact config/result/failure records into the requested evidence root.
It is intentionally not a formal gate: short samples and GPT-2 are used to
obtain directionality before spending time on a full matrix.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/user7/miniconda3/envs/ms_2.5/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def compact_copy(source: Path, destination: Path) -> int:
    copied = 0
    if not source.exists():
        return copied
    for path in source.rglob("*"):
        if not path.is_file() or path.name not in ("config.json", "result.json", "failures.jsonl"):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def run_command(label, command, raw_root, evidence_root, deadline, dry_run=False):
    raw_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    record = {"label": label, "command": [str(x) for x in command],
              "raw_root": str(raw_root), "started": time.strftime("%FT%T%z")}
    if dry_run:
        record.update({"status": "planned", "returncode": None, "copied": 0})
        return record
    remaining = max(1, int(deadline - time.monotonic()))
    log = raw_root / "quick_runner.stdout.log"
    started = time.monotonic()
    try:
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.run(command, cwd=ROOT, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False,
                                  timeout=remaining)
        record.update({"returncode": proc.returncode,
                       "status": "pass" if proc.returncode == 0 else "fail"})
    except subprocess.TimeoutExpired:
        record.update({"returncode": -9, "status": "timeout"})
    except BaseException as exc:
        record.update({"returncode": -1, "status": "error", "error": repr(exc)})
    record.update({"elapsed_s": round(time.monotonic() - started, 3),
                   "copied": compact_copy(raw_root, evidence_root),
                   "log": str(log), "ended": time.strftime("%FT%T%z")})
    return record


def aggregate(command, root, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command + ["--root", str(root), "--output", str(output)],
                          cwd=ROOT, check=False, text=True,
                          capture_output=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr, "output": str(output)}


def write_p5_summary(root: Path):
    rows = []
    for path in root.glob("P5_slots*/**/result.json"):
        result = json.loads(path.read_text())
        rows.append({"source": str(path), "status": result.get("status"),
                     "slots": result.get("slot_count"),
                     "chunk_size": result.get("chunk_size"),
                     "expected_pool_bytes": result.get("expected_pool_bytes"),
                     "baseline_rss": result.get("baseline_rss"),
                     "incremental_rss": result.get("incremental_rss"),
                     "host_rss_peak": result.get("host_rss_peak"),
                     "pinned_dram_peak": result.get("pinned_dram_peak"),
                     "slot_wait_ms": result.get("foreground_wait")})
    output = root / "P5" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"experiment": "P5_quick_trend",
                                  "records": sorted(rows, key=lambda x: str(x["source"]))},
                                 indent=2, sort_keys=True) + "\n")


def write_report(root: Path, records):
    def load(name):
        path = root / name
        return json.loads(path.read_text()) if path.exists() else {}

    p1 = load("P1/summary.json").get("groups", [])
    p3 = load("P3/summary.json").get("groups", [])
    p4 = load("P4/summary.json").get("rows", [])
    p5 = load("P5/summary.json").get("records", [])
    p6 = load("P6/summary.json").get("records", [])
    lines = ["# Quick Trend Round", "", "本轮为方向性探测，不替代正式验收矩阵。",
             "", f"运行状态：`{records and 'completed' or 'no-records'}`", "",
             "| 阶段 | 记录数 | 观察 |", "|---|---:|---|"]
    p1_modes = sum(len(x.get("paths", {})) for x in p1)
    lines.append(f"| P1 | {len(p1)} 组/{p1_modes} 路径 | 4 MiB 路径延迟与吞吐趋势 |")
    overlaps = [x.get("overlap_rate", {}).get("median") for x in p3
                if isinstance(x.get("overlap_rate"), dict)]
    lines.append(f"| P3 | {len(p3)} 组 | overlap median 范围 {min(overlaps):.4g}--{max(overlaps):.4g} |"
                 if overlaps else f"| P3 | {len(p3)} 组 | 无完整 overlap 结果 |")
    p4_pass = sum(x.get("acceptance_status") == "pass" for x in p4)
    lines.append(f"| P4 | {len(p4)} 模式 | step overhead gate pass {p4_pass}/{len(p4)} |")
    lines.append(f"| P5 | {len(p5)} 配置 | RSS/HugePage 随 slots×chunk 趋势 |")
    p6_ok = sum(x.get("status") == "pass" for x in p6)
    lines.append(f"| P6 | {len(p6)} 记录 | auxiliary pass {p6_ok}/{len(p6)}，需 profiler 确认并发 |")
    lines.extend(["", "## Run Records", "", "```json",
                  json.dumps(records, indent=2, ensure_ascii=False), "```", ""])
    (root / "QUICK_TREND_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/quick-trend-20260830")
    parser.add_argument("--raw-root", type=Path,
                        default=Path("/tmp/npu-nvme-quick-trend-20260830"))
    parser.add_argument("--deadline-minutes", type=int, default=105)
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "quick_state.json"
    state = json.loads(state_path.read_text()) if args.resume and state_path.exists() else {
        "experiment": "quick_trend_round", "status": "running", "records": []}
    deadline = time.monotonic() + args.deadline_minutes * 60
    py = str(PYTHON)
    records = state.get("records", [])

    def should_skip(label):
        return args.resume and any(x.get("label") == label and x.get("status") == "pass"
                                   for x in records)

    def execute(label, argv):
        nonlocal records
        if should_skip(label):
            return {"label": label, "status": "skipped_resume"}
        result = run_command(label, argv, args.raw_root / label,
                             args.output_root / label, deadline, args.dry_run)
        records.append(result)
        state.update({"records": records, "status": "running"})
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        return result

    p1_raw = args.raw_root / "P1"
    execute("P1", [py, str(ROOT / "experiments/benchmarks/p1_fair_io.py"),
                    "--path", "all", "--operations", "write", "read",
                    "--sizes", str(4 * 1024 * 1024), "--depths", "1", "4",
                    "--total-bytes", str(256 * 1024 * 1024), "--warmups", "2",
                    "--samples", "8", "--allow-fewer-samples",
                    "--npu", str(args.npu), "--pci", args.pci,
                    "--output-root", str(p1_raw),
                    "--fs-root", "/models/npu_nvme_exp/quick-trend-20260830"])
    if not args.dry_run:
        aggregate([py, str(ROOT / "experiments/benchmarks/p1_aggregate.py")],
                  args.output_root / "P1", args.output_root / "P1" / "summary.json")

    p3_raw = args.raw_root / "P3"
    execute("P3", [py, str(ROOT / "experiments/benchmarks/p3_async_pipeline.py"),
                    "--model", "gpt2", "--modes", "serial", "queue", "async",
                    "--chunks", str(4 * 1024 * 1024), "--depths", "1", "4",
                    "--delays", "0", "1000", "--seeds", "41",
                    "--warmups", "2", "--samples", "5",
                    "--allow-fewer-samples", "--npu", str(args.npu),
                    "--pci", args.pci, "--shm-id", "26000",
                    "--output-root", str(p3_raw)])
    if not args.dry_run:
        aggregate([py, str(ROOT / "experiments/benchmarks/p3_aggregate.py")],
                  args.output_root / "P3", args.output_root / "P3" / "summary.json")

    p4_raw = args.raw_root / "P4"
    execute("P4", [py, str(ROOT / "experiments/benchmarks/p4_training_e2e.py"),
                    "--model", "gpt2", "--modes", "none", "sync", "async",
                    "--intervals", "5", "--checkpoints", "2",
                    "--total-formal-steps", "10", "--seeds", "41",
                    "--warmup-steps", "2", "--chunk-size", str(4 * 1024 * 1024),
                    "--pipeline-depth", "4", "--npu", str(args.npu),
                    "--pci", args.pci, "--shm-id", "27000",
                    "--output-root", str(p4_raw)])
    if not args.dry_run:
        aggregate([py, str(ROOT / "experiments/benchmarks/p4_aggregate.py")],
                  args.output_root / "P4", args.output_root / "P4" / "summary.json")

    for slots in (1, 2, 4):
        for mib in (1, 4, 16):
            label = f"P5_slots{slots}_{mib}MiB"
            execute(label, [py, str(ROOT / "experiments/benchmarks/p5_dma_pool_memory.py"),
                            "--slots", str(slots), "--chunk-size", str(mib * 1024 * 1024),
                            "--total-bytes", str(256 * 1024 * 1024), "--warmups", "1",
                            "--samples", "3", "--npu", str(args.npu), "--pci", args.pci,
                            "--shm-id", str(28000 + slots * 100 + mib),
                            "--output-root", str(args.raw_root / label)])
    if not args.dry_run:
        write_p5_summary(args.output_root)

    p6_raw = args.raw_root / "P6"
    execute("P6", [py, str(ROOT / "experiments/benchmarks/p6_aux_injection.py"),
                    "--model", "gpt2", "--modes", "none", "npu_serial", "npu_parallel",
                    "--tasks", "diff", "--seeds", "41", "--warmups", "1",
                    "--steps", "5", "--npu", str(args.npu), "--pci", args.pci,
                    "--output-root", str(p6_raw)])
    if not args.dry_run:
        aggregate([py, str(ROOT / "experiments/benchmarks/p6_aux_aggregate.py")],
                  args.output_root / "P6", args.output_root / "P6" / "summary.json")
        subprocess.run([py, str(ROOT / "experiments/benchmarks/summarize_quick_trends.py"),
                        "--root", str(args.output_root)], cwd=ROOT, check=False)

    state.update({"status": "planned" if args.dry_run else "complete",
                  "deadline_minutes": args.deadline_minutes,
                  "records": records, "ended": time.strftime("%FT%T%z")})
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_root": str(args.output_root),
                      "raw_root": str(args.raw_root),
                      "records": len(records), "status": state["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
