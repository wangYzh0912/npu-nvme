#!/usr/bin/env python3
"""Discover P6 profiler exports and build one timeline summary per run."""
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); args=p.parse_args(); tasks=sorted(args.root.rglob("task_time_*.csv"))
    if not tasks: raise SystemExit("no task_time CSV found")
    failed=0
    for task in tasks:
        run=next((parent for parent in task.parents if parent.name.startswith("E8_")),task.parent); hbm=list(run.rglob("hbm_*.csv")); op=list(run.rglob("op_summary_*.csv")); child=run/"child_result.json"; cmd=[sys.executable,str(ROOT/"experiments/benchmarks/p6_vector_timeline.py"),"--task-time",str(task),"--output",str(run/"p6_timeline.json")]
        if op: cmd.extend(["--op-summary",*[str(x) for x in op]])
        if child.exists(): cmd.extend(["--steps-json",str(child)])
        if hbm: cmd.extend(["--hbm",*[str(x) for x in hbm]])
        failed+=subprocess.run(cmd,cwd=ROOT,check=False).returncode!=0
    if failed: raise SystemExit(1)
if __name__=="__main__": main()
