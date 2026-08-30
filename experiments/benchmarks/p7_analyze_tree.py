#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); args=p.parse_args(); runs=sorted({path.parent for path in args.root.rglob("samples.jsonl")})
    if len(runs)<3: raise SystemExit(f"expected >=3 P7 runs, found {len(runs)}")
    cmd=[sys.executable,str(ROOT/"experiments/benchmarks/p7_change_summary.py"),"--inputs",*[str(x) for x in runs],"--output",str(args.root/"summary.json")]; raise SystemExit(subprocess.run(cmd,cwd=ROOT,check=False).returncode)
if __name__=="__main__": main()
