#!/usr/bin/env python3
"""Reliability gate: owner pressure, faults, commit cost and crash visibility."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MS_PYTHON = Path("/home/user7/miniconda3/envs/ms_2.5/bin/python")
PYTHON = str(MS_PYTHON if MS_PYTHON.exists() else Path(sys.executable))

def execute(name,argv,root,timeout,dry):
    record={"name":name,"argv":argv,"started":time.strftime("%FT%T%z")}
    if dry: record.update({"status":"planned","returncode":None}); return record
    try:
        proc=subprocess.run(argv,cwd=ROOT,text=True,capture_output=True,check=False,timeout=timeout)
        (root/f"{name}.stdout.log").write_text(proc.stdout+"\n[stderr]\n"+proc.stderr)
        record.update({"returncode":proc.returncode,"status":"pass" if proc.returncode==0 else "fail"})
    except BaseException as exc: record.update({"returncode":-1,"status":"fail","error":repr(exc)})
    return record

def main():
    p=argparse.ArgumentParser(); p.add_argument("--npu",type=int,default=7); p.add_argument("--shm-id",type=int,default=11000); p.add_argument("--output-root",type=Path,default=ROOT/"results/ppt-evidence-20260829/reliability"); p.add_argument("--timeout",type=int,default=7200); p.add_argument("--dry-run",action="store_true"); p.add_argument("--cases",nargs="+",default=None,help="run only the named reliability cases"); args=p.parse_args(); args.output_root.mkdir(parents=True,exist_ok=True)
    cases=[]
    for requests in (1000,10000):
        cases.append((f"owner_{requests}",[PYTHON,str(ROOT/"experiments/benchmarks/sync_ring_ab.py"),"--npu",str(args.npu),"--shm-id",str(args.shm_id+requests),"--warmups","10","--repetitions",str(requests),"--payload-bytes","4096","--output-root",str(args.output_root/f"owner_{requests}")]))
    cases.extend([
        ("fault_unit",["env","PYTHONPATH=/home/user7/.local/lib/python3.9/site-packages",str(Path("/home/user7/.local/bin/pytest")),"-q","tests/python/test_frame_lifecycle.py","tests/python/test_raw_ring.py","tests/python/test_incremental_frame.py","tests/python/test_disk_layout.py"]),
        ("hardware_timeout_and_commit",[PYTHON,"tests/hardware/g0_roundtrip.py","--npu",str(args.npu)]),
        ("hardware_nvme_error",[PYTHON,"tests/hardware/fault_injection_write.py","--npu",str(args.npu)]),
        ("hardware_crash_consistency",[PYTHON,"experiments/benchmarks/s2_raw_ring_matrix.py","--npu",str(args.npu),"--shm-id",str(args.shm_id+2),"--steps","100","--ring-slots","16","--output-root",str(args.output_root/"crash_consistency")]),
        ("metadata_commit_cost",[PYTHON,"experiments/benchmarks/metadata_ab_protocol.py","--npu",str(args.npu),"--shm-id",str(args.shm_id+3),"--warmups","10","--repetitions","30","--output-root",str(args.output_root/"metadata_cost")]),
    ])
    if args.cases:
        known={name for name,_cmd in cases}; unknown=set(args.cases)-known
        if unknown: p.error("unknown cases: "+", ".join(sorted(unknown)))
        cases=[case for case in cases if case[0] in set(args.cases)]
    records=[execute(name,cmd,args.output_root,args.timeout,args.dry_run) for name,cmd in cases]
    result={"experiment":"reliability","records":records,"required_faults":["ring_full","slot_exhaustion","nvme_error","timeout","duplicate_completion","owner_exit"],"crash_rule":"uncommitted generation invisible; prior committed generation recoverable","status":"planned" if args.dry_run else ("pass" if all(r["status"]=="pass" for r in records) else "fail")}
    (args.output_root/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,sort_keys=True))
    if result["status"]=="fail": raise SystemExit(1)
if __name__=="__main__": main()
