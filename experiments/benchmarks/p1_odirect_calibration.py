#!/usr/bin/env python3
"""Read-only O_DIRECT calibration of the two identical SSD namespaces."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"python"))
from ppt_evidence import stats

def sample(device,args,index):
    cmd=["fio",f"--name=cal_{Path(device).name}",f"--filename={device}","--rw=read","--direct=1","--ioengine=io_uring",f"--iodepth={args.depth}",f"--bs={args.block_size}",f"--offset={args.offset}",f"--size={args.total_bytes}","--output-format=json"]
    start=time.perf_counter_ns(); p=subprocess.run(cmd,text=True,capture_output=True,check=False,timeout=args.timeout); elapsed=(time.perf_counter_ns()-start)/1e6
    if p.returncode: raise RuntimeError(p.stderr)
    return {"sample":index,"device":device,"elapsed_ms":elapsed,"fio":json.loads(p.stdout)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--devices",nargs=2,required=True); p.add_argument("--offset",type=int,default=64*1024**3); p.add_argument("--total-bytes",type=int,default=1024**3); p.add_argument("--block-size",type=int,default=4*1024**2); p.add_argument("--depth",type=int,default=4); p.add_argument("--warmups",type=int,default=10); p.add_argument("--samples",type=int,default=30); p.add_argument("--timeout",type=int,default=1800); p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    rows=[]
    for device in args.devices:
        if not Path(device).is_block_device(): raise ValueError(f"not a block device: {device}")
        for i in range(args.warmups+args.samples):
            row=sample(device,args,i); row["warmup"]=i<args.warmups
            if not row["warmup"]: rows.append(row)
    result={"operation":"read-only O_DIRECT calibration; no format or write","devices":args.devices,"rows":rows,"latency_ms":{d:stats([r["elapsed_ms"] for r in rows if r["device"]==d]) for d in args.devices}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result["latency_ms"],sort_keys=True))
if __name__=="__main__": main()
