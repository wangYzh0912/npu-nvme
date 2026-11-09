#!/usr/bin/env python3
"""Derive descriptor/resource geometry from the recorded actual TP4 schema."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from npu_nvme.runtime.resources import qwen_plan as plan

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schema',type=Path,default=ROOT/'results/long-term-v1.3/EN/state_schema-002.json')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--chunk-mib',type=int,default=4);p.add_argument('--depth',type=int,default=4)
    p.add_argument('--capture',choices=['hbm_snapshot','host_snapshot','blocking'],required=True)
    p.add_argument('--control-bytes',type=int,default=0)
    p.add_argument('--active-pins',type=int,default=1)
    for name in ('hbm-free','host-free','hbm-headroom','host-headroom'):p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args();raw=a.schema.read_bytes()
    result=plan(json.loads(raw),chunk=a.chunk_mib*1024**2,depth=a.depth,mode=a.capture,
        hbm_free=a.hbm_free,host_free=a.host_free,hbm_headroom=a.hbm_headroom,host_headroom=a.host_headroom,
        control_bytes=a.control_bytes,active_pins=a.active_pins)
    result['schema_sha256']=hashlib.sha256(raw).hexdigest()
    with a.out.open('x') as stream:json.dump(result,stream,indent=2)
    return 0 if result['admission_per_rank']['admitted'] and result['aggregate_host_admitted'] and result['disk']['admitted'] else 3

if __name__=='__main__':raise SystemExit(main())
