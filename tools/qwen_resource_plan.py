#!/usr/bin/env python3
"""Derive descriptor/resource geometry from the recorded actual TP4 schema."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from npu_nvme.runtime.resources import capture_admission


def plan(schema,*,chunk,depth,mode,hbm_free,host_free,hbm_headroom,host_headroom):
    if chunk not in (1<<20,4<<20,16<<20) or depth not in (1,4,8,16,32,64):raise ValueError('unsupported probe configuration')
    if schema['topology']!={'tp':4,'dp':1,'pp':1}:raise ValueError('expected actual TP4 schema')
    tensors=schema['tensors'];names=set();sizes=[];partitions={}
    for tensor in tensors:
        name=tensor['name'];size=tensor['logical_bytes_per_rank']
        if name in names or type(size) is not int or size<=0:raise ValueError('invalid tensor inventory')
        names.add(name);sizes.append(size)
        partition=tensor['partition'];partitions[partition]=partitions.get(partition,0)+1
    state=sum(sizes);chunks=sum((n+chunk-1)//chunk for n in sizes)
    pool=chunk*depth
    admission=capture_admission(mode=mode,state_bytes=state,hbm_free=hbm_free,host_free=host_free,
        hbm_headroom=hbm_headroom,host_headroom=host_headroom,transport_host=pool)
    aggregate_host=pool+(4*state if mode=='host_snapshot' else 0)
    return dict(aggregate_host_required=aggregate_host,
        aggregate_host_admitted=aggregate_host+host_headroom<=host_free,scope='schema-derived plan; capacities must come from a fresh real probe, not framework max_device_memory',
        rank_count=4,tensor_count_per_rank=len(sizes),partition_counts=partitions,
        state_bytes_per_rank=state,state_bytes_total=state*4,max_tensor_bytes=max(sizes),
        descriptors_per_rank=chunks,descriptors_total=chunks*4,chunk_bytes=chunk,depth=depth,
        shared_spdk_pool_bytes=pool,admission_per_rank=admission,
        required_followup=['per-rank control JSON budget','actual framework peak and ACL free memory',
                           'rank pool mapping/IPC credits','snapshot mode pilot','phase deadlines'],
        caveat='Shared owner pool counted once. Four simultaneous Host snapshots require four state allocations.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schema',type=Path,default=ROOT/'results/long-term-v1.3/EN/state_schema-002.json')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--chunk-mib',type=int,default=4);p.add_argument('--depth',type=int,default=4)
    p.add_argument('--capture',choices=['hbm_snapshot','host_snapshot','blocking'],required=True)
    for name in ('hbm-free','host-free','hbm-headroom','host-headroom'):p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args();raw=a.schema.read_bytes()
    result=plan(json.loads(raw),chunk=a.chunk_mib*1024**2,depth=a.depth,mode=a.capture,
        hbm_free=a.hbm_free,host_free=a.host_free,hbm_headroom=a.hbm_headroom,host_headroom=a.host_headroom)
    result['schema_sha256']=hashlib.sha256(raw).hexdigest()
    with a.out.open('x') as stream:json.dump(result,stream,indent=2)
    return 0 if result['admission_per_rank']['admitted'] and result['aggregate_host_admitted'] else 3

if __name__=='__main__':raise SystemExit(main())
