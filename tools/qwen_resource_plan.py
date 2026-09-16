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
from npu_nvme.d2.region_budget import region_budget


def plan(schema,*,chunk,depth,mode,hbm_free,host_free,hbm_headroom,host_headroom,
         control_bytes=0,active_pins=1,region_bytes=1<<40):
    if chunk not in (1<<20,4<<20,16<<20) or depth not in (1,4,8,16,32,64):raise ValueError('unsupported probe configuration')
    if schema['topology']!={'tp':4,'dp':1,'pp':1}:raise ValueError('expected actual TP4 schema')
    tensors=[row for row in schema['tensors'] if row.get('placement')!='native_container_metadata']
    names=set();sizes=[];partitions={}
    for tensor in tensors:
        name=tensor['name'];size=tensor['logical_bytes_per_rank']
        if name in names or type(size) is not int or size<=0:raise ValueError('invalid tensor inventory')
        names.add(name);sizes.append(size)
        partition=tensor['partition'];partitions[partition]=partitions.get(partition,0)+1
    state=sum(sizes);chunks=sum((n+chunk-1)//chunk for n in sizes)
    if type(control_bytes) is not int or control_bytes<0 or type(active_pins) is not int or not 0<=active_pins<=1:
        raise ValueError('control budget or reader pin limit')
    pool=chunk*depth
    # Four private rank pools and one owner pool. Socket queues and wire/copy
    # temporaries are separate physical allocations, never shared aliases.
    pools=5*pool
    sockets=4*2*2*min(chunk,4<<20)  # both endpoints, send+receive requested budgets
    temporaries=4*3*chunk+3*chunk  # wire payload/hash temporaries and owner staging
    descriptors=chunks*4*2048  # conservative Python metadata admission estimate
    controls=control_bytes*4  # decode/encode residency, not four ranks again
    transport=pools+sockets+temporaries+descriptors+controls
    admission=capture_admission(mode=mode,state_bytes=state,hbm_free=hbm_free,host_free=host_free,
        hbm_headroom=hbm_headroom,host_headroom=host_headroom,transport_host=transport)
    aggregate_host=transport+(4*state if mode=='host_snapshot' else 0)
    disk=region_budget(state_bytes=state*4,descriptor_count=chunks*4,page_bytes=65536,
                       retention=3,active_pins=active_pins,control_bytes=control_bytes)
    disk['safety_margin_bytes']=region_bytes//10
    disk['admitted']=disk['required_region_bytes']+disk['safety_margin_bytes']<=region_bytes
    return dict(aggregate_host_required=aggregate_host,
        aggregate_host_admitted=aggregate_host+host_headroom<=host_free,scope='schema-derived plan; capacities must come from a fresh real probe, not framework max_device_memory',
        rank_count=4,tensor_count_per_rank=len(sizes),partition_counts=partitions,
        state_bytes_per_rank=state,state_bytes_total=state*4,max_tensor_bytes=max(sizes),
        descriptors_per_rank=chunks,descriptors_total=chunks*4,chunk_bytes=chunk,depth=depth,
        transport='socket_host_bridge',owner_pool_bytes=pool,rank_pools_bytes=4*pool,
        socket_buffer_budget_bytes=sockets,temporary_budget_bytes=temporaries,
        descriptor_residency_budget_bytes=descriptors,control_residency_budget_bytes=controls,
        disk=disk,admission_per_rank=admission,
        required_followup=['per-rank control JSON budget','actual framework peak and ACL free memory',
                           'actual socket buffer sizes and native pool allocation','snapshot mode pilot','phase deadlines'],
        caveat='Schema-derived upper estimate, not a measured peak; actual socket buffers and allocator residency must be checked.')


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
