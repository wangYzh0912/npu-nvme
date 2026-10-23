"""Explicit allocation accounting and phase budgets; no framework imports."""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
from pathlib import Path


def positive(value,name):
    if type(value) is not int or value<=0:raise ValueError(name+' must be a positive integer')
    return value


def available_host_bytes():
    meminfo=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available=int(meminfo['MemAvailable'].split()[0])*1024
    limit=Path('/sys/fs/cgroup/memory.max')
    if limit.exists() and limit.read_text().strip()!='max':
        available=min(available,max(0,int(limit.read_text())-int(limit.with_name('memory.current').read_text())))
    return available


def qwen_capacity(config, *, hbm_free=None, resident_host=0):
    import json
    schema=json.loads(Path(config['strategy']).read_text())
    region=json.loads(Path(config['region']).read_text())
    report=qwen_plan(schema,chunk=4<<20,depth=4,mode='blocking',
        hbm_free=0 if hbm_free is None else hbm_free,host_free=available_host_bytes(),
        hbm_headroom=0 if hbm_free is None else config['hbm_headroom_bytes'],
        host_headroom=config['host_headroom_bytes'],control_bytes=4<<20,
        region_bytes=region['length_bytes'])
    # Existing pools are accounted as resident, not charged again to free RAM.
    report['resident_host_bytes']=resident_host
    hugepages=report['owner_pool_bytes']+report['rank_pools_bytes']+report['metadata_pool_bytes']
    resident_normal=max(0,resident_host-hugepages)
    future=max(0,report['aggregate_host_required']-hugepages-resident_normal)
    report['host_required_remaining']=future
    report['aggregate_host_admitted']=future+config['host_headroom_bytes']<=report['admission_per_rank']['host_free']
    report['hbm_measured']=hbm_free is not None
    meminfo=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    report['hugepage_free_bytes']=int(meminfo['HugePages_Free'])*int(meminfo['Hugepagesize'].split()[0])*1024
    report['hugepage_required_remaining']=max(0,hugepages-min(resident_host,hugepages))
    report['admitted']=(report['aggregate_host_admitted'] and report['disk']['admitted'] and
        report['hugepage_required_remaining']<=report['hugepage_free_bytes'])
    if hbm_free is not None:
        report['admitted'] &= report['admission_per_rank']['required_hbm']+config['hbm_headroom_bytes']<=hbm_free
    return report


@dataclass(frozen=True)
class Allocation:
    key: str
    domain: str
    bytes: int
    category: str
    rank: int | None = None
    mapping_id: str | None = None


def allocation_totals(allocations):
    """Shared physical Host mappings count once, distinct HBM ranks never alias."""
    seen={};totals={};categories={}
    for item in allocations:
        positive(item.bytes,'allocation bytes')
        if item.domain not in ('host','hbm'):raise ValueError('unknown allocation domain')
        if not item.key or not item.category:raise ValueError('allocation identity required')
        if item.domain=='hbm' and (type(item.rank) is not int or item.rank<0):raise ValueError('HBM rank required')
        if item.mapping_id and item.domain!='host':raise ValueError('shared mapping must be Host')
        key=('shared',item.mapping_id) if item.mapping_id else (item.domain,item.rank,item.key)
        signature=(item.domain,item.bytes,item.category)
        if key in seen:
            if seen[key]!=signature:raise ValueError('inconsistent shared allocation description')
            continue
        seen[key]=signature
        bucket=item.domain if item.domain=='host' else f'hbm_rank_{item.rank}'
        totals[bucket]=totals.get(bucket,0)+item.bytes
        category=(bucket,item.category);categories[category]=categories.get(category,0)+item.bytes
    return dict(totals=totals,categories=[dict(domain=k[0],category=k[1],bytes=v) for k,v in sorted(categories.items())])


def phase_deadlines(existing_ms,pilots_seconds):
    result={}
    for phase,budget in existing_ms.items():
        positive(budget,'phase budget')
        samples=pilots_seconds.get(phase,[])
        if not samples:raise ValueError('missing successful pilot for '+phase)
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in samples):
            raise ValueError('invalid pilot duration')
        result[phase]=max(budget,math.ceil(3*max(samples)*1000))
    return result


def capture_admission(*,mode,state_bytes,hbm_free,host_free,hbm_headroom,host_headroom,
                      transport_hbm=0,transport_host=0):
    """Select mode before running; never change it based on a failed allocation."""
    if mode not in ('hbm_snapshot','host_snapshot','blocking'):raise ValueError('unknown capture mode')
    positive(state_bytes,'state bytes')
    for value in (hbm_free,host_free,hbm_headroom,host_headroom,transport_hbm,transport_host):
        if type(value) is not int or value<0:raise ValueError('invalid resource capacity')
    hbm=transport_hbm+(state_bytes if mode=='hbm_snapshot' else 0)
    host=transport_host+(state_bytes if mode=='host_snapshot' else 0)
    return dict(mode=mode,admitted=hbm+hbm_headroom<=hbm_free and host+host_headroom<=host_free,
                required_hbm=hbm,required_host=host,hbm_free=hbm_free,host_free=host_free,
                hbm_headroom=hbm_headroom,host_headroom=host_headroom,
                training_blocked_until='source_safe' if mode=='blocking' else 'snapshot_ready')


def framework_memory_snapshot(framework,*,rank,phase,external=()):
    """HAL counters measure framework memory; external ACL/SPDK remain separate."""
    hal=framework.hal
    counters={}
    for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
        fn=getattr(hal,name,None)
        if fn is None:raise RuntimeError('framework peak counter unavailable: '+name)
        counters[name]=int(fn())
    return dict(rank=rank,phase=phase,monotonic_ns=time.monotonic_ns(),framework=counters,
                external=allocation_totals(external),process_status=Path('/proc/self/status').read_text(),
                meminfo=Path('/proc/meminfo').read_text(),
                note='framework reserved includes allocated; do not sum the two; external entries exclude framework allocations')

def qwen_plan(schema,*,chunk,depth,mode,hbm_free,host_free,hbm_headroom,host_headroom,
         control_bytes=0,active_pins=1,region_bytes=1<<40):
    from npu_nvme.d2.region_budget import region_budget
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
    pools=5*pool+(1<<20)
    sockets=4*2*2*min(chunk,4<<20)  # both endpoints, send+receive requested budgets
    temporaries=4*3*chunk+3*chunk  # wire payload/hash temporaries and owner staging
    descriptors=chunks*4*2048  # conservative Python metadata admission estimate
    controls=control_bytes*4  # decode/encode residency, not four ranks again
    scratch=4*3*max(sizes)
    transport=pools+sockets+temporaries+descriptors+controls+scratch
    admission=capture_admission(mode=mode,state_bytes=state,hbm_free=hbm_free,host_free=host_free,
        hbm_headroom=hbm_headroom,host_headroom=host_headroom,transport_host=transport,
        transport_hbm=max(sizes))
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
        transport='socket_host_bridge',owner_pool_bytes=pool,rank_pools_bytes=4*pool,metadata_pool_bytes=1<<20,
        socket_buffer_budget_bytes=sockets,temporary_budget_bytes=temporaries,
        descriptor_residency_budget_bytes=descriptors,control_residency_budget_bytes=controls,
        tensor_scratch_budget_bytes=scratch,
        tensor_hbm_scratch_budget_bytes=max(sizes),
        disk=disk,admission_per_rank=admission,
        required_followup=['per-rank control JSON budget','actual framework peak and ACL free memory',
                           'actual socket buffer sizes and native pool allocation','snapshot mode pilot','phase deadlines'],
        caveat='Schema-derived upper estimate, not a measured peak; actual socket buffers and allocator residency must be checked.')
