"""Explicit allocation accounting and phase budgets; no framework imports."""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
from pathlib import Path


def positive(value,name):
    if type(value) is not int or value<=0:raise ValueError(name+' must be a positive integer')
    return value


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
