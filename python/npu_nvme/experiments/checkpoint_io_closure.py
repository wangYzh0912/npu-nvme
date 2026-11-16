"""Opt-in M1 closure measurements; no device-service or exact-peak claims."""
from contextlib import contextmanager
import csv
import ctypes as C
import json
from pathlib import Path
import re
import subprocess
import time

from .checkpoint_io_analysis import METHODS, process_metrics, distribution
from .checkpoint_io_events import union_ns


class Boundary:
    """Sequential owning-call intervals covering one blocking validation/save."""
    def __init__(self, step, *, clock=time.monotonic_ns):
        self.clock=clock
        self.row=dict(step=step,begin_ns=clock(),stages=[])

    @contextmanager
    def stage(self, name):
        row=dict(name=name,begin_ns=self.clock())
        try:yield row
        finally:
            row['end_ns']=self.clock()
            self.row['stages'].append(row)

    def finish(self):
        self.row['end_ns']=self.clock()
        return self.row


def boundary_coverage(row, required):
    begin,end=row['begin_ns'],row['end_ns']
    if not 0<begin<end:raise ValueError('invalid full boundary')
    stages=row['stages'];names=[v['name'] for v in stages]
    if len(names)!=len(set(names)) or not set(required)<=set(names):
        raise ValueError('missing or duplicate required boundary stage')
    intervals=[(v['begin_ns'],v['end_ns']) for v in stages]
    if any(not begin<=left<=right<=end for left,right in intervals):
        raise ValueError('stage escapes full boundary')
    covered=union_ns(intervals);elapsed=end-begin
    return dict(step=row['step'],elapsed_seconds=elapsed/1e9,
        stage_union_seconds=covered/1e9,unexplained_fraction=1-covered/elapsed,
        status='pass' if covered/elapsed>=.99 else 'failed')


def native_snapshot(transport):
    from npu_nvme.storage.bindings import NPUNVMEStats
    stats=NPUNVMEStats()
    rc=transport.lib.npu_nvme_get_stats(transport.ctx,C.byref(stats))
    if rc:raise RuntimeError('native statistics failed: '+str(rc))
    values={name:int(getattr(stats,name)) for name,_ in stats._fields_}
    return dict(counters=values,retained_buffer_sets=len(transport.retained),
        scopes=dict(request_ring_depth='queued writes only; excludes copies, reads and active requests',
            request_ring_peak='high-water of queued writes only; not global pending high-water',
            nvme_submit_count='native SPDK commands, including metadata and flush',
            async_dma_submit_count='native asynchronous DMA submissions',
            reactor_cpu_us='reactor thread CPU time; cumulative since context open'),
        unavailable=['global native pending high-water','reactor wake notifications (polling API)'])


def verify_drained(snapshot):
    counters=snapshot['counters']
    if (snapshot['retained_buffer_sets'] or counters['nvme_outstanding'] or
            counters['dma_inflight'] or counters['request_ring_depth'] or
            counters['nvme_submit_count']!=counters['nvme_complete_count']):
        raise ValueError('native transport is not drained')


def parse_hbm(text):
    """Parse the installed npu-smi info table, retaining physical device IDs."""
    result={};device=None
    for line in text.splitlines():
        header=re.match(r'[|][ ]*([0-9]+)[ ]+910B[^ ]*[ ]*[|]',line)
        if header:device=int(header.group(1))
        if device is not None and re.search(r'[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}[.][0-9]',line):
            match=re.search(r'([0-9]+)[ ]*/[ ]*([0-9]+)[ ]*[|][ ]*$',line)
            if not match:raise ValueError('npu-smi HBM geometry missing')
            result[str(device)]=dict(used_mib=int(match.group(1)),total_mib=int(match.group(2)))
            device=None
    if not all(str(device) in result for device in (4,5,6,7)):
        raise ValueError('npu-smi physical NPU 4..7 samples missing')
    return result


class ResourceSampler:
    """Supervisor-owned 1 Hz samples; observed maxima are lower bounds on peaks."""
    def __init__(self, path):
        self.path=Path(path);self.last_ns=0

    def due(self):
        return time.monotonic_ns()-self.last_ns>=10**9

    def sample(self, pids):
        now=time.monotonic_ns()
        if now-self.last_ns<10**9:return
        self.last_ns=now;processes=[]
        for pid in sorted(set(pids)):
            try:status=Path('/proc',str(pid),'status').read_text()
            except FileNotFoundError:continue  # exited between process discovery and sampling
            values={}
            for line in status.splitlines():
                key,_,value=line.partition(':')
                if key in ('VmRSS','VmHWM','VmSize','RssAnon','RssFile','RssShmem'):
                    values[key+'_kib']=int(value.split()[0])
            processes.append(dict(pid=pid,**values))
        smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=30)
        hbm=parse_hbm(smi)
        row=dict(begin_ns=now,end_ns=time.monotonic_ns(),processes=processes,
                 hbm={str(device):hbm[str(device)] for device in (4,5,6,7)},
                 peak_semantics='sampled at nominal 1 Hz; HBM is device-wide; VmHWM is kernel RSS high-water')
        with self.path.open('a') as stream:stream.write(json.dumps(row)+chr(10))


def overhead_gate(runs):
    """Eighteen fresh processes, paired medians/elapsed and pooled P95."""
    indexed={}
    for run in runs:
        key=(run['group'],run['mode'],run['replicate'])
        if key in indexed:raise ValueError('duplicate independent process identity')
        indexed[key]=process_metrics(run)
    expected={(method,mode,rep) for method in METHODS for mode in ('A','B') for rep in range(3)}
    if set(indexed)!=expected:raise ValueError('eighteen independent A/B processes required')
    pairs=[];groups=[]
    for method in METHODS:
        for rep in range(3):
            a,b=(indexed[method,mode,rep] for mode in ('A','B'))
            pairs.append(dict(method=method,replicate=rep,
                median_ratio=b['aligned_step_seconds']['median']/a['aligned_step_seconds']['median'],
                formal_elapsed_ratio=b['formal_elapsed_seconds']/a['formal_elapsed_seconds']))
        stats={mode:distribution([window['seconds'] for rep in range(3)
            for window in indexed[method,mode,rep]['windows']]) for mode in ('A','B')}
        groups.append(dict(method=method,modes=stats,p95_ratio=stats['B']['p95']/stats['A']['p95']))
    passed=(all(max(row['median_ratio'],row['formal_elapsed_ratio'])<=1.10 for row in pairs)
            and all(row['p95_ratio']<=1.10 for row in groups))
    return dict(status='pass' if passed else 'failed',limit=1.10,pairs=pairs,groups=groups,
        quantile='pooled 30 aligned windows per method/mode; no per-process P95',
        rule='all preregistered pairs and pooled P95 must pass; no discarded repetitions')


def complete_timeline(events, reports, method, steps):
    from .checkpoint_io_events import timeline
    result=timeline(events);coverage=[];missing=[]
    required={'admission_collectives','framework_synchronize','manifest_read_hash',
              'control_capture','sidecar_write','save_setup','capture_transport'}
    if method=='ours':required|={'reservation','complete_publish_collective'}
    if sorted(row['rank'] for row in reports)!=list(range(4)):
        raise ValueError('four independent rank reports required')
    if [row['step'] for row in result['checkpoints']]!=list(steps):
        raise ValueError('checkpoint event steps differ from registered boundaries')
    for report in reports:
        rows=[row for row in report['boundaries'] if row['operation']=='save']
        if [row['step'] for row in rows]!=list(steps):raise ValueError('missing full save boundary')
        for row in rows:
            measured=boundary_coverage(row,required);measured['rank']=report['rank'];coverage.append(measured)
            ranked=[v for v in events if v['step']==row['step'] and v['role']=='rank' and v['rank']==report['rank']]
            intervals=[(v['begin_ns'],v['end_ns']) for v in row['stages'] if v['name']!='capture_transport']
            def pair(begin,end,*,chunks=False):
                left={v['chunk'] if chunks else None:v['timestamp_ns'] for v in ranked if v['event']==begin}
                right={v['chunk'] if chunks else None:v['timestamp_ns'] for v in ranked if v['event']==end}
                if not left or left.keys()!=right.keys():raise ValueError('missing required critical-path pair: '+begin)
                return [(left[key],right[key]) for key in left]
            intervals+=pair('capture_setup_begin','capture_setup_end')
            intervals+=pair('chunk_source_read_begin' if method=='ours' else 'source_read_begin',
                            'source_released',chunks=True)
            if method=='ours':
                for prefix in ('prepared_journal','prepared_barrier','protocol_admit','protocol_commit'):
                    intervals+=pair(prefix+'_begin',prefix+'_end')
            if any(not row['begin_ns']<=left<=right<=row['end_ns'] for left,right in intervals):
                raise ValueError('critical-path pair escapes full boundary')
            measured['critical_path_unexplained_fraction']=1-union_ns(intervals)/(row['end_ns']-row['begin_ns'])
            if measured['critical_path_unexplained_fraction']>.01:measured['status']='failed'
            manifest=next(v for v in row['stages'] if v['name']=='manifest_read_hash')
            aggregate=manifest.get('aggregate')
            if not aggregate or aggregate['logical_bytes']<=0 or aggregate['tensors']<=0:
                raise ValueError('manifest aggregate read/hash measurements missing')
            if aggregate['read_ns']+aggregate['hash_ns']>manifest['end_ns']-manifest['begin_ns']:
                raise ValueError('manifest sub-timers exceed owning stage')
    for step in steps:
        rows=[row for row in events if row['step']==step]
        for rank in range(4):
            ranked=[row for row in rows if row['role']=='rank' and row['rank']==rank]
            for event in ('capture_setup_begin','capture_setup_end','capture_begin','capture_end',
                          'training_step','training_resume'):
                if sum(row['event']==event for row in ranked)!=1:
                    missing.append(dict(step=step,rank=rank,event=event))
            chunks={name:sum(row['event']==name for row in ranked) for name in
                ('source_read_begin','data_ready','source_released',
                 'sink_consumed_source_safe' if method=='checkpoint_io_sink' else 'chunk_send_begin')}
            if not min(chunks.values()) or len(set(chunks.values()))!=1:
                missing.append(dict(step=step,rank=rank,chunk_counts=chunks))
            if method=='ours' and sum(row['event']=='rank_complete_written' for row in ranked)!=1:
                missing.append(dict(step=step,rank=rank,event='rank_complete_written'))
        if method=='ours':
            for event in ('payload_flush_begin','payload_flush_end','metadata_begin','metadata_flush_end',
                          'anchor_flush_end','raw_commit_receipt','catalog_publish_begin','catalog_published'):
                if sum(row['event']==event for row in rows)!=1:missing.append(dict(step=step,event=event))
    result.update(full_boundaries=coverage,missing_required=missing,
        attribution='Owning host call stages cover the full boundary. capture_transport includes protocol/setup/copy/hash/IPC; nested event unions overlap and are not device service times.')
    result['status']=('pass' if result['status']=='correlated' and not missing
        and all(row['status']=='pass' for row in coverage) else 'failed')
    return result


def lifecycle_gate(output, *, detailed):
    from npu_nvme.runtime.training_catalog import read_checked
    output=Path(output);result=json.loads((output/'result.json').read_text())
    end=result.get('end_to_end',{})
    if (result.get('validation_status')!='pass' or end.get('children_remaining')!=[] or
            end.get('retained')!=[] or end.get('lease_released') is not True):
        raise ValueError('supervisor safe close evidence missing')
    begin=result['lifecycle']['training_launch_ns'];closed=result['lifecycle']['safe_close_end_ns']
    if not 0<begin<closed:raise ValueError('invalid launch to close interval')
    for rank in range(4):
        report=read_checked(output/f'rank_{rank}/training.json');life=report['lifecycle']
        if not report.get('closed') or not (0<life['close_begin_ns']<=life['safe_close_end_ns']<=closed):
            raise ValueError('rank close evidence missing')
        if report['method']!='none':
            session=json.loads((output/f'rank_{rank}/d2-session-status.json').read_text())
            if session.get('status')!='pass' or not session.get('closed'):
                raise ValueError('rank native close not complete')
            if detailed:
                verify_drained(session['native'])
                counters=session['copy_api']
                if counters['submitted']!=counters['completed'] or counters['rejected']:
                    raise ValueError('copy descriptor accounting differs')
    owner_path=output/'owner/result.json'
    if owner_path.exists():
        owner=json.loads(owner_path.read_text())
        if owner.get('status')!='pass' or not owner.get('closed'):raise ValueError('owner not closed')
        if detailed:
            verify_drained(owner['native']);api=owner['transfer_api']
            if api['submitted']!=api['completed'] or api['rejected']:raise ValueError('owner descriptor accounting differs')
    resources=None
    if detailed:
        path=output/'resource-samples.jsonl'
        samples=0;previous=None;gaps=[];rss={};hbm={};observed=set()
        with path.open() as stream:
            for line in stream:
                row=json.loads(line);samples+=1
                if previous is not None:gaps.append((row['begin_ns']-previous)/1e9)
                previous=row['begin_ns']
                for proc in row['processes']:
                    observed.add(proc['pid'])
                    rss[str(proc['pid'])]=max(rss.get(str(proc['pid']),0),proc.get('VmRSS_kib',0))
                for device,spec in row['hbm'].items():hbm[device]=max(hbm.get(device,0),spec['used_mib'])
        rank_pids={read_checked(output/f'rank_{rank}/training.json')['pid'] for rank in range(4)}
        if samples<2 or not rank_pids<=observed:raise ValueError('resource samples missing rank processes')
        resources=dict(samples=samples,maximum_sample_gap_seconds=max(gaps),
            sampled_rss_max_kib=rss,sampled_device_hbm_max_mib=hbm,
            semantics='Nominal 1 Hz samples, not exact allocation peaks; device HBM includes runtime/other contexts. Admission ledger remains a separate budget.')
    return dict(status='pass',end_to_end=end,resources=resources)


def device_summary(directory):
    """Read actual profiler exports; requested capabilities alone are not evidence."""
    directory=Path(directory);tables=[];pmu_columns=set();positive_events=0;categories={};device_tasks=0
    for path in sorted(directory.rglob('*.csv')):
        with path.open(newline='') as stream:
            reader=csv.DictReader(stream);headers=reader.fieldnames or [];count=0
            for row in reader:
                if any(row.values()):count+=1
                if ('Accelerator Core' in row or 'Task Type' in row) and float(
                        row.get('Duration(us)',row.get('Task Duration(us)',0)) or 0)>0:
                    device_tasks+=1
            if count:
                tables.append(dict(path=str(path.relative_to(directory)),rows=count,columns=headers))
                pmu_columns.update(name for name in headers if any(token in name.lower()
                    for token in ('mac_ratio','vec_ratio','mte1','mte2','mte3','aic_mac','aiv_vec')))
    for path in sorted(directory.rglob('*.json')):
        if 'trace' not in path.name.lower():continue
        value=json.loads(path.read_text());events=value.get('traceEvents',[]) if isinstance(value,dict) else value
        if not isinstance(events,list):raise ValueError('unsupported device trace export shape')
        for event in events:
            if event.get('ph')=='X' and float(event.get('dur',0))>0:
                positive_events+=1;category=str(event.get('cat','unspecified'))
                categories[category]=categories.get(category,0)+1
    if not tables or not positive_events or not device_tasks:
        raise ValueError('nonempty device task CSV and duration trace evidence required')
    return dict(status='exports_verified',tables=tables,positive_duration_events=positive_events,
        positive_duration_device_tasks=device_tasks,
        trace_categories=categories,observed_pmu_columns=sorted(pmu_columns),
        pmu_status='columns_observed' if pmu_columns else 'not_in_export',
        clock='Profiler export clock retained in original metadata; host monotonic/realtime anchors are separate. No cross-clock overlap is inferred.')


def dependency_bound(reports):
    """Verify the measured synchronous save -> next optimizer-step dependency."""
    if sorted(row['rank'] for row in reports)!=list(range(4)):raise ValueError('four rank reports required')
    observations=[]
    for report in reports:
        losses={row['step']:row for row in report['losses']}
        profile=report['device_profile']
        if not (profile['start_anchor']['monotonic_ns']<=losses[28]['begin_ns']
                and losses[32]['resume_ns']<=profile['stop_anchor']['monotonic_ns']):
            raise ValueError('profiler does not cover full target steps 28 and 32')
        if report['method']=='none':continue
        boundaries={row['step']:row for row in report['boundaries']}
        for step in (28,32):
            boundary=boundaries[step];loss=losses[step]
            if not loss['end_ns']<=boundary['begin_ns']<boundary['end_ns']<=loss['resume_ns']:
                raise ValueError('save lies outside optimizer-end to resume boundary')
            if step==28 and losses[29]['begin_ns']<boundary['end_ns']:
                raise ValueError('next optimizer step begins before save returns')
            observations.append(dict(rank=report['rank'],step=step,
                blocking_seconds=(boundary['end_ns']-boundary['begin_ns'])/1e9))
    return dict(status='verified',observations=observations,
        limit='The current synchronous callback waits for capture/commit/publication before the next optimizer step. Its explicit framework synchronization and source-lifetime contract prevent overlap with the next training step.',
        interpretation='This is a dependency limit of this execution path, not a device throughput limit or proof that every device engine is idle during saves.')
