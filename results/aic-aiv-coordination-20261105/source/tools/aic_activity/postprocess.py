"""Reproducible task-window extraction and evidence gates for a completed run."""
import argparse,json,hashlib,statistics
from pathlib import Path
from .analysis import read,save
from .flows import annotate
from .quality import training,cross_rank_counts
from .phases import compiled_training_phases

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def process(run,out,kind):
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    evidence=training(run) if kind=='training' else json.loads((run/'result.json').read_text())
    quality=evidence['quality'];rank_summaries=[];files=[]
    for rank in range(4):
        pattern=f'rank_{rank}/activity-profiler/*/ASCEND_PROFILER_OUTPUT' if kind=='training' else f'trace/rank{rank}_*/ASCEND_PROFILER_OUTPUT'
        paths=list(run.glob(pattern))
        if len(paths)!=1:raise ValueError(f'expected one trace for rank {rank}, found {paths}')
        trace=paths[0];rows=read(trace/'kernel_details.csv');phase_quality=None
        flow_path=trace/'trace_view.json'
        if not flow_path.exists():flow_path=trace/'trace_view.json.gz'
        if kind=='training':
            anchors=sorted(r['start'] for r in rows if 'GetNext' in r['name'])
            if len(anchors)!=7:raise ValueError(f'expected seven GetNext anchors, found {len(anchors)}')
            windows=[dict(id=f'step-{18+i}',begin_ns=l,end_ns=h,quality=quality) for i,(l,h) in enumerate(zip(anchors,anchors[1:]))]
            config=json.loads((run/'run-config.json').read_text())
            if config.get('activity_profile',{}).get('activities')=='npu':
                phase_quality=dict(status='unavailable',reason='NPU-only capture omits framework scope correlation; no inferred training phases')
            else:phase_quality=compiled_training_phases(rows,windows)
        else:
            windows=[dict(w,quality=quality) for w in json.loads((run/'windows.json').read_text()) if w['kind']=='measured']
            if len(windows)!=6:raise ValueError('exactly six measured requests required')
            phase_quality=annotate(rows,flow_path,out/f'rank{rank}/phase-links.jsonl.gz')
        target=out/f'rank{rank}'
        phase_source=('unavailable_npu_only' if phase_quality and phase_quality.get('status')=='unavailable' else
                      'compiled_scope_and_first_backward_boundary' if kind=='training' else 'cpu_enqueue_device_flow')
        summaries=save(rows,windows,target,dict(physical_device_id=4+rank,rank=rank,run_id=run.name,phase_evidence=phase_source))
        rank_summaries.append(summaries)
        if phase_quality:(target/'phase_quality.json').write_text(json.dumps(phase_quality,indent=2)+'\n')
        files.extend(dict(path=str(path),sha256=digest(path)) for path in [trace/'kernel_details.csv',flow_path])
        print('PROCESSED',run.name,rank,flush=True)
    completeness=[cross_rank_counts([rank[i] for rank in rank_summaries]) for i in range(6)]
    # Anomalies are explicit diagnostic evidence; task counts alone cannot prove completeness.
    suspect=any(x['suspect'] for x in completeness)
    result=dict(status='pass',measurement_quality=quality,task_count_anomaly=suspect,
                phase_label_quality='partial_attribution' if kind=='inference' else
                    'unavailable' if phase_source=='unavailable_npu_only' else 'scoped_with_inferred_forward_recompute_boundary',
                evidence=evidence,cross_rank_counts=completeness,raw_source_sha256=files)
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--kind',choices=['training','inference'],required=True);a=p.parse_args();process(a.run,a.output,a.kind)
