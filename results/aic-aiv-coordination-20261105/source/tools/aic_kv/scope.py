"""Isolated, resumable graph/backend applicability experiments."""
import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/models/npu_nvme_exp/user7-stack/aic-aiv-scope-20261104')
RAW=Path('/dev/shm/user7-aic-aiv-20261031/scope-20261104')
LOCK=Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lock')


def full_cache_slice(task,expected_elements):
    if task['Name']!='aclnnContiguous_SliceAiCore_Slice' or task['Output Data Types']!='DT_BF16':return False
    outputs=task['Output Shapes'].strip('"').split(';')
    return any(math.prod(int(n) for n in shape.split(',')) in expected_elements for shape in outputs if shape)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def capture_sources(container,dest):
    paths=['/vllm-workspace/vllm-ascend/vllm_ascend/attention/utils.py',
           '/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py',
           '/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
           '/vllm-workspace/vllm-ascend/vllm_ascend/ascend_config.py',
           '/vllm-workspace/vllm/vllm/config/cache.py']
    dest.mkdir(parents=True,exist_ok=True)
    script='import importlib.metadata as m,json;print(json.dumps({k:m.version(k) for k in ("vllm","vllm-ascend","torch","torch-npu")}))'
    save(dest/'packages.json',json.loads(subprocess.check_output(['docker','exec',container,'python','-c',script],text=True)))
    for p in paths:
        subprocess.run(['docker','cp',container+':'+p,str(dest/Path(p).name)],check=True)
    for repo in ('vllm','vllm-ascend'):
        details={}
        for key,args in [('commit',['rev-parse','HEAD']),('status',['status','--porcelain']),('diff',['diff'])]:
            details[key]=subprocess.check_output(['docker','exec',container,'git','-C','/vllm-workspace/'+repo,*args],text=True)
        save(dest/(repo+'-git.json'),details)
    save(dest/'hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.py')})


def run_job(stack,name,arguments,module='aic_kv.run'):
    container='user7-aic-scope-'+stack
    target=RAW/name
    marker=ROOT/'completed'/f'{name}.json'
    if module!='aic_kv.run' and marker.exists():return True
    export_log=RAW/(name+'-export.log')
    exported=(target/'completed.json').exists() and export_log.exists() and export_log.read_text().count('All profiling data parsed')>=4*len(list(target.glob('b*-l*-o*/trace')))
    if exported:
        save(marker,dict(status='pass'))
        print('REUSE',name,flush=True);return True
    if (ROOT/'failures'/f'{name}.json').exists():
        print('FAILED_PREVIOUSLY',name,flush=True);return False
    if target.exists():raise RuntimeError('unfinished output requires explicit new run identity: '+name)
    info=json.loads(subprocess.check_output(['docker','inspect',container]))[0]
    if not info['State']['Running']:subprocess.run(['docker','start',container],check=True)
    env=dict(x.split('=',1) for x in info['Config']['Env'])
    source=ROOT/'source'/name
    shutil.copytree(Path(__file__).parent,source/'aic_kv',ignore=shutil.ignore_patterns('__pycache__'))
    save(source/'container.json',dict(image=info['Image'],mounts=info['Mounts']))
    pythonpath=str(source)+':/experiment:'+env['PYTHONPATH']
    base=['docker','exec','-e','PYTHONPATH='+pythonpath,'-e','PYTHONUNBUFFERED=1',container,'python','-u','-m']
    command=base+[module,'--output','/output/'+name,*arguments]
    save(ROOT/'commands'/f'{name}.json',command)
    print('START',name,flush=True)
    start=time.monotonic()
    with (RAW/(name+'.log')).open('x') as log:
        try:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            subprocess.run(['docker','exec',container,'chmod','-R','a+rX','/output/'+name],check=True)
            if module=='aic_kv.run':
                with (RAW/(name+'-export.log')).open('x') as export:
                    subprocess.run(base+['aic_kv.analyse','--trace','/output/'+name],stdout=export,stderr=subprocess.STDOUT,check=True,timeout=1200)
            subprocess.run(['docker','exec',container,'chmod','-R','a+rX','/output/'+name],check=True)
            save(marker,dict(status='pass'))
        except (subprocess.SubprocessError,TimeoutError) as error:
            # Preserve the original cause and stop only our isolated container.
            log.flush()
            messages=(RAW/(name+'.log')).read_text()
            cause='driver does not support graph operator update (207019)' if 'current driver version does not support to update this op' in messages else 'see retained runtime traceback'
            if 'output changed within one deterministic condition' in messages:
                cause='strict output hash gate failed within the same layout/condition; original outputs retained'
            cleanup=subprocess.run(['docker','exec',container,'chmod','-R','a+rX','/output'])
            stopped=subprocess.run(['docker','stop','--time','10',container])
            save(ROOT/'failures'/f'{name}.json',dict(error=str(error),cause=cause,command=command,
                log=str(RAW/(name+'.log')),cleanup_exit=cleanup.returncode,stop_exit=stopped.returncode))
            print('FAILED',name,error,flush=True)
            if cleanup.returncode or stopped.returncode:raise RuntimeError('failed experiment cleanup') from error
            return False
    print('DONE',name,round(time.monotonic()-start,1),'seconds',flush=True)
    return True


def summarize():
    from .provenance import ns
    rows=[]
    for run in sorted(RAW.iterdir()):
        if not run.is_dir() or run.name.startswith('corun-') or not (ROOT/'completed'/f'{run.name}.json').exists():continue
        cached=ROOT/'evidence'/run.name/'device-summaries-v2.json'
        if cached.exists():
            rows.extend(json.loads(cached.read_text()));continue
        start=len(rows)
        for case in sorted(run.glob('b*-l*-o*')):
            out=ROOT/'evidence'/run.name/case.name;out.mkdir(parents=True,exist_ok=True)
            for p in case.glob('*.json'):shutil.copy2(p,out/p.name)
            for p in (run/'run-config.json',run/'source-identity.json',run/'completed.json'):
                shutil.copy2(p,out.parent/p.name)
            result=json.loads((case/'result.json').read_text());layouts=json.loads((case/'kv-layout.json').read_text())
            config=json.loads((run/'run-config.json').read_text())
            windows=[w for w in json.loads((case/'windows.json').read_text()) if w['kind']=='task']
            traces=sorted(case.glob('trace/rank*_*/ASCEND_PROFILER_OUTPUT'))
            if not traces:
                for layout in layouts:
                    rows.append(dict(run=run.name,case=case.name,device=layout['rank']+4,model=config['model'],**result,
                        graph_mode=layout['graph_mode'],layout=layout['resolved_layout'],
                        k_contiguous=layout['layers'][0]['key']['contiguous'],v_contiguous=layout['layers'][0]['value']['contiguous'],
                        pa_eligible=layout['pa_eligible'],slice_count_per_generate=None,slice_ms_per_generate=None,attention_tasks=[]))
            for trace in traces:
                path=trace/'kernel_details.csv';target=out/trace.parent.name
                target.mkdir(exist_ok=True)
                with path.open('rb') as source,gzip.open(target/'kernel_details.csv.gz','wb') as f:shutil.copyfileobj(source,f)
                with path.open() as f:tasks=list(csv.DictReader(f))
                device=int(tasks[0]['Device_id']);layout=layouts[device-4]
                expected_elements={math.prod(layer[kind]['shape']) for layer in layout['layers'] for kind in ('key','value')}
                slices=[t for t in tasks if full_cache_slice(t,expected_elements) and any(w['begin_ns']<=ns(t['Start Time(us)'])<w['end_ns'] for w in windows)]
                names={t['Name'] for t in tasks}
                rows.append(dict(run=run.name,case=case.name,device=device,model=config['model'],**result,
                    graph_mode=layout['graph_mode'],layout=layout['resolved_layout'],
                    k_contiguous=layout['layers'][0]['key']['contiguous'],v_contiguous=layout['layers'][0]['value']['contiguous'],
                    pa_eligible=layout['pa_eligible'],slice_count_per_generate=len(slices)/len(windows),
                    slice_ms_per_generate=sum(float(t['Duration(us)']) for t in slices)/1000/len(windows),
                    attention_tasks=sorted(n for n in names if any(s in n.lower() for s in ('attention','fusedinfer','paged')))))
        save(cached,rows[start:])
    save(ROOT/'summary.json',rows)
    print('SUMMARY_DEVICE_CASES',len(rows),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','summarize','confirm','version','measure','corun','blocktable','extend','hybrid','correctness','correctness-sync'],required=True)
    a=p.parse_args();ROOT.mkdir(exist_ok=True);RAW.mkdir(exist_ok=True)
    if a.stage=='summarize':summarize();return
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if LOCK.with_suffix('.lease.json').exists():raise RuntimeError('hardware lease occupied')
        smi=subprocess.check_output(['npu-smi','info'],text=True)
        if not all(f'No running processes found in NPU {i}' in smi for i in (4,5,6,7)):raise RuntimeError('cards occupied')
        if a.stage=='screen':
            for stack in ('official','current'):
                capture_sources('user7-aic-scope-'+stack,ROOT/'stack-identity'/stack)
                for mode,pa in [('PIECEWISE',''),('DEFAULT',''),('FULL_DECODE_ONLY',''),('NONE',''),('FULL_DECODE_ONLY','1,8')]:
                    if stack=='current' and mode!='PIECEWISE':continue
                    name=stack+'-'+mode.lower()+('-pa' if pa else '')+'-screen'
                    args=['--graph-mode',mode,'--blocks','1654','--cases','1:128:32,8:128:32','--trace-requests','2','--control-requests','3']
                    if pa:args+=['--pa-shapes',pa]
                    run_job(stack,name,args)
                    summarize()
        elif a.stage=='version':
            capture_sources('user7-aic-scope-v018',ROOT/'stack-identity/v018-runtime')
            for rep in range(3):
                passed=run_job('v018',f'v018-piecewise-r{rep}',[
                    '--worker','aic_kv.worker_v018.V018Worker','--graph-mode','PIECEWISE',
                    '--blocks','1654','--max-model-len','2304','--cases','1:128:32,8:128:32,1:2048:32',
                    '--trace-requests','2','--control-requests','3'])
                summarize()
                if not passed:break
        elif a.stage=='corun':
            for rep in range(3):
                run_job('official',f'corun-r{rep}',[],module='aic_kv.corun')
        elif a.stage=='hybrid':
            run_job('official','hybrid-qwen35-27b-piecewise-screen',[
                '--model','/models/Qwen3.5-27B-original','--graph-mode','PIECEWISE',
                '--blocks','1654','--cases','1:128:32','--trace-requests','1','--control-requests','3'])
            summarize()
        elif a.stage in ('correctness','correctness-sync'):
            for layout in ('native','dense'):
                tag='correctness-sync' if a.stage=='correctness-sync' else 'correctness'
                args=[
                    '--model','/models/Qwen3-8B','--graph-mode','PIECEWISE','--layout',layout,
                    '--prompt-fixture','diverse','--request-metrics','--blocks','1654','--max-model-len','2304',
                    '--cases','8:128:32,8:2048:32','--trace-requests','1','--control-requests','3']
                if a.stage=='correctness-sync':args+=['--sync-scheduling']
                run_job('official',f'{tag}-qwen8b-piecewise-{layout}-r0',args)
                summarize()
        elif a.stage=='blocktable':
            for model in ('8B','4B'):
                for layout in ('native','dense'):
                    run_job('official',f'official-qwen{model.lower()}-piecewise-{layout}-blocktable-r0',[
                        '--model','/models/Qwen3-'+model,'--graph-mode','PIECEWISE','--layout',layout,
                        '--blocks','1654','--max-model-len','2304','--cases','1:128:32,1:2048:32,8:128:32',
                        '--trace-requests','1','--control-requests','3','--metadata-probe',
                        '--metadata-cases','b1-l128-o32,b1-l2048-o32,b8-l128-o32',
                        '--trace-cases','b1-l128-o32,b1-l2048-o32,b8-l128-o32'])
                    summarize()
        elif a.stage=='extend':
            candidates=[('PIECEWISE','native',''),('PIECEWISE','dense',''),('FULL_DECODE_ONLY','native','1,8')]
            for rep in range(3):
                for mode,layout,pa in candidates[rep:]+candidates[:rep]:
                    name=f'extend-qwen8b-{mode.lower()}-{layout}'+('-pa' if pa else '')+f'-r{rep}'
                    args=['--model','/models/Qwen3-8B','--graph-mode',mode,'--layout',layout,'--blocks','1654',
                          '--max-model-len','2304','--cases','1:128:128,8:2048:32',
                          '--trace-requests','1','--control-requests','3']
                    if pa:args+=['--pa-shapes',pa]
                    run_job('official',name,args);summarize()
        elif a.stage=='measure':
            variants=[('native','PIECEWISE',''),('dense','PIECEWISE','')]
            if (ROOT/'completed/official-full_decode_only-pa-screen.json').exists():
                variants.append(('pa','FULL_DECODE_ONLY','1,8'))
            for rep in range(3):
                ordered=variants[rep:]+variants[:rep]
                for label,mode,pa in ordered:
                    arguments=['--graph-mode',mode,'--layout','dense' if label=='dense' else 'native','--blocks','6616',
                        '--cases','1:128:32','--trace-requests','2','--control-requests','6','--system-sample']
                    if pa:arguments+=['--pa-shapes',pa]
                    run_job('official',f'official-system-{label}-r{rep}',arguments)
                    summarize()
        else:
            candidates=[('PIECEWISE','native',''),('PIECEWISE','dense','')]
            pa_run='official-full_decode_only-pa-screen'
            if (ROOT/'completed'/f'{pa_run}.json').exists():candidates.append(('FULL_DECODE_ONLY','native','1,8'))
            for rep in range(3):
                for model in ('8B','4B'):
                    for mode,layout,pa in candidates:
                        name=f'official-qwen{model.lower()}-{mode.lower()}-{layout}'+('-pa' if pa else '')+f'-r{rep}'
                        args=['--model','/models/Qwen3-'+model,'--graph-mode',mode,'--layout',layout,'--blocks','1654','--max-model-len','2304',
                            '--cases','1:128:1,1:128:32,1:2048:32,8:128:32',
                            '--trace-requests','1','--control-requests','3',
                            '--trace-cases','b1-l128-o32,b1-l2048-o32,b8-l128-o32',
                            '--metadata-probe','--metadata-cases','b1-l128-o32,b8-l128-o32']
                        if pa:args+=['--pa-shapes',pa]
                        run_job('official',name,args);summarize()
        for stack in ('official','current'):
            subprocess.run(['docker','stop','--time','10','user7-aic-scope-'+stack],check=True)
        if a.stage=='version':subprocess.run(['docker','stop','--time','10','user7-aic-scope-v018'],check=True)


if __name__=='__main__':main()
