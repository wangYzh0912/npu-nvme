#!/usr/bin/env python3
"""C1 requires software gates and every required fresh-process method/seed."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from npu_nvme.cli.contracts import canonical, comparison, digest, loss_matches, METHODS, validate_restore_report
from npu_nvme.schemas.evidence import validate_manifest


def require(value,message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def validate_hardware(root, formal=True):
    root=Path(root).resolve()
    manifest=read(root/'evidence_manifest.json'); names=set()
    for item in manifest['artifacts']:
        path=(root/item['path']).resolve()
        require(root in path.parents and path.is_file(),'missing/escaping artifact')
        require(item['path'] not in names,'duplicate artifact')
        names.add(item['path'])
        require(path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],'artifact changed')
    require({'result.json','resolved_config.json','preflight.json','source_manifest.json'}<=names,'incomplete hardware evidence')
    result=read(root/'result.json'); config=read(root/'resolved_config.json'); sources=read(root/'source_manifest.json')
    require(result['execution_status']=='completed' and result['validation_status']=='pass','hardware not passed')
    require(result['command']=='benchmark','not a benchmark acceptance')
    require(result['source_digest']==sources['source_digest']==canonical(sources['sources']),'source identity inconsistent')
    require(result['observed_commit']==config['identity']['expected_commit']==sources['observed_commit'],'commit mismatch')
    seeds=config['workload']['seeds'];methods=config['checkpoint']['methods'];m=config['measurement']
    if formal:
        require(seeds==[41,42,43] and set(methods)==set(METHODS),'formal matrix missing seeds/methods')
        require(m['warmup']==5 and m['formal_steps']==30 and m['continue_steps']==3 and m['restore_repetitions']==3 and config['checkpoint']['interval']==3,'formal workload changed')
        require(config['identity']['allow_dirty'] is False,'formal source must be frozen')
    expected={(s,a) for s in seeds for a in methods}
    require(len(result['runs'])==len(expected) and {(r['seed'],r['method']) for r in result['runs']}==expected,'missing/duplicate hardware runs')
    fixtures={}
    def artifact(path):
        p=Path(path).resolve();require(root in p.parents and str(p.relative_to(root)) in names,'unhashed referenced result')
        return read(p)
    for row in result['runs']:
        method=row['method'];source=artifact(row['source'])
        require(source['status']=='trend_measured' and source['formal_steps']==m['formal_steps'],'source incomplete')
        require(source['deterministic']=='ON' and source['transport']=='legacy_sync','wrong execution mode')
        require(row['comparison']==comparison(method),'comparison group changed')
        key=row['seed']; f=source['initial_state_sha256']
        require(fixtures.setdefault(key,f)==f,'methods did not start from same fixture')
        processes=row['processes'];require(processes and all(p['returncode']==0 for p in processes),'failed child process')
        if method=='none':
            require(row['restore_status']=='not_applicable' and source['checkpoint_count']==0,'none produced checkpoint/restore')
            continue
        require(source['checkpoint_count']==m['formal_steps']//config['checkpoint']['interval'],'checkpoint count differs')
        require(all(x['persisted'] is True and x['state']=='PERSISTED' for x in source['checkpoints']),'non-durable checkpoint')
        restored=artifact(row['restore'])
        validate_restore_report(restored,source,m['continue_steps'])
        require(restored['status']=='pass' and restored['byte_exact'] is True and restored['verification_performed'] is True,'not byte-exact restore')
        require(restored['source_pid']==source['source_pid'] and restored['restore_pid']!=source['source_pid'],'not fresh process')
        require(restored['mandatory_integrity'] is True and restored['controls_verified'] is True,'missing mandatory integrity/control verification')
        require(restored['generation']==source['checkpoints'][-1]['generation'],'wrong restored generation')
        require(loss_matches(restored['restored_losses'],restored['source_oracle_losses'],m['continue_steps'],m['loss_rtol'],m['loss_atol']),'invalid continuation')
        measured=[]
        for index in range(m['restore_repetitions']+1):
            timing=artifact(Path(row['restore']).with_name(f'timing-{index}.json'))
            require(timing['status']=='pass' and timing['mandatory_integrity'] is True and timing['controls_verified'] is True,'timing skipped integrity')
            events=timing['timing']['events']; event={x['event']:x['monotonic_ns'] for x in events}
            require(event['restore_begin']<event['model_constructed']<event['integrity_and_apply_done']<=event['controls_verified']<=event['state_ready'],'timing boundary invalid')
            require(loss_matches(timing['restored_losses'],timing['source_oracle_losses'],m['continue_steps']),'timing continuation invalid')
            measured.append((event['state_ready']-event['restore_begin'])/1e9)
        metric=row['restore_seconds']
        require(metric['sample_count']==m['restore_repetitions'],'missing timing samples')
        require(metric['raw_values']==measured[1:] and metric['warmup']==measured[0],'timing values differ from trace')
        require(metric['mean']==sum(measured[1:])/len(measured[1:]) and metric['min']==min(measured[1:]) and metric['max']==max(measured[1:]),'timing summary differs from samples')
        for process in processes:
            require(process.get('resources'),'missing resource samples')
            require(all(x['host_bytes']<=config['runtime']['host_budget_bytes'] and x['hbm_bytes']<=config['runtime']['hbm_budget_bytes'] for x in process['resources']),'resource budget exceeded')
        resources=artifact(Path(row['restore']).with_name('restore.resources.json'))
        if method=='ours':
            require(read(root/'preflight.json')['library_sha256'] in [v for k,v in resources['loaded_libraries'].items() if k.endswith('/libnpu_nvme.so')],'loaded native binary differs')
        require(source['model_revision']==config['workload']['model_revision'],'model revision differs')
    return dict(status='pass',commit=result['observed_commit'],source_digest=result['source_digest'],seeds=seeds,methods=methods,
                library_sha256=read(root/'preflight.json')['library_sha256'],hardware_manifest_sha256=digest(root/'evidence_manifest.json'))


def validate(software,hardware):
    sw=validate_manifest(Path(software)/'evidence_manifest.json')
    require(sw['validation_status']=='pass' and sw['profile_id']=='C1','C1 software gate missing')
    hw=validate_hardware(hardware)
    sw_sources=read(Path(software)/'source_manifest.json')
    hw_sources=read(Path(hardware)/'source_manifest.json')['sources']
    require(all(hw_sources.get(name)==value for name,value in sw_sources.items()),'software and hardware source bytes differ')
    require(sw['environment']['commit']==hw['commit'],'software/hardware commits differ')
    hw.update(stage='C1',software_cases=sw['case_count'],software_manifest_sha256=digest(Path(software)/'evidence_manifest.json'),
        boundaries=['GPT-2 single rank frozen full_state, old environment only','Different raw/filesystem devices: no cross-device ranking','Legacy synchronous transport; no B2/C2, live, TP4, incremental or power-failure claim'])
    return hw


def main():
    p=argparse.ArgumentParser();p.add_argument('--software',required=True,type=Path);p.add_argument('--hardware',required=True,type=Path);p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    try: result=validate(args.software,args.hardware)
    except (ValueError,KeyError,TypeError,OSError) as e: result=dict(status='fail',error=str(e))
    with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result));return int(result['status']!='pass')

if __name__=='__main__':raise SystemExit(main())
