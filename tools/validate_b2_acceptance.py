"""Join final transfer, software, faults and training evidence without phase claims by inference."""
import hashlib
import json
from pathlib import Path
import subprocess


def read(p):return json.loads(Path(p).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)


def matrix(directory,*,copy=False):
    directory=Path(directory);result=read(directory/'result.json');source=read(directory/'source.json')
    require(result['status']=='pass','transfer matrix failed')
    expected={(kind,chunk*1024**2,depth) for kind in ('hbm','host') for chunk in (1,4,16) for depth in (1,4,8,16,32,64)}
    observed=set();binaries=set();environments=set()
    for row in result['runs']:
        r=row['result'];key=(r['kind'],r['chunk'],r['depth'])
        require(key not in observed,'duplicate transfer configuration');observed.add(key)
        require(row['returncode']==0 and r['status']=='pass' and r['byte_exact'] is True and r['checksums_exact'] is True,'bad transfer result')
        require(r['safe_process_exit'] is True and r['close_rc']==0,'unsafe transfer owner exit')
        require(r['capabilities']['pipe_depth']==r['depth'] and r['capabilities']['chunk_size']==r['chunk'],'effective geometry differs')
        require(r['stats']['dma_inflight_peak']<=r['depth'] and r['stats']['dma_inflight']==0,'DMA resources exceed bounds')
        require(r['stats']['nvme_outstanding']==0 and r['stats']['async_event_query_error_count']==0 and r['stats']['stream_sync_fallback_count']==0,'unexpected transfer recovery')
        if r['kind']=='hbm':
            require(r['stats']['async_dma_submit_count']>0,'HBM did not use async')
            if copy:require(r.get('copy_only_byte_exact') is True,'copy-only direction not verified')
        require(r['loaded_libraries'],'missing actual loaded libraries')
        loaded=[(p,h) for p,h in r['loaded_libraries'].items() if '/libnpu_nvme.so' in p]
        require(any(h==r['library_sha256'] for p,h in loaded),'loaded transfer binary differs')
        binaries.add(r['library_sha256']);environments.add(r['python'])
    require(observed==expected,'incomplete transfer matrix')
    require(len(binaries)==len(environments)==1,'mixed matrix environments/binaries')
    return dict(commit=source['commit'],sources=source['source_files'],binary=next(iter(binaries)),cases=len(observed))


def read_faults(directory):
    directory=Path(directory);result=read(directory/'result.json');source=read(directory/'source.json')
    require(result['status']=='pass' and not result['changed_sources'],'read fault source/result invalid')
    cases={'checksum','submit','completion','record','query','late','record-quarantine','query-quarantine'}
    expected={(case,phase) for case in cases for phase in (case,'verify')}
    observed=set()
    for row in result['phases']:
        key=(row['case'],row['phase']);require(key not in observed,'duplicate read fault phase');observed.add(key)
        r=row['worker'];require(row['returncode']==0 and r['status']=='pass' and r['safe_process_exit'] is True,'bad fault phase')
        if 'quarantine' in row['case'] and row['phase']!='verify':
            require(r['driver_stop_rc']==0 and r['production_cleanup_performed'] is False,'quarantine lacked independent stop proof')
            require(r['snapshot']['retained_slots'],'quarantine did not retain buffers')
    require(observed==expected,'missing read fault direction/case')
    return dict(commit=source['commit'],sources=source['files'],binary=source['library_sha256'],cases=len(cases))


def compare_transfer_sources(a,b):
    # Test/probe implementation may evolve between pilot runs; formal join must
    # use identical final core and bindings. Never equate differing C binaries.
    for name,digest in a['sources'].items():
        if name.startswith(('src/','include/','python/')):require(b['sources'].get(name)==digest,'core source differs: '+name)


def join(root,software):
    from npu_nvme.schemas.evidence import validate_manifest
    from tools.validate_c1_acceptance import validate_hardware
    root=Path(root);software=Path(software)
    sw=validate_manifest(software/'evidence_manifest.json')
    require(sw['profile_id']=='B2' and sw['validation_status']=='pass' and not sw['changed_sources'],'software invalid')
    h01=validate_hardware(root/'h01-001')
    checks={}
    for env in ('old','candidate'):
        checks[env]=matrix(root/f'scheduled-{env}-001')
        faults=read_faults(root/f'read-faults-{env}-001')
        compare_transfer_sources(checks[env],faults)
        require(checks[env]['binary']==faults['binary'],'fault binary mismatch')
    compare_transfer_sources(checks['old'],checks['candidate'])
    require(checks['old']['binary']==h01['library_sha256'],'H01 binary mismatch')
    sources=read(root/'h01-001'/'source_manifest.json')['sources']
    core={k:v for k,v in checks['old']['sources'].items() if k.startswith(('src/','include/','python/'))}
    require(all(sources.get(k)==v for k,v in core.items()),'H01 core differs')
    for name,expected in core.items():
        if name.startswith(('src/','include/')):
            raw=subprocess.check_output(['git','show',sw['environment']['commit']+':'+name],cwd=Path(__file__).resolve().parents[1])
            require(hashlib.sha256(raw).hexdigest()==expected,'C_IMPL source commit differs: '+name)
    sw_sources=read(software/'source_manifest.json')
    # Gate runner records Python; C_IMPL additionally records compiled sources.
    require(all(sw_sources.get(k)==v for k,v in core.items() if k.startswith('python/')),'software Python differs')
    h02=read(root/'h02-001'/'result.json');prov=read(root/'h02-001'/'provenance.json')
    require(h02['validation_status']=='pass' and h02['case_count']==11,'H02 incomplete')
    require(prov['binary_sha256']==checks['old']['binary'],'H02 binary mismatch')
    for c in h02['cases']:
        for phase in ('fault','verify'):
            expected=({'before_data_complete':86,'before_metadata_commit':87}.get(c['case'],0) if phase=='fault' else 0)
            require(c[phase]['returncode']==expected and c[phase]['status']=='pass','H02 child failure')
    lifecycle=read(root/'lifecycle-001'/'result.json')
    require(lifecycle['status']=='pass' and not lifecycle['changed_sources'] and lifecycle['library_sha256']==checks['old']['binary'],'lifecycle invalid')
    require(len(lifecycle['phases'])==10,'lifecycle coverage incomplete')
    for phase in lifecycle['phases']:
        require(phase['returncode']==0 and phase['worker']['status']=='pass' and phase['worker']['safe_process_exit'] is True,'unsafe lifecycle phase')
    return dict(status='pass',stage='B2',h01=h01,matrices={e:{k:v for k,v in r.items() if k!='sources'} for e,r in checks.items()},
                core_sources=core,software_commit=sw['environment']['commit'],software_tests=sum(c['counts']['tests'] for c in sw['cases']),
                source_policy='Exact core hashes and per-environment loaded binary hashes; validator-only versioned SONAME fix at 9084a54 separately tested.',
                boundaries=['Single rank frozen H01; no C2 default async claim','Controlled injected failures; no power-loss or automatic takeover claim','Old/candidate transfer environments independently built; no driver promotion'])

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--software',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=join(args.root,args.software)
    with args.out.open('x') as output:json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='core_sources'}))
