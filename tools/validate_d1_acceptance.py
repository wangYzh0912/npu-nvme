#!/usr/bin/env python3
"""Join software, strict H01 and H02 evidence; fail closed on missing identity/cases."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)


def validate(software, h01, h02, lifecycle):
    sw, h, fault, life = [read(p/'result.json') for p in (software,h01,h02,lifecycle)]
    require(sw['execution_status']=='completed' and sw['validation_status']=='pass', 'software failed')
    require(sw['profile_id'] in ('D1','CLEANUP') and not sw['changed_sources'], 'wrong/moving software profile')
    require(digest(software/'profile.json')==sw['profile_sha256'], 'software profile hash differs')
    require(digest(software/'source_manifest.json')==sw['environment']['source_manifest_sha256'],
            'software source manifest hash differs')
    profile=read(software/'profile.json')
    require({x['id'] for x in sw['cases']}=={x['id'] for x in profile['cases']}, 'incomplete software cases')
    for row in sw['cases']:
        require(row['validation_status']=='pass' and row['counts']['tests']>0 and
                not any(row['counts'][k] for k in ('skipped','failures','errors')), 'invalid software case')
    contract=profile['hardware_contract']
    commit=sw['environment']['commit']
    sources=read(software/'source_manifest.json')
    require(h['status']=='pass' and not h['changed_sources'], 'H01 failed/moving source')
    require(sorted(h['seeds'])==sorted(contract['h01_seeds']), 'H01 seeds missing')
    require(len(h['phases'])==2*len(contract['h01_seeds']), 'H01 phases missing')
    libraries=set()
    for seed in contract['h01_seeds']:
        run=h01/f'seed-{seed}'
        for phase in ('save','restore'):
            require(sum(p['seed']==seed and p['phase']==phase and p['returncode']==0 for p in h['phases'])==1,
                    'H01 phase failed/duplicated')
            provenance=read(run/(phase+'-source.json'))
            require(provenance['commit']==commit, 'H01 commit differs')
            libraries.add(provenance['library_sha256'])
        result=read(run/'result.json')
        require(result['status']=='pass' and result['ready'] and result['loaded_state_byte_exact'] and
                result['controls_byte_exact'] and result['final_state_comparison']['allclose'] and
                result['continuation_steps']==contract['continuation_steps'], 'H01 restore contract failed')
    for folder, name in ((h01,'sources.json'),(lifecycle,'sources.json')):
        manifest=read(folder/name)
        require(bool(manifest) and all(sources.get(k)==v for k,v in manifest.items()), 'hardware source differs')
        prefixes = ('python/','src/','include/','tests/','experiments/') if 'experiments/training/full_fixture.py' in sources else ('python/','src/','include/','tests/')
        required={k for k in sources if k.startswith(prefixes) and Path(k).suffix in ('.py','.c','.h')}
        require(set(manifest)==required, 'hardware source coverage differs')
    require(fault['status']=='pass' and fault['execution_status']=='completed', 'H02 failed')
    expected={'acl_copy','event_query','event_record','nvme_submit','nvme_completion','metadata_write',
              'flush','timeout','request_ring_busy','before_data_complete','before_metadata_commit'}
    require({r['case'] for r in fault['cases']}==expected and len(fault['cases'])==len(expected), 'H02 cases missing')
    require(all(r['status']=='pass' and r['verify']['status']=='pass' and r['fault']['status']=='pass'
                for r in fault['cases']), 'H02 phase failed')
    provenance=read(h02/'provenance.json')
    require(provenance['commit']['stdout'].strip()==commit, 'H02 commit differs')
    require(not provenance['dirty_diff']['stdout'].strip(), 'H02 dirty checkout')
    libraries.add(provenance['binary_sha256'])
    require(life['status']=='pass' and not life['changed_sources'], 'lifecycle failed')
    require(len(life['phases'])==2*len(contract['lifecycle_cases']), 'lifecycle phases missing')
    for case in contract['lifecycle_cases']:
        for phase in (case,'verify'):
            rows=[p for p in life['phases'] if p['case']==case and p['phase']==phase]
            require(len(rows)==1 and rows[0]['returncode']==0 and rows[0]['worker']['status']=='pass'
                    and rows[0]['worker']['safe_process_exit'], 'lifecycle unsafe/failed')
            if case==phase=='repeat': require(rows[0]['worker']['cycles']>=contract['repeat_cycles'], 'too few reopen cycles')
    libraries.add(life['library_sha256'])
    require(len(libraries)==1, 'hardware binaries differ')
    return dict(status='pass',commit=commit,library_sha256=libraries.pop(),software_cases=sw['case_count'],
                h01_seeds=contract['h01_seeds'],h02_cases=len(expected),lifecycle_cases=contract['lifecycle_cases'],
                limitations=['Controlled software fault injection; no production quarantine recovery or power-failure guarantee.',
                             'H01 covers this GPT-2 single-card workload and fixed tolerances.'],
                evidence=[dict(path=str(p),result_sha256=digest(p/'result.json')) for p in (software,h01,h02,lifecycle)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('software','h01','h02','lifecycle','out'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try: result=validate(args.software,args.h01,args.h02,args.lifecycle)
    except (ValueError,KeyError,OSError,TypeError) as error: result=dict(status='fail',error=str(error))
    with args.out.open('x') as stream: json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps(result))
    return int(result['status']!='pass')

if __name__=='__main__': raise SystemExit(main())
