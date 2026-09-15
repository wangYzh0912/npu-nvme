#!/usr/bin/env python3
"""Join cleanup software, hardware, native ABI and independent method restores."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from validate_d1_acceptance import validate, read, require, digest

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'none','ours','mindspore_native_save','datastates_acl','pccheck_acl','bytecheckpoint_host','fastpersist_host'}


def validate_cleanup(root):
    result = validate(root/'software', root/'h01', root/'h02', root/'lifecycle')
    require(read(root/'software/result.json')['profile_id']=='CLEANUP', 'cleanup software profile required')
    native = read(root/'native.json')
    require(native['abi']==2 and native['library_sha256']==result['library_sha256'], 'native identity differs')
    declaration = read(ROOT/'results/legacy-cleanup/native-retirement.json')
    require(set(native['exports'])==set(declaration['retained']), 'native export set differs')
    require(not (set(native['exports']) & set(declaration['removed'])), 'retired native symbol exported')
    for name in ('native-smoke','transport-batch','transport-request','inspect'):
        phase = read(root/(name+'-process.json'))
        require(phase['returncode']==0, f'{name} failed')
    benchmark = read(root/'methods-process.json')
    require(benchmark['returncode']==0, 'method matrix failed')
    methods = {}
    for path in (root/'methods').glob('*/*/result.json'):
        row = read(path)
        name = row.get('adapter')
        if name not in METHODS: continue
        require(name not in methods, 'duplicated method evidence')
        require(row['schema_version']==2 and row['project_commit']==result['commit'], 'method source differs')
        require(row['status']=='trend_measured', f'{name} source failed')
        env = read(path.parent/'environment.json')
        require(env['native_library']['sha256']==result['library_sha256'], 'method native library differs')
        if name=='none':
            require(row['restore']['status']=='not_applicable' and row['checkpoint_count']==0, 'invalid none control')
        else:
            restored = row['restore']
            require(restored['status']=='pass' and restored['verification_performed'] and restored['byte_exact'] is True,
                    f'{name} restore not verified')
            require(len(restored['restored_losses'])==len(restored['source_oracle_losses'])==3, 'continuation coverage missing')
            require(row['checkpoint_count']>0 and all(c['persisted'] and not c['failed'] for c in row['checkpoints']), 'durable checkpoint proof missing')
        methods[name] = {'result':str(path),'sha256':digest(path)}
    require(set(methods)==METHODS, 'method coverage missing')
    require(read(root/'transport-batch/result.json')['library_sha256']==result['library_sha256'], 'transport batch binary differs')
    require(read(root/'transport-request/result.json')['library_sha256']==result['library_sha256'], 'transport request binary differs')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==result['commit'], 'checkout differs from frozen acceptance')
    require(not subprocess.check_output(['git','diff','HEAD','--','python','src','include','experiments','tests','tools','config','train.py','CMakeLists.txt'],cwd=ROOT,text=True), 'source changed during acceptance')
    return dict(result, stage='implementation-cleanup', abi=2, native_exports=len(native['exports']), methods=methods)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    try: result=validate_cleanup(args.root)
    except (ValueError,KeyError,OSError,TypeError) as error: result={'status':'fail','error':str(error)}
    with args.out.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))
    return int(result['status']!='pass')

if __name__=='__main__': raise SystemExit(main())
