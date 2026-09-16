#!/usr/bin/env python3
"""Retirement requires joined D1 gates and both migrated fresh-process callers."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
from validate_d1_acceptance import validate, read, require, digest

ROOT=Path(__file__).resolve().parents[1]


def validate_retirement(root, run_id='002'):
    result=validate(root/f'software-{run_id}',root/f'h01-{run_id}',root/f'h02-{run_id}',root/f'lifecycle-{run_id}')
    callers=root/f'callers-{run_id}'
    evidence=read(callers/'result.json')
    require(evidence['status']=='pass' and not evidence['changed_sources'], 'caller gate failed/moving source')
    require(evidence['commit']==result['commit'] and evidence['library_sha256']==result['library_sha256'],
            'caller source/binary differs')
    require({p['phase'] for p in evidence['phases']}=={'single-card','prepare','source','restore'} and
            len(evidence['phases'])==4 and all(p['returncode']==0 for p in evidence['phases']), 'caller phases missing/failed')
    single=read(callers/'single-card/result.json')
    ours=read(callers/'ours/run/restore.json')
    require(single['status']=='pass' and single['loaded_state_byte_exact'] and single['loss_allclose'] and
            single['restore_verified'], 'single-card restore failed')
    require(ours['status']=='pass' and ours['byte_exact'] and ours['verification_performed'], 'Ours restore failed')
    actual, oracle = ours['restored_losses'], ours['source_oracle_losses']
    require(len(actual)==len(oracle)==3 and all(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-6+1e-5*abs(b) for a,b in zip(actual,oracle)),
            'Ours continuation missing or outside fixed tolerance')
    # Check all source bytes, including migrated experiments, against the immutable
    # Git revision rather than accepting a clean-looking result from a dirty tree.
    archive=subprocess.check_output(['git','archive',result['commit'],'python','src','include','tests','experiments'],cwd=ROOT)
    expected={}
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        for member in files:
            if member.isfile() and Path(member.name).suffix in ('.py','.c','.h'):
                expected[member.name]=hashlib.sha256(files.extractfile(member).read()).hexdigest()
    require(read(callers/'sources.json')==expected, 'caller source snapshot differs from accepted Git revision')
    result.update(stage='legacy-retirement',callers=dict(status='pass',path=str(callers),result_sha256=digest(callers/'result.json')))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run-id',default='002')
    args=parser.parse_args()
    try: result=validate_retirement(args.root,args.run_id)
    except (ValueError,KeyError,OSError,TypeError) as error: result=dict(status='fail',error=str(error))
    with args.out.open('x') as stream: json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps(result))
    return int(result['status']!='pass')

if __name__=='__main__': raise SystemExit(main())
