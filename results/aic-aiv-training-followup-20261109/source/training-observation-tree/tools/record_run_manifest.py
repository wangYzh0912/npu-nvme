#!/usr/bin/env python3
"""Record an exact workload invocation and source identity without running it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--repo',required=True,type=Path)
    p.add_argument('argv',nargs=argparse.REMAINDER);a=p.parse_args()
    argv=a.argv[1:] if a.argv[:1]==['--'] else a.argv
    if not argv:p.error('missing workload argv')
    repo=a.repo.resolve()
    git=lambda *args:subprocess.check_output(['git',*args],cwd=repo)
    sources={}
    for directory in ('python','scripts','experiments/training','tools','config'):
        for path in sorted((repo/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.sh','.json','.yaml'):
                sources[str(path.relative_to(repo))]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(schema_version=1,execution_status='planned',validation_status='not_applicable',
                commit=git('rev-parse','HEAD').decode().strip(),
                dirty_diff_sha256=hashlib.sha256(git('diff','HEAD','--binary')).hexdigest(),
                source_sha256=sources,invocation=dict(argv=argv,cwd=str(Path.cwd()),repo=str(repo)))
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__': main()
