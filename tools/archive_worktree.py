#!/usr/bin/env python3
"""Archive a dirty worktree without changing its branch, index or files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

PAYLOAD_SUFFIXES={'.npy','.npz','.safetensors','.ckpt','.pth','.pt','.distcp','.bin','.pkl'}
SKIP_DIRS={'.git','__pycache__','.pytest_cache','build','build_out','node_modules','.venv','venv',
           'kernel_meta','compiler-cache','models','checkpoint_download'}


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):result.update(block)
    return result.hexdigest()


def archive(root,destination,ref,*,publish=False,snapshot_parent=None):
    root=Path(root).resolve();destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    def git(*args,env=None,input=None):
        return subprocess.check_output(['git','-C',str(root),*args],env=env,input=input)
    head=git('rev-parse','HEAD').decode().strip()
    if (destination/'manifest.json').exists():
        previous=json.loads((destination/'manifest.json').read_text())
        if previous['source']!=str(root) or previous['head']!=head or previous['archive_ref']!=ref:
            raise ValueError('existing archive identity differs')
    candidates=set(os.fsdecode(p) for p in git('ls-files','-z','--cached','--others','--exclude-standard').split(b'\0') if p)
    # Logs and planning documents may be ignored but are required archive evidence.
    for directory in ('results','docs'):
        for current,dirs,files in os.walk(root/directory,followlinks=False):
            dirs[:]=[name for name in dirs if name not in SKIP_DIRS and not (Path(current)/name).is_symlink()]
            for name in files:candidates.add(str((Path(current)/name).relative_to(root)))
    small=[];payloads=[];links=[]
    for name in sorted(candidates):
        path=root/name
        if path.name=='.sudo_pw' or path.name.startswith('.env') and path.name!='.env.example':continue
        if path.is_symlink():links.append(dict(path=name,target=os.readlink(path)));continue
        if not path.is_file():continue
        size=path.stat().st_size
        if path.suffix.lower() in PAYLOAD_SUFFIXES or size>=100<<20:
            target=destination/'payloads'/name;target.parent.mkdir(parents=True,exist_ok=True)
            checksum=digest(path)
            if not target.exists():shutil.copy2(path,target)
            if digest(target)!=checksum:raise ValueError('archive copy checksum differs: '+name)
            payloads.append(dict(source=str(path),archive=str(target),bytes=size,sha256=checksum))
        else:small.append(name)
    manifest=dict(source=str(root),head=head,archive_ref=ref,files_in_git=len(small),
                  payload_bytes=sum(row['bytes'] for row in payloads),payloads=payloads,symlinks=links)
    if snapshot_parent:
        manifest['snapshot_parent']=git('rev-parse',snapshot_parent).decode().strip()
        manifest['original_history']='external local Git bundle; original HEAD retained in this manifest'
    manifest_path=destination/'manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='npu-archive-index-',dir=destination) as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/'index'))
        git('read-tree',head,env=env)
        git('add','-u',env=env)
        # Only external manifests represent large generated payloads in the new snapshot.
        excluded=[str(Path(row['source']).relative_to(root)) for row in payloads]
        if excluded:
            git('update-index','--force-remove','-z','--stdin',env=env,
                input=b''.join(os.fsencode(name)+b'\0' for name in excluded))
        readable=[]
        for name in small:
            path=root/name
            if os.access(path,os.R_OK):readable.append(name);continue
            password=Path('/home/user7/npu-nvme/.sudo_pw').read_bytes().rstrip(b'\r\n')+b'\n'
            content=subprocess.check_output(['sudo','-S','-p','','cat','--',str(path)],input=password)
            blob=git('hash-object','-w','--stdin',input=content).decode().strip()
            mode='100755' if path.stat().st_mode&0o111 else '100644'
            git('update-index','--add','--cacheinfo',mode,blob,name,env=env)
        for offset in range(0,len(readable),100):git('add','-f','--',*readable[offset:offset+100],env=env)
        blob=git('hash-object','-w','--stdin',input=manifest_path.read_bytes()).decode().strip()
        git('update-index','--add','--cacheinfo','100644',blob,'archive-manifest.json',env=env)
        tree=git('write-tree',env=env).decode().strip()
        commit=git('commit-tree',tree,'-p',manifest.get('snapshot_parent',head),input=('Archive Qwen baseline predecessor '+root.name+'\n').encode()).decode().strip()
    git('update-ref',ref,commit)
    manifest['archive_commit']=commit
    if publish:
        subprocess.run(['git','-C',str(root),'push','origin',ref+':'+ref],check=True)
        remote=git('ls-remote','origin',ref).decode().split()
        if not remote or remote[0]!=commit:raise RuntimeError('remote archive verification failed')
        manifest['remote_verified']=True
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--ref',required=True)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--snapshot-parent',help='remote-safe parent; original history must be preserved in local bundle')
    args=parser.parse_args()
    if not args.ref.startswith('refs/heads/archive/qwen-baseline-20260916/'):
        parser.error('use the approved archive namespace')
    result=archive(args.worktree,args.destination,args.ref,publish=args.publish,snapshot_parent=args.snapshot_parent)
    print(json.dumps({key:value for key,value in result.items() if key not in ('payloads','symlinks')},indent=2))


if __name__=='__main__':main()
