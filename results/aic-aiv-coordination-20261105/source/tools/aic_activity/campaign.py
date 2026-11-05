"""Serial experiment executor; command success and evidence acceptance differ."""
import argparse,json,subprocess,time,hashlib,os,shutil
from pathlib import Path

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def run(plan,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    state=dict(plan_sha256=fingerprint,status='running',stages=[])
    statefile=out/'campaign.json'
    if statefile.exists():raise FileExistsError('use a unique campaign directory')
    write(statefile,state)
    for row in plan['stages']:
        if not all(isinstance(a,str) for a in row['argv']):raise ValueError('argv must contain strings')
        # Protect persistent storage without placing raw data on disk.
        if shutil.disk_usage('/models').free<2<<30:raise RuntimeError('/models headroom below 2 GiB')
        size=sum(p.stat().st_size for root in plan['budget_roots'] for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())
        if size>256<<30:raise RuntimeError('experiment tmpfs budget exceeded')
        stage=dict(id=row['id'],status='running',begin_ns=time.time_ns(),argv=row['argv'],cwd=row['cwd'])
        state['stages'].append(stage);write(statefile,state)
        print('START',row['id'],flush=True)
        try:
            with (out/(row['id']+'.log')).open('w') as log:
                p=subprocess.run(row['argv'],cwd=row['cwd'],env=dict(os.environ,**row.get('env',{})),stdout=log,stderr=subprocess.STDOUT,timeout=row.get('timeout_seconds',3600))
        except (OSError,subprocess.TimeoutExpired) as error:
            stage.update(status='failed',cause=str(error),end_ns=time.time_ns());state['status']='failed';write(statefile,state)
            raise
        stage.update(exit_code=p.returncode,end_ns=time.time_ns())
        if p.returncode:
            stage['status']='failed';state['status']='failed';write(statefile,state)
            raise RuntimeError(f"stage {row['id']} failed with {p.returncode}: {out/(row['id']+'.log')}")
        if row.get('report'):
            evidence=json.loads(Path(row['report']).read_text())
            if evidence.get('validation_status',evidence.get('status')) not in ('pass','valid'):
                stage['status']='failed_validation';state['status']='failed';write(statefile,state)
                raise RuntimeError(f"stage {row['id']} evidence failed: {evidence}")
            stage['report_sha256']=hashlib.sha256(Path(row['report']).read_bytes()).hexdigest()
        stage['status']='pass';write(statefile,state);print('DONE',row['id'],flush=True)
    state['status']='pass';write(statefile,state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(json.loads(a.plan.read_text()),a.output)
