"""Export actual sample coverage and conditional activity for each physical card."""
import argparse,json,csv,hashlib,gzip,sqlite3
from pathlib import Path
from .pmu import export
from .statistics import core_conditionals

def restore_counter_inputs(archive,device):
    """Recreate the minimal read-only PMU inputs from the durable exact columns."""
    archive=Path(archive);device=Path(device);(device/'sqlite').mkdir(parents=True,exist_ok=False)
    for row in json.loads((archive/'manifest.json').read_text()):
        p=archive.parent/row['archive_path']
        if p.name.endswith('.csv.gz'):
            filename=Path(row['source_path']).name
            c=sqlite3.connect(device/'sqlite'/filename)
            c.execute('create table AICoreOriginalData(timestamp TEXT,coreid INTEGER,task_cyc INTEGER,replayid INTEGER)')
            with gzip.open(p,'rt',newline='') as f:
                r=csv.DictReader(f)
                c.executemany('insert into AICoreOriginalData values(?,?,?,0)',((x['timestamp'],int(x['coreid']),int(x['task_cyc'])) for x in r))
            c.commit();c.close()
        else:
            with gzip.open(p,'rb') as src:(device/Path(row['source_path']).name).write_bytes(src.read())

def process(run,out,kind):
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    evidence=json.loads((run/('sample-evidence.json' if kind=='training' else 'result.json')).read_text())
    reports=[]
    for rank in range(4):
        device=rank+4
        paths=list((run/'pmu').glob(f'PROF_*/device_{device}'))
        if len(paths)!=1:raise ValueError(f'expected one device sample path for {device}: {paths}')
        if kind=='training':windows=evidence['ranks'][rank]['windows']
        else:windows=[w for w in json.loads((run/'windows.json').read_text()) if w['kind']=='measured']
        # Include callback / scheduler gaps between consecutive complete steps or
        # requests in this continuous scope, without startup or collector analysis.
        scope=[dict(id='steady',begin_ns=windows[0]['begin_ns'],end_ns=windows[-1]['end_ns'])]
        target=out/f'rank{rank}';rows,quality=export(paths[0],target,scope)
        conditionals=core_conditionals(rows)
        valid=[r for r in rows if r['valid']]
        means={engine:sum(r[engine+'_activity_pct'] for r in valid)/len(valid) if valid else None for engine in ('aic','aiv')}
        report=dict(physical_device_id=device,rank=rank,run=run.name,kind=kind,windows=windows,continuous_scope=scope,
                    scope_seconds=(scope[0]['end_ns']-scope[0]['begin_ns'])/1e9,pmu_quality=quality,core_means_pct=means,conditionals=conditionals)
        report['quality_reasons']=[]
        if report['scope_seconds']<30:report['quality_reasons'].append('less_than_30_seconds')
        if quality['valid_bin_fraction']<.95:report['quality_reasons'].append('valid_sample_coverage_below_95pct')
        inherited=evidence.get('measurement_quality',evidence.get('quality'))
        if inherited!='valid':report['quality_reasons'].append(inherited)
        report['measurement_quality']='valid' if not report['quality_reasons'] else 'diagnostic'
        (target/'summary.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
        # Preserve exact replay-0 counter columns and metadata, without unrelated
        # SQLite tables/indices. These suffice to independently recompute bins.
        archive=target/'counter-inputs';archive.mkdir(exist_ok=True)
        files=[*paths[0].glob('info.json.*'),*paths[0].glob('start_info.*'),paths[0]/'sample.json']
        hashes=[]
        for p in files:
            if p.name.endswith('.done'):continue
            dst=archive/(p.name+'.gz');h=hashlib.sha256()
            with p.open('rb') as src,gzip.open(dst,'wb') as compressed:
                for block in iter(lambda:src.read(4<<20),b''):h.update(block);compressed.write(block)
            hashes.append(dict(source_path=str(p),sha256=h.hexdigest(),archive_path=str(dst.relative_to(target))))
        for filename in ['aicore.db','ai_vector_core.db']:
            p=paths[0]/'sqlite'/filename
            connection=sqlite3.connect('file:'+str(p)+'?immutable=1',uri=True)
            dst=archive/(filename+'.replay0.csv.gz')
            with gzip.open(dst,'wt',newline='') as f:
                writer=csv.writer(f);writer.writerow(['timestamp','coreid','task_cyc'])
                writer.writerows(connection.execute('select timestamp,coreid,task_cyc from AICoreOriginalData where replayid=0 order by coreid,timestamp'))
            connection.close()
            h=hashlib.sha256()
            with p.open('rb') as src:
                for block in iter(lambda:src.read(4<<20),b''):h.update(block)
            hashes.append(dict(source_path=str(p),sha256=h.hexdigest(),archive_path=str(dst.relative_to(target)),
                               selection='AICoreOriginalData replayid=0, exact timestamp/coreid/task_cyc; unused database tables and indices omitted'))
        (archive/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
        print('SAMPLE_PROCESSED',run.name,device,means,flush=True)
    result=dict(status='pass',measurement_quality='valid' if all(r['measurement_quality']=='valid' for r in reports) else 'diagnostic',evidence=evidence,devices=reports)
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--kind',choices=['training','inference'],required=True);a=p.parse_args();process(a.run,a.output,a.kind)
