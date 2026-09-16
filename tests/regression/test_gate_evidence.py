"""Exercise actual subprocess exit/skip/timeout and evidence tampering paths."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from npu_nvme.schemas.evidence import validate_manifest, validate_profile, sha256_file

ROOT = Path(__file__).resolve().parents[2]


def run(tmp_path, body, *, node='test_case', timeout=10, tier='CPU'):
    test = tmp_path/'test_probe.py'
    test.write_text(body)
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps(dict(schema_version=1,profile_id='probe',cases=[
        dict(id='probe',required=True,tier=tier,nodeid=str(test)+'::'+node,timeout_seconds=timeout)])))
    out = tmp_path/'run'
    p = subprocess.run([sys.executable,str(ROOT/'tools/run_gate.py'),'--profile',str(profile),'--out',str(out)],
                       capture_output=True,text=True,timeout=30)
    return p,out


@pytest.mark.parametrize('body,node,timeout,tier,code,status', [
    ('def test_case(): assert True\n','test_case',10,'CPU',0,('completed','pass')),
    ('def test_case(): assert False\n','test_case',10,'CPU',1,('completed','fail')),
    ('import pytest\ndef test_case(): pytest.skip("dependency")\n','test_case',10,'CPU',3,('blocked','invalid')),
    ('def test_case(): pass\n','missing',10,'CPU',3,('blocked','invalid')),
    ('import time\ndef test_case(): time.sleep(30)\n','test_case',1,'CPU',4,('aborted','invalid')),
    ('def test_case(): pass\n','test_case',10,'HW1',3,('blocked','invalid')),
])
def test_real_exit_aggregation(tmp_path,body,node,timeout,tier,code,status):
    p,out=run(tmp_path,body,node=node,timeout=timeout,tier=tier)
    assert p.returncode==code,p.stdout+p.stderr
    result=validate_manifest(out/'evidence_manifest.json')
    assert (result['execution_status'],result['validation_status'])==status


def test_empty_profile_rejected():
    with pytest.raises(ValueError):
        validate_profile(dict(schema_version=1,profile_id='empty',cases=[]))


@pytest.mark.parametrize('damage',['tamper','missing','escape','bool','zero','tier','xml'])
def test_invalid_artifacts_cannot_pass(tmp_path,damage):
    p,out=run(tmp_path,'def test_case(): pass\n')
    assert p.returncode==0,p.stderr
    manifest=out/'evidence_manifest.json'
    data=json.loads(manifest.read_text())
    result_path=out/'result.json'
    result=json.loads(result_path.read_text())
    if damage=='tamper':
        (out/'case-000/stdout.txt').write_text('changed')
    elif damage=='missing':
        (out/'case-000/stdout.txt').unlink()
    elif damage=='escape':
        data['artifacts'][0]['path']='../profile.json'
    else:
        if damage=='bool': result['cases'][0]['required']=1
        if damage=='zero': result['case_count']=0
        if damage=='tier': result['cases'][0]['tier']='HW4'
        if damage=='xml': (out/'case-000/junit.xml').write_text('<broken')
        result_path.write_text(json.dumps(result))
        for item in data['artifacts']:
            path=out/item['path'];item.update(bytes=path.stat().st_size,sha256=sha256_file(path))
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError): validate_manifest(manifest)
    p=subprocess.run([sys.executable,str(ROOT/'tools/validate_evidence.py'),str(manifest)],capture_output=True,text=True)
    assert p.returncode==2,p.stdout+p.stderr


@pytest.mark.parametrize('damage',['zero_samples','missing_samples','missing_method'])
def test_invalid_performance_metric_cannot_pass(tmp_path,damage):
    p,out=run(tmp_path,'def test_case(): pass\n');assert p.returncode==0
    result_path=out/'result.json';result=json.loads(result_path.read_text())
    metric=dict(unit='ms',sample_count=1,warmup_count=0,raw_values='samples.json',statistics=dict(method='mean'))
    (out/'samples.json').write_text('[1.0]')
    if damage=='zero_samples':metric['sample_count']=0
    if damage=='missing_samples':metric['raw_values']='absent.json'
    if damage=='missing_method':metric['statistics']={}
    result['metrics']=dict(latency=metric);result_path.write_text(json.dumps(result))
    manifest=out/'evidence_manifest.json'
    data=json.loads(manifest.read_text())
    data['artifacts'].append(dict(path='samples.json'))
    for item in data['artifacts']:
        path=out/item['path'];item.update(bytes=path.stat().st_size,sha256=sha256_file(path))
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):validate_manifest(manifest)
