import copy
import importlib.util
import json
from pathlib import Path
import pytest
from npu_nvme.cli.contracts import ROOT,canonical,digest,write,comparison

spec=importlib.util.spec_from_file_location('c1_acceptance',ROOT/'tools/validate_c1_acceptance.py')
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)


def seal(root):
    write(root/'evidence_manifest.json',dict(artifacts=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json']))


@pytest.fixture
def hardware(tmp_path):
    c=json.loads((ROOT/'config/c1/pilot.json').read_text());c['checkpoint']['methods']=['ours'];c['workload']['seeds']=[41]
    h='a'*64;commit=c['identity']['expected_commit'];sources={'train.py':h}
    write(tmp_path/'resolved_config.json',c);write(tmp_path/'source_manifest.json',dict(sources=sources,source_digest=canonical(sources),observed_commit=commit))
    write(tmp_path/'preflight.json',dict(library_sha256=h))
    directory=tmp_path/'seed-41/ours'
    source=dict(status='trend_measured',adapter='ours',source_pid=10,formal_steps=2,deterministic='ON',transport='legacy_sync',initial_state_sha256=h,
        checkpoint_count=1,checkpoints=[dict(generation=3,step=3,sha256=h,persisted=True,state='PERSISTED')],model_revision=c['workload']['model_revision'])
    restore=dict(status='pass',byte_exact=True,verification_performed=True,mandatory_integrity=True,controls_verified=True,
        source_pid=10,restore_pid=20,generation=3,checkpoint_step=3,expected_state_sha256=h,applied_state_sha256=h,
        restored_losses=[1,2,3],source_oracle_losses=[1,2,3])
    write(directory/'source.json',source);write(directory/'restore.json',restore)
    write(directory/'restore.resources.json',dict(loaded_libraries={'/test/libnpu_nvme.so':h}))
    for i in range(2):
        timing=dict(restore,timing=dict(events=[dict(event=name,monotonic_ns=ns) for name,ns in
            [('restore_begin',0),('model_constructed',1),('integrity_and_apply_done',2),('controls_verified',3),('state_ready',1000000000)]]))
        write(directory/f'timing-{i}.json',timing)
    row=dict(seed=41,method='ours',source=str(directory/'source.json'),restore=str(directory/'restore.json'),comparison=comparison('ours'),
        processes=[dict(returncode=0,resources=[dict(host_bytes=1,hbm_bytes=1)]) for _ in range(4)],
        restore_seconds=dict(sample_count=1,raw_values=[1.0],warmup=1.0,mean=1.0,min=1.0,max=1.0))
    write(tmp_path/'result.json',dict(execution_status='completed',validation_status='pass',command='benchmark',observed_commit=commit,source_digest=canonical(sources),runs=[row]))
    seal(tmp_path);return tmp_path


def test_complete_fixture(hardware):
    assert validator.validate_hardware(hardware,formal=False)['status']=='pass'


@pytest.mark.parametrize('kind',['missing_method','failed_child','missing_resource','fake_timing','wrong_binary','missing_oracle','wrong_model','wrong_commit','missing_byte_proof'])
def test_semantically_bad_evidence_rejected_even_with_new_hashes(hardware,kind):
    p=hardware/'result.json';r=json.loads(p.read_text())
    if kind=='missing_method':r['runs']=[]
    elif kind=='failed_child':r['runs'][0]['processes'][0]['returncode']=1
    elif kind=='missing_resource':r['runs'][0]['processes'][0]['resources']=[]
    elif kind=='fake_timing':r['runs'][0]['restore_seconds']['mean']=0.1
    elif kind=='wrong_commit':r['observed_commit']='b'*40
    elif kind in ('missing_oracle','missing_byte_proof'):
        p=hardware/'seed-41/ours/restore.json';r=json.loads(p.read_text());r['source_oracle_losses' if kind=='missing_oracle' else 'byte_exact']=[] if kind=='missing_oracle' else 1
    elif kind=='wrong_binary':
        p=hardware/'seed-41/ours/restore.resources.json';r={'loaded_libraries':{'/test/libnpu_nvme.so':'b'*64}}
    elif kind=='wrong_model':
        p=hardware/'seed-41/ours/source.json';r=json.loads(p.read_text());r['model_revision']='other'
    write(p,r);seal(hardware)
    with pytest.raises(ValueError):validator.validate_hardware(hardware,formal=False)


def test_pilot_cannot_pass_formal_acceptance(hardware):
    with pytest.raises(ValueError):validator.validate_hardware(hardware,formal=True)


def test_missing_artifact_rejected(hardware):
    (hardware/'seed-41/ours/restore.json').unlink()
    with pytest.raises(ValueError):validator.validate_hardware(hardware,formal=False)


@pytest.mark.parametrize('name',['libnpu_nvme.so.1.0','libnpu_nvme.so.2.0'])
def test_versioned_loaded_library_requires_exact_digest(hardware,name):
    path=hardware/'seed-41/ours/restore.resources.json'
    write(path,dict(loaded_libraries={f'/test/{name}':'a'*64}));seal(hardware)
    assert validator.validate_hardware(hardware,formal=False)['status']=='pass'
    write(path,dict(loaded_libraries={f'/test/{name}':'b'*64}));seal(hardware)
    with pytest.raises(ValueError,match='loaded native binary differs'):
        validator.validate_hardware(hardware,formal=False)
