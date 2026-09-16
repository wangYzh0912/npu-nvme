import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import pytest
from npu_nvme.cli.contracts import ROOT, load_config, loss_matches, outcome, comparison
from experiments.baselines.repro import cli


@pytest.fixture
def config():
    return json.loads((ROOT/'config/c1/pilot.json').read_text())


@pytest.mark.parametrize('domain,key,value',[
    ('runtime','wait_ms',True),('runtime','wait_ms',0),('runtime','wait_ms',float('inf')),
    ('runtime','chunk_bytes',4194304),('runtime','chunk_bytes',4097),('runtime','max_requests',2),
    ('checkpoint','transport','unknown'),('checkpoint','capture','live'),('checkpoint','retention',3),
    ('storage','pci','0000:84:00.0'),('storage','offset',4096),('storage','owner','multi'),
    ('workload','seeds',[41,41]),('workload','seeds',[True]),('workload','batch',4),
    ('measurement','loss_atol',1e-3),('measurement','continue_steps',0),
    ('identity','allow_dirty','true'),('identity','expected_commit','anything'),
])
def test_config_rejects_unsupported_contract(config,tmp_path,domain,key,value):
    config[domain][key]=value
    p=tmp_path/'config.json';p.write_text(json.dumps(config))
    with pytest.raises(ValueError): load_config(p)


@pytest.mark.parametrize('change',['unknown','missing','version'])
def test_schema_is_closed(config,tmp_path,change):
    if change=='unknown': config['checkpoint']['unknown']=1
    elif change=='missing': del config['runtime']['close_ms']
    else: config['schema_version']=True
    p=tmp_path/'config.json';p.write_text(json.dumps(config))
    with pytest.raises(ValueError): load_config(p)


@pytest.mark.parametrize('actual,expected',[
    ([],[]),([1,1,1],[]),([1,1],[1,1]),([float('nan'),1,1],[1,1,1]),
    ([float('inf'),1,1],[1,1,1]),([0.000002,0,0],[0,0,0]),([1,1,1],[1,1]),
])
def test_loss_oracle_cannot_be_empty_or_relaxed(actual,expected):
    assert not loss_matches(actual,expected,3)


def test_loss_uses_each_oracle_value():
    assert loss_matches([1+1e-6,2,3],[1,2,3],3)
    assert not loss_matches([1,2,0.000002],[1,2,0],3)


@pytest.mark.parametrize('source_rc,restore_rc,restore',[
    (0,1,{'status':'pass','byte_exact':True,'verification_performed':True}),
    (1,0,{'status':'pass','byte_exact':True,'verification_performed':True}),
    (0,0,{}),(0,0,{'status':'restore_failed'}),
    (0,0,{'status':'pass','byte_exact':1,'verification_performed':True}),
    (0,0,{'status':'pass','byte_exact':True,'verification_performed':False}),
])
def test_failure_aggregation(source_rc,restore_rc,restore):
    assert outcome(dict(status='trend_measured',adapter='ours',restore=restore),source_rc,restore_rc)!=0


@pytest.mark.parametrize('report,rc',[(None,0),({'status':'restore_failed'},0),({'status':'pass','byte_exact':True,'verification_performed':True},1)])
def test_legacy_run_propagates_restore_failure(tmp_path,monkeypatch,report,rc):
    c={'results_root':str(tmp_path),'project_root':str(ROOT),'phase_timeout_seconds':1}
    monkeypatch.setattr(cli,'env_snapshot',lambda c:{})
    monkeypatch.setattr(cli.ADAPTERS['ours'],'preflight',lambda c:{'status':'ready'})
    def process(argv,**kwargs):
        run=Path(argv[argv.index('--run-dir')+1])
        if 'source' in argv:
            (run/'source.json').write_text(json.dumps({'status':'trend_measured','adapter':'ours'}))
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        if report is not None: (run/'restore.json').write_text(json.dumps(report))
        return SimpleNamespace(returncode=rc,stdout='',stderr='')
    monkeypatch.setattr(cli.subprocess,'run',process)
    assert cli.run_one(c,'ours')['status']!='trend_measured'


def test_continue_does_not_change_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'load_config',lambda p:{'results_root':str(tmp_path)})
    calls=[]
    def run(c,name): calls.append(name);return {'status':'roundtrip_failed','adapter':name}
    monkeypatch.setattr(cli,'run_one',run)
    assert cli.main(['suite','--config','unused','--all','--continue-on-failure'])==1
    assert len(calls)==len(cli.ADAPTERS)
    assert json.loads((tmp_path/'suite.json').read_text())['failures']==len(calls)


def test_device_free_cli_import_and_dry_run(tmp_path):
    code='''
import sys, importlib.abc
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mindspore','numpy','ctypes'):
            raise AssertionError('device dependency imported: '+fullname)
sys.meta_path.insert(0,Guard())
from npu_nvme.cli.training import main
raise SystemExit(main(sys.argv[1:]))
'''
    config=json.loads((ROOT/'config/c1/pilot.json').read_text())
    config['identity']['expected_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    proc=subprocess.run([sys.executable,'-c',code,'preflight','--config',str(path),'--dry-run','--out',str(tmp_path/'run')],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    row=json.loads((tmp_path/'run/result.json').read_text())
    assert row['execution_status']=='planned' and row['validation_status']=='not_applicable' and not row['device_initialized']


def test_storage_comparison_groups():
    assert comparison('ours')['comparison_group']!=comparison('mindspore_native_save')['comparison_group']
    assert comparison('bytecheckpoint_host')['comparison_group']==comparison('mindspore_native_save')['comparison_group']
    assert comparison('none')['comparable'] is False


def test_format_missing_flush_fails_before_device(monkeypatch):
    import format_npu_disk as tool
    monkeypatch.setattr(tool,'lib',SimpleNamespace())
    with pytest.raises(RuntimeError,match='flush'): tool.format_disk('0000:83:00.0',force=True)


def test_inspect_read_failure_does_not_return_success(monkeypatch):
    import inspect_npu_disk as tool
    calls=[]
    monkeypatch.setattr(tool,'lib',SimpleNamespace(npu_nvme_init=lambda *a:0,
        npu_nvme_sync_meta_io=lambda *a:-1,npu_nvme_cleanup=lambda *a:calls.append('cleanup')))
    with pytest.raises(RuntimeError,match='Superblock'):tool.inspect_disk('0000:83:00.0')
    assert calls==['cleanup']


def test_export_is_explicitly_retired_without_pickle():
    import export_model
    with pytest.raises(RuntimeError,match='retired'):export_model.export_to_heap()


def test_worker_wait_has_deadline():
    import queue,time
    from experiments.baselines.repro.adapters.worker_semantic import WorkerSemanticAdapter
    worker=object.__new__(WorkerSemanticAdapter);worker._lines=queue.Queue()
    with pytest.raises(TimeoutError,match='deadline'):worker._line(time.monotonic())
