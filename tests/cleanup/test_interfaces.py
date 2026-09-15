import ctypes
from pathlib import Path
from types import SimpleNamespace
import pytest
from npu_nvme.storage import bindings
from experiments.baselines.repro import entry


class Function:
    def __init__(self, value=0): self.value = value
    def __call__(self, *args): return self.value


@pytest.mark.parametrize('version', [0, 1, 3])
def test_wrong_abi_rejects_before_acl(monkeypatch, version):
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: SimpleNamespace(npu_nvme_get_abi_version=Function(version)))
    monkeypatch.setattr(bindings, 'load_acl', lambda *a: pytest.fail('ACL loaded before ABI validation'))
    with pytest.raises(RuntimeError, match='native ABI'):
        bindings.load_backend('/tmp/abi-test.so')


def test_missing_required_symbol_rejects_before_acl(monkeypatch):
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: SimpleNamespace(npu_nvme_get_abi_version=Function(2)))
    monkeypatch.setattr(bindings, 'load_acl', lambda *a: pytest.fail('ACL loaded before symbol validation'))
    with pytest.raises(AttributeError): bindings.load_backend('/tmp/abi-test.so')


def test_one_adapter_per_method():
    assert set(entry.ADAPTERS) == {'none','ours','mindspore_native_save','datastates_acl','pccheck_acl','bytecheckpoint_host','fastpersist_host'}


def test_continue_on_failure_still_returns_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, 'load_config', lambda p: {'methods':{},'results_root':str(tmp_path)})
    monkeypatch.setattr(entry, 'prepare_fixture', lambda *a: None)
    calls = []
    def run(config, name):
        calls.append(name)
        return {'status':'restore_failed'}
    monkeypatch.setattr(entry, 'run_one', run)
    assert entry.main(['benchmark','--config','unused','--all','--continue-on-failure']) == 1
    assert set(calls) == set(entry.ADAPTERS)


@pytest.mark.parametrize('returncode,restore_status', [(1,'pass'), (0,'restore_failed')])
def test_restore_process_failure_cannot_be_reported_as_success(monkeypatch, tmp_path, returncode, restore_status):
    import json
    monkeypatch.setattr(entry, 'env_snapshot', lambda config: {})
    monkeypatch.setattr(entry, 'adapter_evidence', lambda name: '')
    monkeypatch.setattr(entry.ADAPTERS['mindspore_native_save'], 'preflight', lambda config: {'status':'ready'})
    def run(argv, **kwargs):
        out = Path(argv[argv.index('--run-dir')+1])
        if argv[2] == '_source':
            (out/'source.json').write_text(json.dumps({'status':'trend_measured'}))
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        (out/'restore.json').write_text(json.dumps({'status':restore_status}))
        return SimpleNamespace(returncode=returncode, stdout='', stderr='')
    monkeypatch.setattr(entry.subprocess, 'run', run)
    result = entry.run_one({'results_root':str(tmp_path),'timeout_seconds':1}, 'mindspore_native_save')
    assert result['status'] == 'restore_failed'


def test_inspection_validates_catalog_without_writes(monkeypatch):
    from npu_nvme.storage import inspection
    from npu_nvme.storage.layout import make_layout
    calls = []
    def mount(state, capacity, rank):
        state.layout = make_layout(total_bytes=1<<30, full_slot_bytes=1<<20, full_slot_count=3, delta_slot_bytes=4096, delta_slot_count=1)
        state.meta_dict = {'strict_contract':'D1','catalog_revision':0,'checkpoints':{}}
    class Transport:
        total_bytes = 1<<30
        metadata = SimpleNamespace(mount=mount)
        def __init__(self,*a,**kw): pass
        def close(self, timeout): calls.append('close')
        def write(self,*args): pytest.fail('inspection wrote payload')
    monkeypatch.setattr(inspection,'load_backend',lambda p:None)
    monkeypatch.setattr(inspection,'FullTransport',Transport)
    result = inspection.inspect_disk('0000:83:00.0')
    assert result['status']=='pass' and not result['payload_verified']
    assert result['retained_generations']==[] and calls==['close']
