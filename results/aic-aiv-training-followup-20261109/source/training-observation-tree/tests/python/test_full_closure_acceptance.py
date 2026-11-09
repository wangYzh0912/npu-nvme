"""Exercise the hardware acceptance driver without starting hardware workers."""
import json
from types import SimpleNamespace

import pytest

from tools import full_checkpoint_acceptance as driver


def setup(monkeypatch,tmp_path,*,idle=True):
    region=tmp_path/'region.json'
    region.write_text(json.dumps(dict(pci='0000:83:00.0',offset_bytes=256<<30,
        length_bytes=1<<40,npu_ids=[0,1,2,3])))
    model=tmp_path/'model';model.mkdir()
    monkeypatch.setattr(driver,'resolve',lambda raw,root:dict(raw,model=str(model),region=str(region)))
    monkeypatch.setattr(driver.shutil,'disk_usage',lambda path:SimpleNamespace(free=100<<30))
    monkeypatch.setattr(driver.subprocess,'check_output',
        lambda *args,**kwargs:'\n'.join(f'No running processes found in NPU {r}' for r in range(8)) if idle else 'occupied')
    monkeypatch.setattr(driver.os,'geteuid',lambda:0)
    monkeypatch.setattr(driver,'source_identity',lambda root:{'fixture':'sources'})
    return tmp_path/'acceptance'


def test_occupied_devices_do_not_start_any_run(monkeypatch,tmp_path):
    output=setup(monkeypatch,tmp_path,idle=False)
    def unexpected(*args,**kwargs):raise AssertionError('started a worker')
    monkeypatch.setattr(driver.training,'run_fit',unexpected)
    assert driver.acceptance({},output)==3
    assert json.loads((output/'result.json').read_text())['status']=='blocked'


def test_targeted_chain_and_interrupted_publication_recovery(monkeypatch,tmp_path):
    output=setup(monkeypatch,tmp_path);calls=[];physical=[4,5,6,7]
    def fit(config,path,*,resume=False,restore='latest'):
        calls.append((path.name,config,resume,restore))
        path.mkdir()
        start={'ours-explicit4':4,'ours-latest8-to16':8,'ours-latest16-to24':16,'ours-publication-recovery':4}.get(path.name,0)
        driver.training.write(path/'run-config.json',dict(config,start_step=start))
        fault=path.name=='ours-publication-interruption'
        driver.training.write(path/'result.json',dict(validation_status='fail' if fault else 'pass'))
        if fault:
            generation=path/'checkpoints/generation-000000000001'
            driver.catalog.write_checked(generation/'publication-fault.json',{'injected':True})
            driver.training.write(path/'owner/result.json',dict(status='pass',closed=True))
        if path.name=='ours-publication-recovery':
            generation=output/'ours-publication-interruption/checkpoints/generation-000000000001'
            driver.catalog.write_checked(generation/'checkpoint.json',{'recovered':True})
            driver.training.write(path/'reconciliation.json',{'recovered':[1]})
            driver.training.write(path/'restore-selection.json',{'generation':1})
        return int(fault)
    monkeypatch.setattr(driver.training,'run_fit',fit)
    monkeypatch.setattr(driver,'compare',lambda *args:{'status':'pass'})
    assert driver.acceptance({'npu_ids':physical},output)==0
    assert len(calls)==7 and len(json.loads((output/'result.json').read_text())['comparisons'])==5
    assert calls[2][2:] == (True,'1') and calls[2][1]['checkpoint_interval']==24
    assert [row[1]['stop_step'] for row in calls]==[24,8,8,16,24,8,8]
    assert calls[-1][2:] == (True,'latest')
    assert all(row[1]['npu_ids']==physical for row in calls)


def test_acceptance_checks_only_selected_physical_devices(monkeypatch,tmp_path):
    output=setup(monkeypatch,tmp_path)
    devices='\n'.join([*(f'No running processes found in NPU {r}' for r in range(4)),
                        '| 4 0 111 python3 |',*(f'No running processes found in NPU {r}' for r in range(5,8))])
    monkeypatch.setattr(driver.subprocess,'check_output',lambda *args,**kwargs:devices)
    monkeypatch.setattr(driver.training,'run_fit',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('run started')))
    assert driver.acceptance({'npu_ids':[4,5,6,7]},output)==3
    result=json.loads((output/'result.json').read_text())
    assert result['status']=='blocked' and 'NPU [4, 5, 6, 7]' in result['reason']


def test_driver_stops_on_first_failed_run(monkeypatch,tmp_path):
    output=setup(monkeypatch,tmp_path);calls=[]
    def fit(config,path,**kwargs):
        calls.append(path.name);path.mkdir()
        driver.training.write(path/'result.json',dict(validation_status='fail'))
        return 1
    monkeypatch.setattr(driver.training,'run_fit',fit)
    with pytest.raises(RuntimeError,match='acceptance run failed'):
        driver.acceptance({},output)
    assert calls==['trajectory-none']


def test_fault_entry_replaces_only_training_worker(monkeypatch):
    commands=[]
    monkeypatch.setattr(driver.subprocess,'Popen',lambda argv:commands.append(argv))
    original=driver.subprocess.Popen
    with driver.publication_fault():
        driver.subprocess.Popen(['python','owner.py'])
        driver.subprocess.Popen(['msrun','python','-m','npu_nvme.workloads.qwen','--config','config.json'])
    assert driver.subprocess.Popen is original
    assert commands[0]==['python','owner.py']
    assert commands[1][2].endswith('tests/hardware/qwen_publication_fault.py')
    assert commands[1][3:]==['--config','config.json']


def converted(run, files):
    directory=run/'training/qwen3_ms_converted_weight'
    directory.mkdir(parents=True)
    for name,contents in files.items():
        (directory/name).write_bytes(contents)
    return directory


def test_verified_shards_are_linked_without_sharing_other_run_files(tmp_path):
    first,second=tmp_path/'first',tmp_path/'second'
    files={'model.safetensors':b'weights','param_name_map.json':b'{}','run.log':b'first'}
    source=converted(first,files)
    target=converted(second,dict(files,**{'run.log':b'second'}))
    assert driver.reuse_converted_weights(first,second)==2
    assert (source/'model.safetensors').samefile(target/'model.safetensors')
    assert (source/'param_name_map.json').samefile(target/'param_name_map.json')
    assert (target/'run.log').read_bytes()==b'second'
    assert not (source/'run.log').samefile(target/'run.log')
    assert driver.reuse_converted_weights(first,second)==0


def test_any_digest_mismatch_leaves_all_candidates_untouched(tmp_path):
    first,second=tmp_path/'first',tmp_path/'second'
    source=converted(first,{'a.safetensors':b'same','b.safetensors':b'original'})
    target=converted(second,{'a.safetensors':b'same','b.safetensors':b'changed'})
    assert driver.reuse_converted_weights(first,second)==0
    assert not (source/'a.safetensors').samefile(target/'a.safetensors')
    assert (target/'b.safetensors').read_bytes()==b'changed'


def test_incomplete_conversion_is_not_linked(tmp_path):
    first,second=tmp_path/'first',tmp_path/'second'
    source=converted(first,{'a.safetensors':b'same','b.safetensors':b'extra'})
    target=converted(second,{'a.safetensors':b'same'})
    assert driver.reuse_converted_weights(first,second)==0
    assert not (source/'a.safetensors').samefile(target/'a.safetensors')


def test_space_admission_uses_output_filesystem_and_two_caches(monkeypatch,tmp_path):
    model=tmp_path/'model';model.mkdir()
    (model/'a.safetensors').write_bytes(b'a'*16)
    (model/'b.safetensors').write_bytes(b'b'*16)
    output=tmp_path/'output';output.mkdir()
    required=(8<<30)+64
    free=required-1
    def usage(path):
        assert path==output
        return SimpleNamespace(free=free)
    monkeypatch.setattr(driver.shutil,'disk_usage',usage)
    with pytest.raises(OSError,match='two converted model caches'):
        driver.admit_conversion_space(model,output)
    free=required
    driver.admit_conversion_space(model,output)
