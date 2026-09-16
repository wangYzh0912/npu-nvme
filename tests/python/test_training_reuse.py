import json
from pathlib import Path
import subprocess

import pytest
from npu_nvme.runtime.training_catalog import write_checked
from npu_nvme.runtime.training_reuse import verified_runs


def fixture(tmp_path,monkeypatch):
    old=tmp_path/'old';current=tmp_path/'current';previous=tmp_path/'campaign'
    for repo in (old,current):
        (repo/'python').mkdir(parents=True)
        (repo/'python/model.py').write_text('SEED=42\n')
    subprocess.run(['git','init','-q',str(old)],check=True)
    subprocess.run(['git','add','python/model.py'],cwd=old,check=True)
    identity=dict(sources={str(old):dict(commit='original',tracked_sha256='working')},extensions={})
    monkeypatch.setattr('tools.run_campaign.source_snapshot',lambda plan:identity['sources'])
    config=dict(identity={'seed':42},seq_length=128,retention=3)
    def run(name,method,start,stop,interval):
        path=previous/name;path.mkdir(parents=True)
        (path/'result.json').write_text(json.dumps(dict(execution_status='completed',
            validation_status='pass',method=method,final_step=stop)))
        (path/'run-config.json').write_text(json.dumps(dict(config,method=method,
            start_step=start,stop_step=stop,lr_horizon=24,checkpoint_interval=interval)))
        (path/'source-identity.json').write_text(json.dumps(identity))
        for rank in range(4):write_checked(path/f'rank_{rank}/training.json',dict(status='pass',final_step=stop))
        return path
    run('trajectory-none','none',0,24,4)
    for suffix,start,stop,interval in [('source8',0,8,4),('explicit4',4,8,24),
            ('latest8-to16',8,16,4),('latest16-to24',16,24,4)]:
        run('mindspore_native_save-'+suffix,'mindspore_native_save',start,stop,interval)
    return previous,current,config


def test_only_complete_method_groups_reused(tmp_path,monkeypatch):
    previous,current,config=fixture(tmp_path,monkeypatch)
    accepted,_=verified_runs(previous,config,current)
    assert len(accepted)==5
    (previous/'mindspore_native_save-latest16-to24/result.json').write_text('{}')
    accepted,rejected=verified_runs(previous,config,current)
    assert list(accepted)==['trajectory-none']
    assert any(row['method']=='mindspore_native_save' for row in rejected)


@pytest.mark.parametrize('damage',['identity','code','rank'])
def test_changed_inputs_or_completion_prevent_reuse(tmp_path,monkeypatch,damage):
    previous,current,config=fixture(tmp_path,monkeypatch)
    if damage=='identity':config['identity']['seed']=43
    if damage=='code':(current/'python/model.py').write_text('SEED=43\n')
    if damage=='rank':write_checked(previous/'trajectory-none/rank_3/training.json',dict(status='running',final_step=24))
    accepted,rejected=verified_runs(previous,config,current)
    assert not accepted and rejected[0]['method']=='none'
