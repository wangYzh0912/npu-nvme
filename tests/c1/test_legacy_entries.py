import json
import subprocess
import sys
from pathlib import Path
from npu_nvme.cli.contracts import ROOT


def config(tmp_path):
    c=json.loads((ROOT/'config/c1/pilot.json').read_text())
    c['identity']['expected_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    p=tmp_path/'c.json';p.write_text(json.dumps(c));return p


def test_baseline_versioned_config_forwards(tmp_path):
    p=subprocess.run([sys.executable,'-m','experiments.baselines.repro.cli','suite','--config',str(config(tmp_path)),
        '--dry-run','--output',str(tmp_path/'run')],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    result=json.loads((tmp_path/'run/result.json').read_text())
    assert result['execution_status']=='planned' and result['validation_status']=='not_applicable'


def test_single_card_versioned_config_forwards(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'experiments/benchmarks/run_single_card_full.py'),
        '--config',str(config(tmp_path)),'--dry-run','--out',str(tmp_path/'run')],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads((tmp_path/'run/result.json').read_text())['execution_status']=='planned'


def test_unsupported_live_is_capability_code(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'experiments/benchmarks/run_single_card_full.py'),
        '--run-dir',str(tmp_path/'unused'),'--mode','live_async','--dry-run'],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==3,p.stderr
    assert not (tmp_path/'unused').exists()
