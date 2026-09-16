import json
from pathlib import Path
import sys
import pytest
from tools.run_campaign import run

def stage(tmp,name,status='pass',deps=()):
    report=tmp/(name+'.json')
    return dict(id=name,depends_on=list(deps),cwd=str(tmp),argv=[sys.executable,'-c',f'from pathlib import Path; Path({str(report)!r}).write_text({json.dumps(dict(validation_status=status))!r})'],report=str(report),timeout_seconds=5)

def test_zero_exit_failed_verdict_blocks_dependent(tmp_path):
    plan=dict(inputs=[],stages=[stage(tmp_path,'first','fail'),stage(tmp_path,'second',deps=['first'])])
    assert run(plan,tmp_path/'out',tmp_path/'lock')==1
    result=json.loads((tmp_path/'out/result.json').read_text())
    assert result['stages']['second']['status']=='blocked'
    assert not (tmp_path/'second.json').exists()

def test_resume_validates_artifact_and_inputs(tmp_path):
    plan=dict(inputs=[],stages=[stage(tmp_path,'first')])
    assert run(plan,tmp_path/'out',tmp_path/'lock')==0
    assert run(plan,tmp_path/'out',tmp_path/'lock')==0
    (tmp_path/'first.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):run(plan,tmp_path/'out',tmp_path/'lock')

def test_missing_report_is_failure(tmp_path):
    row=stage(tmp_path,'first');row['argv']=[sys.executable,'-c','pass']
    assert run(dict(inputs=[],stages=[row]),tmp_path/'out',tmp_path/'lock')==1

def test_input_drift_requires_new_attempt(tmp_path):
    f=tmp_path/'input';f.write_text('old');plan=dict(inputs=[str(f)],stages=[stage(tmp_path,'first')])
    assert run(plan,tmp_path/'out',tmp_path/'lock')==0
    f.write_text('new')
    with pytest.raises(ValueError,match='input/config changed'):run(plan,tmp_path/'out',tmp_path/'lock')
