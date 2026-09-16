import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('join',ROOT/'tools/validate_d1_acceptance.py')
join=importlib.util.module_from_spec(spec); spec.loader.exec_module(join)


def test_missing_hardware_cannot_pass(tmp_path):
    with pytest.raises(OSError): join.validate(tmp_path,tmp_path,tmp_path,tmp_path)


def test_software_pass_alone_cannot_pass(tmp_path):
    import json
    (tmp_path/'result.json').write_text(json.dumps(dict(execution_status='completed',validation_status='pass',
        profile_id='D1',changed_sources=[],cases=[])))
    (tmp_path/'profile.json').write_text(json.dumps(dict(cases=[],hardware_contract={})))
    with pytest.raises((KeyError,ValueError)): join.validate(tmp_path,tmp_path,tmp_path,tmp_path)
