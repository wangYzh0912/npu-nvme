"""Check that partial or inconsistent evidence cannot open the retirement gate."""
import json
from pathlib import Path
import shutil
import pytest
from tools.validate_d1_acceptance import validate

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def evidence(tmp_path):
    original=ROOT/'results/long-term-v1.3/D1'
    names=('software-002','h01-002','h02-002','lifecycle-003')
    needed={'software-002':['result.json','profile.json','source_manifest.json'],
            'h01-002':['result.json','sources.json']+[f'seed-{seed}/{file}' for seed in (41,42,43)
                      for file in ('save-source.json','restore-source.json','result.json')],
            'h02-002':['result.json','provenance.json'],
            'lifecycle-003':['result.json','sources.json']}
    for folder,files in needed.items():
        for file in files:
            dest=tmp_path/folder/file; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original/folder/file,dest)
    return tuple(tmp_path/name for name in names)


def mutate(path, change):
    data=json.loads(path.read_text()); change(data); path.write_text(json.dumps(data))


def test_completed_d1_evidence_joins(evidence):
    result=validate(*evidence)
    assert result['status']=='pass' and result['software_cases']==116


@pytest.mark.parametrize('folder,file,change',[
    (1,'result.json',lambda d:d.update(seeds=[41,42])),
    (1,'seed-43/restore-source.json',lambda d:d.update(library_sha256='0'*64)),
    (1,'result.json',lambda d:d.update(changed_sources=['src/runtime.c'])),
    (2,'result.json',lambda d:d['cases'].pop()),
    (3,'result.json',lambda d:d['phases'][0]['worker'].update(safe_process_exit=False)),
    (3,'result.json',lambda d:d['phases'][0]['worker'].update(cycles=3)),
    (0,'profile.json',lambda d:d.update(hardware_contract={})),
])
def test_partial_or_inconsistent_evidence_rejects(evidence,folder,file,change):
    mutate(evidence[folder]/file,change)
    with pytest.raises(ValueError): validate(*evidence)
