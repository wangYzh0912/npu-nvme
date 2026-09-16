import json
from pathlib import Path
import pytest
from npu_nvme.cli.contracts import load_config,CapabilityBlocked
ROOT=Path(__file__).resolve().parents[2]
def test_xl_capacity_is_not_c1_ten_gib_slot_limit(tmp_path):
    c=json.loads((ROOT/'config/c1/pilot.json').read_text())
    c['workload']['model_id']='gpt2_xl';c['checkpoint']['transport']='async';c['storage']['slot_size_gb']=32
    p=tmp_path/'xl.json';p.write_text(json.dumps(c));assert load_config(p)['storage']['slot_size_gb']==32
    c['checkpoint']['transport']='legacy_sync';p.write_text(json.dumps(c))
    with pytest.raises(CapabilityBlocked):load_config(p)
