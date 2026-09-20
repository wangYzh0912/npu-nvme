"""Regression coverage for migration and idempotent session completion."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import pytest
from npu_nvme.d2.backend import stage_config_from_config
from npu_nvme.storage.d1_image import D1Image
from npu_nvme.d2.format import Region
from npu_nvme.d2.v2_migration import migrate

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('field,value', [('offset_bytes',256<<30),('length_bytes',1<<40),('region_id','qwen'),('pci','0000:84:00.0')])
def test_validation_marker_cannot_relabel_production(field,value):
    config=json.loads((ROOT/'config/d2_validation_region.json').read_text())
    config[field]=value
    with pytest.raises(ValueError,match='physical'):stage_config_from_config(config,required_purpose='validation')


class Disk:
    def __init__(self):self.data=bytearray(80<<20);self.writes=0
    def read(self,o,n):return bytes(self.data[o:o+n])
    def write(self,o,data):self.writes+=1;self.data[o:o+len(data)]=data
    def flush(self):pass


def test_real_d1_commit_image_migration(tmp_path):
    image=tmp_path/'source.img'
    subprocess.run([sys.executable,str(ROOT/'tools/d1_image_fixture.py'),'--image',str(image)],check=True,capture_output=True)
    source=D1Image(image)
    try:
        _,record=source.select();assert record['state_step']==12
        assert source.select(8)[1]['state_step']==8
        with pytest.raises(ValueError):source.select(999)
        disk=Disk();region=Region(disk,offset=0,length=len(disk.data),retention=3);region.format(region_id='file-migration')
        kwargs=dict(source=source,record=record,destination=region,request_id='migration',identity={'source':record['manifest_sha256']},chunk_bytes=4<<20,source_layout=source.state.layout)
        receipt=migrate(**kwargs)
        fresh=Region(disk,offset=0,length=len(disk.data),retention=3);fresh.mount()
        with fresh.selected(receipt['generation']) as selected:
            assert fresh.verify_payloads(selected)
        writes=disk.writes
        assert migrate(**kwargs)['replayed'] and disk.writes==writes
        corrupted=copy.deepcopy(record);corrupted['params']['model/weight']['sha256']='0'*64
        with pytest.raises(ValueError):migrate(**dict(kwargs,record=corrupted,request_id='bad'))
        assert disk.writes==writes
    finally:source.close()
