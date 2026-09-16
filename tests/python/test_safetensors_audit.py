import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import pytest

spec=importlib.util.spec_from_file_location('audit',Path(__file__).resolve().parents[2]/'tools/audit_safetensors.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def save(path,header,payload):
    raw=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw+payload)


def test_hashes_cover_file_and_each_tensor(tmp_path):
    p=tmp_path/'state.safetensors'
    save(p,{'x':dict(dtype='F32',shape=[2],data_offsets=[0,8]),'step':dict(dtype='I64',shape=[],data_offsets=[8,16])},struct.pack('<ffq',1.,2.,8))
    result=audit.inspect(p)
    assert result['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    assert result['tensors'][0]['sha256']==hashlib.sha256(struct.pack('<ff',1.,2.)).hexdigest()
    assert result['tensors'][1]['scalar_value']==8


@pytest.mark.parametrize('change',['size','negative','bool','overlap','hole','trailing','dtype','truncated'])
def test_bad_container_rejected_before_payload_read(tmp_path,change):
    p=tmp_path/'state.safetensors';entry=dict(dtype='F32',shape=[1],data_offsets=[0,4]);header={'x':entry};payload=b'\0'*4
    if change=='size':entry['shape']=[2]
    if change=='negative':entry['shape']=[-1]
    if change=='bool':entry['shape']=[True]
    if change=='overlap':header['y']=dict(entry)
    if change=='hole':entry['data_offsets']=[4,8];payload=b'\0'*8
    if change=='trailing':payload+=b'\0'
    if change=='dtype':entry['dtype']='OBJECT'
    if change=='truncated':payload=payload[:-1]
    save(p,header,payload)
    with pytest.raises(ValueError):audit.inspect(p)


def test_duplicate_tensor_name_rejected(tmp_path):
    p=tmp_path/'state.safetensors'
    raw=b'{"x":{},"x":{}}';p.write_bytes(struct.pack('<Q',len(raw))+raw)
    with pytest.raises(ValueError,match='duplicate'):audit.inspect(p)
