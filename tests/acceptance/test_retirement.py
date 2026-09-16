import binascii
import json
import struct
from pathlib import Path
from types import SimpleNamespace
import pytest
from npu_nvme.checkpoint import DirectCheckpoint, MigrationRequired
from npu_nvme.strict_checkpoint import StrictCheckpoint
from npu_nvme.storage.format import unpack_metadata, pack_metadata
from npu_nvme.storage.layout import METADATA_MAGIC, META_SLOT_BYTES


def test_only_strict_default_and_alias():
    import direct_checkpoint
    import npu_nvme.checkpoint
    assert direct_checkpoint is npu_nvme.checkpoint
    assert DirectCheckpoint is StrictCheckpoint


@pytest.mark.parametrize('kwargs',[{'strict':False},{'world_size':2},{'rank_id':1},{'keep_last_n':3},{'checkpoint_slots':2}])
def test_legacy_options_reject_before_framework_or_device(kwargs):
    with pytest.raises(MigrationRequired): DirectCheckpoint(**kwargs)


@pytest.mark.parametrize('method',['save','load','load_state','recover','delta_save','_persist_metadata','register_tasks','write_host_frame'])
def test_old_methods_cannot_dispatch(method):
    store=object.__new__(DirectCheckpoint)
    with pytest.raises(MigrationRequired): getattr(store,method)(object())


def test_old_envelope_is_rejected_with_valid_crc():
    body=json.dumps({'checkpoints':{}}).encode()
    raw=struct.pack('<8sIIQQII',METADATA_MAGIC,1,0,7,len(body),binascii.crc32(body)&0xffffffff,0)+body
    with pytest.raises(ValueError,match='unsupported metadata envelope'): unpack_metadata(raw.ljust(META_SLOT_BYTES,b'\0'))
    assert unpack_metadata(pack_metadata({'strict_contract':'D1','checkpoints':{}},7))[0]==7


def test_catalog_property_is_detached():
    store=object.__new__(DirectCheckpoint)
    state=SimpleNamespace(meta_dict={'checkpoints':{'x':{'generation':1}}})
    store._runtime=SimpleNamespace(commit=SimpleNamespace(state=state))
    snapshot=store.meta_dict; snapshot['checkpoints']['x']['generation']=2
    assert state.meta_dict['checkpoints']['x']['generation']==1


def test_no_executable_legacy_implementations():
    root=Path(__file__).resolve().parents[2]/'python/npu_nvme'
    for name in ('runtime/worker.py','runtime/restore.py','runtime/owner.py','experimental/legacy.py','framework/weights.py','framework/restore.py'):
        assert not (root/name).exists()
