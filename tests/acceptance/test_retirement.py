import binascii
import json
import struct
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import pytest
from npu_nvme import StrictCheckpoint
from npu_nvme.storage.format import unpack_metadata, pack_metadata
from npu_nvme.storage.layout import METADATA_MAGIC, META_SLOT_BYTES


@pytest.mark.parametrize('kwargs', [{'strict': False}, {'world_size': 2}, {'rank_id': 1}, {'keep_last_n': 3}, {'checkpoint_slots': 2}, {'request_slots': 1}])
def test_removed_options_reject_before_framework_or_device(kwargs):
    with pytest.raises(TypeError):
        StrictCheckpoint(**kwargs)


@pytest.mark.parametrize('method', ['save','load','load_state','recover','delta_save','_persist_metadata','register_tasks','write_host_frame','cleanup','wait_for_io_completion','live_async_capability'])
def test_removed_methods_are_absent(method):
    assert not hasattr(StrictCheckpoint, method)


@pytest.mark.parametrize('module', ['direct_checkpoint','c_bindings','chunk_helpers','disk_layout','full_checkpoint_protocol','training_cell','delta_protocol','r0_pipeline','multirank_protocol','npu_nvme.checkpoint','npu_nvme.runtime.scheduler','npu_nvme.runtime.handle','npu_nvme.storage.transport'])
def test_removed_modules_are_absent(module):
    assert importlib.util.find_spec(module) is None


def test_old_envelope_is_rejected_with_valid_crc():
    body=json.dumps({'checkpoints':{}}).encode()
    raw=struct.pack('<8sIIQQII',METADATA_MAGIC,1,0,7,len(body),binascii.crc32(body)&0xffffffff,0)+body
    with pytest.raises(ValueError,match='unsupported metadata envelope'): unpack_metadata(raw.ljust(META_SLOT_BYTES,b'\0'))
    assert unpack_metadata(pack_metadata({'strict_contract':'D1','checkpoints':{}},7))[0]==7


def test_catalog_property_is_detached():
    store=object.__new__(StrictCheckpoint)
    state=SimpleNamespace(meta_dict={'checkpoints':{'x':{'generation':1}}})
    store._runtime=SimpleNamespace(commit=SimpleNamespace(state=state))
    snapshot=store.meta_dict; snapshot['checkpoints']['x']['generation']=2
    assert state.meta_dict['checkpoints']['x']['generation']==1
