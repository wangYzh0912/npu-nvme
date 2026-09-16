"""V2 metadata writes and rollback match fixtures captured before ownership migration."""
import ctypes
import hashlib
import json
from pathlib import Path

import pytest

from npu_nvme.runtime.commit import LegacyCommitCoordinator, LegacyCommitSpec, MetadataState
from npu_nvme.storage.layout import make_layout, SUPERBLOCK_OFFSET, META_SLOT_A_OFFSET, META_SLOT_B_OFFSET
from npu_nvme.storage.metadata import MetadataIO


class MemoryBinding:
    def __init__(self, fault):
        self.fault = fault
        self.blocks = {}
        self.trace = []
        self.flushes = 0

    def npu_nvme_sync_meta_io(self, ctx, offset, size, read, pointer):
        if read:
            ctypes.memmove(pointer, self.blocks[offset], size)
            return 0
        data = ctypes.string_at(pointer, size)
        self.trace.append(dict(operation='write', offset=offset, size=size,
                               sha256=hashlib.sha256(data).hexdigest()))
        if self.fault == 'metadata_write' and offset != SUPERBLOCK_OFFSET: return -5
        if self.fault == 'superblock_write' and offset == SUPERBLOCK_OFFSET: return -5
        self.blocks[offset] = data
        return 0

    def npu_nvme_flush(self, ctx):
        self.flushes += 1
        self.trace.append(dict(operation='flush'))
        return -5 if self.fault == 'flush'+str(self.flushes) else 0


def input_state():
    return MetadataState(layout=make_layout(16<<30, 1<<30, 4, 256<<20, 8))


def layout_items():
    return [dict(name='model/x', offset=1<<30, size=4, shape=[1], dtype='float32', sha256='a'*64)]


def summary(state, binding, error):
    return dict(trace=binding.trace, error=error,
                blocks={str(k):hashlib.sha256(v).hexdigest() for k,v in binding.blocks.items()},
                metadata=state.meta_dict,generation=state.metadata_generation,slot=state.active_meta_slot)


@pytest.mark.parametrize('fault', ['none', 'metadata_write', 'superblock_write', 'flush1', 'flush2'])
def test_v2_commit_matches_pre_migration_fixture(tmp_path, fault):
    binding=MemoryBinding(fault);state=input_state()
    coordinator=LegacyCommitCoordinator(MetadataIO(binding),state)
    spec=LegacyCommitSpec(0,1,4096,3,str(tmp_path/'meta.pkl'))
    error=None
    try:coordinator.publish_full(spec,8,layout_items())
    except RuntimeError as exc:error=str(exc)
    actual=summary(state,binding,error)
    expected=json.loads((Path(__file__).parents[1]/'fixtures/v2_metadata_commit.json').read_text())['cases'][fault]
    assert actual==expected
    if fault=='none':
        # Fresh catalog must read the committed V2 bytes through the store.
        # Initialize the unused slot with a valid older replica for the mount.
        binding.blocks[META_SLOT_A_OFFSET]=binding.blocks[META_SLOT_B_OFFSET]
        mounted=MetadataState()
        MetadataIO(binding).mount(mounted,16<<30,0)
        assert mounted.metadata_generation==1 and mounted.meta_dict==state.meta_dict
