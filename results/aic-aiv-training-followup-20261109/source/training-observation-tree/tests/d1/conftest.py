from copy import deepcopy
import hashlib
from types import SimpleNamespace
import pytest
from npu_nvme.runtime.d1_schema import REQUIRED_CONTROLS, manifest_digest
from npu_nvme.runtime.commit import MetadataState
from npu_nvme.runtime.d1_commit import D1CommitCoordinator
from npu_nvme.storage.layout import make_layout


@pytest.fixture
def spec():
    return dict(identity={'model':'test','optimizer':'test','world_size':1},
        parameters={name:dict(shape=[8192], dtype='uint8', size=8192) for name in ('model/x','optimizer/m')},
        control_names=sorted(REQUIRED_CONTROLS),
        applicability={name:('not_applicable:fixed LR' if name=='scheduler' else 'required')
                       for name in ('model','optimizer','rng','data_cursor','scheduler','loss_scale')})


class IO:
    def __init__(self): self.history=[]; self.fail=False
    def persist(self, state, generation=None):
        self.history.append(deepcopy(state.meta_dict))
        if self.fail: raise IOError('injected metadata failure')
        state.metadata_generation=generation
        state.active_meta_slot=1-state.active_meta_slot


@pytest.fixture
def coordinator():
    layout=make_layout(total_bytes=1<<30,full_slot_bytes=1<<20,full_slot_count=3,
                       delta_slot_bytes=4096,delta_slot_count=1)
    state=MetadataState(layout=layout,meta_dict={'strict_contract':'D1','catalog_revision':0,'checkpoints':{}})
    return D1CommitCoordinator(metadata_io=IO(),state=state)


def make_record(c, r, spec):
    params={}; disk={}; overall=hashlib.sha256()
    offset=c.state.layout.full_base + r.slot*c.state.layout.full_slot_bytes
    descriptors=deepcopy(spec['parameters'])
    descriptors.update({'control/'+name:dict(shape=[4],dtype='uint8',size=4,codec='json-tagged-v1') for name in spec['control_names']})
    for name, info in sorted(descriptors.items()):
        raw=bytes([r.generation % 251 + 1])*info['size']; whole=hashlib.sha256(raw).hexdigest()
        chunks=[]
        for co in range(0,len(raw),4096):
            data=raw[co:co+4096]; disk[offset+co]=data
            chunks.append(dict(offset=co,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
        params[name]=dict(info,offset=offset,sha256=whole,chunks=chunks)
        overall.update(raw); offset+=(len(raw)+4095)//4096*4096
    record=dict(strict_contract='D1',type='TRAINING_STATE_FULL',schema_version=1,rank_id=0,world_size=1,
        generation=r.generation,state_step=r.step,slot=r.slot,request_id=r.request_id,writer_epoch=r.writer_epoch,
        chunk_size=4096,spec=deepcopy(spec),params=params,data_sha256=overall.hexdigest())
    record['manifest_sha256']=manifest_digest(record)
    return record,disk


def publish(c,spec,step):
    from npu_nvme.types import TransferReceipt
    r=c.reserve(step=step); record,disk=make_record(c,r,spec)
    receipt=TransferReceipt(r.request_id,r.generation,sum(p['size'] for p in record['params'].values()),
                           sum(len(p['chunks']) for p in record['params'].values()),True,record['data_sha256'])
    result=c.publish(r,receipt,record)
    return r,receipt,record,disk,result
