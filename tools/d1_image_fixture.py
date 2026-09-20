#!/usr/bin/env python3
"""Produce a strict D1 transport fixture through the real FULL commit chain."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from npu_nvme.storage.metadata import MetadataIO
from npu_nvme.storage.layout import make_layout
from npu_nvme.runtime.commit import MetadataState
from npu_nvme.runtime.d1_commit import D1CommitCoordinator
from npu_nvme.runtime.d1_runtime import FullRuntime
from npu_nvme.runtime.d1_schema import REQUIRED_CONTROLS
from training_state import encode_control_value


class FileTransport:
    def __init__(self, path):
        self.fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.ftruncate(self.fd, 128 << 20)

    def npu_nvme_sync_meta_io(self, ctx, offset, size, read, ptr):
        if read:
            ctypes.memmove(ptr, self.read(offset, size), size)
        else:
            self.write(offset, ctypes.string_at(ptr, size))
        return 0

    def npu_nvme_flush(self, ctx):
        self.flush()
        return 0

    def write(self, offset, data):
        if offset < 0 or offset+len(data) > 128 << 20:
            raise ValueError('fixture write extent')
        if os.pwrite(self.fd, data, offset) != len(data):
            raise OSError('short fixture write')

    def read(self, offset, size): return os.pread(self.fd, size, offset)
    def flush(self): os.fsync(self.fd)
    def quiescent(self): return True
    def frozen_bytes(self, item, offset, size): return item['raw'][offset:offset+size]
    def close(self, timeout=120):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def create(path):
    transport = FileTransport(path)
    layout = make_layout(128 << 20, 32 << 20, 3, 4096, 1)
    state = MetadataState(layout=layout, meta_dict={'strict_contract':'D1', 'catalog_revision':0, 'checkpoints':{}})
    io = MetadataIO(transport)
    io.persist(state, generation=0)
    coordinator = D1CommitCoordinator(metadata_io=io, state=state)
    size = (4 << 20)+8196
    spec = dict(identity={'model':'strict-D1-host-fixture','optimizer':'adam-fixture','world_size':1},
        parameters={name:dict(shape=[size//4],dtype='float32',size=size)
                    for name in ('model/weight','optimizer/m','optimizer/v')},
        control_names=sorted(REQUIRED_CONTROLS),
        applicability={name:('not_applicable:fixed LR' if name=='scheduler' else 'required')
                       for name in ('model','optimizer','rng','data_cursor','scheduler','loss_scale')})
    def prepare(components, controls, expected):
        values=[dict(info,name=name,raw=components[name]) for name,info in expected['parameters'].items()]
        for name,value in controls.items():
            array,_=encode_control_value(value)
            raw=array.tobytes()
            values.append(dict(name='control/'+name,shape=[len(raw)],dtype='uint8',size=len(raw),raw=raw))
        return values
    runtime=FullRuntime(coordinator,transport,prepare=prepare,freeze=lambda p,g:p,release=lambda p:None)
    try:
        for step in (4,8,12):
            components={name:(bytes([step+i])*size) for i,name in enumerate(spec['parameters'])}
            controls={name:{'fixture':name,'step':step} for name in spec['control_names']}
            controls.update(global_step=step,loss_scale=1.0,mindspore_seed=42,data_cursor={'next':step})
            runtime.save(components,controls,step,spec).wait()
    finally:
        runtime.close()
    with open(path, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path':str(path),'bytes':Path(path).stat().st_size,'sha256':digest,
            'steps':[4,8,12],'scope':'strict D1 persisted host-state fixture; not historical Qwen recovery'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(create(args.image),indent=2))
