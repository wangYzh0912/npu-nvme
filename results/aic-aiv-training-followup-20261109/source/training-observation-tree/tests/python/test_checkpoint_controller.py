from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from types import SimpleNamespace
import json
import struct

from npu_nvme.framework.checkpoint_controller import CheckpointController
from npu_nvme.runtime.training_catalog import committed,selected,verify_files


def test_collective_periodic_publication_retains_complete_generations(tmp_path):
    output=tmp_path/'run';output.mkdir()
    root=tmp_path/'checkpoints'
    config=dict(checkpoint_root=str(root),method='mindspore_native_save',identity={'model':'test'},
                run_id='run',operation_timeout_seconds=5,retention=3)
    def save_checkpoint(entries,path,**kwargs):
        assert len(entries)==1 and entries[0]['name']=='weight'
        header=json.dumps({'weight':dict(shape=[7],dtype='U8',data_offsets=[0,7])}).encode()
        Path(path).write_bytes(struct.pack('<Q',len(header))+header+entries[0]['data'])
    parameter=SimpleNamespace(name='weight',data=b'payload')
    network=SimpleNamespace(parameters_and_names=lambda:[('weight',parameter)])
    framework=SimpleNamespace(hal=SimpleNamespace(synchronize=lambda:None),
        Tensor=lambda value:value,save_checkpoint=save_checkpoint)
    collective=threading.Barrier(4,timeout=5)
    controllers=[CheckpointController(config,rank=r,npu_id=r,output=output) for r in range(4)]
    def save(rank):
        results=[]
        for step in (4,8,12,16):
            results.append(controllers[rank].save(framework,network,step=step,
                state={'weight':{'bytes':7,'shape':[7]}},controls={'step':step},barrier=collective.wait))
        return results
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(save,range(4)))
    assert all([row['generation'] for row in rows]==[1,2,3,4] for rows in results)
    assert [generation for generation,_,_ in committed(root)]==[2,3,4]
    with selected(root) as (path,value):
        assert value['step']==16
        assert len(value['ranks'])==4
        for rank in value['ranks']:
            verify_files(path/f"rank_{rank['rank']}",rank['files'])
