from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from types import SimpleNamespace

from npu_nvme.framework.checkpoint_controller import CheckpointController
from npu_nvme.runtime.training_catalog import committed,selected,verify_files


def test_collective_periodic_publication_retains_complete_generations(tmp_path):
    output=tmp_path/'run';output.mkdir()
    root=tmp_path/'checkpoints'
    config=dict(checkpoint_root=str(root),method='mindspore_native_save',identity={'model':'test'},
                run_id='run',operation_timeout_seconds=5,retention=3)
    framework=SimpleNamespace(hal=SimpleNamespace(synchronize=lambda:None),
        save_checkpoint=lambda network,path,**kwargs:Path(path).write_bytes(network))
    collective=threading.Barrier(4,timeout=5)
    controllers=[CheckpointController(config,rank=r,npu_id=r,output=output) for r in range(4)]
    def save(rank):
        results=[]
        for step in (4,8,12,16):
            results.append(controllers[rank].save(framework,b'payload',step=step,
                state={'weight':{'bytes':7}},controls={'step':step},barrier=collective.wait))
        return results
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(save,range(4)))
    assert all([row['generation'] for row in rows]==[1,2,3,4] for rows in results)
    assert [generation for generation,_,_ in committed(root)]==[2,3,4]
    with selected(root) as (path,value):
        assert value['step']==16
        assert len(value['ranks'])==4
        for rank in value['ranks']:
            verify_files(path/f"rank_{rank['rank']}",rank['files'])
