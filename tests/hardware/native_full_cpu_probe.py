"""Exercise the baseline Native serializer on real model/Adam/control tensors."""
import argparse
import json
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import nn,Tensor,Parameter

from qwen_native_state import parameter_manifest,compare_manifests,capture_control
from npu_nvme.framework.checkpoint_controller import CheckpointController
from npu_nvme.runtime import training_catalog as catalog


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    ms.set_context(device_target='CPU',mode=ms.PYNATIVE_MODE)
    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight=Parameter(Tensor(np.arange(16,dtype=np.float32).reshape(4,4)),name='weight')
        def construct(self,x):return (x@self.weight).sum()
    class Training(nn.Cell):
        def __init__(self):
            super().__init__();self.model=Model()
            self.optimizer=nn.Adam(self.model.trainable_params(),learning_rate=.001)
            self.control=Parameter(Tensor(np.asarray(7,np.int64)),name='control',requires_grad=False)
        def construct(self,x):return self.model(x)
    network=Training()
    network.optimizer((Tensor(np.ones((4,4),np.float32)),))
    expected,_=parameter_manifest(network)
    path=args.out/'full.safetensors'
    ms.save_checkpoint(network,str(path),integrated_save=False,async_save=False,format='safetensors')
    restored=Training()
    values=ms.load_checkpoint(str(path),format='safetensors')
    missing,unexpected=ms.load_param_into_net(restored,values,strict_load=True)
    if missing or unexpected:raise ValueError((missing,unexpected))
    actual,_=parameter_manifest(restored);compare_manifests(expected,actual)
    if not any('moment' in name for name in actual):raise ValueError('Adam state missing')
    ms.manual_seed(42)
    # Exercise the exact production serializer twice after parameter replacement.
    config=dict(checkpoint_root=str(args.out/'catalog'),method='mindspore_native_save',
        identity={'model':'cpu'},run_id='cpu',operation_timeout_seconds=10,retention=3)
    original_publish=catalog.publish
    catalog.publish=lambda path,world:original_publish(path,world=2)
    from concurrent.futures import ThreadPoolExecutor
    import threading
    for step in (2,3):
        controllers=[];networks=[]
        for rank in range(2):
            candidate=Training()
            values=ms.load_checkpoint(str(path),format='safetensors')
            assert ms.load_param_into_net(candidate,values,strict_load=True)==([],[])
            for _,parameter in candidate.parameters_and_names():
                parameter.set_data(ms.Tensor(parameter.asnumpy(),dtype=parameter.dtype))
                parameter.sliced=False
                parameter.param_info.is_param_init=True
            networks.append(candidate)
            controllers.append(CheckpointController(config,rank=rank,npu_id=rank,output=args.out/f'cycle-{step}'))
        barrier=threading.Barrier(2,timeout=10)
        def save(rank):
            state,small=parameter_manifest(networks[rank])
            return controllers[rank].save(ms,networks[rank],step=step,state=state,
                controls=capture_control(ms,step=step,data_sha256='cpu',lr_horizon=4,small=small),barrier=barrier.wait)
        with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(save,range(2)))
        path=Path(rows[0]['path'])/'rank_0/native/full.safetensors'
        candidate=Training();values=ms.load_checkpoint(str(path),format='safetensors')
        assert ms.load_param_into_net(candidate,values,strict_load=True)==([],[])
        state,_=parameter_manifest(candidate);compare_manifests(expected,state)
    (args.out/'result.json').write_text(json.dumps(dict(status='pass',mindspore=ms.__version__,state=actual,
        replaced_parameter_save_restore_cycles=2),indent=2)+'\n')


if __name__=='__main__':main()
