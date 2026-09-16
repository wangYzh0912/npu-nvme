"""Exercise the baseline Native serializer on real model/Adam/control tensors."""
import argparse
import json
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import nn,Tensor,Parameter

from qwen_native_state import parameter_manifest,compare_manifests


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
    (args.out/'result.json').write_text(json.dumps(dict(status='pass',mindspore=ms.__version__,state=actual),indent=2)+'\n')


if __name__=='__main__':main()
