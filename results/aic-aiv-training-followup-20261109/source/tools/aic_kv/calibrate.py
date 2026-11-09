"""Real idle/copy/matrix segments for simultaneous PMU/HBM capability."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    import torch_npu
    torch.npu.set_device(0)
    backing=torch.randn((1024,2,128,256),device='npu',dtype=torch.bfloat16)
    matrix=torch.randn((2048,2048),device='npu',dtype=torch.bfloat16)
    for _ in range(3):
        copied=backing[:,0].contiguous()
        product=matrix@matrix
    torch.npu.synchronize()
    command=['msprof','--output='+str(a.output/'system'),'--sys-devices=4','--sys-period=14',
             '--ai-core=on','--aic-mode=sample-based','--aic-freq=100','--aic-metrics=PipeUtilization',
             '--task-time=off','--sys-hardware-mem=on','--sys-hardware-mem-freq=50','--type=text']
    (a.output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    windows=[]
    with (a.output/'collector.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            time.sleep(2)
            if child.poll() is not None:raise RuntimeError('collector exited early')
            for name,duration in [('idle',2),('copy',4),('matrix',4)]:
                begin=time.time_ns();deadline=time.monotonic()+duration;count=0
                while time.monotonic()<deadline:
                    if name=='idle':time.sleep(.05)
                    elif name=='copy':copied=backing[:,0].contiguous()
                    else:product=matrix@matrix
                    torch.npu.synchronize();count+=1
                windows.append(dict(id=name,begin_ns=begin,end_ns=time.time_ns(),iterations=count))
            code=child.wait(timeout=180)
            if code:raise RuntimeError(f'collector exit {code}')
        finally:
            if child.poll() is None:
                child.terminate();child.wait(timeout=30)
    (a.output/'windows.json').write_text(json.dumps(windows,indent=2)+'\n')


if __name__=='__main__':main()
