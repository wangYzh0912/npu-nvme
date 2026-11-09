"""Actual tensor operations for serial/concurrent measurement calibration."""
import argparse
import json
import time
from pathlib import Path
import torch
import torch_npu

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=2);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.npu.set_device(0)
    x=torch.randn((4096,4096),device='npu',dtype=torch.float16);y=x.clone();z=torch.empty_like(x)
    v=torch.ones(32*1024*1024,device='npu',dtype=torch.float16)
    c=torch.npu.Stream();s=torch.npu.Stream()
    def matrix():
        with torch.npu.stream(c):torch.mm(x,y,out=z)
    def vector():
        with torch.npu.stream(s):v.mul_(1.00001);v.add_(.00001)
    matrix();vector();torch.npu.synchronize()
    records=[]
    for phase in ('idle','matrix','vector','serial','concurrent'):
        torch.npu.synchronize();begin=time.time_ns();deadline=time.monotonic()+a.seconds;count=0
        while time.monotonic()<deadline:
            if phase=='idle':time.sleep(.01)
            elif phase=='matrix':matrix()
            elif phase=='vector':vector()
            elif phase=='serial':
                matrix();c.synchronize()
                for _ in range(4):vector();s.synchronize()
            elif phase=='concurrent':
                matrix()
                for _ in range(4):vector()
            else:vector()
            count+=1
            if count%2==0:torch.npu.synchronize()
        torch.npu.synchronize();records.append(dict(id=phase,begin_ns=begin,end_ns=time.time_ns(),iterations=count))
    (a.output/'windows.json').write_text(json.dumps(records,indent=2)+'\n')
    (a.output/'correctness.json').write_text(json.dumps(dict(matrix_finite=bool(torch.isfinite(z).all().item()),vector_finite=bool(torch.isfinite(v).all().item()))))
    print(json.dumps(records),flush=True)

if __name__=='__main__':main()
