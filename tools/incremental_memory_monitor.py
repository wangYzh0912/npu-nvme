#!/usr/bin/env python3
"""Observe board memory independently; polling can miss short peaks."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def sample():
    raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=10)
    rows=[];device=None
    for line in raw.splitlines():
        match=re.search(r'\|\s*(\d+)\s+910',line)
        if match:device=int(match.group(1))
        match=re.search(r'(\d+)\s*/\s*(65536)\s*\|',line)
        if match and device is not None:rows.append(dict(device=device,used_mib=int(match.group(1)),capacity_mib=int(match.group(2))))
    return dict(monotonic_ns=time.monotonic_ns(),devices=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-file',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('a',buffering=1) as f:
        while not a.stop_file.exists():
            try:f.write(json.dumps(sample())+'\n')
            except Exception as e:f.write(json.dumps(dict(error=repr(e)))+'\n')
            time.sleep(1)
