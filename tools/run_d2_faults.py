#!/usr/bin/env python3
"""Explicit inspect/hash/initialize sequence for authorized D2 fault scratch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--shm-base',type=int,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    prefix=[sys.executable,'scripts/run_user_environment.py','--manifest',str(a.manifest),'--profile','old','--','python','tests/hardware/d2_fault_campaign.py']
    for phase in ('inspect','execute'):
        cmd=prefix+['--out',str(a.out/phase),'--shm-id',str(a.shm_base+(phase=='execute'))]
        if phase=='inspect':cmd+=['--inspect']
        else:cmd+=['--expected-header-sha256',json.loads((a.out/'inspect/result.json').read_text())['header_sha256']]
        with (a.out/(phase+'.log')).open('w') as log:rc=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if rc:return rc
    result=json.loads((a.out/'execute/result.json').read_text());(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return int(result['status']!='pass')
if __name__=='__main__':raise SystemExit(main())
