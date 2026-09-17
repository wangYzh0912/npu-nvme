#!/usr/bin/env python3
"""Resume explicit graph experiment configurations, never skip a failed gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--matrix',type=Path,required=True)
    args=parser.parse_args();matrix=json.loads(args.matrix.read_text())
    for item in matrix:
        target=args.campaign/item['name']
        if (target/'result.json').exists():
            old=json.loads((target/'result.json').read_text())
            if old.get('status')=='pass':continue
            raise RuntimeError('failed run requires diagnosis, not automatic retry: '+str(target))
        if target.exists():raise RuntimeError('existing run may be active: '+str(target))
        (args.campaign/'queue-state.json').write_text(json.dumps(dict(status='running',configuration=item),indent=2)+'\n')
        command=[sys.executable,str(ROOT/'tools/graph_topk_run.py'),'--output',str(target)]
        for key,value in item.items():
            if key=='name':continue
            option='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value:command.append(option)
            else:command.extend([option,str(value)])
        result=subprocess.run(command,cwd=ROOT)
        subprocess.run([sys.executable,str(ROOT/'tools/graph_topk_report.py'),'--campaign',str(args.campaign)],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'tools/publish_graph_topk.py'),'--campaign',str(args.campaign)],cwd=ROOT,check=True)
        if result.returncode:raise RuntimeError('configuration failed: '+item['name'])
    (args.campaign/'queue-state.json').write_text(json.dumps(dict(status='completed',configurations=matrix),indent=2)+'\n')

if __name__=='__main__':main()
