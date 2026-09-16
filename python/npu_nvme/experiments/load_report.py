"""Collect independently executed boundary probes."""
import json
from pathlib import Path


def collect(paths, output):
    rows=[]
    for scale,path in paths:
        value=json.loads((Path(path)/'result.json').read_text())
        if value.get('status')!='pass' or not value.get('digest'):
            raise ValueError('load probe did not consume valid output')
        rows.append(dict(scale=scale,elapsed_ns=value['elapsed_ns'],select_ns=value['select_ns'],
                         scan_bytes=value['score_scan_bytes'],scalar_ops=value['score_scalar_ops'],
                         peak_bytes=value['peak_bytes'],selection=value['select']))
    result=dict(status='pass',probe='standalone boundary; no training overlap claim',rows=rows,
                slowdown_references=[.01,.03,.05],training_slowdown=None)
    Path(output).write_text(json.dumps(result,indent=2)+'\n');return result
