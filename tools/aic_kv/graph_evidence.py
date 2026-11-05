"""Extract static IR dependencies; static stream order is not runtime readiness."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re


def extract(path):
    graph='unknown';nodes=[];current=None
    for line_no,line in enumerate(path.read_text().splitlines(),1):
        m=re.match(r'subgraph @([^ (]+)',line)
        if m:graph=m[1]
        m=re.match(r'\s+%(\d+)(?:\([^)]*\))? = ([^(]+)\(',line)
        if m:
            # References are only taken from the operand portion before attrs.
            rhs=line[m.end():].split('primitive_attrs:',1)[0].split('cnode_attrs:',1)[0]
            stream=re.search(r'stream_id: I32\((\d+)\)',line)
            current=dict(graph=graph,node=m[1],op=m[2],source_line=line_no,
                operands=re.findall(r'%(\d+)\b',rhs),
                stream_id=int(stream[1]) if stream else None)
            nodes.append(current)
        elif current is not None and '# Fullname with scope:' in line:
            current['scope']=line.split('# Fullname with scope:',1)[1].strip()
    if not nodes:raise ValueError('no static IR nodes: '+str(path))
    return nodes


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);ranks=[]
    for rank in range(4):
        candidates=list((a.run/f'rank_{rank}/dependency-graphs').rglob('hwopt_d_after_stream_assign_1_*.ir'))
        if len(candidates)!=1:raise ValueError(f'expected one final main stream graph: {candidates}')
        path=candidates[0];nodes=extract(path);target=a.output/f'rank{rank}';target.mkdir(exist_ok=True)
        with gzip.open(target/(path.name+'.gz'),'wb') as f:f.write(path.read_bytes())
        with gzip.open(target/'static-dependencies.jsonl.gz','wt') as f:
            for node in nodes:f.write(json.dumps(node,separators=(',',':'))+'\n')
        ranks.append(dict(rank=rank,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            nodes=len(nodes),ops=dict(Counter(n['op'] for n in nodes)),
            static_streams=dict(Counter(str(n['stream_id']) for n in nodes)),
            explicit_operand_edges=sum(len(n['operands']) for n in nodes),
            runtime_task_mapping='unavailable',readiness='unknown',
            interpretation='IR operands/control nodes and assigned streams only; no device completion timestamps'))
    (a.output/'graph-evidence.json').write_text(json.dumps(dict(run=str(a.run),ranks=ranks),indent=2)+'\n')

if __name__=='__main__':main()
