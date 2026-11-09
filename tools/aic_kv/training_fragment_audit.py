"""Extract static evidence for optimizer copies and paired backward MatMul."""
import argparse
from collections import Counter,defaultdict,deque
import gzip
import json
import math
from pathlib import Path
import re


def operand_path(nodes,start,target):
    queue=deque([[start]]);seen=set()
    while queue:
        path=queue.popleft();key=path[-1]
        if key==target:return path
        if key in seen:continue
        seen.add(key)
        node=nodes[key]
        queue.extend(path+[(node['graph'],v)] for v in node['operands']
                     if (node['graph'],v) in nodes and (node['graph'],v) not in seen)
    return None


def audit(graphs,output):
    output.mkdir(parents=True,exist_ok=True)
    irs=list((graphs/'rank0').glob('*.ir.gz'))
    if len(irs)!=1:raise ValueError('expected one archived final stream graph')
    lines=gzip.open(irs[0],'rt').read().splitlines()
    nodes=[json.loads(line) for line in gzip.open(graphs/'rank0/static-dependencies.jsonl.gz','rt')]
    byid={(n['graph'],n['node']):n for n in nodes};moves=[];pairs=defaultdict(list)
    for node in nodes:
        i=node['source_line']-1;line=lines[i]
        if node['op']=='TensorMove':
            match=re.search(r'-> \(<Tensor\[Float32\], \(([^)]*)\)',lines[i+1])
            if match:
                shape=[int(x.strip()) for x in match[1].split(',') if x.strip()]
                moves.append(dict(node=node['node'],ir_line=i+1,shape=shape,logical_output_bytes=math.prod(shape)*4,
                                  operand_nodes=node['operands'],source_ir=line.strip()))
        if node['op']=='PrimFunc_MatMul' and 'Gradients' in node.get('scope',''):
            match=re.search(r'forward_unique_id: "([^"]+)"',line)
            if match:pairs[match[1]].append(node)
    result=dict(source_ir=str(irs[0]),tensor_move_count=len(moves),
                logical_output_bytes_sum=sum(m['logical_output_bytes'] for m in moves),moves=moves,
                interpretation='static Float32 output geometry; not measured bytes or per-step execution proof')
    (output/'optimizer-tensor-moves.json').write_text(json.dumps(result,indent=2)+chr(10))
    candidates=[]
    for identity,group in pairs.items():
        if len(group)!=2:continue
        operands=[]
        for node in group:
            inputs=[byid[(node['graph'],v)] for v in node['operands'] if (node['graph'],v) in byid]
            operands.append(dict(node=node,inputs=inputs,
                duplicated_inputs=[n for n in inputs if 'duplicated: Bool(1)' in lines[n['source_line']-1]],
                input_output_types=lines[node['source_line']]))
        candidates.append(dict(forward_unique_id=identity,branches=operands,runtime_readiness='unknown',
            constraint='Static operands only. Check recomputation, collectives, activation lifetime and alias effects before legal co-run.'))
    (output/'backward-paired-matmul-audit.json').write_text(json.dumps(dict(
        source_ir=str(irs[0]),paired_gradient_operations=len(candidates),candidates=candidates),indent=2)+chr(10))
    (output/'static-operation-counts.json').write_text(json.dumps(dict(Counter(n['op'] for n in nodes)),indent=2)+chr(10))
    norms=[n for n in nodes if 'clip_grad_norm-ClipGradNorm/AddN-' in n.get('scope','')]
    if len(norms)!=1:raise ValueError('expected one explicit global gradient norm sum')
    norm=norms[0];target=(norm['graph'],norm['node']);chains=[]
    updates=[n for n in nodes if n['op']=='GraphKernel_Mul_Add_Assign_fusion']
    for update in updates:
        path=operand_path(byid,(update['graph'],update['node']),target)
        if path is not None:chains.append(path)
    if not chains:raise ValueError('no optimizer update reaches the global norm')
    shortest=min(chains,key=len)
    (output/'global-clip-optimizer-dependency.json').write_text(json.dumps(dict(
        source_ir=str(irs[0]),norm_sum=norm,gradient_norm_inputs=len(norm['operands']),
        first_moment_fused_update_nodes=len(updates),updates_with_norm_ancestor=len(chains),
        representative_operand_path=[byid[key] for key in shortest],
        runtime_ready_timestamp='unknown',
        interpretation='Current graph requires all norm contributions and AllReduce before clipped-gradient updates. Static dependency, not a measured critical-path duration; do not remove global clipping to fabricate legal overlap.'),indent=2)+chr(10))
    print('audited',len(moves),'TensorMove nodes and',len(candidates),'paired backward operations')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--graphs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    audit(args.graphs,args.output)


if __name__=='__main__':main()
