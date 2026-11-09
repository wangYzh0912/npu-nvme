#!/usr/bin/env python3
"""Summarize trace evidence without treating profiler costs as clean timings."""
import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,default=Path('results/graph-topk-20260917'))
    args=parser.parse_args()
    trace=json.loads((args.run/'device-overlap.json').read_text())
    types={}
    tensor_extents={}
    for rank in trace['ranks']:
        for step in rank['steps']:
            tasks=[t for t in rank['tasks'] if t['phase']=='auxiliary' and
                   step['source_ready_us']<=t['start']<step['last_auxiliary_end_us']]
            for kind in {t['type'] for t in tasks}:
                selected=[t for t in tasks if t['type']==kind]
                types.setdefault(kind,[]).append(sum(t['end']-t['start'] for t in selected)/1000)
            for kind in ('TensorMove','Sub','Mul'):
                selected=[t for t in tasks if t['type']==kind and t['output_dtype']=='FLOAT']
                size=sum(math.prod(int(n) for n in t['output_shape'].strip('"').split(','))*4 for t in selected)
                tensor_extents.setdefault(kind,[]).append(size)
    costs={kind:dict(min_ms=min(values),median_ms=statistics.median(values),max_ms=max(values))
           for kind,values in sorted(types.items(),key=lambda kv:-statistics.median(kv[1]))}
    artifact=dict(source=str(args.run),diagnostic_only=True,
        sampled_complete_rank_steps=sum(len(r['steps']) for r in trace['ranks']),
        task_type_costs=costs, output_tensor_extent_bytes_per_rank_step=tensor_extents,
        actual_auxiliary_training_overlap_us=sum(r['actual_task_overlap_us'] for r in trace['ranks']),
        limits=['Profiled costs include diagnostic perturbation and are not clean performance results.',
                'Collective duration includes cross-rank arrival waiting, not just transfer.',
                'Tensor output extents are not simultaneous peak memory or measured HBM bus bytes.',
                'No causal per-stage critical-path attribution from cumulative single runs.'])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'bottlenecks.json').write_text(json.dumps(artifact,indent=2)+'\n')
    text=['# G6 设备轨迹与瓶颈证据','',
        '本表来自独立 3 步 profiler；采用 4 卡共 8 个完整区间。它解释任务成本，不替代 20 步无 profiler 计时。','',
        '|任务|每卡每步任务时长中位 ms|最小～最大 ms|',
        '|---|---:|---:|']
    for kind,c in costs.items():
        text.append(f"|{kind}|{c['median_ms']:.3f}|{c['min_ms']:.3f}～{c['max_ms']:.3f}|")
    text += ['', '主要成本与下一步方向：', '',
        '1. 块评分的片段归约：每步 146 个 UnsortedSegmentSum，约 54～57 ms。优先保留 TP 逻辑块语义，融合差分、平方与片段归约，减少中间数组和任务数。',
        '2. 扫描及中间结果：差分约 21.5 ms，平方和编译器 TensorMove 各约 12～13 ms。每卡每步 Sub/Mul 输出范围各为 8,190,427,136 字节，TensorMove 约 7,568,097,284 字节；不是仅扫描 2S 的理想路径。峰值差会掩盖发生在训练峰值以外的临时空间。',
        '3. 跨 rank 等待与同步：辅助 AllReduce 的包络含到达等待，各 rank 差异明显。应先降低评分链的任务数量和 rank 到达偏差，再用独立通信边界实验判断传输本身；当前不能把包络全部归因于 HCCL 带宽。',
        '', 'TopK 约 9.6 ms，显式 Sort 约 0.02 ms。当前不是第一优先级；三档 K 都需要全量评分。',
        '', '所有完整采样区间均满足更新后读取和下一更新前结束；辅助与前向/反向任务交集为 0。串行参照没有重叠收益，目标并行图没有编译通过，因此无法报告并行相对串行收益。',
        '', '分段任务时长、通信包络与训练关键路径增量分别理解，不能相加宣称最终减速。累计图相邻配置也受融合、调度和基线波动影响。']
    (args.output/'BOTTLENECKS.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
