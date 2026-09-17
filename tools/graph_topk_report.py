#!/usr/bin/env python3
"""Build a conservative, auditable progress report from completed timing runs."""
import argparse
import json
import statistics
from pathlib import Path


def build(campaign, output):
    runs = []
    for run in sorted(campaign.iterdir()):
        result = run / 'result.json'
        if not result.exists(): continue
        d = json.loads(result.read_text())
        if d.get('status') != 'pass' or (run/'validation-exclusion.json').exists() or (run/'timing-exclusion.json').exists(): continue
        cfg = json.loads((run / 'run-config.json').read_text())
        ranks = [json.loads((run / f'rank_{r}/result.json').read_text()) for r in range(4)]
        row = dict(reference_only=cfg.get('reference_only',False), compute_iterations=cfg.get('compute_iterations',0), name=run.name, role=cfg['role'], level=cfg['level'], layout=cfg['layout'],
                   scan_fraction=cfg['scan_fraction'], ratio=cfg['ratio'], diagnostic=cfg['profile'],
                   warmup_steps=cfg['warmup_steps'], steps=cfg['formal_steps'], seconds=d['completion_seconds'], training_seconds=d['training_seconds'],
                   drain_seconds=max((r['all_done_ns']-r['training_end_ns'])/1e9 for r in ranks),
                   peak_hbm_bytes=max(r['memory_peak_bytes'] for r in ranks),
                   post_restore_warmup_seconds=[(x['end_ns']-x['begin_ns'])/1e9 for x in ranks[0]['warmup'][-cfg['warmup_steps']:]],
                   numerical_loss=[r['loss'] for r in ranks[0]['losses']],
                   maximum_first_step_seconds=max((r['losses'][0]['end_ns']-r['losses'][0]['begin_ns'])/1e9 for r in ranks),
                   auxiliary_seconds=statistics.median([v/1e9 for r in ranks for v in r.get('standalone_auxiliary_ns',[])]) if cfg['level'] else None,
                   score_oracle=all(r.get('score_oracle',{}).get('status')=='pass' for r in ranks) if cfg['level'] else None,
                   actual_parallel_verified=False)
        memory = run / 'board-memory.jsonl'
        if memory.exists():
            begin=min(r['begin_ns'] for r in ranks);end=max(r['all_done_ns'] for r in ranks)
            samples=[json.loads(line) for line in memory.read_text().splitlines()]
            points=[x for x in samples if begin<=x.get('monotonic_ns',0)<=end]
            row['board_memory_peak_mib']={str(rank):max((v['used_mib'] for sample in points for v in sample.get('devices',[]) if v['device']==rank), default=None) for rank in range(4)}
            row['board_sampling_limits']='One-second sampling may miss short peaks; includes allocations outside framework allocator.'
        profile = run / 'device-overlap.json'
        if profile.exists():
            p=json.loads(profile.read_text());row['overlap_us']=[r['actual_task_overlap_us'] for r in p['ranks']]
        runs.append(row)
    formal = [r for r in runs if not r['diagnostic'] and r['steps']==20 and r['warmup_steps']==12 and r['layout']=='serial']
    for row in formal:
        base = [r for r in formal if r['role']==row['role'] and r['level']==0 and not r['reference_only']]
        if base:
            median=statistics.median(r['seconds'] for r in base)
            row['baseline_count']=len(base)
            row['T0_seconds']=median;row['slowdown']=(row['seconds']-median)/median
            row['baseline_spread']=(max(r['seconds'] for r in base)-min(r['seconds'] for r in base))/median
            row['loss_matches_G0']=all(abs(a-b)<=1e-6+1e-6*abs(b) for a,b in zip(row['numerical_loss'],base[0]['numerical_loss']))
            row['extra_peak_hbm_bytes']=row['peak_hbm_bytes']-max(r['peak_hbm_bytes'] for r in base)
        controls=[r for r in formal if r['role']==row['role'] and r['level']==1 and not r['compute_iterations'] and r['layout']==row['layout']]
        if controls and row['level'] > 1:row['increment_over_G1']=row['seconds']/statistics.median(r['seconds'] for r in controls)-1
    output.mkdir(parents=True,exist_ok=True)
    (output/'measurements.json').write_text(json.dumps(dict(runs=runs),indent=2)+'\n')
    text=['# 图内 Top-K 实验进度','',
          '结果仅属于图内检测成本实验。目标并行布局必须通过设备轨迹验收；未验收数据不作为实际并行容量证据。固定参考不代表持续保存参考语义，设备结果不代表持久化。','',
          '|配置|模型|20 步总完成秒|相对 G0|相对 G1|排空秒|峰值 HBM GiB|训练 loss 一致|',
          '|---|---|---:|---:|---:|---:|---:|---|']
    for r in formal:
        text.append('|'+ '|'.join([r['name'],r['role'],f"{r['seconds']:.6f}",
            f"{100*r['slowdown']:.3f}%" if 'slowdown' in r and r.get('baseline_count',0)>=2 and r.get('baseline_spread',1)<=.03 else '未定',
            f"{100*r['increment_over_G1']:.3f}%" if 'increment_over_G1' in r else '待定',
            f"{r['drain_seconds']:.6f}",f"{r['peak_hbm_bytes']/2**30:.3f}",str(r.get('loss_matches_G0','待定'))])+'|')
    if formal:
        spreads=[r.get('baseline_spread') for r in formal if r.get('baseline_spread') is not None and sum(x['role']==r['role'] and x['level']==0 for x in formal)>=2]
        text += ['', '当前同模型 G0 spread：'+(f'{100*max(spreads):.3f}%' if spreads else '待第二次基线')+'。超过 3% 时，减速预算结论标记为未定。']
    text += ['', '每配置默认一次、必要时最多两次；4 个 rank 不作为 4 次独立重复。未完成和失败启动不进入上表。',
             '', '主预算 3%，参考 1%/5%。在稳定基线、输出校验及依赖验证完成前，不宣布预算通过。',
             '', '暂未获得的截止点等待、训练计算段变化、设备实际重叠和窗口 HBM 指标保持缺失，不用 Host 提交耗时代替。']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,default=Path('/models/npu_nvme_exp/user7-stack/graph-topk-20260917-001'))
    p.add_argument('--output',type=Path,default=Path('results/graph-topk-20260917'));a=p.parse_args();build(a.campaign,a.output)
