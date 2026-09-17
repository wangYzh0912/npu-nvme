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
        step_seconds=[(x['end_ns']-x['begin_ns'])/1e9 for x in ranks[0]['losses']]
        row['step_seconds_rank0']=dict(minimum=min(step_seconds),median=statistics.median(step_seconds),
                                       maximum=max(step_seconds),all_steps=step_seconds)
        if cfg['level']>=2:
            row['working_set']=dict(candidate_blocks=ranks[0]['geometry']['candidate_blocks'],
                global_W_R_input_bytes=sum(r['geometry']['local_input_bytes'] for r in ranks),
                reference_bytes_per_rank=[r['geometry']['local_reference_bytes'] for r in ranks],
                TP_fragments_per_rank=[r['geometry']['tp_fragments'] for r in ranks],
                selected_blocks=ranks[0]['output_check']['selected_count'] if cfg['level']>=6 else 0)
        if cfg['level']>=7:
            row['device_consumer']=dict(budgets=[r.get('consumer_budget') for r in ranks],
                validation_passed=all(r.get('consumer_oracle',{}).get('status')=='pass' for r in ranks))
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
            row['slowdown_baseline_range']=[row['seconds']/max(r['seconds'] for r in base)-1,
                                            row['seconds']/min(r['seconds'] for r in base)-1]
            row['budget_assessment']={str(budget): (
                'above_budget_for_both_baselines' if len(base)>=2 and row['slowdown_baseline_range'][0]>budget
                else 'undetermined_baseline_variation' if len(base)<2 or row['baseline_spread']>.03
                else 'measured_feasible_point' if row['slowdown']<=budget else 'above_budget')
                for budget in (.01,.03,.05)}
            row['loss_matches_G0']=all(abs(a-b)<=1e-6+1e-6*abs(b) for a,b in zip(row['numerical_loss'],base[0]['numerical_loss']))
            row['extra_peak_hbm_bytes']=row['peak_hbm_bytes']-max(r['peak_hbm_bytes'] for r in base)
        controls=[r for r in formal if r['role']==row['role'] and r['level']==1 and not r['compute_iterations'] and r['layout']==row['layout']]
        if controls and (row['level'] > 1 or row['compute_iterations']):row['increment_over_G1']=row['seconds']/statistics.median(r['seconds'] for r in controls)-1
    output.mkdir(parents=True,exist_ok=True)
    (output/'measurements.json').write_text(json.dumps(dict(runs=runs),indent=2)+'\n')
    text=['# 图内 Top-K 实验进度','',
          '结果仅属于图内检测成本实验。目标并行布局必须通过设备轨迹验收；未验收数据不作为实际并行容量证据。固定参考不代表持续保存参考语义，设备结果不代表持久化。','',
          '|配置|模型|20 步总完成秒|相对 G0 区间|相对 G1|独立辅助 ms|峰值 HBM GiB|训练 loss 一致|',
          '|---|---|---:|---:|---:|---:|---:|---|']
    for r in formal:
        text.append('|'+ '|'.join([r['name'],r['role'],f"{r['seconds']:.6f}",
            '～'.join(f'{100*v:.3f}%' for v in r['slowdown_baseline_range']) if r.get('baseline_count',0)>=2 else '待第二次基线',
            f"{100*r['increment_over_G1']:.3f}%" if 'increment_over_G1' in r else '待定',
            f"{1000*r['auxiliary_seconds']:.3f}" if r['auxiliary_seconds'] is not None else '—',
            f"{r['peak_hbm_bytes']/2**30:.3f}",str(r.get('loss_matches_G0','待定'))])+'|')
    if formal:
        spreads=[r.get('baseline_spread') for r in formal if r.get('baseline_spread') is not None and sum(x['role']==r['role'] and x['level']==0 for x in formal)>=2]
        text += ['', '当前同模型 G0 spread：'+(f'{100*max(spreads):.3f}%' if spreads else '待第二次基线')+'。超过 3% 时，减速预算结论标记为未定。']
    text += ['', '每配置默认一次、必要时最多两次；4 个 rank 不作为 4 次独立重复。未完成和失败启动不进入上表。',
             '', 'G0 区间是相对两次独立基线的敏感性范围，不是统计置信区间。独立辅助时间在正式区间之外测量，不能直接等同训练关键路径增量。',
             '', '主预算 3%，参考 1%/5%。在稳定基线、输出校验及依赖验证完成前，不宣布预算通过。',
             '', '暂未获得的截止点等待、训练计算段变化、设备实际重叠和窗口 HBM 指标保持缺失，不用 Host 提交耗时代替。']
    text += ['', '## 扫描负载与预算', '',
             '|模型|q|候选块|全局 W/R 输入 GB|TP 片段合计|3% 判定|',
             '|---|---:|---:|---:|---:|---|']
    for r in sorted(formal,key=lambda r:(r['role'],r['scan_fraction'])):
        if r['level']!=6 or r['ratio']!=.1:continue
        work=r['working_set']
        assessment=r.get('budget_assessment',{}).get('0.03','undetermined')
        label={'above_budget_for_both_baselines':'对两次基线均超预算',
               'undetermined_baseline_variation':'未定：基线不足或波动',
               'measured_feasible_point':'已测可行点','above_budget':'超预算'}.get(assessment,assessment)
        text.append(f"|{r['role']}|{r['scan_fraction']}|{work['candidate_blocks']}|{work['global_W_R_input_bytes']/1e9:.6f}|{sum(work['TP_fragments_per_rank'])}|{label}|")
    text += ['', '表中 q<1 是跨层、跨参数类型的真实逻辑块子集；仍常驻完整参考。K 比例固定为候选块数的 10%，与 q 独立。曲线不预设单调，不外推未测负载。',
             '', '## 结论边界', '',
             '两次目标并行图未通过 MindSpore StepParallel 编译；因此本表仅报告串行图附加成本，不能回答有多少成本可以被下一步前向/反向隐藏。',
             '', 'G7/G8 按条件入口决定：当前主模型最小 q=1/8 对两次基线均超出 5%，没有通过或接近预算的配置，未进入其硬件验证。设备端实现及 CPU 几何测试保留，不作为持久化或持续运行证据。',
             '', '仅分配参考控制因未能证实正式区间 HBM 驻留而排除；完整参考的有效内存证据来自 G2–G6 实测额外峰值约 7.63 GiB/卡。峰值差不能证明没有差分、平方等中间张量：串行辅助临时空间可低于训练峰值而被掩盖。',
             '', 'NVMe、D2H、socket、payload SHA、介质回读和影子 loss 不在本轮主计时中。旧完整保存路径仅保留为独立工程证据。']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,default=Path('/models/npu_nvme_exp/user7-stack/graph-topk-20260917-001'))
    p.add_argument('--output',type=Path,default=Path('results/graph-topk-20260917'));a=p.parse_args();build(a.campaign,a.output)
