"""Summarize measured coordination probes and conditional stop decisions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from .coordination import ROOT, RAW


def collect(root=ROOT):
    rows=[]
    for run in sorted((root/'evidence').iterdir()):
        for p in sorted(run.glob('b*/result.json')):
            row=json.loads(p.read_text())
            row.update(run=run.name,case=p.parent.name,
                latency_accepted=row['control_cv']<=.05,
                intervention_accepted=run.name.startswith(('cold-','diagnostic-')),
                independent_unit='one_model_process')
            rows.append(row)
    return rows


def compare(rows):
    result=[]
    for case in ('b1-l128-o32','b8-l128-o32'):
        pairs=[]
        for rep in range(3):
            selected={r['run']:r for r in rows if r['case']==case}
            names=[f'cold-full-pa-cache-{backend}-r{rep}' for backend in ('pa','fia')]
            if not all(n in selected for n in names):continue
            pa,fia=(selected[n] for n in names)
            pairs.append(dict(replicate=rep,pa_ms=pa['control_median_ms'],fia_ms=fia['control_median_ms'],
                output_matches=pa['output_sha256']==fia['output_sha256'],
                stability_passed=pa['latency_accepted'] and fia['latency_accepted'],
                fia_vs_pa_change_pct=100*(fia['control_median_ms']/pa['control_median_ms']-1)))
        accepted=len(pairs)==3 and all(p['output_matches'] and p['stability_passed'] for p in pairs)
        result.append(dict(case=case,pairs=pairs,accepted=accepted,
            note='same FULL graph mode and native PA cache; backend intervention checked by device probe'))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();root=a.root
    rows=collect(root);comparisons=compare(rows)
    (root/'inference-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    (root/'backend-comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    fields=['run','case','control_median_ms','control_cv','latency_accepted','intervention_accepted','output_sha256']
    with (root/'inference-results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants=[('cold-piecewise-dense','PIECEWISE / shared KV / FIA'),('cold-piecewise-pa-cache','PIECEWISE / PA cache / FIA'),('cold-full-pa-cache-fia','FULL / PA cache / FIA'),('cold-full-pa-cache-pa','FULL / PA cache / PA')]
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,case in zip(axes,('b1-l128-o32','b8-l128-o32')):
        for index,(prefix,label) in enumerate(variants):
            values=[r for r in rows if r['case']==case and r['run'] in [f'{prefix}-r{n}' for n in range(3)]]
            valid=[r['control_median_ms'] for r in values if r['latency_accepted']]
            invalid=[r['control_median_ms'] for r in values if not r['latency_accepted']]
            if valid:ax.scatter([index]*len(valid),valid,color='#2563eb',s=38)
            if invalid:ax.scatter([index]*len(invalid),invalid,color='#dc2626',marker='x',s=48)
        ax.set_xticks(range(len(variants)),[x[1].replace(' / ','\n') for x in variants]);ax.set(ylabel='Unprofiled generate latency (ms)',title=case)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Independent process repeats; red crosses fail CV <= 5% (diagnostic only)')
    (root/'figures').mkdir(exist_ok=True)
    for suffix in ('png','svg','pdf'):fig.savefig(root/'figures'/f'graph-backend-decomposition.{suffix}',dpi=180)
    plt.close(fig)
    lines=['# AIC/AIV 协同探索：图与 attention 后端拆解','',
        '固定 Qwen3-8B、TP4、BF16、物理卡4–7、1654 KV blocks、greedy seed42、输入128/输出32；每次模型进程为独立重复。',
        '原生 PIECEWISE dense 是受限布局实验；FULL 使用官方 PA 连续缓存。跨图模式差异仍包含融合/发射/图回放变化，不直接归因于单项 CPU 或 AIC/AIV 调度。','',
        '初次共享 AOT cache 的运行只保留诊断，不进入后端因果结论；cold 条件按进程使用独立编译缓存。',
        '设备探针四卡每卡1152个 FIA 任务、0个 PA 任务、0个整缓存 Slice，支持 FULL 模式下连续PA cache/FIA路径可执行。其任务追踪约20.5%开销，只用于路径识别。','',
        '|配置|负载|独立进程|无profiler中位时延 ms|稳定条件数|','|---|---|---:|---:|---:|']
    for prefix,label in variants:
        for case in ('b1-l128-o32','b8-l128-o32'):
            values=[r for r in rows if r['case']==case and r['run'] in [f'{prefix}-r{n}' for n in range(3)]]
            if values:lines.append(f"|{label}|{case}|{len(values)}|{statistics.median(r['control_median_ms'] for r in values):.3f}|{sum(r['latency_accepted'] for r in values)}/{len(values)}|")
    lines+=['', '三次独立b1重复中，FULL/PA和FULL/FIA输出一致；后端变化远小于PIECEWISE/FULL路径差异。',
        '官方独立K/V分配的PIECEWISE实验用于检查storage owner差异；与FULL/FIA比较仍只隔离到图执行路径及伴随融合行为，尚不能归因为单个CPU发射环节。',
        'b1四卡的32个实际请求step对应block table序列完全一致；终止后的空调度迭代数为2/1/2，完整scheduler行并不完全相同，差异保留在b1-block-table-equivalence.json。']
    systems=json.loads((root/'system-summary.json').read_text()) if (root/'system-summary.json').exists() else []
    if systems:
        lines+=['', '同段系统采样（每条件一次诊断，不冒充三次PMU重复）：', '',
            '|路径|负载|物理卡|AIC忙碌 %|AIV忙碌 %|HBM读 GiB/调用|HBM写 GiB/调用|',
            '|---|---|---:|---:|---:|---:|---:|']
        for row in systems:
            if row['device']!=4:continue
            lines.append(f"|{row['run']}|{row['case']}|4|{row['aic_mean_pct']:.2f}|{row['aiv_mean_pct']:.2f}|{row['hbm_read_bytes_per_request']/2**30:.3f}|{row['hbm_write_bytes_per_request']/2**30:.3f}|")
        lines+=['', '任务trace的attention是MIX_AIC，未拆解内部AIC/AIV子事件；不把混合算子外层区间算作双引擎同时忙。系统样本与task trace来自不同请求，不作逐点拼接。']
    lines+=['','表格保留不稳定条件，不将其时延作为性能排名；图中红叉单独标出。输出哈希与全部控制请求见 inference-results.json 和 evidence。',
        '', '训练既有 trace 审计覆盖 seq128/512 × 四卡；API事件标识不足以恢复生产者/消费者关系，readiness 仍为 unknown。高开销 trace 的算子任务时长不能当作正常训练关键路径。',
        '', '[后端配对比较](backend-comparisons.json) · [训练依赖审计](training/retained-trace-dependency-audit.json) · [验收](acceptance.json)',
        '', '未确认真实模型的合法 ready AIC 候选时，不进入真实模型 co-run；未通过候选正确性/稳定性门槛时，不进入多请求优化验收。合成并发既有约4%改善不替代这两项门槛。']
    correctness=root/'full-path-diverse-output-analysis.json'
    if correctness.exists():
        lines+=['', '## 多提示词严格门槛', '',
            '原生FULL/PA及实验FULL/FIA均出现同条件重复输出差异；这不单独证明内存损坏，也不归咎于dense布局。原始token和首次差异记录在full-path-diverse-output-analysis.json。',
            '未经过的长输入条件保留未完成，不当作通过。当前只接受固定fixture下的路径和输出配对证据。']
    for length in (128,512):
        sample=root/f'training/seq{length}/sample-evidence.json'
        if not sample.exists():continue
        quality=json.loads(sample.read_text())
        lines+=['', f'## 训练 seq{length} 新观测', '',
            f"96步执行通过；sample-only采样质量：{quality['measurement_quality']}。每条件本轮一个新进程，不冒充三个独立重复。"]
        for row in quality['ranks']:
            if row['rank']!=0:continue
            lines.append(f"物理卡4：采样前CV {100*row['warmup_cv']:.2f}%，系统采集耗时变化 {100*row['profile_overhead_fraction']:.2f}%，完整step窗口 {len(row['windows'])}。")
            with (root/f'training/seq{length}/rank0/pmu/core_timeseries.csv').open() as stream:
                bins=list(csv.DictReader(stream))
            valid=[b for b in bins if b['valid']=='True']
            weights=[int(b['window_end_ns'])-int(b['window_start_ns']) for b in valid]
            means={engine:sum(float(b[engine+'_activity_pct'])*w for b,w in zip(valid,weights))/sum(weights) for engine in ('aic','aiv')}
            lines.append(f"物理卡4按有效bin时长加权：AIC {means['aic']:.2f}%，AIV {means['aiv']:.2f}%；有效bin {len(valid)}/{len(bins)}。")
        lines+=['静态IR及操作数依赖单独归档；静态stream ID不等价于运行时队列就绪，缺少同次设备task映射时readiness仍为unknown。']
    lines+=['', '## 后续切入点', '',
        '优先分解连续KV/FIA下FULL与PIECEWISE的图边界、回放间隙和伴随融合变化；其次调查多提示词greedy输出漂移（固定调度与logit差距）。',
        '训练侧将优化器/张量转换链的融合或分块作为待验证假设；未证明HBM饱和、未证明真实AIC任务已ready，不将这些假设列为已实现加速。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
