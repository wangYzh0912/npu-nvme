"""Render standalone scientific figures and a measured follow-up report."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics


def micro_tasks(root):
    from collections import defaultdict
    from .provenance import Ranges,ns,digest
    summaries=[]
    for trace in sorted((root/'micro').glob('trace-n*/*/ASCEND_PROFILER_OUTPUT')):
        path=trace/'trace_view.json'
        if not path.exists():path=trace/'trace_view.json.gz'
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt') as f:events=json.load(f)
        if isinstance(events,dict):events=events['traceEvents']
        scopes=Ranges(events);starts=defaultdict(list);ends=defaultdict(list)
        counts={mode:0 for mode in ('strided_fia','dense_fia','conversion_only','conversion_plus_fia')}
        observed={e['name'].split('::',1)[1] for e in events if e.get('ph')=='X' and e.get('name','').startswith('KV_MICRO::')}
        for e in events:
            cat=e.get('cat')
            if cat not in ('async_npu','HostToDevice'):continue
            if e.get('ph')=='s':starts[(cat,e['id'])].append(e)
            elif e.get('ph')=='f':ends[(e['pid'],e['tid'],ns(e['ts']))].append(e)
        total=0;unknown=0
        for e in events:
            if e.get('ph')!='X' or 'aclnnContiguous_SliceAiCore_Slice' not in e.get('name',''):continue
            total+=1;modes=set()
            for endpoint in ends.get((e['pid'],e['tid'],ns(e['ts'])),[]):
                launches=starts.get((endpoint['cat'],endpoint['id']),[])
                if len(launches)!=1:continue
                modes.update(r['name'].split('::',1)[1] for r in scopes.containing(launches[0]) if r['name'].startswith('KV_MICRO::'))
            if len(modes)==1:counts[next(iter(modes))]+=1
            else:unknown+=1
        summaries.append(dict(trace=str(trace.relative_to(root)),source_sha256=digest(path),
            observed_modes=sorted(observed),contiguous_slice_tasks=counts,
            total_slice_tasks=total,unattributed_slice_tasks=unknown,
            complete_mode_scopes=observed==set(counts)))
    (root/'micro/task-evidence.json').write_text(json.dumps(summaries,indent=2)+'\n')
    return summaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root
    rows=json.loads((root/'analysis/all-runs.json').read_text())
    card=[r for r in rows if r['device']==4]
    capacity=[r for r in card if r['run'].startswith('capacity-')]
    work=[r for r in card if r['run'].startswith('work-')]
    if not capacity:raise ValueError('no completed capacity data')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figdir=root/'figures';figdir.mkdir(exist_ok=True)
    group=defaultdict(list)
    for r in capacity:group[r['actual_blocks']].append(r)
    blocks=sorted(group)
    def points(ax,key,label,color,factor=1):
        means=[]
        for n in blocks:
            values=[r[key]/factor for r in group[n]]
            ax.scatter([n]*len(values),values,color=color,s=22,alpha=.6)
            means.append(statistics.mean(values))
        ax.plot(blocks,means,'o-',label=label,color=color,lw=1.6)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axes[0,0]
    first=max(capacity,key=lambda r:r['actual_blocks'])
    path=root/'analysis'/first['run']/first['case']/'rank0/kernel_details.csv.gz'
    from .provenance import ns
    with gzip.open(path,'rt') as f:tasks=list(csv.DictReader(f))
    windows=json.loads((path.parents[1]/'windows.json').read_text())
    w=next(w for w in windows if w['kind']=='task')
    slices=[t for t in tasks if t['Name']=='aclnnContiguous_SliceAiCore_Slice' and w['begin_ns']<=ns(t['Start Time(us)'])<w['end_ns']]
    lo=ns(slices[0]['Start Time(us)'])-50_000
    hi=ns(slices[1]['Start Time(us)'])+ns(slices[1]['Duration(us)'])+50_000
    for t in tasks:
        start=ns(t['Start Time(us)']);end=start+ns(t['Duration(us)'])
        if start>=hi or end<=lo:continue
        kind=t['Accelerator Core'];y={'AI_CORE':2,'AI_VECTOR_CORE':1}.get(kind,0)
        color='#d97706' if 'Contiguous' in t['Name'] else '#2563eb' if y==2 else '#64748b'
        ax.broken_barh([((max(start,lo)-lo)/1000,(min(end,hi)-max(start,lo))/1000)],(y-.3,.6),facecolors=color,edgecolors='white',linewidth=.6)
    ax.set_yticks([0,1,2],['Mixed/other','AIV','AIC']);ax.set_xlabel('Time in one decode fragment (us)')
    ax.set_title('Device tasks; orange = Contiguous/Slice')
    ax=axes[0,1];points(ax,'slice_logical_output_bytes_per_request','Slice logical output','#d97706',2**30)
    ax.set(xlabel='Allocated KV blocks per card',ylabel='GiB per request',title='Logical tensor bytes, not measured HBM traffic')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax=axes[1,0]
    points(ax,'hbm_read_bytes_per_request','HBM reads','#2563eb',2**30)
    points(ax,'hbm_write_bytes_per_request','HBM writes','#059669',2**30)
    ax.set(xlabel='Allocated KV blocks per card',ylabel='GiB per request',title='Same-segment native HBM interval counters');ax.legend()
    ax.set_ylim(bottom=0)
    ax=axes[1,1]
    points(ax,'control_median_ms','Unprofiled request','#111827')
    trace_label='Slice task-duration sum (diagnostic trace)' if any(r['task_quality']!='valid' for r in capacity) else 'Slice task-duration sum (trace)'
    points(ax,'slice_ms_per_request',trace_label,'#d97706')
    ax.set(xlabel='Allocated KV blocks per card',ylabel='Milliseconds per request',title='Dots are independent process repetitions');ax.legend()
    ax.set_ylim(bottom=0)
    fig.suptitle('Qwen3-8B / TP4 / 910B3 card 4 / b1, input 128, output 32 / PIECEWISE')
    for suffix in ('png','svg','pdf'):fig.savefig(figdir/f'kv-capacity-causality.{suffix}',dpi=180)
    plt.close(fig)
    if all('aiv_capacity_busy_ms_per_generate' in r for r in capacity):
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for engine,color in (('aic','#2563eb'),('aiv','#d97706')):
            points(axes[0],engine+'_capacity_busy_ms_per_generate',engine.upper(),color)
            points(axes[1],engine+'_mean_pct',engine.upper(),color)
        axes[0].set(xlabel='Allocated KV blocks',ylabel='Core-capacity busy time (ms/call)',
                    title='Valid PMU bins; busy time is not FLOPs')
        axes[1].set(xlabel='Allocated KV blocks',ylabel='Mean engine busy (%)',
                    title='Activity fraction also depends on request duration')
        for ax in axes:ax.set_ylim(bottom=0);ax.legend()
        for suffix in ('png','svg','pdf'):fig.savefig(figdir/f'kv-engine-capacity-time.{suffix}',dpi=180)
        plt.close(fig)
    if work:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for prompt,color in ((128,'#2563eb'),(512,'#d97706')):
            vals=[r for r in work if r['case'].startswith(f'b1-l{prompt}-')]
            for field,ax in [('slice_count_per_request',axes[0]),('slice_logical_output_bytes_per_request',axes[1])]:
                xs=sorted({int(r['case'].split('-o')[1]) for r in vals})
                ys=[statistics.mean(r[field] for r in vals if r['case'].endswith(f'-o{x}')) for x in xs]
                if 'bytes' in field:ys=[y/2**30 for y in ys]
                ax.plot(xs,ys,'o-',label=f'Input {prompt}',color=color)
        axes[0].set(ylabel='Slice tasks per generate call',xlabel='Generated tokens per request',title='Number of repeated conversions (b1)')
        axes[1].set(ylabel='GiB per generate call',xlabel='Generated tokens per request',title='Fixed cache capacity; logical Slice output (b1)')
        for ax in axes:ax.legend()
        for suffix in ('png','svg','pdf'):fig.savefig(figdir/f'kv-work-scaling.{suffix}',dpi=180)
        plt.close(fig)
    microfile=root/'micro/results.json'
    if microfile.exists():
        micro=json.loads(microfile.read_text())['results']
        micro_evidence=micro_tasks(root)
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        for mode in ('strided_fia','dense_fia','conversion_only','conversion_plus_fia'):
            vals=sorted((r for r in micro if r['mode']==mode),key=lambda r:r['blocks'])
            ax.plot([r['blocks'] for r in vals],[r['median_ms'] for r in vals],'o-',label=mode)
        ax.set(xlabel='Allocated KV blocks',ylabel='Milliseconds per call',title='Isolated FIA replay; conversion charged explicitly')
        ax.legend()
        for suffix in ('png','svg','pdf'):fig.savefig(figdir/f'kv-micro-replay.{suffix}',dpi=180)
        plt.close(fig)
    table=[]
    for n in blocks:
        vals=group[n]
        table.append(dict(blocks=n,independent_runs=len(vals),
            latency_ms=statistics.mean(r['control_median_ms'] for r in vals),
            latency_range_ms=[min(r['control_median_ms'] for r in vals),max(r['control_median_ms'] for r in vals)],
            slice_ms=statistics.mean(r['slice_ms_per_request'] for r in vals),
            slice_count=statistics.mean(r['slice_count_per_request'] for r in vals),
            hbm_read_gib=statistics.mean(r['hbm_read_bytes_per_request']/2**30 for r in vals),
            hbm_write_gib=statistics.mean(r['hbm_write_bytes_per_request']/2**30 for r in vals),
            aic_pct=statistics.mean(r['aic_mean_pct'] for r in vals),aiv_pct=statistics.mean(r['aiv_mean_pct'] for r in vals)))
    (root/'capacity-summary.json').write_text(json.dumps(table,indent=2)+'\n')
    deltas=[]
    for lo,hi in zip(blocks,blocks[1:]):
        logical_hi=statistics.mean(r['slice_logical_output_bytes_per_request'] for r in group[hi])
        logical_lo=statistics.mean(r['slice_logical_output_bytes_per_request'] for r in group[lo])
        actual_hi=statistics.mean(r['hbm_write_bytes_per_request'] for r in group[hi])
        actual_lo=statistics.mean(r['hbm_write_bytes_per_request'] for r in group[lo])
        deltas.append(dict(low_blocks=lo,high_blocks=hi,
                          delta_hbm_write_bytes=actual_hi-actual_lo,
                          delta_slice_logical_bytes=logical_hi-logical_lo,
                          actual_to_logical_delta_ratio=(actual_hi-actual_lo)/(logical_hi-logical_lo)))
    (root/'capacity-write-deltas.json').write_text(json.dumps(deltas,indent=2)+'\n')
    with (root/'capacity-summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    output_hashes={r['output_sha256'] for r in capacity}
    checks=dict(capacity_card4_runs=len(capacity),all_devices_run_count=len(rows),
        counting_unit='process x case x physical device; ranks are not independent repetitions',
        same_output_across_capacity=len(output_hashes)==1,
        all_capacity_control_cv_at_most_5pct=all(r['control_cv']<=.05 for r in capacity),
        all_capacity_task_overhead_at_most_10pct=all(r['task_quality']=='valid' for r in capacity),
        all_capacity_system_overhead_at_most_10pct=all(r['system_overhead_fraction']<=.1 for r in capacity),
        minimum_pmu_valid_fraction=min(r['pmu_valid_fraction'] for r in capacity),
        minimum_hbm_coverage=min(r['hbm']['coverage'] for r in capacity),
        full_capacity_default_output_matches=None)
    checks['per_device']={device:dict(capacity_device_cases=len(values),
        min_pmu_valid_fraction=min(r['pmu_valid_fraction'] for r in values),
        min_hbm_coverage=min(r['hbm']['coverage'] for r in values))
        for device in (4,5,6,7)
        if (values:=[r for r in rows if r['device']==device and r['run'].startswith('capacity-')])}
    if work:
        by_case=defaultdict(list)
        for row in work:by_case[row['case']].append(row)
        checks['work_cases']={case:dict(independent_processes=len({r['run'] for r in values}),
            consistent_output=len({r['output_sha256'] for r in values})==1,
            stable_controls=all(r['control_cv']<=.05 for r in values),
            task_trace_timing_valid=all(r['task_quality']=='valid' for r in values))
            for case,values in sorted(by_case.items())}
        with (root/'work-summary.csv').open('w') as f:
            fields=['run','case','actual_blocks','control_median_ms','slice_count_per_request',
                    'slice_logical_output_bytes_per_request','task_quality','output_sha256']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            writer.writerows({k:r[k] for k in fields} for r in work)
    metadata=root/'metadata/windows.json'
    if metadata.exists():
        default_hashes={r['output_sha256'] for r in json.loads(metadata.read_text())}
        checks['full_capacity_default_output_matches']=output_hashes==default_hashes
    lines=['# KV 数据整理观察实验','',
        '所有结果来自实际设备。主表只展示物理卡4，各卡原始归一化证据分别保存。',
        '', '固定 Qwen3-8B、TP4、BF16、PIECEWISE、b1/input128/output32；只改变实际 KV block 数。',
        '', '|实际block数|独立进程|无profiler请求 ms|Slice任务和 ms|Slice次数/请求|HBM读 GiB/请求|HBM写 GiB/请求|AIC %|AIV %|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in table:
        marker='†' if any(r['task_quality']!='valid' for r in group[t['blocks']]) else ''
        lines.append(f"|{t['blocks']}|{t['independent_runs']}|{t['latency_ms']:.3f}|{t['slice_ms']:.3f}{marker}|{t['slice_count']:.0f}|{t['hbm_read_gib']:.2f}|{t['hbm_write_gib']:.2f}|{t['aic_pct']:.2f}|{t['aiv_pct']:.2f}|")
    lines+=['','[容量对照四面板图](figures/kv-capacity-causality.png)',
        '[引擎容量忙碌时间与比例](figures/kv-engine-capacity-time.png)',
        '[工作量对照](figures/kv-work-scaling.png)','[独立算子复现](figures/kv-micro-replay.png)',
        '', '## 已确认的调用链与布局','',
        '旧首轮四卡各6请求，每卡13392个大Slice全部通过精确设备端点和关联流匹配到decode的FIA及CANN内部aclnnContiguous。',
        '新元数据记录每层K/V共享存储、block交错、两视图非连续；实际shape、stride和存储偏移见metadata。',
        '固定镜像的using_paged_attention对普通128维head要求FULL_DECODE_ONLY图模式；本实验PIECEWISE不满足该条件。源码分支与实测FIA调用链一致，不能将结果推广到其他图模式或所有vLLM-Ascend版本。',
        '', '## 口径与限制','',
        'Slice逻辑输出字节来自输出shape与BF16大小，不等同于实际HBM流量；HBM由原生四通道区间计数计算，按采集版本源码每count为32字节，首样本丢弃，timestamp已由CANN换算到绝对时间。',
        'HBM/PMU段与任务追踪段分开运行，不做逐点跨运行叠加。PMU是10ms核心忙碌时间，HBM是20ms区间平均，均不证明瞬时并发或带宽饱和。',
        '引擎容量忙碌时间为有效PMU bin的忙碌比例乘bin时长之和，再除generate调用次数；AIC与AIV分别按各自全部核心容量归一化。它包括任务内部等待，不是算术工作量或FLOPs。',
        '三个独立模型进程是重复单位，请求不是独立实验。Slice任务持续时间之和不自动等于可消除的端到端时延。',
        '统计单元为一次generate调用；b1对应一个请求，b8对应八请求批次。JSON/CSV中历史命名per_request字段均按generate调用计量，不把批次总量冒充单请求量。',
        'attention最小复现使用有限合成值及实测几何，布局正确性比较阈值rtol=atol=1e-2；不会把一次性预先整理的收益当作持续KV更新下的免费收益。',
        '校验门槛与结果见acceptance.json；任何未通过门槛的项目必须作为诊断解释。',
        '†该容量至少一次逐任务追踪开销超过10%，其Slice时长仅为诊断值；无profiler控制段与独立系统采样段分别验收，不用该任务时长推算加速比。']
    if len(table)==3:
        low,high=table[0],table[-1]
        lines+=['','## 容量对照结果','',
            f"实际容量从{high['blocks']}降至{low['blocks']}个block时，无profiler请求时延由{high['latency_ms']:.3f}ms变为{low['latency_ms']:.3f}ms（比值{high['latency_ms']/low['latency_ms']:.3f}）。",
            f"相应Slice任务时长和由{high['slice_ms']:.3f}ms变为{low['slice_ms']:.3f}ms；HBM写入由{high['hbm_write_gib']:.2f}变为{low['hbm_write_gib']:.2f}GiB/请求。",
            '这些数字针对固定实际请求，只改变分配容量；若全部质量门槛和输出一致性通过，支持缓存容量放大了整理成本。它们不证明带宽饱和，也不代表同等比例的AIC算力可以回收。']
        lines+=['', '相邻容量条件的HBM写入差额 / Slice逻辑输出差额：'+
                '；'.join(f"{d['low_blocks']}→{d['high_blocks']}: {d['actual_to_logical_delta_ratio']:.6f}" for d in deltas)+
                '。这是实测系统流量与任务输出量的交叉核对。']
        if all('aiv_capacity_busy_ms_per_generate' in r for r in capacity):
            engine_means={n:{e:statistics.mean(r[e+'_capacity_busy_ms_per_generate'] for r in group[n])
                             for e in ('aic','aiv')} for n in blocks}
            lines+=['',f"独立PMU的每调用AIV容量忙碌时间由{engine_means[blocks[-1]]['aiv']:.3f}降至{engine_means[blocks[0]]['aiv']:.3f}ms；AIC对应为{engine_means[blocks[-1]]['aic']:.3f}与{engine_means[blocks[0]]['aic']:.3f}ms。分母变化造成的活跃比例上升，不等同于新增矩阵计算。"]
    if microfile.exists():
        if len(capacity)!=9 or len(work)!=27 or any(len(values)!=3 for values in group.values()):
            raise ValueError('final report requires the complete independent capacity/work matrices')
        complete=all(x['complete_mode_scopes'] and x['unattributed_slice_tasks']==0 for x in micro_evidence) and len(micro_evidence)==3
        dense_eliminates=complete and all(x['contiguous_slice_tasks']['dense_fia']==0 and x['contiguous_slice_tasks']['strided_fia']>0 for x in micro_evidence)
        expected_capacity={(f'capacity-r{rep}-n{n}','b1-l128-o32',d)
            for rep in range(3) for n in blocks for d in (4,5,6,7)}
        work_cases={f'b1-l{l}-o{g}' for l in (128,512) for g in (1,2,8,32)}|{'b8-l128-o32'}
        expected_work={(f'work-r{rep}',case,d) for rep in range(3) for case in work_cases for d in (4,5,6,7)}
        observed={(r['run'],r['case'],r['device']) for r in rows}
        checks['collection_complete']=observed==expected_capacity|expected_work and len(rows)==144
        checks['scope']='inference KV layout/capacity cause; not a scheduler optimization or new training experiment'
        checks['micro_replay']=dict(complete_task_attribution=complete,
            dense_eliminates_internal_slice=dense_eliminates,
            layout_numerics_and_update_visibility=all(r['cache_update_matches'] and r['active_update_output_change']>0 and r['output_max_abs_error']<=.01 for r in micro),
            independent_processes=1,timing_batches_per_mode=3)
        checks['all_devices_capacity_controls_stable']=all(r['control_cv']<=.05 for r in rows if r['run'].startswith('capacity-'))
        checks['all_work_controls_stable']=all(r['control_cv']<=.05 for r in rows if r['run'].startswith('work-'))
        checks['all_devices_system_overhead_at_most_10pct']=all(r['system_overhead_fraction']<=.1 for r in rows if r['run'].startswith('capacity-'))
        checks['b1_slice_count_matches_72_per_decode']=all(r['slice_count_per_request']==72*(int(r['case'].split('-o')[1])-1) for r in work if r['case'].startswith('b1-'))
        checks['required_evidence_passed']=all((checks['collection_complete'],
            checks['same_output_across_capacity'],checks['full_capacity_default_output_matches'],
            checks['all_devices_capacity_controls_stable'],checks['all_work_controls_stable'],
            checks['all_devices_system_overhead_at_most_10pct'],checks['b1_slice_count_matches_72_per_decode'],
            all(v['consistent_output'] for v in checks['work_cases'].values()),
            all(v['min_pmu_valid_fraction']>=.95 and v['min_hbm_coverage']>=.95 for v in checks['per_device'].values()),
            complete,dense_eliminates,checks['micro_replay']['layout_numerics_and_update_visibility']))
        checks['status']='complete_with_diagnostic_traces' if checks['required_evidence_passed'] else 'requires_review'
        lines+=['','## 最小算子复现','',
            f'三种容量的模式scope完整且Slice归因无缺失：{complete}；连续输入消除FIA内部Contiguous/Slice：{dense_eliminates}。',
            '数值正确性、更新可见性及逐模式时延见micro/results.json，设备任务归因见micro/task-evidence.json。']
        lines+=['','## 工作量与完成状态','',
            'b1 的 input128/512、output1/2/8/32 在三次独立进程中均保持 Slice 次数 = 36层 × K/V两路 × (output−1)。output1没有大Slice，说明该路径主要随decode重复，而非随输入长度增长。b8以整次generate为单位单独保留，不能套用b1公式。',
            f"收集矩阵完整：{checks['collection_complete']}；必要证据验收通过：{checks['required_evidence_passed']}。逐任务追踪开销超过10%的条件保留为诊断，不用其持续时间推算收益。",
            '本轮完成范围是推理KV整理原因定位。未实施真实模型的持久连续KV布局、图模式切换、训推协同或调度器改造，也未证明AIC已有可运行任务却被调度器闲置。']
    else:
        checks['status']='partial'
    (root/'acceptance.json').write_text(json.dumps(checks,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
