"""Applicability report: configuration coverage is not deployment prevalence."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from .scope import ROOT,save


def classify(row):
    if row['slice_count_per_generate'] is None:return 'task_trace_not_collected'
    if row['slice_count_per_generate']>0:
        return 'observed_full_cache_slice'
    if row['k_contiguous'] and row['v_contiguous']:
        return 'no_slice_with_dense_cache'
    return 'no_slice_observed_requires_path_evidence'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();root=a.root
    rows=json.loads((root/'summary.json').read_text())
    card=[r for r in rows if r['device']==4]
    if not card:raise ValueError('no complete hardware experiments')
    for row in rows:row['classification']=classify(row)
    save(root/'classified-results.json',rows)
    fields=['run','case','device','graph_mode','k_contiguous','v_contiguous','slice_count_per_generate','control_median_ms','control_cv','task_quality','classification','output_sha256']
    with (root/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    screen=[r for r in card if r['run'].endswith('screen')]
    labels=sorted({r['run'] for r in screen})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for batch,color,offset in ((1,'#2563eb',-.16),(8,'#d97706',.16)):
        subset={r['run']:r for r in screen if r['case']==f'b{batch}-l128-o32'}
        ids=[i for i,label in enumerate(labels) if label in subset]
        for field,ax in [('slice_count_per_generate',axes[0]),('control_median_ms',axes[1])]:
            ax.barh([i+offset for i in ids],[subset[labels[i]][field] for i in ids],height=.3,label=f'b{batch}',color=color)
    for ax in axes:
        ax.set_yticks(range(len(labels)),[l.replace('-screen','').replace('official-','release/').replace('current-','custom/') for l in labels]);ax.invert_yaxis();ax.legend()
    axes[0].set(xlabel='Large Contiguous/Slice tasks per generate',title='Observed device tasks; no copy inferred from names alone')
    axes[1].set(xlabel='Unprofiled generate latency (ms)',title='Screen: one process per configuration, diagnostic')
    figs=root/'figures';figs.mkdir(exist_ok=True)
    for suffix in ('png','svg','pdf'):fig.savefig(figs/f'applicability-screen.{suffix}',dpi=170)
    plt.close(fig)
    confirm=[r for r in card if r['run'].startswith('official-qwen') and '-r' in r['run'] and '-blocktable-' not in r['run']]
    if confirm:
        cases=['b1-l128-o1','b1-l128-o32','b1-l2048-o32','b8-l128-o32']
        fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained',sharey=True)
        variants=(('native','PIECEWISE native','#2563eb'),('dense','PIECEWISE dense','#059669'),('pa','FULL_DECODE_ONLY + PA','#d97706'))
        for ax,model in zip(axes,('/models/Qwen3-8B','/models/Qwen3-4B')):
            selected=[r for r in confirm if r['model']==model]
            ax.set_title(model.rsplit('/',1)[-1])
            for key,label,color in variants:
                values={}
                for row in selected:
                    is_key=('-dense-' in row['run']) if key=='dense' else ('full_decode_only' in row['run']) if key=='pa' else ('-native-' in row['run'] and 'full_decode_only' not in row['run'])
                    if is_key:values.setdefault(row['case'],[]).append(row)
                xs=[];means=[]
                for i,case in enumerate(cases):
                    if case not in values:continue
                    valid=[r['control_median_ms'] for r in values[case] if r['control_cv']<=.05]
                    invalid=[r['control_median_ms'] for r in values[case] if r['control_cv']>.05]
                    if valid:
                        xs.append(i);means.append(statistics.mean(valid));ax.scatter([i]*len(valid),valid,color=color,s=17,alpha=.55)
                    if invalid:ax.scatter([i]*len(invalid),invalid,color=color,s=30,marker='x')
                if xs:ax.plot(xs,means,'o-',label=label,color=color,lw=1.6)
            ax.set_xticks(range(len(cases)),cases,rotation=25,ha='right');ax.legend()
        axes[0].set_ylabel('Unprofiled milliseconds per generate call')
        fig.suptitle('vLLM-Ascend 0.30: lines average controls with CV <= 5%; x = unstable, excluded')
        for suffix in ('png','svg','pdf'):fig.savefig(figs/f'layout-path-confirmation.{suffix}',dpi=170)
        plt.close(fig)
    groups=defaultdict(list)
    for r in card:
        if '-r' in r['run']:groups[(r['run'].rsplit('-r',1)[0],r['case'])].append(r)
    comparisons=[]
    for (variant,case),values in sorted(groups.items()):
        comparisons.append(dict(variant=variant,case=case,independent_processes=len(values),
            stable_controls=all(v['control_cv']<=.05 for v in values),
            consistent_output=len({v['output_sha256'] for v in values})==1,
            mean_control_ms=statistics.mean(v['control_median_ms'] for v in values),
            min_control_ms=min(v['control_median_ms'] for v in values),max_control_ms=max(v['control_median_ms'] for v in values),
            slices=statistics.mean(v['slice_count_per_generate'] for v in values) if all(v['slice_count_per_generate'] is not None for v in values) else None))
    save(root/'independent-comparisons.json',comparisons)
    extension=[r for r in card if r['run'].startswith('extend-qwen8b-')]
    if extension:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,case in zip(axes,('b1-l128-o128','b8-l2048-o32')):
            modes=('piecewise-native','piecewise-dense','full_decode_only-native-pa')
            for i,mode in enumerate(modes):
                vals=[r for r in extension if r['case']==case and r['run'].rsplit('-r',1)[0]=='extend-qwen8b-'+mode]
                valid=[r['control_median_ms'] for r in vals if r['control_cv']<=.05]
                invalid=[r['control_median_ms'] for r in vals if r['control_cv']>.05]
                if valid:ax.scatter([i]*len(valid),valid,color='#2563eb',s=25);ax.plot([i-.2,i+.2],[statistics.mean(valid)]*2,color='#111827')
                if invalid:ax.scatter([i]*len(invalid),invalid,color='#d97706',marker='x')
            ax.set_title(case);ax.set_xticks(range(3),['PIECEWISE native','PIECEWISE dense','FULL_DECODE + PA'],rotation=20,ha='right');ax.set_ylim(bottom=0)
        axes[0].set_ylabel('Unprofiled ms per generate call')
        fig.suptitle('Qwen3-8B extension: rotated process order; x = control CV > 5%')
        for suffix in ('png','svg','pdf'):fig.savefig(figs/f'long-workload-extension.{suffix}',dpi=170)
        plt.close(fig)
    paired={}
    layout_rows=[r for r in card if r['run'].startswith(('official-qwen','extend-qwen')) and '-r' in r['run'] and '-blocktable-' not in r['run']]
    for r in layout_rows:
        if 'full_decode_only' in r['run']:continue
        key=(r['model'],r['case'],r['run'].rsplit('-r',1)[1])
        item=paired.setdefault(key,{})
        variant='dense' if '-dense-' in r['run'] else 'native'
        item[variant]=dict(output_sha256=r['output_sha256'],latency_ms=r['control_median_ms'],
                           control_cv=r['control_cv'],slices=r['slice_count_per_generate'])
    layout_output_pairs=[dict(model=k[0],case=k[1],replicate=k[2],variants=v,
        matches=v['native']['output_sha256']==v['dense']['output_sha256'])
        for k,v in sorted(paired.items()) if {'native','dense'}<=v.keys()]
    layout_latency=[dict(model=x['model'],case=x['case'],replicate=x['replicate'],
        native_ms=x['variants']['native']['latency_ms'],dense_ms=x['variants']['dense']['latency_ms'],
        dense_latency_change_pct=(x['variants']['dense']['latency_ms']/x['variants']['native']['latency_ms']-1)*100,
        output_matches=x['matches'],controls_stable=max(x['variants']['native']['control_cv'],x['variants']['dense']['control_cv'])<=.05,
        native_full_cache_slices=x['variants']['native']['slices'],dense_full_cache_slices=x['variants']['dense']['slices'])
        for x in layout_output_pairs]
    latency_groups=defaultdict(list)
    for row in layout_latency:
        if row['controls_stable'] and row['output_matches']:
            latency_groups[(row['model'],row['case'])].append(row['dense_latency_change_pct'])
    save(root/'layout-latency-comparisons.json',layout_latency)
    block_table_pairs=[]
    for pair in layout_output_pairs:
        rep=pair['replicate'];case=pair['case']
        model_slug=pair['model'].split('Qwen3-')[1].lower()
        prefix=f'official-qwen{model_slug}-piecewise-'
        native=f'{prefix}native-r{rep}';dense=f'{prefix}dense-r{rep}'
        if rep=='0':
            candidate=f'{prefix}native-blocktable-r0'
            if (root/'evidence'/candidate/case/'block-table-probe-rank0.json').exists():native=candidate
            candidate=f'{prefix}dense-blocktable-r0'
            if (root/'evidence'/candidate/case/'block-table-probe-rank0.json').exists():dense=candidate
        probe_files=[(root/'evidence'/run/case/f'block-table-probe-rank{rank}.json')
                     for run in (native,dense) for rank in range(4)]
        if not all(path.exists() for path in probe_files):continue
        rank_matches={}
        for rank in range(4):
            rel=Path('evidence')/native/case/f'block-table-probe-rank{rank}.json'
            other=Path('evidence')/dense/case/f'block-table-probe-rank{rank}.json'
            left,right=root/rel,root/other
            a=json.loads(left.read_text());b=json.loads(right.read_text())
            normalize=lambda rows:[dict(num_reqs=row['num_reqs'],groups=row['groups']) for row in rows if row['num_reqs']]
            rank_matches[rank]=bool(a) and normalize(a)==normalize(b)
        exact=all(rank_matches.values())
        block_table_pairs.append(dict(model=pair['model'],case=case,replicate=rep,ranks=rank_matches,
            same_logical_block_ids=exact,
            interpretation='exact_host_block_table_match' if exact else
            'different_batch_schedule_or_block_assignment; output equivalence is separate evidence',
            native_run=native,dense_run=dense))
    save(root/'block-table-equivalence.json',block_table_pairs)
    failures={p.stem:json.loads(p.read_text()) for p in (root/'failures').glob('*.json')}
    unstable=[dict(run=r['run'],case=r['case'],device=r['device'],control_cv=r['control_cv']) for r in rows if r['control_cv']>.05]
    save(root/'unstable-controls.json',unstable)
    system_path=root/'system-summary.json'
    system=json.loads(system_path.read_text()) if system_path.exists() else []
    system_card=[r for r in system if r['device']==4]
    system_groups=defaultdict(list)
    for r in system_card:
        variant=next((x for x in ('native','dense','pa') if f'official-system-{x}-' in r['run']),None)
        if variant:system_groups[variant].append(r)
    system_comparisons=[]
    for variant in ('native','dense','pa'):
        if variant not in system_groups:continue
        values=system_groups[variant]
        fields=('aic_capacity_busy_ms_per_generate','aiv_capacity_busy_ms_per_generate',
                'hbm_read_bytes_per_request','hbm_write_bytes_per_request','system_overhead_fraction')
        system_comparisons.append(dict(variant=variant,independent_processes=len(values),
            **{name+'_mean':statistics.mean(v[name] for v in values) for name in fields},
            min_pmu_valid_fraction=min(v['pmu_valid_fraction'] for v in values),
            min_hbm_coverage=min(v['hbm']['coverage'] for v in values),
            min_hbm_valid_interval_fraction=min(v['hbm_valid_fraction'] for v in values)))
    save(root/'system-comparisons.json',system_comparisons)
    if system_groups:
        variants=[v for v in ('native','dense','pa') if v in system_groups]
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        panels=[(('control_median_ms','Unprofiled request','#111827',1),),
            (('aic_capacity_busy_ms_per_generate','AIC','#2563eb',1),('aiv_capacity_busy_ms_per_generate','AIV','#d97706',1)),
            (('hbm_read_bytes_per_request','HBM reads','#2563eb',2**30),('hbm_write_bytes_per_request','HBM writes','#059669',2**30))]
        for ax,items in zip(axes,panels):
            for field,label,color,divisor in items:
                means=[]
                for i,variant in enumerate(variants):
                    ys=[r[field]/divisor for r in system_groups[variant]]
                    ax.scatter([i]*len(ys),ys,color=color,s=18,alpha=.55);means.append(statistics.mean(ys))
                ax.plot(range(len(variants)),means,'o-',label=label,color=color)
            ax.set_xticks(range(len(variants)),['PIECEWISE native' if x=='native' else 'PIECEWISE dense' if x=='dense' else 'FULL_DECODE + PA' for x in variants],rotation=20,ha='right');ax.legend();ax.set_ylim(bottom=0)
        axes[0].set_ylabel('Request latency (ms/generate)')
        axes[1].set_ylabel('Engine capacity busy time\n(ms/generate; not FLOPs)')
        axes[2].set_ylabel('HBM traffic (GiB/generate)')
        fig.suptitle('Qwen3-8B / TP4 / 910B3 card 4 / KV 6616 blocks / b1 input128 output32')
        for suffix in ('png','svg','pdf'):fig.savefig(figs/f'layout-engine-hbm.{suffix}',dpi=170)
        plt.close(fig)
    corun_path=root/'corun/task-evidence.json'
    corun=json.loads(corun_path.read_text()) if corun_path.exists() else []
    corun_timings=[]
    for p in sorted((root/'corun').glob('corun-r*/results.json')):
        result=json.loads(p.read_text());timings={r['mode']:r['median_ms'] for r in result['results']}
        corun_timings.append(dict(run=p.parent.name,timings=timings,
            parallel_to_serial_ratio=timings['parallel_pair']/timings['serial_pair'],
            output_matches=result['output_matches']))
    save(root/'corun/timing-comparisons.json',corun_timings) if (root/'corun').exists() else None
    if corun and corun_timings:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        modes=('serial_pair','parallel_pair');colors={'AI_CORE':'#2563eb','AI_VECTOR_CORE':'#d97706'}
        for ax,mode in zip(axes,modes):
            tasks=[t for t in corun[0]['tasks'] if t['mode']==mode and t['engine'] in colors]
            lo=min(t['start_ns'] for t in tasks)
            for t in tasks:
                y=1 if t['engine']=='AI_CORE' else 0
                ax.broken_barh([((t['start_ns']-lo)/1e6,(t['end_ns']-t['start_ns'])/1e6)],(y-.25,.5),facecolors=colors[t['engine']])
            ax.set_yticks([0,1],['AIV copy','AIC matrix']);ax.set_xlabel('Device task time (ms)');ax.set_title(mode)
        fig.suptitle('Independent synthetic operations: task-envelope overlap, not core-level PMU')
        for suffix in ('png','svg','pdf'):fig.savefig(figs/f'independent-engine-overlap.{suffix}',dpi=170)
        plt.close(fig)
    acceptance=dict(device_cases=len(rows),screen_configurations=len(labels),
        screen_is_single_process_not_speedup_acceptance=True,
        all_controls_cv_at_most_5pct=all(r['control_cv']<=.05 for r in rows),
        unstable_controls=len(unstable),
        repetitions=sorted({x['independent_processes'] for x in comparisons}),
        failures=failures,deployments_sampled=False,
        no_population_majority_claim=True,
        layout_output_pairs=len(layout_output_pairs),all_native_dense_outputs_match=bool(layout_output_pairs) and all(x['matches'] for x in layout_output_pairs),
        stable_layout_comparisons=sum(x['controls_stable'] and x['output_matches'] for x in layout_latency),
        block_table_pairs=len(block_table_pairs),all_native_dense_block_tables_match=bool(block_table_pairs) and all(x['same_logical_block_ids'] for x in block_table_pairs),
        simultaneous_system_capture_rows=len(system),
        minimum_system_pmu_valid_fraction=min((r['pmu_valid_fraction'] for r in system),default=None),
        minimum_system_hbm_coverage=min((r['hbm']['coverage'] for r in system),default=None),
        minimum_system_hbm_valid_interval_fraction=min((r['hbm_valid_fraction'] for r in system),default=None),
        system_variants=len(system_comparisons),
        system_repeats={r['variant']:r['independent_processes'] for r in system_comparisons},
        synthetic_corun_repetitions=len(corun),
        synthetic_corun_flow_attribution_complete=bool(corun) and all(r['unattributed_device_tasks']==0 for r in corun))
    expected=set()
    cases=('b1-l128-o1','b1-l128-o32','b1-l2048-o32','b8-l128-o32')
    variants=('piecewise-native','piecewise-dense','full_decode_only-native-pa')
    for model in ('8b','4b'):
        for variant in variants:
            for rep in range(3):
                expected.update((f'official-qwen{model}-{variant}-r{rep}',case,d) for case in cases for d in (4,5,6,7))
    for variant in variants:
        for rep in range(3):
            expected.update((f'extend-qwen8b-{variant}-r{rep}',case,d) for case in ('b1-l128-o128','b8-l2048-o32') for d in (4,5,6,7))
    for rep in range(3):
        expected.update((f'v018-piecewise-r{rep}',case,d) for case in ('b1-l128-o32','b1-l2048-o32','b8-l128-o32') for d in (4,5,6,7))
        for variant in ('native','dense','pa'):
            expected.update((f'official-system-{variant}-r{rep}','b1-l128-o32',d) for d in (4,5,6,7))
    expected.update((f'correctness-sync-qwen8b-piecewise-{layout}-r0',case,d)
        for layout in ('native','dense') for case in ('b8-l128-o32','b8-l2048-o32') for d in (4,5,6,7))
    observed={(r['run'],r['case'],r['device']) for r in rows}
    missing=sorted(expected-observed)
    failed_conditions=[row for row in missing if row[0] in failures]
    unaccounted=[row for row in missing if row[0] not in failures]
    acceptance['required_model_device_cases']=len(expected)
    acceptance['missing_model_device_cases']=missing
    acceptance['failed_model_device_cases']=failed_conditions
    acceptance['unaccounted_model_device_cases']=unaccounted
    acceptance['collection_complete']=not unaccounted and len(corun)==3
    acceptance['b1_block_table_pairs_exact']=bool(block_table_pairs) and all(r['same_logical_block_ids'] for r in block_table_pairs if r['case'].startswith('b1-'))
    acceptance['b8_block_table_pairs_exact']=bool(block_table_pairs) and all(r['same_logical_block_ids'] for r in block_table_pairs if r['case'].startswith('b8-'))
    version_rows=[r for r in rows if r['run'].startswith('v018-piecewise-r')]
    acceptance['version_018_all_native_cache_contiguous_without_full_cache_slice']=len(version_rows)==36 and all(r['k_contiguous'] and r['v_contiguous'] and r['slice_count_per_generate']==0 for r in version_rows)
    diverse_pairs=[]
    for case in ('b8-l128-o32','b8-l2048-o32'):
        probes=[root/'evidence'/f'correctness-sync-qwen8b-piecewise-{layout}-r0'/case for layout in ('native','dense')]
        if not all((p/'input-fixture.json').exists() for p in probes):continue
        fixtures=[json.loads((p/'input-fixture.json').read_text()) for p in probes]
        windows=[json.loads((p/'windows.json').read_text()) for p in probes]
        control_tokens=[[w['output_token_ids'] for w in seq if w['kind'].startswith('control')] for seq in windows]
        metrics=[[w['request_metrics'] for w in seq if w['kind'].startswith('control')] for seq in windows]
        diverse_pairs.append(dict(case=case,distinct_inputs=len(set(fixtures[0]['input_sha256'])),
            same_inputs=fixtures[0]==fixtures[1],outputs_match=bool(control_tokens[0]) and control_tokens[0][0]==control_tokens[1][0],
            native_request_metrics=metrics[0],dense_request_metrics=metrics[1]))
    save(root/'diverse-batch-correctness.json',diverse_pairs)
    acceptance['diverse_batch_correctness']=len(diverse_pairs)==2 and all(x['distinct_inputs']==8 and x['same_inputs'] and x['outputs_match'] for x in diverse_pairs)
    hybrid_done=(root/'completed/hybrid-qwen35-27b-piecewise-screen.json').exists()
    hybrid_failure=(root/'failures/hybrid-qwen35-27b-piecewise-screen.json').exists()
    acceptance['hybrid_architecture_probe']='completed' if hybrid_done else 'failed_with_cause' if hybrid_failure else 'not_run'
    acceptance['required_evidence_passed']=all((acceptance['collection_complete'],
        not missing,
        acceptance['all_native_dense_outputs_match'],acceptance['b1_block_table_pairs_exact'],
        acceptance['diverse_batch_correctness'],len(system)==36,
        all(r['pmu_valid_fraction']>=.95 and r['hbm']['coverage']>=.95 and abs(r['system_overhead_fraction'])<=.1 for r in system)))
    acceptance['investigation_complete']=acceptance['collection_complete'] and (hybrid_done or hybrid_failure)
    acceptance['layout_general_correctness_accepted']=acceptance['diverse_batch_correctness']
    acceptance['status']='complete_with_diagnostics' if acceptance['required_evidence_passed'] else 'investigation_complete_correctness_unresolved' if acceptance['investigation_complete'] else 'in_progress'
    save(root/'acceptance.json',acceptance)
    save(root/'layout-output-equivalence.json',layout_output_pairs)
    lines=['# KV 整理适用范围实验','',
        '本轮优先定位触发边界。官方镜像标签与定制镜像是镜像来源，不自动代表两个独立代码版本；实际身份见 stack-identity。',
        '统计单位为一次 generate，b8表示八请求批次。筛查每配置一个模型进程，只作路径证据和诊断时延；独立重复结果另表列出。','',
        '|配置|工作负载|实际图模式|K/V连续|Slice/调用|无profiler ms|控制CV|',
        '|---|---|---|---|---:|---:|---:|']
    table_start=len(lines)-2
    for r in card:
        count='未采集' if r['slice_count_per_generate'] is None else f"{r['slice_count_per_generate']:.0f}"
        lines.append(f"|{r['run']}|{r['case']}|{r['graph_mode']}|{r['k_contiguous']}/{r['v_contiguous']}|{count}|{r['control_median_ms']:.3f}|{r['control_cv']:.2%}|")
    (root/'RESULTS.md').write_text('# 逐条件结果（物理卡4）\n\n'+'\n'.join(lines[table_start:])+'\n')
    lines=lines[:table_start]+[
        '整缓存整理具有明确的版本、物理布局和 attention 路径边界；当前证据不支持“大多数 vLLM 推理请求都会遇到”。',
        '同模式布局对照用于定位转换开销；PA 对照同时改变图与 attention 路径，收益单独解释。真实 AIC ready 队列不可见，不能把低活跃度直接归因于调度遗漏。',
        f"归档状态：{acceptance['status']}；已完成设备级条件 {len(rows)}，原布局/连续布局输出配对 {len(layout_output_pairs)}。",
        '多提示词批次的严格重复输出门槛未通过，原布局自身也出现差异；连续布局只能作为受控工作负载下的实验对照，尚未通过通用正确性验收。失败输出与TTFT/decode时间戳保留在 failure-evidence 和 diverse-failed-run-analysis.json。',
        '', '[逐条件表格](RESULTS.md) · [机器可读数据](results.csv) · [验收结果](acceptance.json)']
    if system_comparisons:
        lines+=['','完整容量 6616 blocks；以下数值只展示物理卡4，各物理卡分别留证。','',
            '|路径|独立进程|请求 ms|AIC容量忙碌 ms/调用|AIV容量忙碌 ms/调用|HBM读 GiB/调用|HBM写 GiB/调用|',
            '|---|---:|---:|---:|---:|---:|---:|']
        for item in system_comparisons:
            vals=system_groups[item['variant']]
            lines.append(f"|{item['variant']}|{len(vals)}|{statistics.mean(x['control_median_ms'] for x in vals):.2f}|{item['aic_capacity_busy_ms_per_generate_mean']:.2f}|{item['aiv_capacity_busy_ms_per_generate_mean']:.2f}|{item['hbm_read_bytes_per_request_mean']/2**30:.3f}|{item['hbm_write_bytes_per_request_mean']/2**30:.3f}|")
    links=['','[配置筛查图](figures/applicability-screen.png)']
    if confirm:links.append('[布局与 attention 路径重复对照](figures/layout-path-confirmation.png)')
    if system_groups:links.append('[完整容量下的引擎忙碌时间与 HBM 流量](figures/layout-engine-hbm.png)')
    if extension:links.append('[长生成与长输入批次扩展](figures/long-workload-extension.png)')
    if corun_timings:links.append('[独立矩阵与复制的任务重叠](figures/independent-engine-overlap.png)')
    lines+=links+['',
        '## 适用范围的实测证据','',
        '官方发布镜像与定制镜像的 vLLM-Ascend attention、cache allocator 和 graph gate 源文件哈希完全一致，因此它们不是两份独立版本证据。PIECEWISE 在两者上均复现整缓存 Slice。',
        'vLLM-Ascend 0.18 使用独立源码和 CANN 8.5.1 的真实模型对照；该版同一 Qwen3-8B ordinary GQA cache 是连续 K/V，实测没有整缓存 Slice。版本结论必须同时注明 Python/PyTorch/CANN/driver 组合，不能声称仅由版本号导致。',
        'v0.30 显式 PIECEWISE 与 NONE 均观察到整缓存 Slice；显式 FULL_DECODE_ONLY 加 pa_shape_list=[1,8] 走连续 PA cache 且没有整缓存 Slice，b1/b8 token 输出一致。显式 FULL_DECODE_ONLY（默认空 pa_shape_list）和 DEFAULT graph path 在当前 driver 上因 FusedInferAttentionScore graph-update 错误退出，不作为无 Slice 的负例。',
        'PA资格会在分配阶段使 requires_contiguous_pa_kv_cache 成立，从而改变缓存布局；某些运行时形状即使仍回到 FIA，也可收到连续输入。因此 PA 条件没有整缓存 Slice 不只是一项算子替换的证据。',
        '官方发布身份已通过 GitHub 标签目标与远端提交文件核对：[v0.30.0rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0rc1)；本地关键源文件与该标签提交逐字节一致。镜像本地ID、包版本和远端核验原件保存在 stack-identity。',
        'same-mode dense layout control 使用相同 zero-initialized backing allocation，在 cache bind 前改为 [K][V] 连续 views；实际 cache write/read 仍调用原生路径。block IDs 和 output equivalence 单独验收。',
        '异步批次下，请求进入/退出时机和物理 block 分配可能跨进程不同；b8 block table 不相等不能归因于 layout patch，也不冒充严格相同调度对照。b1 的精确 block table 配对与 b8 的输出等价分别保留。',
        '主矩阵每轮按 native→dense→PA 固定顺序执行，未随机化；长负载扩展和系统采样矩阵轮转顺序。控制CV反映进程内稳定性，不能消除宿主负载或跨进程漂移，也不是显著性检验。',
        f"已核对 native/dense 输出哈希对数：{len(layout_output_pairs)}，全部一致：{acceptance['all_native_dense_outputs_match']}；实际 block table 对数：{len(block_table_pairs)}，一致：{acceptance['all_native_dense_block_tables_match']}。证据见 layout-output-equivalence.json 与 block-table-equivalence.json。", '',
        f"无 profiler、控制稳定且输出一致的布局对照数：{acceptance['stable_layout_comparisons']}；逐条件延迟变化见 layout-latency-comparisons.json。", '',
        *[f"{model.rsplit('/',1)[-1]} {case}：{len(values)} 个独立进程配对，连续视图 latency 变化均值 {statistics.mean(values):.1f}%，范围 {min(values):.1f}% 至 {max(values):.1f}%。"
          for (model,case),values in sorted(latency_groups.items())], '',
        f"独立容量条件 PMU/HBM 采样设备行数：{len(system)}；最小 PMU 有效覆盖率：{acceptance['minimum_system_pmu_valid_fraction']}；最小 HBM 覆盖率：{acceptance['minimum_system_hbm_coverage']}。system-summary.json 不与 task trace 跨执行逐点对齐。", '',
        'same-mode native/dense 的 AIC/AIV 容量忙碌时间与原生 HBM 读写流量见 system-comparisons.json；每项是独立系统采样段中的每次 generate 均值，不能解释为可回收 FLOPs。',
        f"独立矩阵/向量复制控制重复数：{len(corun)}；trace flow 归因完整：{acceptance['synthetic_corun_flow_attribution_complete']}。这只是两个合成任务的设备任务 envelope 并发，不证明模型里存在已 ready 的 AIC 工作。", '',
        *([f"合成分流 co-run / 串行时延比范围：{min(r['parallel_to_serial_ratio'] for r in corun_timings):.3f}–{max(r['parallel_to_serial_ratio'] for r in corun_timings):.3f}。三次运行均存在纯AIC/AIV任务envelope重叠，但重叠后的kernel持续时间变长，不能将重叠时长当作直接收益。"] if corun_timings else []),'',
        f"八个不同提示词的批次正确性验收：{acceptance['diverse_batch_correctness']}。异步调度条件中原布局自身曾出现重复token差异，失败原件保留；同步调度诊断与主性能矩阵分别解释。", '',
        '## 结论约束','',
        'Slice是否消失按实际设备任务确认，FIA/PA分支与图捕获任务必须结合元数据及任务清单解释。更换图模式同时可能改变融合、后端和缓存布局，因此跨图模式差值不能归因于单一布局。',
        '未抽样真实部署或请求分布，不计算“大多数vLLM请求”的比例。Qwen3-4B与8B共享同类GQA结构，不能作为跨attention架构的证明。',
        '当前910B3数据不外推CUDA、ROCm或其他NPU；旧版本源码对照与旧版本硬件通过是两类证据。',
        '完整失败日志与命令保留；图模式启动失败、输出变化和未知归因不伪装成无整理的负例。',
        f"控制段 CV 超过5%的设备条件数：{len(unstable)}，清单见 unstable-controls.json。这些条件可提供路径诊断，不能进入稳定延迟比较；不通过删掉差重复来报告全通过。",
        '真实布局对照在cache绑定前一次建立连续缓存，后续由原生写入路径原地更新；不把长期复用过期副本视为优化。']
    if failures:lines+=['','## 未通过的配置','']+[f'- {name}: {value.get("cause",value["error"])}；[完整错误与命令](failures/{name}.json)。' for name,value in sorted(failures.items())]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',root/'REPORT.md',flush=True)


if __name__=='__main__':main()
