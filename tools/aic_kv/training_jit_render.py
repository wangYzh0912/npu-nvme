"""Standalone scientific figures and evidence-bounded training report."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from .training_jit_campaign import RESULTS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=RESULTS);args=parser.parse_args();root=args.root
    rows=json.loads((root/'measurements.json').read_text());selected=[r for r in rows if r['physical_device']==4]
    groups={}
    for r in selected:groups.setdefault((r['jit_level'],r['sequence_length'],r['micro_batch']),[]).append(r)
    fragments=[]
    suite='resident-fragments' if list((root/'resident-fragments/runs').glob('*/result.json')) else 'fragments'
    for p in sorted((root/suite/'runs').glob('*/result.json')):
        d=json.loads(p.read_text());v=d['timings_ms_per_update'];d.update(run=p.parent.name,median_ms=statistics.median(v),cv=statistics.pstdev(v)/statistics.mean(v));fragments.append(d)
    (root/'fragment-measurements.json').write_text(json.dumps(fragments,indent=2)+chr(10))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    keys=sorted(groups);labels=[f'{mode}\nS{seq} B{batch}' for mode,seq,batch in keys]
    for i,k in enumerate(keys):
        g=groups[k]
        axes[0].scatter([i]*len(g),[r['control_ms'] for r in g],
                        color=['#2563eb' if r['measurement_accepted'] else '#dc2626' for r in g])
        axes[1].scatter([i-.12]*len(g),[r['aic_mean_pct'] for r in g],color='#2563eb',label='AIC' if i==0 else None)
        axes[1].scatter([i+.12]*len(g),[r['aiv_mean_pct'] for r in g],color='#ea580c',label='AIV' if i==0 else None)
        axes[2].scatter([i-.12]*len(g),[r['low_aic_fraction']*100 for r in g],color='#16a34a',label='AIC <= 10%' if i==0 else None)
        axes[2].scatter([i+.12]*len(g),[r['joint_fraction']*100 for r in g],color='#9333ea',label='AIC <=10%, AIV >=50%' if i==0 else None)
    for ax in axes:ax.set_xticks(range(len(keys)),labels,rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
    axes[0].set(ylabel='Unprofiled training step (ms)');axes[1].set(ylabel='Sampled engine core busy (%)',ylim=(0,100));axes[2].set(ylabel='Fraction of valid sampled time (%)',ylim=(0,105));axes[1].legend();axes[2].legend()
    fig.suptitle('Qwen3-8B TP4/DP1; physical card 4; one point per independent model process')
    (root/'figures').mkdir(exist_ok=True)
    for suffix in ('png','svg','pdf'):fig.savefig(root/'figures'/('training-jit-workload.'+suffix),dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for i,k in enumerate(keys):
        for row in groups[k]:
            read=row.get('hbm_read_bytes_per_complete_step');write=row.get('hbm_write_bytes_per_complete_step')
            if read is None or write is None:continue
            axes[0].scatter(i-.12,read/1e9,color='#2563eb',label='Read' if i==0 and row==groups[k][0] else None)
            axes[0].scatter(i+.12,write/1e9,color='#ea580c',label='Write' if i==0 and row==groups[k][0] else None)
            axes[1].scatter(i,(read+write)/(k[1]*k[2])/1e9,color='#7c3aed')
    for ax in axes:ax.set_xticks(range(len(keys)),labels,rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
    axes[0].set(ylabel='HBM traffic (GB / complete training step)');axes[0].legend()
    axes[1].set(ylabel='HBM read + write (GB / input token)')
    fig.suptitle('Physical card 4; native HBM counters, windows >=95% covered; no saturation claim')
    for suffix in ('png','svg','pdf'):fig.savefig(root/'figures'/('training-hbm-workload.'+suffix),dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    for i,k in enumerate(keys):
        for index,row in enumerate(groups[k]):
            ax.scatter(i-.12,row['joint_state_sampled_ms_per_step'],color='#9333ea',
                       label='Total joint time / sampled step' if i==0 and index==0 else None)
            ax.scatter(i+.12,row['mean_longest_joint_interval_ms'],color='#0d9488',
                       label='Mean longest continuous interval / step' if i==0 and index==0 else None)
    ax.set_xticks(range(len(keys)),labels,rotation=30,ha='right')
    ax.set(ylabel='AIC <=10% and AIV >=50% (ms)',
           title='Physical card 4; 10 ms windows; phase/readiness not implied',ylim=(0,None))
    ax.grid(axis='y',alpha=.2);ax.legend()
    for suffix in ('png','svg','pdf'):fig.savefig(root/'figures'/('training-joint-window-duration.'+suffix),dpi=170)
    plt.close(fig)
    representatives=[]
    for mode in ('O0','O1','O2'):
        candidates=[r for r in selected if r['jit_level']==mode and r['sequence_length']==128 and
                    r['micro_batch']==1 and r['measurement_accepted']]
        if candidates:representatives.append(sorted(candidates,key=lambda r:r['replicate'])[0])
    if representatives:
        fig,axes=plt.subplots(len(representatives),1,figsize=(13,2.8*len(representatives)),
                              layout='constrained',squeeze=False)
        for ax,row in zip(axes.flat,representatives):
            run=root/'training/runs'/row['run']
            windows=json.loads((run/'sample-evidence.json').read_text())['ranks'][0]['windows'][:4]
            origin=windows[0]['begin_ns']
            bins=list(csv.DictReader((run/'rank0/pmu/core_timeseries.csv').open()))
            for engine,color in (('aic','#2563eb'),('aiv','#ea580c')):
                for i,window in enumerate(windows):
                    points=[b for b in bins if b['window_id']==window['id'] and b['valid']=='True']
                    ax.plot([(int(b['window_start_ns'])-origin+5_000_000)/1e6 for b in points],
                            [float(b[engine+'_activity_pct']) for b in points],color=color,
                            label=engine.upper() if i==0 else None,linewidth=1.3)
            for window in windows:
                start=(window['begin_ns']-origin)/1e6;end=(window['end_ns']-origin)/1e6
                ax.axvline(start,color='#94a3b8',alpha=.5,linewidth=.6)
                ax.text((start+end)/2,103,window['id'],ha='center',fontsize=9)
            ax.set(ylim=(0,113),ylabel='Core busy (%)',xlabel='Elapsed time (ms)',title=row['run'])
            ax.legend(loc='upper right');ax.grid(axis='y',alpha=.2)
        fig.suptitle('Steady training; native 10 ms PMU bins. Step boundaries known; substep phases unresolved.')
        for suffix in ('png','svg','pdf'):
            fig.savefig(root/'figures'/('training-activity-timeline.'+suffix),dpi=170)
        plt.close(fig)
    if fragments:
        fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
        modes=('adamw-o0','adamw-o1','fused-o0');shapes=((128,),(4096,),(4096,3072))
        for ax,shape in zip(axes,shapes):
            for i,mode in enumerate(modes):
                v=[r for r in fragments if tuple(r['shape'])==shape and r['mode']==mode]
                for r in v:ax.scatter(i,r['median_ms'],color='#2563eb' if r['cv']<=.05 and r['oracle']['passed'] else '#dc2626')
            ax.set_xticks(range(3),modes,rotation=15);ax.set(title=str(shape),ylabel='Update latency (ms)');ax.grid(axis='y',alpha=.2)
        fig.suptitle(('Resident gradient' if suite=='resident-fragments' else 'Ordinary Tensor input: includes transfers')+'; actual optimizer; red = diagnostic')
        for suffix in ('png','svg','pdf'):fig.savefig(root/'figures'/('optimizer-fragments.'+suffix),dpi=170)
        plt.close(fig)
    native_path=root/'fragment-native-tasks.json'
    if native_path.is_file():
        traces=[t for t in json.loads(native_path.read_text()) if t['suite']==suite]
        if traces:
            fig,axes=plt.subplots(len(traces),1,figsize=(14,2.5*len(traces)),
                                  layout='constrained',squeeze=False)
            colors={'move':'#a855f7','vector':'#ea580c','copy':'#64748b','callback':'#0d9488'}
            for ax,row in zip(axes.flat,traces):
                tasks=row['timeline']['tasks']
                for t in tasks:
                    if t['kind']=='KERNEL_AIVEC':
                        lane=1;kind='move' if 'TensorMove' in t['name'] else 'vector'
                    elif t['kind']=='MEMCPY_ASYNC':lane=0;kind='copy'
                    elif t['kind']=='HOSTFUNC_CALLBACK':lane=0;kind='callback'
                    else:raise ValueError('unclassified native task '+t['kind'])
                    ax.broken_barh([(t['start_us']/1000,t['duration_us']/1000)],
                                   (lane-.3,.6),facecolors=colors[kind])
                ax.set_yticks([0,1,2],['DMA / callback','AIV','AIC'])
                ax.set(ylim=(-.6,2.6),title=row['run'],xlabel='Time from first device task (ms)')
                ax.grid(axis='x',alpha=.2)
                ax.text(.99,.92,'8 diagnostic updates; no AIC tasks in native trace',
                        transform=ax.transAxes,ha='right',fontsize=9)
            from matplotlib.patches import Patch
            fig.legend(handles=[Patch(color=colors[k],label=v) for k,v in
                [('move','AIV TensorMove'),('vector','AIV update'),
                 ('copy','MEMCPY (payload unreported)'),('callback','Host callback')]],
                loc='outside upper center',ncols=4)
            for suffix in ('png','svg','pdf'):
                fig.savefig(root/'figures'/('optimizer-native-timeline.'+suffix),dpi=170)
            plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    ax.set(xlim=(0,12),ylim=(0,4));ax.axis('off')
    boxes=[(.2,2.6,'Clipped gradient / FP32 cast'),(3.2,2.6,'First moment update'),(3.2,.8,'Second moment update'),(6.4,1.7,'Bias correction / sqrt / div'),(9.6,1.7,'Parameter update / assign')]
    for x,y,label in boxes:ax.text(x,y,label,fontsize=10,bbox=dict(boxstyle='round,pad=.5',facecolor='#eff6ff',edgecolor='#2563eb'))
    for x1,y1,x2,y2 in [(2.2,2.6,3.1,2.6),(2.2,2.5,3.1,.9),(5.4,2.6,6.3,1.9),(5.4,.9,6.3,1.6),(8.7,1.7,9.5,1.7)]:
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#334155'))
    ax.text(.2,.25,'Real model prerequisite: 291 norm contributions -> AddN -> AllReduce -> clipping scale -> gradient update.',fontsize=9)
    ax.text(.2,-.05,'TensorMove state snapshots persist in O1. This optimizer fragment contains no intrinsic matrix work.',fontsize=9)
    fig.suptitle('Problem fragment: optimizer state movement and vector update')
    for suffix in ('png','svg','pdf'):fig.savefig(root/'figures'/('optimizer-update-chain.'+suffix),dpi=170)
    plt.close(fig)
    lines=['# 训练低 AIC 与问题片段观察','',
        '本轮固定 Qwen3-8B、TP4/DP1、序列并行、AdamW、FP32 参数/状态与 BF16 计算、全重计算、固定重复文本；不是跨模型或真实训练数据分布的普遍性调查。',
        'O0 旧基线与 O1 的 resolved config 已逐字段核对：实质差异只有 JIT 等级，另外两项差异是输出路径；初始参数/状态哈希一致。O1/O0 的 loss 存在差异，不能宣称整模型逐位等价或生产验收通过。','',
        '目前已定位的具体问题是全局梯度裁剪之后的优化器状态移动与向量更新尾段。六种O1负载下，卡4每步低AIC/高AIV窗口约149.5–163.4ms；平均AIC活跃随负载从约3.85%升至34.68%。这支持继续优化明确尾段，不能推广为整个训练普遍有90%可回收AIC算力。',
        '官方融合已经减少部分开销；安装版本的FusedAdamW在驻留大分片上进一步缩短更新链。但本轮未做整模型优化器替换或真实co-run，不宣称协同收益。','',
        '[官方方案核验](official/AUDIT.md) · [测量口径](MEASUREMENT_CONTRACT.md) · [逐卡测量](measurements.csv) · [配置覆盖](recurrence.json) · [验收](acceptance.json) · [下一步](NEXT_STEPS.md)','',
        '## 模型测量（只展示物理卡4）','',
        '[工作负载对照](figures/training-jit-workload.png) · [AIC/AIV 时间曲线](figures/training-activity-timeline.png) · [HBM 流量](figures/training-hbm-workload.png) · [联合窗口绝对时长](figures/training-joint-window-duration.png)','',
        '|模式|序列|micro-batch|进程数|通过质量门槛|控制 step ms 中位数|AIC %均值|AIV %均值|低AIC时间 %|低AIC且高AIV时间 %|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,g in sorted(groups.items()):
        lines.append(f"|{k[0]}|{k[1]}|{k[2]}|{len(g)}|{sum(r['measurement_accepted'] for r in g)}/{len(g)}|{statistics.median(r['control_ms'] for r in g):.3f}|{statistics.mean(r['aic_mean_pct'] for r in g):.2f}|{statistics.mean(r['aiv_mean_pct'] for r in g):.2f}|{statistics.mean(r['low_aic_fraction'] for r in g)*100:.2f}|{statistics.mean(r['joint_fraction'] for r in g)*100:.2f}|")
    finished=[g for k,g in groups.items() if k[0]=='O1' and len(g)==3 and all(r['measurement_accepted'] for r in g)]
    if finished:
        joint=[statistics.mean(r['joint_state_sampled_ms_per_step'] for r in g) for g in finished]
        continuous=[statistics.mean(r['mean_longest_joint_interval_ms'] for r in g) for g in finished]
        lines+=['',f"更具体的问题窗口：在已完成三重复的{len(finished)}个O1配置中，每步联合状态时间均值为{min(joint):.1f}–{max(joint):.1f}ms，平均最长连续区间为{min(continuous):.1f}–{max(continuous):.1f}ms。连续区间遇无效bin、时间缺口或step边界立即断开。这是阈值窗口的绝对时长，不等于已证明可回收的协同时间，也不能仅凭曲线命名其阶段。"]
    recurrence=json.loads((root/'recurrence.json').read_text())['O1']
    lines+=['','逐卡配置覆盖：','']
    for card in recurrence:
        cells=card['cells'];complete=sum(c['complete'] for c in cells)
        lines.append(f"- 卡{card['physical_device']}：{complete}/6条件各有3次有效独立重复；低AIC占多数时间的条件{sum(c['low_aic_majority_in_every_repeat'] for c in cells)}/6；低AIC且高AIV占多数时间的条件{sum(c['majority_in_every_repeat'] for c in cells)}/6。")
    diagnostic_path=root/'numerical-diagnostics.json'
    if diagnostic_path.is_file():
        diagnostic=json.loads(diagnostic_path.read_text())
        repeated=[r for r in diagnostic['within_mode_repeats'] if len(r['runs'])==3]
        lines+=['',f"数值重复性：{sum(r['all_final_hashes_identical'] and r['loss_trajectories_identical'] for r in repeated)}/{len(repeated)}个配置/卡组合的三进程loss轨迹与最终状态哈希一致。跨JIT只保留描述性差值，没有事后放宽容差来宣布整模型等价；详见[numerical-diagnostics.json](numerical-diagnostics.json)。"]
    o2_path=root/'o2-observation-status.json'
    if o2_path.is_file():
        o2=json.loads(o2_path.read_text())
        if o2.get('smoke_exit_code'):
            lines+=['','O2兼容性探测在首个训练step前失败；直接编译错误为`Get scalar or string value from None with check failed`。其后续基线未进入性能比较，详见[原始失败和原因](training/failures/o2-seq128-mb1-r0/cause-audit.json)。这不是O2普遍无效或性能差的证据。']
    lines+=['','每点为一次独立模型进程，卡5–7另有独立数据。低AIC≤10%、高AIV≥50%；10ms bin 内统计不证明瞬时并行。联合比例下降不等于低AIC消失，也不能把空闲比例换算成可回收FLOPs。','',
        'HBM 采用与该 CANN 8.3 官方 viewer 相同的设备→主机→wall 时钟转换；锚点、原生计数和覆盖单独保留。流量大不证明带宽饱和；无检查点区间不能据此归因于 NVMe I/O。',
        'O1使用96步/horizon128；O2为保证高速图下沉路径也有足够采样时长，使用1024步/horizon1024。两者均为无warmup恒定1e-6学习率；只能比较公共前96步的数值轨迹，不能比较不同最终step状态。','',
        '## 问题片段','','[优化器更新链](figures/optimizer-update-chain.png) · [片段重复对照](figures/optimizer-fragments.png) · [原生任务时间线](figures/optimizer-native-timeline.png)','',
        '保留的seq128/O1图已融合多条 AdamW 逐元素更新链，但仍有1166个 Float32 TensorMove，逻辑输出合计32.77GB。该数是静态几何，不能当作实测每步流量。首批复制直接来自一阶/二阶优化器状态，随后进入乘法、平方、归一化及回写。原始节点、shape和IR来源见[TensorMove证据](fragments/optimizer-tensor-moves.json)。',
        '当前还存在明确的调度约束：[全局梯度裁剪依赖](fragments/global-clip-optimizer-dependency.json)将291项范数贡献经AddN、AllReduce与缩放因子送入优化器更新。不能在保持相同全局裁剪语义时，直接将某层完整AdamW更新提前到其余梯度就绪之前。范数汇总之前可合法提前的局部工作仍需单独验证，静态依赖没有给出运行时可回收时长。',
        '同一O1图中已有291个Mul/Mul/ReduceSum融合节点，说明逐梯度范数相关计算已有官方融合；仍存在的全局汇总与状态更新依赖不能通过重复加入局部融合来解释或消除。',
        '片段复现运行安装版本的真实 AdamW / FusedAdamW 实现，输入值是固定有限合成值，尺寸来自本模型实际参数分片；包含参数和一阶/二阶状态的8步Float64 oracle。它验证局部更新及执行成本，不等于从模型捕获了完整激活/梯度，也不证明模型AIC任务已ready。',
        '片段计时包含其Cell调用路径和每40次更新一次的同步；整模型在同一训练图中更新多个分片，固定梯度与缓冲区重复更新也可能具有不同的缓存局部性，不能用片段时长乘以分片数来预测整步耗时。普通Tensor与驻留Parameter的调用结构也不同，两套计时不能被解释成只改变H2D传输的严格单变量对照。',
        '普通Tensor梯度输入版本在每次更新中含MEMCPY_ASYNC，不能当作设备驻留梯度下的模型优化器成本。普通输入结果仅作诊断，驻留版本需结合原生任务与传输字节验证。当前表格来源：'+suite+'。','',
        '|实现|shape|独立进程数|数值oracle通过|中位 update ms|CV通过进程|','|---|---|---:|---:|---:|---:|']
    for shape in ((128,),(4096,),(4096,3072)):
        for mode in ('adamw-o0','adamw-o1','fused-o0'):
            g=[r for r in fragments if tuple(r['shape'])==shape and r['mode']==mode]
            if g:lines.append(f"|{mode}|{shape}|{len(g)}|{sum(r['oracle']['passed'] for r in g)}/{len(g)}|{statistics.median(r['median_ms'] for r in g):.6f}|{sum(r['cv']<=.05 for r in g)}/{len(g)}|")
    if native_path.is_file():
        lines+=['','原生任务诊断（每条trace包含8次更新；任务耗时求和不等于正常关键路径）：','',
            '|套件/实现|AIV任务/更新|MEMCPY任务/更新|MEMCPY最大耗时 µs|传输字节|',
            '|---|---:|---:|---:|---|']
        for t in json.loads(native_path.read_text()):
            counts=t['timeline']['task_count_by_kind']
            lines.append(f"|{t['suite']}/{t['run']}|{counts.get('KERNEL_AIVEC',0)/8:g}|{t['timeline']['memcpy_count']/8:g}|{t['timeline']['memcpy_max_duration_us']:.3f}|Level0未报告|")
    model_tail=root/'model-fragment/optimizer-tail-evidence.json'
    if model_tail.is_file():
        tails=json.loads(model_tail.read_text())['rows']
        lines+=['','### 已定位的真实模型片段','',
            '[实际模型阶段与尾段图](figures/real-model-optimizer-tail.png) · [逐任务身份对应与统计](model-fragment/optimizer-tail-evidence.json)','',
            '另一次O1/seq128/micro-batch1真实模型诊断在四卡、每卡三个完整step内，以stream/task ID/精确起始时间将kernel scope对应到原生任务类型。每个优化器尾段均有2621个KERNEL_AIVEC任务，其中1165个TensorMove、291组一阶更新与291组二阶更新；检查了跨入尾段边界的原生任务，没有AIC/MIX任务，也没有未对应的尾段kernel。',
            '尾段从具名的第一个优化器状态TensorMove-op0到末个尾段任务结束定义，包含状态复制和更新。第一处明确Gradients scope仅用于推断前向/反向边界，反向与重计算/范数归约合并标注。该次整步计时扰动约45%，不得将其阶段时长直接移植到正常PMU运行，也不得证明独立矩阵工作已ready。','',
            '|物理卡|诊断step数|尾段包络中位ms|AIV任务union中位ms|TensorMove任务和中位ms|','|---|---:|---:|---:|---:|']
        for device in (4,5,6,7):
            own=[r for r in tails if r['physical_device']==device]
            lines.append(f"|{device}|{len(own)}|{statistics.median(r['tail_envelope_ms'] for r in own):.3f}|{statistics.median(r['tail_aiv_union_ms'] for r in own):.3f}|{statistics.median(r['tensor_move_duration_sum_ms'] for r in own):.3f}|")
        lines+=['','这使问题从整步低活跃曲线收敛为可复查的“全局范数/裁剪之后的优化器状态移动与向量更新尾段”。正常路径中约150ms的阈值窗口与该片段相符，但两次执行不能做逐时刻拼接或精确因果时长替换。FusedAdamW的局部改善可作为下一轮整模型干预候选；尚无整模型替换或协同加速收益。']
        lines+=['','关于I/O：该任务使用内存生成的固定数据，正式观测区间不写检查点；优化器尾段的原生任务只有AIV，没有PCIe DMA或HCCL任务。TensorMove及更新链反映设备内状态整理/读写，不能等同于NVMe或主机输入传输。整步HBM流量可直接测得，但尚未给出逐算子的流量归因或带宽饱和校准。']
    old_summary=root/'reused-o0-baseline-summary.json'
    qualified_o0_runs={r['run'] for r in rows if r['jit_level']=='O0' and
                      sum(other['run']==r['run'] for other in rows)==4 and
                      all(other['measurement_accepted'] and other['hbm_measurement_accepted']
                          for other in rows if other['run']==r['run'])}
    matched_o0=[r for r in selected if r['run'] in qualified_o0_runs]
    matched_o1=[r for r in selected if r['jit_level']=='O1' and r['sequence_length']==128 and r['micro_batch']==1]
    if len(matched_o0)==1 and len(matched_o1)==3:
        a=matched_o0[0];b={key:statistics.mean(r[key] for r in matched_o1) for key in
            ('hbm_read_bytes_per_complete_step','hbm_write_bytes_per_complete_step','capacity_busy_aic_ms_per_step','capacity_busy_aiv_ms_per_step')}
        old_flow=a['hbm_read_bytes_per_complete_step']+a['hbm_write_bytes_per_complete_step']
        new_flow=b['hbm_read_bytes_per_complete_step']+b['hbm_write_bytes_per_complete_step']
        lines+=['',f"新增同口径O0/HBM补充控制（1进程，O1为3进程）在卡4每步读写合计{old_flow/1e9:.1f}GB，O1均值{new_flow/1e9:.1f}GB，差值{(new_flow/old_flow-1)*100:.1f}%；AIV容量忙碌时间分别{a['capacity_busy_aiv_ms_per_step']:.1f}ms与{b['capacity_busy_aiv_ms_per_step']:.1f}ms，AIC分别{a['capacity_busy_aic_ms_per_step']:.2f}ms与{b['capacity_busy_aic_ms_per_step']:.2f}ms。该单轮补充支持融合减少向量/访存开销，不证明新增AIC计算或跨引擎协同；实验顺序固定，不能将其包装为三轮随机对照。"]
    if old_summary.exists():
        old=next(r for r in json.loads(old_summary.read_text()) if r['physical_device']==4)
        new=[r for r in selected if r['jit_level']=='O1' and r['sequence_length']==128 and r['micro_batch']==1]
        if new:
            lines+=['','按每步容量忙碌时间核对，旧O0样本AIC约'+f"{old['aic_capacity_busy_ms_per_step']:.2f}"+'ms，O1独立重复均值约'+f"{statistics.mean(r['capacity_busy_aic_ms_per_step'] for r in new):.2f}"+'ms；AIV分别约'+f"{old['aiv_capacity_busy_ms_per_step']:.2f}"+'ms与'+f"{statistics.mean(r['capacity_busy_aiv_ms_per_step'] for r in new):.2f}"+'ms。百分比变化受整步分母缩短影响，不能解释为新增矩阵工作或回收同等FLOPs。旧O0系统采样是一轮新观测，历史三次重复见旧报告。']
    lines+=['','反向图中已配对同一前向操作的 dX/dW MatMul，但 dW 依赖后续重计算的激活；未证明在前面的AIV窗口中输入已可用，因此没有以删掉依赖来强行做co-run。精确节点见[反向候选依赖](fragments/backward-paired-matmul-candidate.json)。','',
        '## 质量与未完成事项','',
        '整模型完成、采样质量、配置覆盖、跨JIT数值等价、片段oracle和真实协同收益是不同结论。失败及未完成条件不计作通过，不删除不稳定重复。各配置质量见measurements.json和status。','',
        'seq512首个O1尝试曾在模型启动前被硬件锁拒绝（片段任务与下一模型条件竞争）；原件保留为基础设施失败。另一启动尝试遭SIGTERM且未进入preflight，恢复使用独立目录；不作为模型不兼容。',
        '首轮O0/HBM补充控制的卡6只有50/53个step达到95%流量覆盖，比例94.3%未通过HBM门槛；其PMU仍有效。原始结果保留，另以完全相同配置进行独立确认；各卡HBM验收与通过step数量见measurements.json，未放宽原门槛。',
        '本轮观测和片段实验的完成状态不等于生产优化验收。后续实施顺序见[NEXT_STEPS.md](NEXT_STEPS.md)。']
    (root/'REPORT.md').write_text(chr(10).join(lines)+chr(10))
    print('rendered',len(groups),'model cells',len(fragments),'fragment processes')


if __name__=='__main__':main()
