"""Summarize completed independent repeats with explicit unresolved gates."""
import json,csv,statistics
from pathlib import Path
from .statistics import aggregate,describe

CASES=['b1-l128-greedy','b1-l512-greedy','b8-l128-greedy','b8-l512-greedy','b8-l128-sample']

def inference(analysis,out):
    analysis=Path(analysis);out=Path(out);out.mkdir(parents=True,exist_ok=True);allcards={};flat=[]
    for rank in range(4):
        configs={}
        for case in CASES:
            runs={};qualities=[];durations=[];overhead=[];phase=[]
            for repeat in range(3):
                root=analysis/f'infer-matrix-r{repeat}'/case
                accept=json.loads((root/'acceptance.json').read_text())
                windows=json.loads((root/f'rank{rank}/summary.json').read_text())
                runs[f'r{repeat}']=windows
                qualities.append(dict(run=f'r{repeat}',quality=accept['measurement_quality'],task_count_anomaly=accept['task_count_anomaly']))
                overhead.append(accept['evidence']['overhead_fraction'])
                durations.append(accept['evidence']['measured_median_ms'])
                phase.append(json.loads((root/f'rank{rank}/phase_quality.json').read_text())['attributed_engine_task_fraction'])
            result=aggregate(runs);result.update(quality=qualities,latency_ms=describe(durations),overhead_fraction=describe(overhead),phase_attribution_fraction=describe(phase),physical_device_id=4+rank)
            configs[case]=result
            for state,values in result['statistics'].items():flat.append(dict(physical_device_id=4+rank,configuration=case,state=state,**values))
        allcards[str(4+rank)]=configs
    (out/'inference-independent-runs.json').write_text(json.dumps(allcards,indent=2)+'\n')
    with (out/'inference-independent-runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .figures import COLORS,output
    for card,configs in allcards.items():
        fig,ax=plt.subplots(figsize=(13,4.5));y=list(range(len(CASES)));left=[0]*len(y)
        for state,color in COLORS.items():
            values=[100*configs[c]['statistics'][state]['mean'] for c in CASES]
            ax.barh(y,values,left=left,color=color,label=state)
            left=[l+v for l,v in zip(left,values)]
        ax.set(yticks=y,yticklabels=CASES,xlim=(0,100),xlabel='Mean task-envelope time across 3 independent processes (%)',title=f'Inference task states — physical NPU {card}')
        ax.legend(ncol=5,loc='lower center',bbox_to_anchor=(.5,-.35),fontsize=8)
        fig.tight_layout();output(fig,out/f'inference-states-device{card}')
        fig,ax=plt.subplots(figsize=(13,4.5))
        for engine,color in [('aic','#275dad'),('aiv','#d97912')]:
            vals=[];low=[];high=[]
            for case in CASES:
                perrun=[v[engine+'_only']+v['both'] for v in configs[case]['independent_runs'].values()]
                m=statistics.mean(perrun);vals.append(m*100);low.append((m-min(perrun))*100);high.append((max(perrun)-m)*100)
            ax.errorbar(CASES,vals,yerr=[low,high],fmt='o-',color=color,label=engine.upper(),capsize=4)
        ax.set(ylabel='Task-envelope fraction (%)',ylim=(0,100),title=f'Independent-run mean and min–max range — physical NPU {card}');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();output(fig,out/f'inference-repeats-device{card}')
    return allcards

def training(analysis,out):
    analysis=Path(analysis);out=Path(out);out.mkdir(parents=True,exist_ok=True);cards={};table=[]
    for rank in range(4):
        configs={}
        for length in (128,512):
            runs=[]
            for repeat in range(3):
                root=analysis/f'train-{length}-pmu-r{repeat}'
                report=json.loads((root/f'rank{rank}/summary.json').read_text())
                acceptance=json.loads((root/'acceptance.json').read_text())
                report['overhead_fraction']=acceptance['evidence']['ranks'][rank]['profile_overhead_fraction']
                report['warmup_cv']=acceptance['evidence']['ranks'][rank]['warmup_cv'];runs.append(report)
            stats={engine:describe([r['core_means_pct'][engine] for r in runs]) for engine in ('aic','aiv')}
            configs[str(length)]=dict(independent_runs=runs,core_means_pct=stats,overhead_fraction=describe([r['overhead_fraction'] for r in runs]),
                                      all_runs_valid=all(r['measurement_quality']=='valid' for r in runs))
            for engine,values in stats.items():table.append(dict(physical_device_id=4+rank,seq_length=length,engine=engine,**values))
        cards[str(4+rank)]=configs
    (out/'training-independent-runs.json').write_text(json.dumps(cards,indent=2)+'\n')
    with (out/'training-independent-runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    from .figures import core_curves,core_joint,output
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for card,configs in cards.items():
        selected=[]
        for length in (128,512):
            runs=configs[str(length)]['independent_runs'];median=statistics.median(r['core_means_pct']['aiv'] for r in runs)
            r=min(runs,key=lambda r:abs(r['core_means_pct']['aiv']-median))
            selected.append((f'Training seq{length}, median AIV-mean independent run; '+r['measurement_quality'],analysis/r['run']/f'rank{int(card)-4}/core_timeseries.csv'))
        for case in ['b1-l128-greedy','b8-l128-greedy']:
            root=analysis/'infer-pmu-r0'/case
            if not root.exists() and case=='b1-l128-greedy':root=analysis/'infer-pmu-r0/b1-l128-greedy-exact'
            selected.append(('Inference PIECEWISE '+case+' / out32 — separate sample-only run',root/f'rank{int(card)-4}/core_timeseries.csv'))
        core_curves(selected,out/f'training-inference-core-device{card}',device=int(card))
        if card=='4':core_joint(selected,out/'core-joint-device4')
        fig,ax=plt.subplots(figsize=(9,4.5))
        for engine,color in [('aic','#275dad'),('aiv','#d97912')]:
            vals=[];lower=[];upper=[]
            for length in (128,512):
                d=configs[str(length)]['core_means_pct'][engine];vals.append(d['mean']);lower.append(d['mean']-d['min']);upper.append(d['max']-d['mean'])
            ax.errorbar(['seq128','seq512'],vals,yerr=[lower,upper],fmt='o-',color=color,label=engine.upper(),capsize=5)
        ax.set(ylabel='Sampled core capacity busy (%)',ylim=(0,100),title=f'Training independent-run mean and min–max — physical NPU {card}');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();output(fig,out/f'training-repeats-device{card}')
    return cards

def threshold_table(analysis,out):
    analysis=Path(analysis);rows=[]
    roots=[analysis/f'train-{length}-pmu-r{repeat}' for length in (128,512) for repeat in range(3)]
    roots+=[analysis/'infer-pmu-r0'/case for case in ('b1-l128-greedy-exact','b8-l128-greedy')]
    for root in roots:
        for rank in range(4):
            d=json.loads((root/f'rank{rank}/summary.json').read_text())
            for r in d['conditionals']['threshold_sensitivity']:
                rows.append(dict(run=root.name,physical_device_id=4+rank,quality=d.get('measurement_quality','valid'),**r))
    path=Path(out)/'core-threshold-sensitivity.csv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
