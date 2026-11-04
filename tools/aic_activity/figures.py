"""Publication artifacts from measured per-device data; never average cards."""
import csv,json,statistics,gzip
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .analysis import partition

COLORS={'aic_only':'#275dad','aiv_only':'#d97912','both':'#2a9d8f','neither':'#d1d5db','unknown':'#8e55aa'}
PHASE_COLORS={'forward':'#93c5fd','backward':'#fca5a5','recompute':'#c4b5fd','optimizer':'#86efac','input':'#67e8f9',
              'prefill':'#93c5fd','decode':'#86efac','token_selection':'#fbbf24','unclassified':'#d1d5db'}

def csvrows(path):
    path=Path(path)
    if not path.exists():path=Path(str(path)+'.gz')
    with (gzip.open(path,'rt',newline='') if path.suffix=='.gz' else path.open(newline='')) as f:return list(csv.DictReader(f))

def output(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):fig.savefig(path.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

def comparison(cases,out):
    """Each case is an actual single window closest to that run's median width."""
    fig,axes=plt.subplots(len(cases)*2,1,figsize=(14,4.2*len(cases)),squeeze=False,
                          gridspec_kw={'height_ratios':[3,1.5]*len(cases)})
    for i,(name,path) in enumerate(cases):
        path=Path(path);summaries=json.loads((path/'summary.json').read_text())
        median=statistics.median(w['duration_ns'] for w in summaries)
        w=min(summaries,key=lambda w:abs(w['duration_ns']-median));wid=w['window_id']
        rows=[r for r in csvrows(path/'timeseries.csv') if r['window_id']==wid]
        edges=[int(r['start_ns'])/1e6 for r in rows]+[int(rows[-1]['end_ns'])/1e6]
        ax=axes[2*i,0];phaseax=axes[2*i+1,0]
        for engine,color in [('aic','#275dad'),('aiv','#d97912')]:
            ax.stairs([100*(float(r[engine+'_only'])+float(r['both'])) for r in rows],edges,color=color,label=engine.upper(),linewidth=1.2)
        ax.fill_between(edges[:-1],[0]*len(rows),[100*float(r['unknown']) for r in rows],step='post',color=COLORS['unknown'],alpha=.25,label='Mixed/unknown')
        ax.set(ylabel='Task active / 1 ms (%)',ylim=(0,100),xlim=(0,edges[-1]),title=f'{name} | physical NPU 4 | {wid} | {w["quality"]}')
        ax.legend(loc='upper right',ncol=3);ax.grid(alpha=.2)
        stages=[r for r in csvrows(path/'stages.csv') if r['window_id']==wid];phases=sorted({r['stage'] for r in stages})
        for lane,phase in enumerate(phases):
            spans=[((int(r['device_start_ns'])-w['begin_ns'])/1e6,(int(r['device_end_ns'])-int(r['device_start_ns']))/1e6) for r in stages if r['stage']==phase]
            phaseax.broken_barh(spans,(lane-.35,.7),facecolors=PHASE_COLORS.get(phase,'#d1d5db'),rasterized=True)
        phaseax.set(yticks=range(len(phases)),yticklabels=phases,ylim=(-.7,len(phases)-.3),xlim=(0,edges[-1]),xlabel='Elapsed device-window time (ms)')
    fig.suptitle('AIC / AIV task activity in training and inference',weight='bold')
    fig.tight_layout(rect=(0,.04,1,.95));fig.text(.01,.01,'Two nonzero curves within a display bin can be serial tasks. Use the both state for interval intersection. Unknown mixed kernels are not idle; training above the overhead gate is diagnostic.',fontsize=8)
    output(fig,out)

def states_and_cdf(cases,out):
    from .analysis import read
    fig,(ax,cdf)=plt.subplots(1,2,figsize=(14,4.5));labels=[]
    for i,(name,path) in enumerate(cases):
        path=Path(path);windows=json.loads((path/'summary.json').read_text());total=sum(w['duration_ns'] for w in windows)
        left=0
        for state,color in COLORS.items():
            value=100*sum(w['state_ns'][state] for w in windows)/total
            ax.barh(i,value,left=left,color=color,label=state if i==0 else None);left+=value
        labels.append(name);rows=read(path/'events.jsonl.gz');values=[]
        for w in windows:
            values.extend((p['end']-p['start'])/1e3 for p in partition(rows,w['begin_ns'],w['end_ns']) if p['state']=='aiv_only')
        values=sorted(values)
        if values:cdf.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=name)
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Observed window time (%)',xlim=(0,100),title='Five disjoint task-envelope states')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=8)
    cdf.set(xscale='log',xlabel='Contiguous AIV-only task window (µs)',ylabel='Empirical CDF',title='AIV-only windows; task envelopes only');cdf.legend(fontsize=8);cdf.grid(alpha=.2)
    fig.tight_layout();output(fig,out)

def core_curves(cases,out,device=4):
    fig,axes=plt.subplots(len(cases),1,figsize=(14,3*len(cases)),squeeze=False)
    for (name,path),ax in zip(cases,axes[:,0]):
        rows=csvrows(path);base=int(rows[0]['window_start_ns'])
        edges=[(int(r['window_start_ns'])-base)/1e9 for r in rows]+[(int(rows[-1]['window_end_ns'])-base)/1e9]
        for engine,color in [('aic','#275dad'),('aiv','#d97912')]:
            values=[float(r[engine+'_activity_pct']) if r['valid']=='True' else np.nan for r in rows]
            ax.stairs(values,edges,color=color,label=engine.upper(),linewidth=1)
        ax.set(title=name,ylabel='Core capacity busy (%)',ylim=(0,100),xlabel='Elapsed time in sampled run (s)');ax.grid(alpha=.2);ax.legend(loc='upper right')
    fig.suptitle(f'Independent sample-only PMU — physical NPU {device}',weight='bold')
    fig.tight_layout(rect=(0,.04,1,.95));fig.text(.01,.01,'10 ms samples; busy cycles / (core clock × all engine cores × bin width). Missing coverage stays blank. Separate runs from task/phase traces.',fontsize=8)
    output(fig,out)

def core_joint(cases,out):
    """Show the joint distribution without calling a sample instantaneous."""
    fig,axes=plt.subplots(1,len(cases),figsize=(4.5*len(cases),4.4),squeeze=False,sharex=True,sharey=True)
    for (name,path),ax in zip(cases,axes[0]):
        rows=[r for r in csvrows(path) if r['valid']=='True']
        x=[float(r['aic_activity_pct']) for r in rows];y=[float(r['aiv_activity_pct']) for r in rows]
        plot=ax.hexbin(x,y,gridsize=25,extent=(0,100,0,100),mincnt=1,bins='log',cmap='viridis')
        ax.axvline(10,color='#dc2626',linestyle='--',linewidth=.8);ax.axhline(50,color='#dc2626',linestyle='--',linewidth=.8)
        label=(name.split(',')[0] if name.startswith('Training') else
               'Inference b1 / in128' if 'b1-l128' in name else 'Inference b8 / in128' if 'b8-l128' in name else name)
        ax.set(title=label,xlabel='AIC core capacity busy (%)',xlim=(0,100),ylim=(0,100));ax.set_aspect('equal');fig.colorbar(plot,ax=ax,label='10 ms bin count',shrink=.7)
    axes[0,0].set_ylabel('AIV core capacity busy (%)')
    fig.suptitle('Joint sampled activity — physical NPU 4; temporal resolution 10 ms',weight='bold')
    fig.tight_layout(rect=(0,.05,1,.93));fig.text(.01,.01,'Each point aggregates a 10 ms bin and may contain serial AIC/AIV work. Dashed thresholds: AIC 10%, AIV 50%; full threshold sensitivity is in JSON/CSV.',fontsize=8)
    output(fig,out)
