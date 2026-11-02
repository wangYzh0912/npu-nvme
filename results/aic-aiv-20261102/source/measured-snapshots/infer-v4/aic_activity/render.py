"""Render task-envelope and sampled-core busy activity as separate panels."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def render(summary_path,task_path,stage_path,out,title,core_path=None):
    summaries=json.loads(Path(summary_path).read_text());tasks=load(task_path);stages=load(stage_path) if stage_path and Path(stage_path).exists() else []
    cores=load(core_path) if core_path and Path(core_path).exists() else []
    fig,axs=plt.subplots(2 if cores else 1,1,figsize=(14,7 if cores else 4),squeeze=False,gridspec_kw={'height_ratios':[2,2]} if cores else None)
    for index,summary in enumerate(summaries):
        wid=summary['window_id'];begin=summary['begin_ns'];width=(summary['end_ns']-begin)/1e6
        tr=[r for r in tasks if r['window_id']==wid];shift=index*(width+150)
        x=[(int(r['start_ns'])/1e6)+shift for r in tr];edges=[*x,(int(tr[-1]['end_ns'])/1e6)+shift] if tr else []
        if tr:
            axs[0,0].stairs([100*(float(r['aic_only'])+float(r['both'])) for r in tr],edges,color='#275dad',label='AIC task envelopes' if index==0 else None)
            axs[0,0].stairs([100*(float(r['aiv_only'])+float(r['both'])) for r in tr],edges,color='#d97912',label='AIV task envelopes' if index==0 else None)
            axs[0,0].stairs([100*float(r['unknown']) for r in tr],edges,color='#8e55aa',alpha=.6,linestyle='--',label='Mixed/unknown task envelope' if index==0 else None)
        # Aggregate stage marks into collections to avoid thousands of artists.
        for phase in sorted({s['stage'] for s in stages if s['window_id']==wid}):
            spans=[((int(s['device_start_ns'])-begin)/1e6+shift,(int(s['device_end_ns'])-int(s['device_start_ns']))/1e6)
                   for s in stages if s['window_id']==wid and s['stage']==phase]
            color={'forward':'#bfdbfe','backward':'#fecaca','recompute':'#e9d5ff','optimizer':'#bbf7d0','input':'#cffafe','communication':'#e7e5e4'}.get(phase,'#e5e7eb')
            axs[0,0].broken_barh(spans,(0,100),facecolors=color,alpha=.16,rasterized=True,zorder=0)
        axs[0,0].axvline(shift,color='#334155',alpha=.35);axs[0,0].text(shift+8,96,wid,fontsize=8,va='top')
        if cores:
            cr=[r for r in cores if r['window_id']==wid and r['valid']=='True']
            if cr:
                cx=[(int(r['window_start_ns'])-begin)/1e6+shift for r in cr];ce=[*cx,(int(cr[-1]['window_end_ns'])-begin)/1e6+shift]
                axs[1,0].stairs([float(r['aic_activity_pct']) for r in cr],ce,color='#275dad',label='AIC core busy' if index==0 else None)
                axs[1,0].stairs([float(r['aiv_activity_pct']) for r in cr],ce,color='#d97912',label='AIV core busy' if index==0 else None)
    axs[0,0].set(ylabel='Task active time per 1 ms (%)',ylim=(0,100),title='Device task envelopes; mixed intervals remain unknown')
    axs[0,0].grid(alpha=.2);axs[0,0].legend(loc='upper right',ncol=3,fontsize=8)
    axs[0,0].set_xticks([])
    if cores:
        axs[1,0].set(ylabel='Sampled core busy (%)',ylim=(0,100),xlabel='Elapsed time in consecutive measured windows (ms; gap between windows is visual only)')
        axs[1,0].grid(alpha=.2);axs[1,0].legend(loc='upper right')
    else:axs[0,0].set_xlabel('Elapsed time in consecutive measured windows (ms; gaps omitted between windows)')
    fig.suptitle(title,fontsize=14,weight='bold')
    fig.text(.01,.01,'Task-envelope activity and PMU core busy time are distinct measurements. Core sampling resolution is 10 ms; no interpolation. Stage shading comes from scoped kernel names.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95));out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):fig.savefig(out.with_suffix('.'+ext),dpi=180)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True,type=Path);p.add_argument('--tasks',required=True,type=Path);p.add_argument('--stages',type=Path);p.add_argument('--cores',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--title',required=True);a=p.parse_args();render(a.summary,a.tasks,a.stages,a.output,a.title,a.cores)
if __name__=='__main__':main()
