"""Render task-envelope and sampled-core busy activity as separate panels."""
import argparse,csv,json,gzip
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load(path):
    path=Path(path)
    if not path.exists():path=Path(str(path)+'.gz')
    with (gzip.open(path,'rt',newline='') if path.suffix=='.gz' else path.open(newline='')) as f:return list(csv.DictReader(f))

def render(summary_path,task_path,stage_path,out,title,core_path=None):
    summaries=json.loads(Path(summary_path).read_text());tasks=load(task_path);stages=load(stage_path) if stage_path else []
    cores=load(core_path) if core_path else []
    phases=sorted({s['stage'] for s in stages})
    heights=[3]+([3] if cores else [])+([max(1.2,len(phases)*.22)] if stages else [])
    fig,axs=plt.subplots(len(heights),1,figsize=(14,sum(heights)+1.5),squeeze=False,sharex=True,gridspec_kw={'height_ratios':heights})
    shift=0
    for index,summary in enumerate(summaries):
        wid=summary['window_id'];begin=summary['begin_ns'];width=(summary['end_ns']-begin)/1e6
        tr=[r for r in tasks if r['window_id']==wid]
        x=[(int(r['start_ns'])/1e6)+shift for r in tr];edges=[*x,(int(tr[-1]['end_ns'])/1e6)+shift] if tr else []
        if tr:
            axs[0,0].stairs([100*(float(r['aic_only'])+float(r['both'])) for r in tr],edges,color='#275dad',label='AIC task envelopes' if index==0 else None)
            axs[0,0].stairs([100*(float(r['aiv_only'])+float(r['both'])) for r in tr],edges,color='#d97912',label='AIV task envelopes' if index==0 else None)
            axs[0,0].stairs([100*float(r['unknown']) for r in tr],edges,color='#8e55aa',alpha=.6,linestyle='--',label='Mixed/unknown task envelope' if index==0 else None)
        # Aggregate stage marks into collections to avoid thousands of artists.
        for lane,phase in enumerate(phases):
            spans=[((int(s['device_start_ns'])-begin)/1e6+shift,(int(s['device_end_ns'])-int(s['device_start_ns']))/1e6)
                   for s in stages if s['window_id']==wid and s['stage']==phase]
            color={'forward':'#93c5fd','backward':'#fca5a5','recompute':'#c4b5fd','optimizer':'#86efac','input':'#67e8f9','prefill':'#93c5fd','decode':'#86efac','token_selection':'#fbbf24'}.get(phase,'#d1d5db')
            axs[-1,0].broken_barh(spans,(lane-.35,.7),facecolors=color,rasterized=True)
        axs[0,0].axvline(shift,color='#334155',alpha=.35);axs[0,0].text(shift+8,96,wid,fontsize=8,va='top')
        if cores:
            cr=[r for r in cores if r['window_id']==wid]
            if cr:
                cx=[(int(r['window_start_ns'])-begin)/1e6+shift for r in cr];ce=[*cx,(int(cr[-1]['window_end_ns'])-begin)/1e6+shift]
                axs[1,0].stairs([float(r['aic_activity_pct']) if r['valid']=='True' else np.nan for r in cr],ce,color='#275dad',label='AIC core busy' if index==0 else None)
                axs[1,0].stairs([float(r['aiv_activity_pct']) if r['valid']=='True' else np.nan for r in cr],ce,color='#d97912',label='AIV core busy' if index==0 else None)
        shift+=width+150
    axs[0,0].set(ylabel='Task active time per 1 ms (%)',ylim=(0,100),title='Device task envelopes; mixed intervals remain unknown')
    axs[0,0].grid(alpha=.2);axs[0,0].legend(loc='upper right',ncol=3,fontsize=8)
    if cores:
        axs[1,0].set(ylabel='Sampled core busy (%)',ylim=(0,100),xlabel='Elapsed time in consecutive measured windows (ms; gap between windows is visual only)')
        axs[1,0].grid(alpha=.2);axs[1,0].legend(loc='upper right')
    if stages:axs[-1,0].set(yticks=range(len(phases)),yticklabels=phases,ylim=(-.7,len(phases)-.3),ylabel='Device phase lanes')
    axs[-1,0].set_xlabel('Elapsed time within consecutive measured windows (ms; 150 ms visual gap between windows)')
    fig.suptitle(title,fontsize=14,weight='bold')
    fig.text(.01,.01,'Two nonzero display bins do not prove instant overlap. Task envelopes and core busy are distinct. Core sampling: 10 ms; missing bins stay blank. Phase lanes may overlap.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95));out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):fig.savefig(out.with_suffix('.'+ext),dpi=180)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True,type=Path);p.add_argument('--tasks',required=True,type=Path);p.add_argument('--stages',type=Path);p.add_argument('--cores',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--title',required=True);a=p.parse_args();render(a.summary,a.tasks,a.stages,a.output,a.title,a.cores)
if __name__=='__main__':main()
