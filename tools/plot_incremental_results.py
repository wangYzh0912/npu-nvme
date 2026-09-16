#!/usr/bin/env python3
"""Plot measured results, leaving absent experiments absent."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('campaign',type=Path);args=parser.parse_args();root=args.campaign
    out=root/'figures';out.mkdir(exist_ok=True)
    profile=json.loads((root/'profile-summary-rank0.json').read_text())
    fig,ax=plt.subplots(figsize=(9,3));keys=list(profile['fractions']);ax.bar(keys,[100*profile['fractions'][key] for key in keys]);ax.set_ylabel('Task interval fraction (%)');ax.set_title('Main model: 19 complete GetNext intervals; mixed-task envelopes');fig.tight_layout();fig.savefig(out/'resource-overlap.png',dpi=180);plt.close(fig)
    first=profile['step_boundaries'][5]
    fig,axes=plt.subplots(2,1,figsize=(12,5),sharex=True)
    colors=dict(cube_only='#4477aa',vector_only='#ee6677',both='#228833',neither='#dddddd')
    phase_colors=dict(forward='#66ccee',backward='#aa3377',optimizer='#ccbb44',other='#bbbbbb')
    for ax,rows,key,mapping in [(axes[0],profile['segments'],'state',colors),
                                 (axes[1],profile.get('phase_tasks',[]),'phase',phase_colors)]:
        batches={name:[] for name in mapping}
        for row in rows:
            left=max(first[0],row['start']);right=min(first[1],row['end'])
            if right>left:batches[row[key]].append(((left-first[0])/1000,(right-left)/1000))
        for name,spans in batches.items():ax.broken_barh(spans,(0,1),facecolors=mapping[name])
    from matplotlib.patches import Patch
    for ax,mapping in zip(axes,[colors,phase_colors]):
        ax.set_yticks([]);ax.legend(handles=[Patch(color=c,label=n) for n,c in mapping.items()],loc='upper right',ncol=4)
    axes[0].set_title('Main model, step 6: task activity and named-kernel phases (same wall clock)')
    axes[1].set_xlabel('Time from GetNext (ms)');fig.tight_layout();fig.savefig(out/'phase-timeline-main.png',dpi=180);plt.close(fig)
    loads=json.loads((root/'load-boundary.json').read_text())['rows']
    fig,ax=plt.subplots(figsize=(7,4));ax.plot([row['scale'] for row in loads],[row['elapsed_ns']/1e6 for row in loads],'o-');ax.set_xlabel('Approximate per-rank scoring load');ax.set_ylabel('Standalone duration (ms)');ax.set_title('Boundary probe (not training slowdown)');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'score-boundary.png',dpi=180);plt.close(fig)
    validation=root/'validation-summary.json'
    if validation.exists():
        rows=json.loads(validation.read_text())['runs']
        canonical=[r for r in rows if r['canonical'] and r['status']=='pass' and r['role']=='main']
        if canonical:
            fig,axes=plt.subplots(2,2,figsize=(11,7))
            for row in canonical:
                curves=row['fidelity'];steps=[v['step'] for v in curves]
                for ax,key in zip(axes.flat,['relative_l2','loss_difference','maximum_block_age']):
                    ax.plot(steps,[v[key] for v in curves],label=row['group']);ax.set_ylabel(key);ax.set_xlabel('Optimizer step');ax.legend();ax.grid(alpha=.25)
                run=root/'runs'/row['run'];payload=[]
                from npu_nvme.runtime.training_catalog import read_checked
                reports=[read_checked(run/f'rank_{i}'/'training.json') for i in range(4)]
                budget=json.loads((root/'resource-budget-main.json').read_text())['weight_bytes']
                for i in range(20):payload.append(sum(r['incremental']['steps'][i]['payload_bytes'] for r in reports)/budget)
                axes[1,1].plot(steps,payload,label=row['group'])
            axes[1,1].set_ylabel('Payload / full weight bytes');axes[1,1].set_xlabel('Optimizer step');axes[1,1].legend()
            fig.tight_layout();fig.savefig(out/'fidelity-curves.png',dpi=180);plt.close(fig)
    copies=json.loads((root/'copy-boundary.json').read_text())['rows']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for grain in sorted({row['granularity'] for row in copies}):
        rows=sorted([row for row in copies if row['granularity']==grain],key=lambda row:row['bytes'])
        for ax,key in zip(axes,['d2d_ns','d2h_ns']):
            ax.plot([row['bytes']/1e9 for row in rows],[row[key]/1e6 for row in rows],'o-',label=f'{grain//1024} KiB');ax.set_xlabel('Transferred GB');ax.set_ylabel('Standalone ms');ax.set_title(key[:-3].upper());ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'copy-boundary.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
