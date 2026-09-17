#!/usr/bin/env python3
"""Standalone scientific figures from graph measurements and same-run traces."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,'/models/npu_nvme_exp/user7-stack/incremental-report-packages')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,default=Path('/models/npu_nvme_exp/user7-stack/graph-topk-20260917-001'))
    p.add_argument('--output',type=Path,default=Path('results/graph-topk-20260917/figures'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    palette={'forward':'#3074a4','backward':'#df9232','optimizer':'#8370b5','auxiliary':'#359b70'}
    for profile in a.campaign.glob('*/device-overlap.json'):
        doc=json.loads(profile.read_text());fig,axes=plt.subplots(4,1,figsize=(12,7),sharex=True)
        for ax,rank in zip(axes,doc['ranks']):
            intervals=rank['phase_intervals'];begin=min(left for rows in intervals.values() for left,right in rows);end=max(right for rows in intervals.values() for left,right in rows)
            for y,(phase,color) in enumerate(palette.items()):
                spans=[((left-begin)/1000,(right-left)/1000) for left,right in intervals[phase]]
                ax.broken_barh(spans,(y-.3,.6),facecolors=color)
            ax.set_yticks(range(4),palette.keys(),fontsize=7);ax.set_ylabel('rank '+str(rank['rank']));ax.grid(axis='x',alpha=.2)
        axes[-1].set_xlabel('Same-run device timeline (ms from first task)')
        fig.suptitle(profile.parent.name+'\nTask envelopes only; no active-core or instruction-overlap claim',fontsize=11)
        fig.tight_layout();fig.savefig(a.output/(profile.parent.name+'-timeline.png'),dpi=160);plt.close(fig)
    measurements=a.output.parent/'measurements.json'
    if not measurements.exists():return
    rows=json.loads(measurements.read_text())['runs']
    rows=[r for r in rows if not r['diagnostic'] and r['steps']==20 and r['warmup_steps']==12 and 'slowdown' in r]
    if rows:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for role in ('main','auxiliary'):
            selected=[r for r in rows if r['role']==role and r['level'] in (0,1,2,3,4,5,6) and r['scan_fraction']==1 and r['ratio']==.1 and not r['reference_only'] and not r['compute_iterations']]
            selected.sort(key=lambda r:r['level'])
            axes[0].plot([r['level'] for r in selected],[r['seconds'] for r in selected],'o-',label=role)
        axes[0].set_xlabel('Cumulative graph level');axes[0].set_ylabel('20-step completion time (s)');axes[0].legend()
        for level in (4,6):
            selected=sorted([r for r in rows if r['level']==level and r['role']=='main' and r['ratio']==.1],key=lambda r:r['scan_fraction'])
            if selected:axes[1].plot([r['scan_fraction'] for r in selected],[100*r['slowdown'] for r in selected],'o-',label='G'+str(level))
        for budget,color in ((1,'gray'),(3,'#bf4747'),(5,'gray')):
            axes[1].axhline(budget,color=color,ls='--',lw=.8,label=str(budget)+'% budget')
        axes[1].set_xlabel('Real block scan fraction q');axes[1].set_ylabel('Slowdown vs G0 (%)');axes[1].legend(fontsize=8)
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle('Serial graph costs; parallel execution not established')
        fig.tight_layout();fig.savefig(a.output/'graph-costs.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
