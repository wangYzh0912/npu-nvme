"""Assert positive/negative calibration controls and plot measured results."""
import json,csv
from pathlib import Path

def certify(analysis,windows,out):
    analysis=Path(analysis);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results=json.loads((analysis/'calibration.json').read_text());by={r['id']:r for r in results}
    checks={
        'idle_both_below_1pct':max(by['idle']['core_mean'].values())<1,
        'matrix_aic_above_50pct':by['matrix']['core_mean']['aic']>50,
        'matrix_aiv_below_1pct':by['matrix']['core_mean']['aiv']<1,
        'vector_aiv_above_20pct':by['vector']['core_mean']['aiv']>20,
        'vector_aic_below_1pct':by['vector']['core_mean']['aic']<1,
        'serial_task_overlap_below_1pct':by['serial']['task']['both']<.01,
        'concurrent_task_overlap_above_20pct':by['concurrent']['task']['both']>.2,
        'concurrent_each_engine_above_20pct':min(by['concurrent']['core_mean'].values())>20,
        'each_phase_at_least_190_valid_10ms_bins':all(r['valid_bins']>=190 for r in results)}
    evidence=dict(status='pass' if all(checks.values()) else 'fail',checks=checks,results=results,
                  limitation='Core mean activity alone cannot establish simultaneous busy cycles inside one 10 ms sample; independent device task intervals provide the positive overlap control.')
    (out/'calibration-acceptance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if not all(checks.values()):raise ValueError('device calibration failed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (analysis/'core_timeseries.csv').open() as f:rows=list(csv.DictReader(f))
    phases=json.loads(Path(windows).read_text());begin=phases[0]['begin_ns']
    fig,(ax,bars)=plt.subplots(2,1,figsize=(13,7),gridspec_kw={'height_ratios':[3,2]})
    for engine,color in [('aic','#275dad'),('aiv','#d97912')]:
        x=[(int(r['window_start_ns'])-begin)/1e9 for r in rows]
        edges=x+[(int(rows[-1]['window_end_ns'])-begin)/1e9]
        ax.stairs([float(r[engine+'_activity_pct']) for r in rows],edges,label=engine.upper(),color=color)
    for phase in phases:
        l=(phase['begin_ns']-begin)/1e9;h=(phase['end_ns']-begin)/1e9
        ax.axvline(l,color='#888',alpha=.5);ax.text((l+h)/2,98,phase['id'],ha='center',va='top')
    ax.set(ylabel='Core capacity busy (%)',ylim=(0,100),xlabel='Elapsed time (s)',title='Real NPU calibration: idle → matrix → vector → serial → concurrent');ax.legend();ax.grid(alpha=.2)
    states=['aic_only','aiv_only','both','neither','unknown'];colors=['#275dad','#d97912','#2a9d8f','#d1d5db','#8e55aa'];left=[0]*len(results)
    for state,color in zip(states,colors):
        values=[100*r['task'][state] for r in results];bars.barh([r['id'] for r in results],values,left=left,label=state,color=color);left=[l+v for l,v in zip(left,values)]
    bars.set(xlabel='Measured task-envelope time (%)',xlim=(0,100));bars.legend(ncol=5,loc='lower center',bbox_to_anchor=(.5,-.5),fontsize=8)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(out/('calibration.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    return evidence
