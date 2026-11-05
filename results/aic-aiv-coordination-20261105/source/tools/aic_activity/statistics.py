"""Independent-run aggregation; windows within a run are not replicate runs."""
import statistics
from .analysis import STATES

def describe(values):
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),min=min(values),max=max(values)) if values else None

def aggregate(runs):
    result={}
    for key,windows in runs.items():
        duration=sum(w['duration_ns'] for w in windows)
        result[key]={state:sum(w['state_ns'][state] for w in windows)/duration for state in STATES}
    return dict(independent_runs=result,statistics={state:describe([r[state] for r in result.values()]) for state in STATES})

def core_conditionals(rows):
    valid=[r for r in rows if r['valid'] is True or r['valid']=='True']
    result=[]
    for low in (5,10,20):
        for high in (30,50,70):
            for active,idle in [('aiv','aic'),('aic','aiv')]:
                selected=[r for r in valid if float(r[active+'_activity_pct'])>=high]
                n=sum(float(r[idle+'_activity_pct'])<=low for r in selected)
                result.append(dict(active=active,relatively_idle=idle,high_threshold_pct=high,low_threshold_pct=low,
                                   conditioning_bins=len(selected),matching_bins=n,conditional_fraction=n/len(selected) if selected else None))
    return dict(valid_bins=len(valid),total_bins=len(rows),threshold_sensitivity=result)
