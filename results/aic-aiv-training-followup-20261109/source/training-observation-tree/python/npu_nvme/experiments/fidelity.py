"""Streaming committed-fragment fidelity and logical block ages."""
import math
import re
import numpy as np


class Ages:
    def __init__(self, identities):
        self.saved={key:0 for key in identities}
        self.step=0

    def advance(self, step, selected):
        if step!=self.step+1:raise ValueError('nonsequential fidelity step')
        selected=set(selected)
        if not selected<=self.saved.keys():raise ValueError('unknown saved block')
        for key in selected:self.saved[key]=step
        self.step=step
        values=[step-value for value in self.saved.values()]
        return dict(maximum=max(values,default=0),mean=sum(values)/max(1,len(values)))


def fragment_error(shadow, current, fragments):
    """Count owned fragments only, with float64 accumulation."""
    totals={}
    for row in fragments:
        name=row['name'];start=row['local_element_offset'];end=start+row['element_count']
        actual=np.asarray(current[name]).reshape(-1)[start:end].astype(np.float64)
        saved=np.asarray(shadow[name]).reshape(-1)[start:end].astype(np.float64)
        if len(actual)!=row['element_count'] or saved.shape!=actual.shape:raise ValueError('fragment target mismatch')
        error=float(np.dot(actual-saved,actual-saved));norm=float(np.dot(actual,actual))
        bucket=totals.setdefault(name,[0.,0.]);bucket[0]+=error;bucket[1]+=norm
    return totals


def reduce_error(rank_totals):
    combined={}
    for totals in rank_totals:
        for name,(error,norm) in totals.items():
            bucket=combined.setdefault(name,[0.,0.]);bucket[0]+=error;bucket[1]+=norm
    error=sum(row[0] for row in combined.values());norm=sum(row[1] for row in combined.values())
    layers={}
    for name,(e,n) in combined.items():
        match=re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)',name)
        layer='layer.'+match.group(1) if match else name
        bucket=layers.setdefault(layer,[0.,0.]);bucket[0]+=e;bucket[1]+=n
    layer_errors={name:math.sqrt(e)/max(math.sqrt(n),1e-12) for name,(e,n) in layers.items()}
    return dict(layers=layer_errors,worst_layer=max(layer_errors,key=layer_errors.get,default=None),
                worst_layer_relative_l2=max(layer_errors.values(),default=0),relative_l2=math.sqrt(error)/max(math.sqrt(norm),1e-12),
                parameters={name:math.sqrt(e)/max(math.sqrt(n),1e-12) for name,(e,n) in combined.items()})
