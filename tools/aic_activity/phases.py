"""Classify launches from the current scheduler snapshot, before runner mutation."""
def launch_phase(scheduler,requests):
    new={r.req_id for r in scheduler.scheduled_new_reqs}
    cached=scheduler.scheduled_cached_reqs
    computed=dict(zip(cached.req_ids,cached.num_computed_tokens))
    phases=set()
    for req,n in scheduler.num_scheduled_tokens.items():
        if not n:continue
        if req in new:phases.add('prefill')
        else:
            state=requests.get(req)
            if state is None or req not in computed:phases.add('unclassified')
            else:phases.add('prefill' if computed[req]<len(state.prompt_token_ids or []) else 'decode')
    if not phases:return 'no_tokens'
    return next(iter(phases)) if len(phases)==1 else 'mixed_prefill_decode'

def compiled_training_phases(rows,windows):
    """A compiled recompute tag also names the original forward invocation.

    The first explicit Gradients/ task in a complete step separates original
    forward from backward. Tagged invocations before it are original forward;
    tagged invocations after it are recomputation during backward. This is an
    inference from scoped task order, recorded explicitly as such.
    """
    boundaries=[]
    for w in windows:
        backward=[r['start'] for r in rows if r['phase']=='backward' and w['begin_ns']<=r['start']<w['end_ns']]
        if not backward:raise ValueError('no explicit backward task in training window')
        boundary=min(backward);boundaries.append(dict(window_id=w['id'],first_backward_ns=boundary))
        for r in rows:
            if r['phase']=='recompute' and w['begin_ns']<=r['start']<boundary:
                r['phase']='forward';r['phase_evidence']='compiled_recompute_tag_before_first_gradient'
    return dict(method='compiled scope plus first explicit Gradients task per step; inferred phase boundary',boundaries=boundaries)
