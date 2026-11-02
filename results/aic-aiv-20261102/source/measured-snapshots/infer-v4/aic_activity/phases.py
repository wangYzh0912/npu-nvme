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
