from types import SimpleNamespace as N
from tools.aic_activity.phases import launch_phase
from tools.aic_activity.phases import compiled_training_phases

def scheduler(tokens,new=(),cached=(),computed=()):
    return N(num_scheduled_tokens=tokens,scheduled_new_reqs=[N(req_id=r) for r in new],
             scheduled_cached_reqs=N(req_ids=list(cached),num_computed_tokens=list(computed)))

def test_cached_scheduler_progress_owns_decode_classification():
    # Runner state is stale before execute_model refreshes it.
    state={'r':N(prompt_token_ids=[1]*128,num_computed_tokens=0)}
    assert launch_phase(scheduler({'r':1},cached=['r'],computed=[128]),state)=='decode'
    assert launch_phase(scheduler({'r':64},cached=['r'],computed=[64]),state)=='prefill'

def test_new_mixed_and_empty_launches():
    state={'r':N(prompt_token_ids=[1]*128)}
    assert launch_phase(scheduler({'n':128},new=['n']),state)=='prefill'
    assert launch_phase(scheduler({'n':128,'r':1},new=['n'],cached=['r'],computed=[128]),state)=='mixed_prefill_decode'
    assert launch_phase(scheduler({}),state)=='no_tokens'
    assert launch_phase(scheduler({'x':1}),state)=='unclassified'

def test_compiled_recompute_tag_does_not_mislabel_original_forward():
    rows=[dict(start=1,phase='recompute'),dict(start=5,phase='backward'),dict(start=6,phase='recompute')]
    evidence=compiled_training_phases(rows,[dict(id='step',begin_ns=0,end_ns=10)])
    assert [r['phase'] for r in rows]==['forward','backward','recompute']
    assert evidence['boundaries'][0]['first_backward_ns']==5
