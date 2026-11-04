from tools.aic_activity.statistics import aggregate,core_conditionals

def test_independent_runs_get_equal_weight_after_window_duration_weighting():
    def window(n,v):return dict(duration_ns=n,state_ns=dict(aic_only=n*v,aiv_only=n*(1-v),both=0,neither=0,unknown=0))
    result=aggregate({'r0':[window(10,1),window(30,0)],'r1':[window(1,1)]})
    assert result['independent_runs']['r0']['aic_only']==.25
    assert result['statistics']['aic_only']['mean']==.625

def test_threshold_conditional_excludes_missing_samples_and_empty_conditions():
    result=core_conditionals([dict(valid=True,aic_activity_pct=0,aiv_activity_pct=60),dict(valid=False,aic_activity_pct=100,aiv_activity_pct=0)])
    assert result['valid_bins']==1
    selected=[x for x in result['threshold_sensitivity'] if x['active']=='aiv' and x['high_threshold_pct']==50]
    assert all(x['conditional_fraction']==1 for x in selected)
    assert any(x['conditional_fraction'] is None for x in result['threshold_sensitivity'])
