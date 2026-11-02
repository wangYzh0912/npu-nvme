import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]))
from tools.aic_activity.pmu import bins

def test_busy_time_is_core_capacity_normalized_and_respects_coverage():
    sample={'aic':[dict(core=0,start=0,end=10_000_000,busy_fraction=.5,valid=True)],
            'aiv':[dict(core=25,start=0,end=10_000_000,busy_fraction=.25,valid=True)]}
    rows=bins(sample,0,10_000_000)
    assert rows[0]['aic_activity_pct']==50
    assert rows[0]['aiv_activity_pct']==25
    assert rows[0]['valid'] is True

def test_partial_sample_coverage_is_invalid_not_idle():
    sample={'aic':[dict(core=0,start=0,end=5_000_000,busy_fraction=1,valid=True)],
            'aiv':[dict(core=25,start=0,end=10_000_000,busy_fraction=0,valid=True)]}
    row=bins(sample,0,10_000_000)[0]
    assert row['aic_activity_pct'] is None and row['aiv_activity_pct'] is None
    assert row['aic_coverage']==.5

def test_pmu_plot_resolution_never_exceeds_sampling_resolution():
    try:bins({'aic':[],'aiv':[]},0,10_000_000,1_000_000)
    except ValueError as exc:assert '>=10ms' in str(exc)
    else:assert False
