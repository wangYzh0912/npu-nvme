from pathlib import Path
from npu_nvme.experiments import campaign

def rows(seconds):
 return [dict(incremental=dict(formal_begin_ns=1,formal_end_ns=int(seconds*1e9)+1),losses=[dict(loss=1.)]*24) for _ in range(4)]

def test_one_timing_sample_cannot_prove_repeatability(monkeypatch):
 monkeypatch.setattr(campaign,'validate_training_run',lambda *a:rows(20))
 result=campaign.validate_baseline([Path('one')],dict(warmup_steps=4,formal_steps=20,loss_atol=1e-6,loss_rtol=1e-5,baseline_spread_limit=.03))
 assert not result['stable']

def test_two_close_samples_are_stable(monkeypatch):
 monkeypatch.setattr(campaign,'validate_training_run',lambda p,*a:rows(20 if str(p)=='one' else 20.2))
 result=campaign.validate_baseline([Path('one'),Path('two')],dict(warmup_steps=4,formal_steps=20,loss_atol=1e-6,loss_rtol=1e-5,baseline_spread_limit=.03))
 assert result['stable']
