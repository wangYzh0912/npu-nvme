import importlib.util,sys
from types import SimpleNamespace
import pytest
from npu_nvme.runtime.training_catalog import read_checked

@pytest.fixture
def suite(tmp_path,monkeypatch):
 from npu_nvme.experiments import probe_suite as module
 restored=[]
 def restore(*a,**kw):restored.append(kw['identity']);return dict(verified=True,tensors=2,bytes=8)
 monkeypatch.setattr(module,'restore',restore)
 monkeypatch.setitem(sys.modules,'mindspore.communication.comm_func',SimpleNamespace(barrier=lambda:None))
 runtime=SimpleNamespace(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,memory_allocated=lambda:100,max_memory_allocated=lambda:120)
 options=dict(probe=dict(specs=[dict(name='before',probe=None),dict(name='after',probe=None)]),initial_full=str(tmp_path/'initial'),initial_identity='same')
 instance=module.ProbeSuite(SimpleNamespace(runtime=runtime),object(),options,rank=0,output=tmp_path/'run')
 instance.warmup();return instance,restored

def test_reset_separated_twenty_step_intervals(suite):
 instance,restored=suite
 for i in range(40):
  instance.observe(dict(step=i+5,loss=float(i%20),begin_ns=i+1,end_ns=i+2))
  instance.save(i+1)
 assert instance.close()==0 and restored==['same','same']
 for name in ('before','after'):
  r=read_checked(instance.output/'suite'/name/'rank_0/training.json')
  assert len(r['losses'])==20 and len(r['incremental']['steps'])==20
  assert [v['logical_step'] for v in r['losses']]==list(range(1,21))
  assert r['initial_full_restore']['verified']

def test_suite_rejects_changed_trajectory(suite):
 instance,_=suite
 for i in range(20):
  instance.observe(dict(step=i+5,loss=float(i)));instance.save(i+1)
 with pytest.raises(ValueError,match='numerical trajectory'):
  instance.observe(dict(step=25,loss=9.0))

def test_suite_cannot_complete_partial_interval(suite):
 with pytest.raises(ValueError,match='before all configurations'):suite[0].close()
