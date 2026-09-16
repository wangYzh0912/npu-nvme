import importlib.util,json,subprocess,types,tempfile
from pathlib import Path
root=Path.cwd()
spec=importlib.util.spec_from_file_location('fixture_helpers',root/'tests/gates/test_metadata_ownership.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
old=types.ModuleType('legacy_metadata_baseline');old.__file__=str(root/'python/direct_checkpoint.py')
exec(subprocess.check_output(['git','show','5b7ef43:python/direct_checkpoint.py'],text=True),old.__dict__)
cases={}
for fault in ['none','metadata_write','superblock_write','flush1','flush2']:
 with tempfile.TemporaryDirectory() as out:
  binding=helper.MemoryBinding(fault);old.lib=binding
  manager=object.__new__(old.DirectCheckpoint);manager._closed=True
  state=helper.input_state()
  for key in ['layout','meta_dict','metadata_generation','active_meta_slot']:setattr(manager,key,getattr(state,key))
  manager.rank_id=0;manager.world_size=1;manager.chunk_size=4096;manager.keep_last_n=3
  manager.ctx=None;manager._meta_pkl=str(Path(out)/'meta.pkl');error=None
  try:manager._commit_metadata(8,helper.layout_items())
  except RuntimeError as exc:error=str(exc)
  cases[fault]=helper.summary(manager,binding,error)
(root/'tests/fixtures/v2_metadata_commit.json').write_text(json.dumps(dict(source_commit='5b7ef43',cases=cases),indent=2)+'\n')
