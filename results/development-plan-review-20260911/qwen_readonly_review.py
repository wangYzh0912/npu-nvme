import json, pathlib, subprocess, hashlib, struct, importlib.util
repo=pathlib.Path('/home/user7/npu-nvme')
base=pathlib.Path('/models/npu_nvme_exp/user7-stack/qwen3-8b-full-restart-20260911-153007')
commit='ef21f922deeaa9f6d684f7dc3b14b9484a8f74fe'
prefix='results/qwen3-8b-training-20260911/'
def sha(b): return hashlib.sha256(b).hexdigest()
def remote(name): return subprocess.check_output(['git','show',commit+':'+prefix+name],cwd=repo)
spec=importlib.util.spec_from_file_location('audit',repo/'experiments/training/check_qwen_training_run.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
result={'scope':'report consistency, local file presence and checkpoint headers/scalars only; no payload hash, restore, oracle, or device training','source_commit':commit,'raw_directory':str(base),'files':[],'ranks':[]}
reports=[]
for name in ['acceptance.json']+[f'rank_{r}/acceptance.json' for r in range(4)]:
    b=remote(name); raw=(base/name).read_bytes(); assert json.loads(b)==json.loads(raw)
    result['files'].append({'path':name,'remote_sha256':sha(b),'raw_sha256':sha(raw),'json_equal':True})
    if name != 'acceptance.json':reports.append(json.loads(b))
assert [r['rank'] for r in reports]==list(range(4))
for key in ['losses','environment_id','config_sha256','data_sha256','parallel','versions']:
    assert all(r[key]==reports[0][key] for r in reports)
assert len(reports[0]['losses'])==8 and all(not x['overflow'] for x in reports[0]['losses'])
for rank in range(4):
    directory=base/f'training/checkpoint/rank_{rank}'
    meta=json.loads((directory/'meta.json').read_text());path=directory/meta['last_ckpt_file']
    record=mod.inspect_checkpoint(path,8)
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]; raw=f.read(length); header=json.loads(raw)
    record.update(rank=rank,header_sha256=sha(raw),header_bytes=length,payload_sha256=None)
    record['control_names']=[k for k in header if k not in ('__metadata__',) and not k.endswith('.weight') and not k.startswith(('adam_m.','adam_v.'))]
    result['ranks'].append(record)
model=pathlib.Path('/models/Qwen3-8B');index=json.loads((model/'model.safetensors.index.json').read_text())
result['model_shards']=[{'path':str(model/n),'bytes':(model/n).stat().st_size,'sha256':None} for n in sorted(set(index['weight_map'].values()))]
result['config_sha256']=sha((model/'config.json').read_bytes());assert result['config_sha256']==reports[0]['config_sha256']
result['total_checkpoint_bytes']=sum(x['bytes'] for x in result['ranks'])
result['total_weight_file_bytes']=sum(x['bytes'] for x in result['model_shards'])
result['historical_environment_id']=reports[0]['environment_id']
result['status']='report_and_local_structure_verified'
print(json.dumps(result,indent=2))
