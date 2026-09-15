from pathlib import Path
import subprocess,json,hashlib,sys,time
root=Path('/models/npu_nvme_exp/user7-stack/checkouts/legacy-cleanup')
out=root/'results/legacy-cleanup/acceptance-001'
out.mkdir(parents=True,exist_ok=False)
py='/home/user7/miniconda3/envs/ms_2.5/bin/python'
pw=Path('/home/user7/npu-nvme/.sudo_pw').read_bytes().rstrip(b'\n')+b'\n'
config=json.loads((root/'experiments/baselines/repro/configs/gpt2_30step.json').read_text())
config.update(results_root=str(out/'methods'),fs_test_dir=str(out/'method-files'),warmup_steps=1,formal_steps=2,checkpoint_every=2,continue_steps=3,spdk_shm_id=56100)
(out/'methods-config.json').write_text(json.dumps(config,indent=2)+'\n')
phases=[
('native-smoke',[str(root/'build_out/bin/v2_smoke_test'),'0000:83:00.0','7']),
('transport-batch',[py,'tests/hardware/transport_matrix.py','--api','batch','--output',str(out/'transport-batch'),'--shm-id','55201']),
('transport-request',[py,'tests/hardware/transport_matrix.py','--api','request','--output',str(out/'transport-request'),'--shm-id','55203']),
('h01',[py,'tests/hardware/d1_full_state.py','--out',str(out/'h01'),'--shm-id','55300']),
('h02',[py,'tests/hardware/stage4_fault_lifecycle.py','--output',str(out/'h02'),'--shm-id','55400']),
('lifecycle',[py,'tests/hardware/d1_lifecycle.py','--output',str(out/'lifecycle'),'--shm-id','55500']),
('methods',[py,'train.py','benchmark','--config',str(out/'methods-config.json'),'--all','--continue-on-failure']),
('inspect',[py,'train.py','inspect','--config',str(out/'methods-config.json')])]
lib=root/'build_out/lib/libnpu_nvme.so'
syms=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True)
native={'abi':2,'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'exports':sorted(line.split()[-1] for line in syms.splitlines() if line.split()[-1].startswith('npu_nvme_'))}
(out/'native.json').write_text(json.dumps(native,indent=2)+'\n')
for name,cmd in phases:
 print('START',name,flush=True);start=time.monotonic()
 with (out/(name+'.log')).open('wb') as log:
  p=subprocess.run(['sudo','-S','-p','','bash','/tmp/legacy-cleanup-root.sh',*cmd],cwd=root,input=pw,stdout=log,stderr=subprocess.STDOUT)
 record={'phase':name,'argv':cmd,'returncode':p.returncode,'seconds':time.monotonic()-start}
 (out/(name+'-process.json')).write_text(json.dumps(record,indent=2)+'\n')
 print('END',record,flush=True)
 if p.returncode:sys.exit(p.returncode)
