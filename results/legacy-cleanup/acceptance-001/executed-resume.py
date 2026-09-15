from pathlib import Path
import subprocess,json,sys,time,os
root=Path('/models/npu_nvme_exp/user7-stack/checkouts/legacy-cleanup')
out=root/'results/legacy-cleanup/acceptance-001'
py='/home/user7/miniconda3/envs/ms_2.5/bin/python'
pw=Path('/home/user7/npu-nvme/.sudo_pw').read_bytes().rstrip(b'\n')+b'\n'
while Path('/proc/648082').exists():
 time.sleep(5)
h01=json.loads((out/'h01/result.json').read_text())
if h01['status']!='pass': sys.exit('H01 did not pass')
(out/'sequencer-resume.json').write_text(json.dumps({'reason':'parent sequencer interrupted; H01 child completed independently','h01_status':h01['status'],'parent_returncode':'unavailable; use recorded H01 phase return codes'},indent=2)+'\n')
phases=[
('native-smoke',[str(root/'build_out/bin/v2_smoke_test'),'0000:83:00.0','7']),
('transport-batch',[py,'tests/hardware/transport_matrix.py','--api','batch','--output',str(out/'transport-batch'),'--shm-id','55201']),
('transport-request',[py,'tests/hardware/transport_matrix.py','--api','request','--output',str(out/'transport-request'),'--shm-id','55203']),
('h01',[py,'tests/hardware/d1_full_state.py','--out',str(out/'h01'),'--shm-id','55300']),
('h02',[py,'tests/hardware/stage4_fault_lifecycle.py','--output',str(out/'h02'),'--shm-id','55400']),
('lifecycle',[py,'tests/hardware/d1_lifecycle.py','--output',str(out/'lifecycle'),'--shm-id','55500']),
('methods',[py,'train.py','benchmark','--config',str(out/'methods-config.json'),'--all','--continue-on-failure']),
('inspect',[py,'train.py','inspect','--config',str(out/'methods-config.json')])]
for name,cmd in phases[4:]:
 print('START',name,flush=True);start=time.monotonic()
 with (out/(name+'.log')).open('wb') as log:
  p=subprocess.run(['sudo','-S','-p','','bash','/tmp/legacy-cleanup-root.sh',*cmd],cwd=root,input=pw,stdout=log,stderr=subprocess.STDOUT)
 record={'phase':name,'argv':cmd,'returncode':p.returncode,'seconds':time.monotonic()-start}
 (out/(name+'-process.json')).write_text(json.dumps(record,indent=2)+'\n')
 print('END',record,flush=True)
 if p.returncode:sys.exit(p.returncode)
