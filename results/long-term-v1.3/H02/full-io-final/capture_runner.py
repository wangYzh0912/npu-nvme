from pathlib import Path
import subprocess,json,hashlib,sys,os,time
root=Path('/models/npu_nvme_exp/user7-stack/checkouts/long-term-v1.3')
out=root/'results/long-term-v1.3/H02/full-io-final'
out.mkdir(parents=True,exist_ok=False)
script=root/'tests/hardware/full_io_roundtrip.py'
cmd=[sys.executable,str(script),'--pci','0000:83:00.0','--npu','7','--shm-id','313101','--output',str(out/'profiling')]
record={'argv':cmd,'cwd':str(root),'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((root/'build_out/lib/libnpu_nvme.so').read_bytes()).hexdigest()}
(out/'invocation.json').write_text(json.dumps(record,indent=2)+'\n')
(out/'executed_full_io_roundtrip.py').write_bytes(script.read_bytes())
start=time.monotonic()
with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
 try:
  result=subprocess.run(cmd,cwd=root,stdout=stdout,stderr=stderr,timeout=900)
  record.update(returncode=result.returncode,status='pass' if result.returncode==0 and '[FULL-IO] PASS' in (out/'stdout.txt').read_text() else 'fail')
 except subprocess.TimeoutExpired:
  record.update(returncode=None,status='fail',timeout=True)
record['seconds']=time.monotonic()-start
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
sys.exit(0 if record['status']=='pass' else 1)
