#!/usr/bin/env python3
"""One authorized eight-NPU device reset; never reboot the host."""
import fcntl
import json
from pathlib import Path
import subprocess
import time


def main():
    root=Path('/models/npu_nvme_exp/user7-stack')
    out=root/'recovery-eight-npus-20260916-001';out.mkdir(exist_ok=False)
    with (root/'hardware-campaign.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        boot=Path('/proc/sys/kernel/random/boot_id').read_text()
        (out/'boot-id-before.txt').write_text(boot)
        before=subprocess.check_output(['npu-smi','info'],text=True)
        (out/'npu-before.txt').write_text(before)
        if not all(f'No running processes found in NPU {i}' in before for i in range(8)):
            raise RuntimeError('NPU process appeared; reset cancelled')
        (out/'topology-before.txt').write_text(subprocess.check_output(['npu-smi','info','-t','topo'],text=True))
        command=['npu-smi','set','-t','reset','-i','0','-c','0','-m','1']
        result=subprocess.run(command,input='Y\ny\n',text=True,capture_output=True,timeout=180)
        (out/'reset.json').write_text(json.dumps(dict(command=command,returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
        print(result.stdout,flush=True)
        if result.returncode:raise RuntimeError('device reset failed; no host reboot fallback')
        time.sleep(25)
        for attempt in range(12):
            probe=subprocess.run(['npu-smi','info'],text=True,capture_output=True,timeout=20)
            (out/f'npu-after-{attempt}.txt').write_text(probe.stdout+probe.stderr)
            if probe.returncode==0 and all(f'No running processes found in NPU {i}' in probe.stdout for i in range(8)):
                break
            time.sleep(5)
        else:raise RuntimeError('eight devices did not return')
        if Path('/proc/sys/kernel/random/boot_id').read_text()!=boot:raise RuntimeError('host boot identity changed')
        (out/'topology-after.txt').write_text(subprocess.check_output(['npu-smi','info','-t','topo'],text=True))
        (out/'result.json').write_text(json.dumps(dict(status='reset_completed',host_boot_unchanged=True,
            next='health, communication and DMA probes; old lease remains'),indent=2)+'\n')
        print(str(out),flush=True)


if __name__=='__main__':main()
