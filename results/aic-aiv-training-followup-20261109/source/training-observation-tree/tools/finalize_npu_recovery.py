#!/usr/bin/env python3
"""Reconcile the failed pilot lease only after recorded recovery probes pass."""
import fcntl
import json
from pathlib import Path
import subprocess


def main():
    root=Path('/models/npu_nvme_exp/user7-stack')
    out=root/'recovery-eight-npus-20260916-001'
    with (root/'hardware-campaign.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=json.loads((out/'result.json').read_text())
        if result['status']!='reset_completed' or not result['host_boot_unchanged']:
            raise ValueError('reset proof missing')
        if (out/'boot-id-before.txt').read_text()!=Path('/proc/sys/kernel/random/boot_id').read_text():
            raise ValueError('host boot changed')
        for rank in range(8):
            report=json.loads((root/'recovery-eight-hccl-20260916-001'/f'rank-{rank}.json').read_text())
            if report['status']!='pass' or report['rank']!=rank:raise ValueError('HCCL probe failed')
        for rank in range(4):
            report=json.loads((root/f'placement-npu{rank}-20260916-001/result.json').read_text())
            if report['status']!='pass' or report['npu']!=rank:raise ValueError('DMA probe failed')
        processes=json.loads((root/'recovery-qwen-20260916-001/processes.json').read_text())
        for row in processes:
            path=Path('/proc')/str(row['pid'])/'stat'
            if path.exists():
                fields=path.read_text().rsplit(')',1)[1].split()
                if fields[0]!='Z' and fields[19]==str(row['start_ticks']):raise ValueError('old process still alive')
        smi=subprocess.check_output(['npu-smi','info'],text=True)
        if not all(f'No running processes found in NPU {rank}' in smi for rank in range(8)):
            raise ValueError('device still occupied')
        (out/'npu-final.txt').write_text(smi)
        lease=root/'hardware-campaign.lease.json'
        value=json.loads(lease.read_text())
        if value['process_group']!=1798749:raise ValueError('unexpected hardware lease')
        lease.rename(out/'reconciled-hardware-campaign.lease.json')
        (out/'reconciliation.json').write_text(json.dumps(dict(status='pass',lease_released=True,
            hugepage_files='preserved; new jobs use distinct shm ids',host_boot_unchanged=True),indent=2)+'\n')
        print('Eight-card recovery verified; failed pilot lease archived.',flush=True)


if __name__=='__main__':main()
