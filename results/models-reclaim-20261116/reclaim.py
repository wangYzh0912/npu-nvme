"""Delete only previously hashed obsolete payloads after verified remote archival."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

BASE = Path('/models/npu_nvme_exp/user7-stack')
ARCHIVE = Path('/tmp/npu-nvme-checkpoint-io-closure-results/results/models-reclaim-20261116')
NAMES = {'qwen3-v13-source-001', 'qwen3-v13-source-002', 'qwen3-v13-replay-001',
         'qwen3-v13-restore-001', 'qwen3-v13-restore-002', 'qwen3-v13-restore-003'}


def verify_payload(row):
    path = Path(row['path'])
    relative = path.relative_to(BASE)
    if (relative.parts[0] not in NAMES or relative.parts[1:3] != ('training', 'checkpoint')
            or path.suffix != '.safetensors' or path.resolve() != path):
        raise ValueError('payload outside exact historical checkpoint scope')
    s = path.lstat()
    actual = dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                  allocated_bytes=s.st_blocks * 512, mtime_ns=s.st_mtime_ns, links=s.st_nlink)
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or any(row[key] != value for key, value in actual.items()):
        raise ValueError('payload changed since archival: ' + str(path))
    return path, (s.st_dev, s.st_ino)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    content = (ARCHIVE / 'cleanup-plan.json').read_bytes()
    plan = json.loads(content)
    remote = json.loads((ARCHIVE / 'remote-verified-before-delete.json').read_text())
    if (plan['status'] != 'archived_pending_remote_verification' or len(plan['payloads']) != 32
            or remote['status'] != 'pass'
            or remote['cleanup_plan_sha256'] != hashlib.sha256(content).hexdigest()):
        raise ValueError('remote archival proof does not cover this exact payload plan')
    with (BASE / 'hardware-campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (BASE / 'hardware-campaign.lease.json').exists():
            raise RuntimeError('hardware campaign has a lease')
        selected = [verify_payload(row) for row in plan['payloads']]
        identities = {identity for _, identity in selected}
        if len(identities) != 32:
            raise ValueError('duplicate payload inode')
        for descriptor in Path('/proc').glob('[0-9]*/fd/*'):
            try:
                s = descriptor.stat()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if (s.st_dev, s.st_ino) in identities:
                raise RuntimeError('checkpoint payload is open: ' + str(descriptor))
        before = shutil.disk_usage('/models')._asdict()
        if not args.execute:
            print(json.dumps(dict(status='dry_run_pass', payloads=len(selected), reclaimable_bytes=plan['reclaimable_bytes'])))
            return
        report = dict(status='deleting', remote_archive_commit=remote['remote_commit'],
                      started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), disk_before=before, deleted=[])
        target = ARCHIVE / 'cleanup-execution.json'
        def save():
            temporary = target.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2) + chr(10))
            temporary.replace(target)
            os.chown(target, 22633, 22633)
        save()
        for row in plan['payloads']:
            path, _ = verify_payload(row)
            path.unlink()
            report['deleted'].append(dict(path=str(path), sha256=row['sha256'], allocated_bytes=row['allocated_bytes']))
            save()
        report.update(status='completed', completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      deleted_allocated_bytes=sum(row['allocated_bytes'] for row in report['deleted']),
                      disk_after=shutil.disk_usage('/models')._asdict(),
                      retained=dict(model=Path('/models/Qwen3-8B').is_dir(),
                                    compressed_archive=(BASE / 'archive-qwen-baseline-20260916/compressed-checkpoints/manifest.json').is_file(),
                                    environment=(BASE / 'full-closure-abi3-20261021-005/environments.json').is_file()))
        save()
        print(json.dumps(report))


if __name__ == '__main__':
    main()
