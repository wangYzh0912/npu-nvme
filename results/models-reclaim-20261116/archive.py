import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import time

BASE = Path('/models/npu_nvme_exp/user7-stack')
DEST = Path('/tmp/npu-nvme-checkpoint-io-closure-results/results/models-reclaim-20261116')
NAMES = ('qwen3-v13-source-001', 'qwen3-v13-source-002', 'qwen3-v13-replay-001',
         'qwen3-v13-restore-001', 'qwen3-v13-restore-002', 'qwen3-v13-restore-003')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + chr(10))
    temporary.replace(path)

def snapshot(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or path.resolve() != path:
        raise ValueError('non-regular or redirected payload: ' + str(path))
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                allocated_bytes=s.st_blocks * 512, mtime_ns=s.st_mtime_ns, links=s.st_nlink)

def payload(path):
    before = snapshot(path)
    if before['links'] != 1:
        raise ValueError('unexpected shared checkpoint payload')
    with path.open('rb') as f:
        header_bytes = struct.unpack('<Q', f.read(8))[0]
        if not 0 < header_bytes < 4 << 20:
            raise ValueError('unsafe safetensors header size')
        header = json.loads(f.read(header_bytes))
    target = DEST / 'legacy' / path.relative_to(BASE)
    write(target.with_suffix('.safetensors.header.json'), header)
    sha = digest(path)
    if snapshot(path) != before:
        raise RuntimeError('payload changed during hashing')
    result = dict(path=str(path), sha256=sha, **before,
                  header_archive=str(target.with_suffix('.safetensors.header.json').relative_to(DEST)))
    print(json.dumps(dict(hashed=str(path), sha256=sha)), flush=True)
    return result

def main():
    DEST.mkdir(exist_ok=False)
    if (BASE / 'hardware-campaign.lease.json').exists():
        raise RuntimeError('hardware campaign active')
    r1 = Path('/home/user7/npu-nvme-evidence/aic-aiv-io-20261116-r1')
    shutil.copytree(r1, DEST / 'completed-e1-saves')
    files = []
    for name in NAMES:
        root = BASE / name
        for source in sorted(root.rglob('*')):
            if source.is_symlink():
                continue
            if not source.is_file() or source.suffix == '.safetensors':
                continue
            if source.stat().st_size > 32 << 20:
                raise ValueError('unexpected large sidecar: ' + str(source))
            target = DEST / 'legacy' / source.relative_to(BASE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        files.extend(sorted((root / 'training/checkpoint').rglob('*.safetensors')))
    if len(files) != 32:
        raise ValueError('expected exactly 32 historical checkpoint payloads')
    record = dict(status='hashing', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  scope='Only six obsolete September Qwen v1.3 run checkpoint payloads; retain metadata, converted weights, compressed baseline archive, models and runtimes.',
                  deleted_payloads_reconstructable_from_git=False,
                  preserved_original_model='/models/Qwen3-8B',
                  payloads=[], disk_before=shutil.disk_usage('/models')._asdict())
    write(DEST / 'cleanup-plan.json', record)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for item in executor.map(payload, files):
            record['payloads'].append(item)
            write(DEST / 'cleanup-plan.json', record)
    record.update(status='archived_pending_remote_verification',
                  reclaimable_bytes=sum(x['allocated_bytes'] for x in record['payloads']))
    write(DEST / 'cleanup-plan.json', record)
    archived=[]
    for p in sorted(DEST.rglob('*')):
        if p.is_file() and p.name != 'SHA256SUMS.json':
            archived.append(dict(path=str(p.relative_to(DEST)), bytes=p.stat().st_size, sha256=digest(p)))
    write(DEST / 'SHA256SUMS.json', dict(files=archived))
    for p in [DEST, *DEST.rglob('*')]:
        os.chown(p, 22633, 22633)
    print(json.dumps(dict(status='archive_complete', files=len(archived), reclaimable_GiB=record['reclaimable_bytes'] / (1 << 30))), flush=True)

if __name__ == '__main__':
    main()
