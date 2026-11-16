"""Resume the frozen M1 queue on /models after publishing/reclaiming old payloads.

This is experiment orchestration, outside the frozen execution source tree.
The former tmpfs E1 timings keep their original storage identity. All newly
executed restores, profiler captures and A/B runs use /models output paths.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/tmp/npu-nvme-checkpoint-io-closure')
OLD_RAW = Path('/dev/shm/user7-aic-aiv-io-closure-20261116-r1')
OLD_OUT = Path('/home/user7/npu-nvme-evidence/aic-aiv-io-20261116-r1')
BASE = Path('/models/npu_nvme_exp/user7-stack/aic-aiv-io-closure-20261116-r2')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT)]
from tools.checkpoint_io_closure import Campaign, METHODS, complete_timeline, lifecycle_gate, read_events
from tools.checkpoint_io import base_config, save, chunk_signatures, archive_run
from npu_nvme.runtime import training_catalog as catalog


def wait_for_idle(config):
    status = BASE / 'execution/admission-wait.json'
    while True:
        info = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=30)
        lease = Path(config['hardware_lock']).with_suffix('.lease.json')
        idle = all(f'No running processes found in NPU {device}' in info for device in config['npu_ids'])
        ready = idle and not lease.exists()
        save(status, dict(status='ready' if ready else 'waiting_for_hardware',
            pid=os.getpid(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            devices=config['npu_ids'], devices_idle=idle, lease_present=lease.exists(),
            npu_info=info, behavior='Wait without terminating or changing other workloads. Recheck admission before each run.'))
        if ready:
            return
        time.sleep(30)


class ModelsCampaign(Campaign):
    def run(self, *args, **kwargs):
        wait_for_idle(self.config)
        return super().run(*args, **kwargs)


def file_map(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('unexpected symlink in E1 source: ' + str(path))
        if path.is_file():
            result[str(path.relative_to(root))] = dict(bytes=path.stat().st_size,
                                                     sha256=catalog.digest_file(path))
    return result


def main():
    old = json.loads((OLD_OUT / 'result.json').read_text())
    if (old['status'], old['error']) != ('failed', "OSError('durable evidence headroom below 8 GiB')"):
        raise ValueError('previous queue did not stop at the expected admission gate')
    expected = ['e1-' + method for method in METHODS]
    if [row['name'] for row in old['runs']] != expected or old.get('active_run'):
        raise ValueError('expected three completed E1 saves and no active run')
    if (BASE / 'raw').exists() or (BASE / 'evidence').exists():
        raise FileExistsError('continuation output already exists; inspect it before retry')
    config = base_config(ROOT / 'config/qwen_training.json')
    wait_for_idle(config)
    campaign = ModelsCampaign(config, BASE / 'raw', BASE / 'evidence')
    try:
        campaign.record.update(phase='verify_and_relocate_completed_e1',
            continuation=dict(previous_result=str(OLD_OUT / 'result.json'),
                previous_result_sha256=catalog.digest_file(OLD_OUT / 'result.json'),
                previous_status=old['status'], previous_error=old['error'],
                original_e1_output_filesystem='tmpfs /dev/shm',
                new_output_filesystem='/models mounted filesystem',
                performance_scope='Reused E1 timings retain original tmpfs identity. E4 and all eighteen E2 A/B processes use the same new /models placement; do not pool old/new placement timings.',
                raw_media='Existing raw83 registered [256,1280) GiB, no formatting.',
                orchestrator_sha256=catalog.digest_file(Path(__file__))))
        campaign.write()
        if campaign.identity != old['source_identity'] or campaign.binaries != old['binary_sha256']:
            raise ValueError('frozen execution source or binaries differ from completed E1')
        campaign.frozen()
        before = file_map(OLD_RAW)
        save(campaign.out / 'relocation-source-files.json', before)
        subprocess.run(['cp', '-a', str(OLD_RAW) + '/.', str(campaign.raw) + '/'], check=True)
        after = file_map(campaign.raw)
        if after != before:
            raise ValueError('relocated E1 files differ by path, size or full SHA256')
        shutil.copytree(OLD_OUT, campaign.out / 'reused-e1-evidence')
        campaign.cache = campaign.raw / 'e1-none'
        for row in old['runs']:
            reused = copy.deepcopy(row)
            reused.update(path=str(campaign.raw / row['name']), reused_from=row['path'],
                          executed_output_filesystem='tmpfs /dev/shm')
            campaign.record['runs'].append(reused)
        proofs = []
        for method in METHODS:
            path = campaign.raw / ('e1-' + method)
            if json.loads((path / 'source-identity.json').read_text())['execution_sources'] != campaign.identity:
                raise ValueError('reused run execution source differs')
            lifecycle = lifecycle_gate(path, detailed=True)
            report = dict(method=method, lifecycle=lifecycle)
            if method != 'none':
                reports = [catalog.read_checked(path / f'rank_{rank}/training.json') for rank in range(4)]
                timeline = complete_timeline(read_events(path), reports, method, range(4, 33, 4))
                if timeline['status'] != 'pass':
                    raise ValueError('reused E1 full boundary proof failed')
                save(campaign.out / 'revalidated' / (method + '-timeline.json'), timeline)
                campaign.initial_comparison(campaign.cache, path)
                campaign.comparison(campaign.cache, path, stage='e1_reused', method=method)
            proofs.append(report)
        sink = chunk_signatures(campaign.raw / 'e1-checkpoint_io_sink', 'sink_consumed_source_safe')
        full = chunk_signatures(campaign.raw / 'e1-ours', 'chunk_send_begin')
        if sink != full:
            raise ValueError('relocated E1 capture equivalence failed')
        chunk = dict(status='pass', streams=len(sink), chunks=sum(map(len, sink.values())),
                     logical_bytes=sum(v[2] for rows in sink.values() for v in rows))
        if chunk != json.loads((OLD_OUT / 'e1-chunk-equivalence.json').read_text()):
            raise ValueError('revalidated chunk totals differ from published E1 evidence')
        save(campaign.out / 'e1-chunk-equivalence.json', chunk)
        save(campaign.out / 'relocation-verification.json', dict(status='pass', files=len(before),
            logical_bytes=sum(row['bytes'] for row in before.values()),
            source=str(OLD_RAW), target=str(campaign.raw), lifecycles=proofs,
            source_files_manifest_sha256=catalog.digest_file(campaign.out / 'relocation-source-files.json')))
        campaign.frozen()
        # Restore the retained step-28 generation before any new T2 writes can retire it.
        campaign.record['phase'] = 'e1_fresh_restore28_on_models'
        campaign.write()
        restored = campaign.restore(campaign.raw / 'e1-ours', 'e1-restore28-to32', 28)
        campaign.comparison(campaign.cache, restored, stage='e1', method='fresh_restore28')
        campaign.record['gates']['e1'] = 'pass'
        campaign.write()
        campaign.e4(campaign.cache)
        campaign.e2()
        campaign.record.update(status='closure_gates_passed', phase='completed',
            limitation='E4 verifies the synchronous dependency limit. Device engine/PMU attribution remains separate. No M2 implementation authorized.')
        campaign.write()
    except BaseException as error:
        campaign.record.update(status='failed', error=repr(error))
        campaign.write()
        active = campaign.record.get('active_run')
        if active and Path(active['raw']).exists():
            try:
                archive_run(active['raw'], campaign.out / 'failed' / active['name'])
            except Exception as archive_error:
                campaign.record['failure_archive_error'] = repr(archive_error)
                campaign.write()
        raise


if __name__ == '__main__':
    main()
