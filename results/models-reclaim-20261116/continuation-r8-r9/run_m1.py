#!/usr/bin/env python3
"""Continue M1 using role-specific DPDK arguments and isolated hugepages."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/tmp/npu-nvme-checkpoint-io-closure-owner-env')
PREVIOUS = Path('/models/npu_nvme_exp/user7-stack/aic-aiv-io-closure-20261116-r2')
BASE = Path('/models/npu_nvme_exp/user7-stack/aic-aiv-io-closure-20261116-r9')
LIBRARY = Path('/models/npu_nvme_exp/user7-stack/aic-aiv-io-closure-20261116-r5/execution/native-build/libnpu_nvme.so.3.0')
HUGEPAGES = Path('/models/npu_nvme_exp/user7-stack/aic-aiv-io-closure-20261116-r8/execution/hugepages')
EXPECTED_COMMIT = 'e227968cdbf126d0f14bf5525916af361959246f'
SHM_BASE = 82000
DPDK_ARGS = f'--iova-mode=va --huge-dir={HUGEPAGES}'

sys.path[:0] = [str(ROOT / 'python'), str(ROOT)]
from tools.checkpoint_io import base_config, save
from tools.checkpoint_io_closure import Campaign
from npu_nvme.cli.contracts import source_identity
from npu_nvme.runtime import training_catalog as catalog


def admission_wait(config):
    path = BASE / 'execution/admission-wait.json'
    while True:
        info = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=30)
        lock_path = Path(config['hardware_lock'])
        lease = lock_path.with_suffix('.lease.json')
        idle = all(f'No running processes found in NPU {device}' in info
                   for device in config['npu_ids'])
        with lock_path.open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock_available = False
            else:
                lock_available = True
        ready = idle and lock_available and not lease.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(status='ready' if ready else 'waiting_for_hardware',
            updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(),
            devices=config['npu_ids'], devices_idle=idle,
            hardware_lock_available=lock_available, lease_present=lease.exists(),
            npu_info=info), indent=2) + '\n')
        if ready:
            return
        time.sleep(30)


class WaitingCampaign(Campaign):
    def run(self, name, method, **kwargs):
        admission_wait(self.config)
        previous = os.environ.get('NPU_NVME_DPDK_ARGS')
        if kwargs.get('profile'):
            os.environ['NPU_NVME_DPDK_ARGS'] = DPDK_ARGS
        else:
            os.environ.pop('NPU_NVME_DPDK_ARGS', None)
        try:
            return super().run(name, method, **kwargs)
        finally:
            if previous is None:
                os.environ.pop('NPU_NVME_DPDK_ARGS', None)
            else:
                os.environ['NPU_NVME_DPDK_ARGS'] = previous


def main():
    previous = json.loads((PREVIOUS / 'evidence/result.json').read_text())
    if previous.get('gates', {}).get('e1') != 'pass':
        raise ValueError('previous E1 gate is not passed')
    if previous.get('phase') != 'e4_device_captures' or previous.get('active_run', {}).get('name') != 'e4-none':
        raise ValueError('previous queue did not stop at the expected E4 profiler failure')
    if previous.get('gates', {}).get('e2') != 'not_run' or previous.get('gates', {}).get('e3') != 'not_run':
        raise ValueError('unexpected E2/E3 state in previous queue')
    restored = PREVIOUS / 'raw/e1-restore28-to32/result.json'
    restored_result = json.loads(restored.read_text())
    if (restored_result.get('validation_status') != 'pass' or
            restored_result.get('final_step') != 32 or
            restored_result.get('end_to_end', {}).get('lease_released') is not True):
        raise ValueError('step 28 to 32 restore did not safely pass')
    if (BASE / 'raw').exists() or (BASE / 'evidence').exists():
        raise FileExistsError('M1 r9 raw/evidence output exists; inspect it before retry')
    if not HUGEPAGES.is_dir() or not LIBRARY.is_file():
        raise FileNotFoundError('private hugepage mount or rebuilt ABI3 library is missing')
    if source_identity(ROOT)['observed_commit'] != EXPECTED_COMMIT:
        raise RuntimeError('role-specific DPDK source commit differs')

    (BASE / 'execution').mkdir(parents=True, exist_ok=True)
    manifest_path = Path('/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/environments.json')
    manifest = json.loads(manifest_path.read_text())
    manifest['profiles']['candidate']['library'] = str(LIBRARY)
    private_manifest = BASE / 'execution/environments.json'
    private_manifest.write_text(json.dumps(manifest, indent=2) + '\n')

    config = base_config(ROOT / 'config/qwen_training.json')
    config['environment_manifest'] = str(private_manifest)
    config['shm_base'] = SHM_BASE
    os.environ['NPU_NVME_DPDK_ARGS'] = DPDK_ARGS
    admission_wait(config)
    campaign = WaitingCampaign(config, BASE / 'raw', BASE / 'evidence')

    oracle_source = PREVIOUS / 'raw/e1-none'
    oracle = campaign.raw / 'e1-none'
    oracle.mkdir()
    copied = []
    for rank in range(4):
        source_rank = oracle_source / f'rank_{rank}'
        target_rank = oracle / f'rank_{rank}'
        target_rank.mkdir()
        for name in ('training.json', 'state-initial.json', 'control-initial.json',
                     'state-step-32.json', 'control-step-32.json'):
            source = source_rank / name
            target = target_rank / source.name
            shutil.copy2(source, target)
            source_sha = catalog.digest_file(source)
            target_sha = catalog.digest_file(target)
            if source_sha != target_sha:
                raise ValueError('copied E1 oracle evidence digest differs: ' + str(source))
            copied.append(dict(path=str(target.relative_to(campaign.raw)),
                bytes=target.stat().st_size, sha256=target_sha, source_sha256=source_sha))
        report = catalog.read_checked(target_rank / 'training.json')
        if report.get('status') != 'pass' or report.get('final_step') != 32:
            raise ValueError('T0 oracle requires a passed full 32-step trajectory')

    comparisons = [row for row in previous['comparisons'] if row.get('status') == 'pass']
    restore_comparison = next((row for row in comparisons
        if row.get('stage') == 'e1' and row.get('method') == 'fresh_restore28'), None)
    if restore_comparison is None:
        raise ValueError('fresh restore comparison is missing')
    campaign.record.update(phase='e4_device_captures',
        gates=dict(e1='pass', e2='not_run', e3='not_run', e4='not_run'),
        prior_e1=dict(result=str(PREVIOUS / 'evidence/result.json'),
            result_sha256=catalog.digest_file(PREVIOUS / 'evidence/result.json'),
            passing_comparisons=comparisons,
            restore_run=previous['runs'][-1], restore_comparison=restore_comparison,
            note='E1 is the independently recorded frozen result; E4 and E2/E3 run the role-specific DPDK source.'),
        e1_initial_oracle=dict(source=str(oracle_source), copied_files=copied),
        continuation=dict(source_commit=EXPECTED_COMMIT,
            native_library=str(LIBRARY), native_library_sha256=catalog.digest_file(LIBRARY),
            dpdk_args_for_profiled_copy_rank=DPDK_ARGS, shm_base=SHM_BASE,
            raw_owner_dpdk='default PA mode and /dev/hugepages; copy-rank override removed by source fix',
            private_hugetlbfs=str(HUGEPAGES), original_abi3_manifest_preserved=True,
            output_filesystem='/models', raw_format_authorized=False))
    campaign.cache = None
    campaign.write()
    try:
        campaign.e4(oracle)
        campaign.e2()
        campaign.record.update(status='closure_gates_passed', phase='completed',
            limitation='E4 verifies synchronous dependency limits. Device-engine/PMU attribution remains separate; no M2 implementation authorized.')
        campaign.write()
    except BaseException as error:
        campaign.record.update(status='failed', error=repr(error))
        campaign.write()
        active = campaign.record.get('active_run')
        if active and Path(active['raw']).exists():
            try:
                from tools.checkpoint_io import archive_run
                archive_run(active['raw'], campaign.out / 'failed' / active['name'])
            except Exception as archive_error:
                campaign.record['failure_archive_error'] = repr(archive_error)
                campaign.write()
        raise


if __name__ == '__main__':
    main()
