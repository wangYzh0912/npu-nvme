"""Continue M1 from passed E1 after fixing candidate-CANN profiler ownership."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/tmp/npu-nvme-checkpoint-io-closure')
PREVIOUS = Path('/models/npu_nvme_exp/user7-stack/aic-aiv-io-closure-20261116-r2')
BASE = Path('/models/npu_nvme_exp/user7-stack/aic-aiv-io-closure-20261116-r3')
EXPECTED_COMMIT = '7666596964ef4b76d7100b7e00b47a158904ef8a'
sys.path[:0] = [str(ROOT / 'python'), str(ROOT)]
from tools.checkpoint_io import base_config, save
from tools.checkpoint_io_closure import Campaign, schedule
from npu_nvme.runtime import training_catalog as catalog
from npu_nvme.cli.contracts import source_identity


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + chr(10))
    temporary.replace(path)


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
        write(path, dict(status='ready' if ready else 'waiting_for_hardware',
            updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(),
            devices=config['npu_ids'], devices_idle=idle, hardware_lock_available=lock_available,
            lease_present=lease.exists(), npu_info=info,
            behavior='Wait without terminating or changing other workloads. Recheck before each run.'))
        if ready:
            return
        time.sleep(30)


class WaitingCampaign(Campaign):
    def run(self, *args, **kwargs):
        admission_wait(self.config)
        return super().run(*args, **kwargs)


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
    if BASE.exists():
        raise FileExistsError('M1 r3 output exists; inspect it before retry')
    (BASE / 'execution').mkdir(parents=True)
    config = base_config(ROOT / 'config/qwen_training.json')
    if source_identity(ROOT)['observed_commit'] != EXPECTED_COMMIT:
        raise RuntimeError('profiler-owner fix source commit differs')
    admission_wait(config)
    campaign = WaitingCampaign(config, BASE / 'raw', BASE / 'evidence')
    oracle_source = PREVIOUS / 'raw/e1-none'
    oracle = campaign.raw / 'e1-none'
    oracle.mkdir()
    oracle_rows = []
    for rank in range(4):
        source_rank = oracle_source / f'rank_{rank}'
        target_rank = oracle / f'rank_{rank}'
        target_rank.mkdir()
        for kind in ('state', 'control'):
            source = source_rank / f'{kind}-initial.json'
            target = target_rank / source.name
            shutil.copy2(source, target)
            oracle_rows.append(dict(path=str(target.relative_to(campaign.raw)),
                bytes=target.stat().st_size, sha256=catalog.digest_file(target),
                source_sha256=catalog.digest_file(source)))
    prior_comparisons = [row for row in previous['comparisons'] if row.get('status') == 'pass']
    restore_comparison = next((row for row in prior_comparisons
        if row.get('stage') == 'e1' and row.get('method') == 'fresh_restore28'), None)
    if restore_comparison is None:
        raise ValueError('fresh restore comparison is missing')
    campaign.record.update(phase='e4_device_captures', gates=dict(e1='pass',e2='not_run',e3='not_run',e4='not_run'),
        prior_e1=dict(source_commit=previous['runs'][0]['source_identity']['observed_commit'],
            result=str(PREVIOUS / 'evidence/result.json'),
            result_sha256=catalog.digest_file(PREVIOUS / 'evidence/result.json'),
            passing_comparisons=prior_comparisons,restore_run=previous['runs'][-1],
            restore_comparison=restore_comparison,
            note='E1 was completed under its separately recorded frozen source. E4/E2/E3 use the new frozen profiler-owner fix commit.'),
        e1_initial_oracle=dict(source=str(oracle_source),copied_files=oracle_rows),
        continuation=dict(fixed_source_commit=EXPECTED_COMMIT,
            profiler_worker_uid=os.stat('/models/npu_nvme_exp/user7-stack/cann-install/ascend-toolkit/latest').st_uid,
            previous_profiler_failure=str(PREVIOUS / 'evidence/failed/e4-none'),
            output_filesystem='/models', raw_format_authorized=False))
    campaign.cache = None
    campaign.write()
    try:
        campaign.e4(oracle)
        campaign.e2()
        campaign.record.update(status='closure_gates_passed',phase='completed',
            limitation='E4 verifies the synchronous dependency limit. Device-engine/PMU attribution remains separate; no M2 implementation authorized.')
        campaign.write()
    except BaseException as error:
        campaign.record.update(status='failed',error=repr(error))
        campaign.write()
        active=campaign.record.get('active_run')
        if active and Path(active['raw']).exists():
            try:
                from tools.checkpoint_io import archive_run
                archive_run(active['raw'],campaign.out/'failed'/active['name'])
            except Exception as archive_error:
                campaign.record['failure_archive_error']=repr(archive_error)
                campaign.write()
        raise


if __name__ == '__main__':
    main()
