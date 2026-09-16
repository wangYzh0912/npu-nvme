"""CPU-only regression checks for the isolated Qwen training entry."""
import importlib.util
import json
from pathlib import Path
import subprocess
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'experiments/training/train_qwen3_full_restart.py'


def test_report_replace_is_complete(tmp_path):
    spec = importlib.util.spec_from_file_location('qwen_entry', ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'acceptance.json'
    module.write_report(path, {'rank': 0, 'status': 'running'})
    module.write_report(path, {'rank': 0, 'status': 'failed'})
    assert json.loads(path.read_text()) == {'rank': 0, 'status': 'failed'}
    assert not path.with_suffix('.tmp').exists()


def test_invalid_steps_rejected_before_loading_runtime(tmp_path):
    result = subprocess.run([sys.executable, str(ENTRY), '--output', str(tmp_path), '--steps', '0'],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'steps must be positive' in result.stderr
    assert not list(tmp_path.iterdir())


def test_launcher_uses_candidate_module_and_joins_workers():
    launcher = ROOT / 'scripts/run_qwen3_four_rank.sh'
    subprocess.run(['bash', '-n', str(launcher)], check=True)
    script = launcher.read_text()
    assert '--profile candidate' in script
    assert 'python -m mindspore.parallel.cluster.run' in script
    assert '--join=True' in script
    assert '--worker_num=4 --local_worker_num=4' in script
    assert '/usr/local/bin/msrun' not in script
    assert 'run_qwen_worker.sh' not in script


def test_training_uses_native_nonlegacy_context():
    script = ENTRY.read_text()
    assert 'config.use_legacy = False' in script
    assert 'TransformerConfig.swap' not in script
    assert "rank_dir = out / f'rank_{rank}'" in script
    assert 'trainer.finetune(resume_from_checkpoint=str(source)' in script
    assert 'training_pass_restart_not_tested' in script


def test_checkpoint_audit_rejects_truncation_and_wrong_step(tmp_path):
    spec = importlib.util.spec_from_file_location('qwen_audit', ENTRY.with_name('check_qwen_training_run.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    header = {}
    data = bytearray()
    for key, dtype, fmt, value in (
        ('layer.weight', 'F32', '<f', 1.0),
        ('adam_m.layer.weight', 'F32', '<f', 0.1),
        ('adam_v.layer.weight', 'F32', '<f', 0.2),
        ('global_step', 'I32', '<i', 8), ('step_num', 'I64', '<q', 8),
        ('epoch_num', 'I64', '<q', 8), ('loss_scale', 'F32', '<f', 1.0),
    ):
        start = len(data)
        data.extend(struct.pack(fmt, value))
        header[key] = {'dtype': dtype, 'shape': [], 'data_offsets': [start, len(data)]}
    raw = json.dumps(header).encode()
    path = tmp_path / 'state.safetensors'
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + data)
    assert module.inspect_checkpoint(path, 8)['adam_parameter_pairs'] == 1
    with pytest.raises(ValueError, match='Wrong checkpoint step'):
        module.inspect_checkpoint(path, 7)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match='Truncated'):
        module.inspect_checkpoint(path, 8)
