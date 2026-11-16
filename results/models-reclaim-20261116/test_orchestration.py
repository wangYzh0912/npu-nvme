import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


continuation = load('models_continuation', 'continue_on_models.py')
reclaim = load('models_reclaim', 'reclaim.py')


class OrchestrationTests(unittest.TestCase):
    def test_copy_proof_detects_content_change_and_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / 'evidence.json'
            file.write_text('first')
            before = continuation.file_map(root)
            file.write_text('other')
            self.assertNotEqual(before, continuation.file_map(root))
            (root / 'redirect').symlink_to(file)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                continuation.file_map(root)

    def test_waits_for_busy_devices_without_launching_or_killing_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            idle = chr(10).join('No running processes found in NPU ' + str(i) for i in range(4, 8))
            config = dict(npu_ids=[4, 5, 6, 7], hardware_lock=str(root / 'hardware.lock'))
            saved = []
            with patch.object(continuation, 'BASE', root), patch.object(continuation, 'save', side_effect=lambda p, v: saved.append(v)), patch.object(continuation.subprocess, 'check_output', side_effect=['busy', idle]), patch.object(continuation.time, 'sleep') as sleep:
                continuation.wait_for_idle(config)
            self.assertEqual([r['status'] for r in saved], ['waiting_for_hardware', 'ready'])
            sleep.assert_called_once_with(30)

    def test_delete_guard_rejects_changed_and_out_of_scope_payload(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(reclaim, 'BASE', Path(temporary)):
            root = Path(temporary)
            path = root / 'qwen3-v13-source-001/training/checkpoint/rank_0/old.safetensors'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'original')
            s = path.stat()
            row = dict(path=str(path), device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                       allocated_bytes=s.st_blocks * 512, mtime_ns=s.st_mtime_ns, links=s.st_nlink)
            self.assertEqual(reclaim.verify_payload(row)[0], path)
            path.write_bytes(b'changed content')
            with self.assertRaisesRegex(ValueError, 'changed since archival'):
                reclaim.verify_payload(row)
            with self.assertRaisesRegex(ValueError, 'outside exact'):
                reclaim.verify_payload(dict(row, path=str(root / 'Qwen3-8B/weights.safetensors')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
