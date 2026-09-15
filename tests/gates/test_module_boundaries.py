import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_v2_bytes_unchanged_from_A_entry():
    import hashlib
    import json
    fixture=json.loads((ROOT/'tests/fixtures/v2_compat.json').read_text())
    from npu_nvme.storage import layout,format
    geometry=layout.make_layout(*fixture['layout_args'])
    assert hashlib.sha256(format.pack_superblock(geometry)).hexdigest()==fixture['superblock_sha256']
    for item in fixture['metadata']:
        encoded=format.pack_metadata(item['payload'],item['generation'])
        assert hashlib.sha256(encoded).hexdigest()==item['sha256']
        assert format.unpack_metadata(encoded)==(item['generation'],item['payload'])


def test_new_binding_import_does_not_open_a_library():
    program='''import ctypes
ctypes.CDLL=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('eager load'))
import npu_nvme.storage.bindings
'''
    result=subprocess.run([sys.executable,'-c',program],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr




def test_runtime_has_no_framework_import_or_facade_proxy():
    import ast
    for path in (ROOT / 'python/npu_nvme/runtime').glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(part in (node.module or '').split('.')
                               for part in ('framework', 'direct_checkpoint', 'checkpoint', 'mindspore', 'numpy'))
            if isinstance(node, ast.Import):
                assert not any(part in alias.name.split('.')
                               for alias in node.names for part in ('framework', 'direct_checkpoint', 'mindspore', 'numpy'))
            if isinstance(node, ast.ClassDef):
                assert not any(isinstance(base, ast.Name) and 'Mixin' in base.id
                               for base in node.bases)


def test_core_imports_without_framework_or_native_library():
    program = """import builtins, ctypes
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('mindspore', 'numpy')):
        raise AssertionError('runtime dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
ctypes.CDLL = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('eager load'))
from npu_nvme import StrictCheckpoint
import npu_nvme.types, npu_nvme.runtime.d1_runtime
import npu_nvme.runtime.commit, npu_nvme.runtime.d1_restore
import npu_nvme.storage.metadata, npu_nvme.storage.full_transport
"""
    result = subprocess.run([sys.executable, '-c', program], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
