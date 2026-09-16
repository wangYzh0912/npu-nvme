from types import SimpleNamespace
import subprocess
import sys
import pytest
from npu_nvme.storage import bindings


def test_missing_library_is_typed(monkeypatch):
    def missing(path): raise OSError('missing test library')
    monkeypatch.setattr(bindings.ctypes,'CDLL',missing)
    with pytest.raises(bindings.BackendUnavailable):bindings.load_backend('/missing/lib.so')


class Symbols:
    def __init__(self,missing=()):self.missing=set(missing);self.symbols={}
    def __getattr__(self,name):
        if name in self.missing:raise AttributeError(name)
        return self.symbols.setdefault(name,SimpleNamespace())


@pytest.mark.parametrize('symbol',['npu_nvme_init','npu_nvme_flush','npu_nvme_close','npu_nvme_read_batch_host','npu_nvme_write_batch_host','npu_nvme_wait_quiescent'])
def test_missing_required_symbol_is_typed(monkeypatch,symbol):
    monkeypatch.setattr(bindings.ctypes,'CDLL',lambda path:Symbols([symbol]) if path=='/test/lib.so' else Symbols())
    with pytest.raises(bindings.BackendUnavailable):bindings.load_backend('/test/lib.so')


@pytest.mark.parametrize('symbol',['npu_nvme_submit_write_batch_host','npu_nvme_poll_request','npu_nvme_release_request','npu_nvme_delta_get_area_offset','npu_nvme_get_io_timeout_ms','npu_nvme_register_tasks'])
def test_optional_symbols_independent(monkeypatch,symbol):
    monkeypatch.setattr(bindings.ctypes,'CDLL',lambda path:Symbols([symbol]) if path=='/test/lib.so' else Symbols())
    loaded=bindings.load_backend('/test/lib.so')
    assert not hasattr(loaded.lib,symbol)
    assert hasattr(loaded.lib,'npu_nvme_init')


def test_legacy_import_never_loads_library():
    code="import ctypes; ctypes.CDLL=lambda *a,**k: (_ for _ in ()).throw(AssertionError('eager load')); import c_bindings; from c_bindings import lib,acl_lib"
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
