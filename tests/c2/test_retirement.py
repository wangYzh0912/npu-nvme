"""Preview-only ABI2 symbol and source audit; no hardware."""
from pathlib import Path
import subprocess
import re
import ast
import pytest
ROOT=Path(__file__).resolve().parents[2]


def test_no_blocking_batch_exports_and_versioned_soname():
    library=ROOT/'build/libnpu_nvme.so'
    symbols=subprocess.check_output(['nm','-D','--defined-only',str(library)],text=True)
    assert not re.search(r'\bnpu_nvme_(?:write_batch(?:_crc|_host)?|read_batch(?:_host)?)\b',symbols)
    assert 'npu_nvme_abi_version' in symbols and 'npu_nvme_submit_transfer' in symbols
    assert 'libnpu_nvme.so.2' in subprocess.check_output(['readelf','-d',str(library)],text=True)


def test_no_runtime_sync_bulk_or_retired_callers():
    for name in ('dma.c','read_pipeline.c','write_pipeline.c'):
        assert not re.search(r'\baclrtMemcpy\s*\(', (ROOT/'src'/name).read_text())
    pattern=re.compile(r'\bnpu_nvme_(?:write_batch(?:_crc|_host)?|read_batch(?:_host)?)\b')
    for top in ('src','include','python','tests','experiments'):
        for p in (ROOT/top).rglob('*'):
            if p.suffix in ('.c','.h','.py') and p.resolve()!=Path(__file__).resolve():
                assert not pattern.search(p.read_text()),str(p)



def test_abi_version_is_checked_without_device_initialization():
    import ctypes
    library=ctypes.CDLL(str(ROOT/'build/libnpu_nvme.so'))
    library.npu_nvme_abi_version.argtypes=[]
    library.npu_nvme_abi_version.restype=ctypes.c_uint32
    assert library.npu_nvme_abi_version()==2
