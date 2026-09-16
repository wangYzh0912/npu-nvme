"""C_IMPL: production C under ASan/UBSan, external calls stubbed; no hardware."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def binary(tmp_path_factory):
    out = Path(os.environ['GATE_ARTIFACT_DIR'])/'compiled' if os.environ.get('GATE_ARTIFACT_DIR') else tmp_path_factory.mktemp('c-host')
    out.mkdir(parents=True,exist_ok=True)
    spdk = Path(os.environ.get('SPDK_DIR', '/home/user7/npu-nvme/third_party/spdk'))
    cann = Path(os.environ.get('CANN_INCLUDE', '/usr/local/Ascend/ascend-toolkit/latest/include'))
    if not shutil.which('cc') or not shutil.which('clang') or not (spdk/'include/spdk/nvme.h').is_file() or not (cann/'acl/acl.h').is_file():
        pytest.skip('C_IMPL requires cc, clang compiler-rt, SPDK and CANN headers')
    obj = out/'host.o'
    exe = out/'host.elf'
    compile_cmd = ['cc','-std=gnu11','-g','-O1','-ffunction-sections','-fdata-sections',
                   '-fsanitize=address,undefined','-fno-omit-frame-pointer',
                   '-I'+str(ROOT/'include'),'-I'+str(spdk/'include'),
                   '-I'+str(spdk/'dpdk/build/include'),'-I'+str(cann),
                   '-c',str(ROOT/'tests/c/host_safety_test.c'),'-o',str(obj)]
    # GCC's local sanitizer symlinks are broken; compiler-rt supplies compatible
    # v8 runtime. Suppress export-dynamic so uncalled hardware entry points are GC'd.
    link_cmd = ['clang','-fsanitize=address,undefined',str(obj),
                '-Wl,--gc-sections,--no-export-dynamic','-lpthread','-lcrypto','-o',str(exe)]
    for cmd in (compile_cmd, link_cmd):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    print('C_IMPL compile:', compile_cmd, 'link:', link_cmd, flush=True)
    from hashlib import sha256
    print('C_IMPL binary_sha256:', sha256(exe.read_bytes()).hexdigest(), flush=True)
    return exe


@pytest.mark.parametrize('case', ['role_permissions','metadata_failed_flush','pointer_overflow','copy_d2h','copy_h2d','copy_corrupt','streaming_windows','flush_prior_failure','capabilities','pending_limit','write_rotation','read_rotation','checksum_slices','flush_dependency','flush_failed_dependency','read_valid','read_corrupt','read_late','read_query_quarantine','read_record_quarantine','transfer_digest','transfer_invalid','delta_overflow','batch_bound','batch_bytes',
                                  'geometry_overflow','metadata_capacity','legal_bounds',
                                  'queued_close','early_release','late_completion',
                                  'write_completion','query_quarantine','dma_stopped','copy_rejected',
                                  'close_deadline','close_retry','metadata_timeout','lost_completion',
                                  'spdk_exit','native_owner','concurrent_close','closed_admission'])
def test_input_bounds(binary, case):
    result = subprocess.run([str(binary),case],capture_output=True,text=True,timeout=10,
                            env=dict(os.environ, ASAN_OPTIONS='detect_leaks=1',
                                     UBSAN_OPTIONS='halt_on_error=1'))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS' in result.stdout


def test_dma_quarantine(binary):
    result = subprocess.run([str(binary),'dma_quarantine'],capture_output=True,text=True,timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
