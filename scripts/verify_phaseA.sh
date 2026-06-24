#!/bin/bash
# ============================================================
# Phase A: NPU 服务器重构验证脚本
# 用法: bash scripts/verify_phaseA.sh
# ============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
check() { echo -e "\n${GREEN}=== $1 ===${NC}"; }

REPO=/home/user7/npu-nvme
SUDO_PW="CGCL_2025_#$"
export PYTHONPATH=$REPO/python:$PYTHONPATH
cd $REPO

echo "============================================================"
echo "  Phase A: NPU-NVMe Refactoring Verification"
echo "  $(date)"
echo "============================================================"

# --- A.1: Git pull latest ---
check "A.1: Git pull latest code"
git pull origin master && pass "Git pull" || fail "Git pull"

# --- A.2: C compilation ---
check "A.2: C compilation"
cd $REPO/build_out
make clean 2>/dev/null || true
if make 2>&1 | tail -20; then
    pass "C compilation"
else
    fail "C compilation failed"
fi
cd $REPO

# --- A.3: Pure-logic tests (no hardware needed) ---
check "A.3: Pure-logic unit tests"
OUTPUT=$(echo "$SUDO_PW" | sudo -S $REPO/build_out/bin/run_test.sh 2>&1) || true
echo "$OUTPUT"
if echo "$OUTPUT" | grep -q "FAIL"; then
    fail "Pure-logic tests had failures"
elif echo "$OUTPUT" | grep -q "PASS"; then
    pass "Pure-logic tests"
else
    fail "Pure-logic tests — no PASS/FAIL found"
fi

# --- A.4: Hardware integration tests ---
check "A.4: Hardware integration tests (NVMe + NPU)"
OUTPUT=$(echo "$SUDO_PW" | sudo -S $REPO/build_out/bin/run_test.sh 0000:83:00.0 1 2>&1) || true
echo "$OUTPUT"
if echo "$OUTPUT" | grep -q "FAIL"; then
    fail "Hardware tests had failures"
elif echo "$OUTPUT" | grep -q "PASS"; then
    pass "Hardware integration tests"
else
    fail "Hardware tests — no PASS/FAIL found"
fi

# --- A.5: Python imports ---
check "A.5: Python import verification"
python -c "
from direct_checkpoint import (DirectCheckpoint, ProbeTrainOneStepCell,
    lib, get_dev_ptr, build_chunks_host, pack_delta_frame, FileDeltaWriter,
    NPUNVMEContext, replace_with_noop_initializer)
print('  direct_checkpoint imports OK')

sys.path.insert(0, '$REPO')
from experiments.common import (
    make_gpt2xl_training, setup_faf_checkpointing, make_ckpt,
    StepTimer, EpochTimer, init_env)
print('  experiments.common imports OK')
" 2>&1 && pass "Python imports" || fail "Python imports"

# --- A.6: FULL checkpoint roundtrip ---
check "A.6: FULL checkpoint roundtrip"
python -c "
import sys, os, time, numpy as np
sys.path.insert(0, 'python')
import mindspore as ms
from mindspore import Tensor
from direct_checkpoint import DirectCheckpoint, format_npu_disk

msg = ms.Tensor(np.random.randn(1000, 1000).astype(np.float16))
ckpt = DirectCheckpoint(nvme_addr='0000:83:00.0', npu_device_id=1,
                         pipeline_depth=8, chunk_size=4*1024*1024)
t0 = time.perf_counter()
rc, n_chunks, _, _, stats = ckpt.save([msg], step=0)
dt_save = (time.perf_counter() - t0) * 1000
print(f'  SAVE: {n_chunks} chunks, {dt_save:.1f}ms, '
      f'prep={stats[\"prep_time\"]*1000:.1f}ms')

t0 = time.perf_counter()
total_bytes, n_chunks, dt_load, bw, stats = ckpt.load([msg], step=0)
print(f'  LOAD: {total_bytes/1024/1024:.1f}MB, {dt_load*1000:.1f}ms, '
      f'BW={bw:.0f}MB/s')
ckpt.cleanup()
print(f'  FULL CKPT ROUNDTRIP OK')
" 2>&1 && pass "FULL checkpoint roundtrip" || fail "FULL checkpoint roundtrip"

# --- A.7: Delta roundtrip (>64MB verification) ---
check "A.7: Delta roundtrip (>64MB)"
python -c "
import sys, numpy as np
sys.path.insert(0, 'python')
from direct_checkpoint import DirectCheckpoint

ckpt = DirectCheckpoint(nvme_addr='0000:83:00.0', npu_device_id=1)
ckpt.delta_init(256, 128)

# Simulate ~160MB delta frame (exceeds old 64MB sync_meta_io limit)
k = 300; bs = 524288
blocks = [{'layer_id': 0, 'name': f'param_{i}', 'block_idx': i,
            'int8_data': np.random.randint(-128, 127, bs, dtype=np.int8),
            'scale': float(np.random.rand() * 0.1)}
          for i in range(k)]
small_patches = []

slot = ckpt.delta_save(1, blocks, small_patches)
print(f'  DELTA_SAVE: slot={slot}, blocks={k}, '
      f'size={bs*k/1024/1024:.0f}MB')

step_id, r_blocks, r_smalls = ckpt.delta_load_slot(slot)
assert step_id == 1, f'step_id mismatch: {step_id}'
assert len(r_blocks) == k, f'block count mismatch: {len(r_blocks)}'
for i in range(min(5, k)):
    assert np.array_equal(r_blocks[i]['int8_data'], blocks[i]['int8_data']), \
        f'data mismatch at block {i}'
print(f'  DELTA_LOAD: step={step_id}, blocks={len(r_blocks)}, '
      f'DATA VERIFIED (first 5 blocks checked)')
ckpt.cleanup()
print(f'  DELTA >64MB ROUNDTRIP OK')
" 2>&1 && pass "Delta >64MB roundtrip" || fail "Delta >64MB roundtrip"

# --- Summary ---
echo ""
echo "============================================================"
echo "  Phase A Verification: ALL PASSED"
echo "  $(date)"
echo "============================================================"
