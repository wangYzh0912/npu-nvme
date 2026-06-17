#!/bin/bash
# Phase 1b Driver — runs baseline + inject=50 for each model.
# Uses the same pattern as Phase 1a: run standalone first for graph compilation,
# then msprof for PMU collection (cache hit avoids TBE subprocess crash).
#
# Usage: sudo bash phase1b_driver.sh
# Logs:  experiments/output/phase1b_driver.log

set -euo pipefail
REPO="/home/user7/npu-nvme"
SCRIPT="${REPO}/experiments/baselines/phase1b_profile.py"
LOG="${REPO}/experiments/output/phase1b_driver.log"
PASSWD="CGCL_2025_#\$"

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export GLOG_v=0
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# ── Helpers ──
run_standalone() {
    local label="$1" preset="$2" inject="$3"
    echo "[$(date +%T)] Standalone: ${label}" | tee -a "$LOG"
    # Use expect or pipe password
    echo "$PASSWD" | sudo -S bash -c "
        source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
        export GLOG_v=0 ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=0
        /root/miniconda3/envs/ms_2.5/bin/python '$SCRIPT' \
            --label '$label' --preset '$preset' --inject '$inject' \
            --steps 16 --sink 4 --epochs 2
    " 2>&1 | tail -15 | tee -a "$LOG"
    echo "[$(date +%T)] DONE: ${label}" | tee -a "$LOG"
}

run_msprof() {
    local label="$1" preset="$2" inject="$3"
    local prof_dir="${REPO}/output/profiling_vec/${label}"

    echo "[$(date +%T)] msprof: ${label}" | tee -a "$LOG"
    rm -rf "$prof_dir" 2>/dev/null
    echo "$PASSWD" | sudo -S bash -c "
        source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
        export GLOG_v=0 ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=0
        /usr/local/Ascend/ascend-toolkit/latest/bin/msprof \
            --output='$prof_dir' \
            -- /root/miniconda3/envs/ms_2.5/bin/python '$SCRIPT' \
                --label '$label' --preset '$preset' --inject '$inject' \
                --steps 16 --sink 4 --epochs 2
    " 2>&1 | tail -20 | tee -a "$LOG"
    echo "[$(date +%T)] msprof DONE: ${label}" | tee -a "$LOG"
}

# ── Experiment Matrix ──
# Model presets sorted by size
PRESETS=(
    "gpt2_12L"       # ~124M params, ~250MB FP16
    "gpt2_large"     # 36L d=1280, ~0.85B params
    "gpt2_1_2b"      # 24L d=1536, ~1.2B
    "gpt2_xl"        # 48L d=1600, ~1.56B
    "gpt2_2_5b"      # 48L d=2048, ~2.5B
    "gpt2_3_3b"      # 64L d=2048, ~3.3B
)

# gpt2_12L is already covered by V2 (81M). Let's use different sizes.
# Reordered: start from medium-large
PRESETS=(
    "gpt2_12L"       # ~124M params — V1b_12L
    "gpt2_large"     # 36L d=1280, ~0.85B — V5
    "gpt2_1_2b"      # 24L d=1536, ~1.2B — V5b
    "gpt2_2_5b"      # 48L d=2048, ~2.5B — V6
    "gpt2_3_3b"      # 64L d=2048, ~3.3B — V7
)

LABELS=("V1b_12L" "V5" "V5b" "V6" "V7")

echo "============================================" | tee "$LOG"
echo "Phase 1b Driver — $(date)" | tee -a "$LOG"
echo "Models: ${PRESETS[*]}" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

for i in "${!PRESETS[@]}"; do
    preset="${PRESETS[$i]}"
    label="${LABELS[$i]}"
    bl_label="${label}_baseline"
    ij_label="${label}_inject"

    echo "" | tee -a "$LOG"
    echo "===== ${preset} -> ${label} =====" | tee -a "$LOG"

    # 1) Standalone baseline (compile model, get step time)
    run_standalone "$bl_label" "$preset" 0

    # 2) msprof baseline (PMU data for baseline)
    run_msprof "$bl_label" "$preset" 0

    # 3) Standalone inject-50 (compile model with Vector ops)
    run_standalone "${ij_label}_50" "$preset" 50

    # 4) msprof inject-50 (PMU data for inject)
    run_msprof "${ij_label}_50" "$preset" 50

    echo "===== ${preset} COMPLETE =====" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "Phase 1b Driver COMPLETE — $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
