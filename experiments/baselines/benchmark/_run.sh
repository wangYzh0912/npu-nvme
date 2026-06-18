#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Step 1 Benchmark: GPT-2 XL Pure Training Baseline
# ═══════════════════════════════════════════════════════════════════════
# Orchestrates: msprof wrapper → training → CSV parse → merged results
#
# Usage:
#   bash _run.sh [MODE] [STEPS]
#
# Modes:
#   full      — msprof + training + parse  (default)
#   quick     — training only, no msprof    (for validation)
#   parse     — parse existing msprof output (requires PROF_DIR)
#
# Examples:
#   bash _run.sh full 20     # Full benchmark, 20 steps
#   bash _run.sh quick 5     # Quick test, 5 steps, no profiler
#   PROF_DIR=/path/to/PROF_* bash _run.sh parse
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

MODE="${1:-full}"
STEPS="${2:-20}"
DEVICE_ID="${DEVICE_ID:-1}"

REPO="/home/user7/npu-nvme"
PYTHON="/root/miniconda3/envs/ms_2.5/bin/python"
SUDO_PW="CGCL_2025_#$"
ASCEND_SETUP="/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"

BENCHMARK_DIR="$REPO/experiments/baselines/benchmark"
OUTPUT_DIR="$REPO/experiments/output/benchmark"
PROFILING_BASE="$REPO/output/profiling_vec/step1"
SCRIPT="$BENCHMARK_DIR/step1_benchmark.py"
WRAPPER="$BENCHMARK_DIR/pmu_wrapper.sh"

echo "═══════════════════════════════════════════════════════════════"
echo "Step 1 Benchmark: GPT-2 XL Pure Training Baseline"
echo "  Mode: $MODE  |  Steps: $STEPS  |  Device: $DEVICE_ID"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROFILING_BASE"

case "$MODE" in

  full)
    echo ""
    echo "[Phase 1/2] Running training under msprof..."
    echo "  Profiler output → $PROFILING_BASE"
    echo ""

    # Clean stale output
    find "$PROFILING_BASE" -name "PROF_*" -mmin -120 -exec rm -rf {} + 2>/dev/null || true

    echo "$SUDO_PW" | sudo -S bash -c "
      source $ASCEND_SETUP && \
      msprof --output=$PROFILING_BASE --application=$WRAPPER $STEPS $DEVICE_ID
    "

    # Copy output to project dir (msprof writes with root ownership)
    echo "$SUDO_PW" | sudo -S chown -R user7:user7 "$PROFILING_BASE" 2>/dev/null || true

    # Find latest PROF directory
    PROF_DIR=$(ls -dt "$PROFILING_BASE"/PROF_* 2>/dev/null | head -1)
    if [ -z "$PROF_DIR" ]; then
      echo "ERROR: No PROF_* directory found under $PROFILING_BASE"
      ls -la "$PROFILING_BASE" || true
      exit 1
    fi

    echo ""
    echo "[Phase 2/2] Parsing msprof output..."
    echo "  PROF directory: $PROF_DIR"
    echo ""

    echo "$SUDO_PW" | sudo -S bash -c "
      source $ASCEND_SETUP && \
      $PYTHON $SCRIPT --parse-only --profiler-dir $PROF_DIR --output $OUTPUT_DIR/step1_benchmark.json
    "

    echo ""
    echo "Done! Results:"
    echo "  $OUTPUT_DIR/step1_benchmark.json"
    echo "  $OUTPUT_DIR/step1_benchmark_partial.json"
    echo "  $OUTPUT_DIR/step1_hbm_usage.log"
    echo "  $OUTPUT_DIR/step1_hbm_usage.json"
    echo "  $PROF_DIR/mindstudio_profiler_output/op_summary_*.csv"
    ;;

  quick)
    echo ""
    echo "[Quick] Training only, no msprof..."
    echo ""

    echo "$SUDO_PW" | sudo -S bash -c "
      source $ASCEND_SETUP && \
      $PYTHON $SCRIPT --steps $STEPS --device-id $DEVICE_ID
    "

    echo ""
    echo "Done! Results:"
    echo "  $OUTPUT_DIR/step1_benchmark_partial.json"
    echo "  $OUTPUT_DIR/step1_hbm_usage.log"
    echo "  $OUTPUT_DIR/step1_hbm_usage.json"
    ;;

  parse)
    PROF_DIR="${PROF_DIR:-$(ls -dt "$PROFILING_BASE"/PROF_* 2>/dev/null | head -1)}"
    if [ -z "$PROF_DIR" ]; then
      echo "ERROR: Set PROF_DIR=/path/to/PROF_* or run 'full' first"
      exit 1
    fi
    echo ""
    echo "[Parse] Parsing msprof output: $PROF_DIR"
    echo ""

    echo "$SUDO_PW" | sudo -S bash -c "
      source $ASCEND_SETUP && \
      $PYTHON $SCRIPT --parse-only --profiler-dir $PROF_DIR --output $OUTPUT_DIR/step1_benchmark.json
    "

    echo ""
    echo "Done! Merged results → $OUTPUT_DIR/step1_benchmark.json"
    ;;

  *)
    echo "ERROR: Unknown mode '$MODE'. Use: full | quick | parse"
    exit 1
    ;;
esac
