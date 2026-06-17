#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# Wrapper for msprof profiling — uses --application syntax (CANN 8.0)
# Source: experiments/baselines/benchmark/step1_benchmark.py
# Usage: sudo bash prof_wrapper.sh [steps] [device_id]
# ════════════════════════════════════════════════════════════════════
STEPS="${1:-20}"
DEVICE_ID="${2:-1}"
REPO="/home/user7/npu-nvme"
PYTHON="/root/miniconda3/envs/ms_2.5/bin/python"
OUTDIR="/home/user7/npu-nvme/output/profiling_vec/step1"

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
mkdir -p "$OUTDIR"

# Clean stale output if any
find "$OUTDIR" -name "PROF_*" -mtime -1 -exec rm -rf {} + 2>/dev/null

echo "=== PROFILER START ==="
echo "STEPS=$STEPS DEVICE=$DEVICE_ID PID=$$ OUT=$OUTDIR"
echo "PYTHON=$PYTHON"
echo "Kernel: $(uname -r)"

# Use --application mode (CANN 8.0 RC3 traditional msprof)
"$PYTHON" "$REPO/experiments/baselines/benchmark/step1_benchmark.py" \
  --steps "$STEPS" --device-id "$DEVICE_ID" &
APP_PID=$!
echo "App PID: $APP_PID"

# Give the python process time to start
sleep 2

# Start profiling against the running PID
msprof --output="$OUTDIR" --application="$APP_PID" 2>&1 &
MSPROF_PID=$!
echo "msprof PID: $MSPROF_PID"

# Wait for the app to finish
wait $APP_PID 2>/dev/null
APP_RC=$?
echo "App exit: $APP_RC"

# Wait a bit for msprof to flush, then kill it if still running
sleep 10
kill $MSPROF_PID 2>/dev/null
wait $MSPROF_PID 2>/dev/null

echo "=== PROFILER DONE ==="

# Show what was produced
ls -la "$OUTDIR/" 2>/dev/null
if ls "$OUTDIR"/PROF_*/ 2>/dev/null; then
    for d in "$OUTDIR"/PROF_*/; do
        echo "Contents of $d:"
        find "$d" -type f | head -20
    done
fi

exit $APP_RC
