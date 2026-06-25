#!/bin/bash
# ============================================================
# Delta Checkpoint Full Benchmark
# ============================================================
# Usage: bash bench_full.sh [DEVICE_ID] [--steps N]
# ============================================================
set -euo pipefail

DEVICE_ID="${1:-1}"
shift 1 2>/dev/null || true
EXTRA_ARGS=("$@")

REPO="/home/user7/npu-nvme"
PYTHON="/root/miniconda3/envs/ms_2.5/bin/python"
ASCEND_SETUP="/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"
PW_FILE="$REPO/.sudo_pw"
SUDO_PW="$(cat "$PW_FILE" | tr -d '\n')"
SCRIPT="$REPO/experiments/bench_full.py"

echo "============================================================"
echo "Delta Checkpoint Full Benchmark"
echo "  Device: $DEVICE_ID  |  Args: ${EXTRA_ARGS[*]:-default}"
echo "============================================================"

TMP_SCRIPT="$(mktemp /tmp/bench_full_XXXXXX.sh)"
cat > "$TMP_SCRIPT" << SCRIPT_EOF
#!/bin/bash
set -e; set +u
source "$ASCEND_SETUP"
set -u
export PYTHONPATH="$REPO/python:\$PYTHONPATH"
"$PYTHON" "$SCRIPT" --device-id "$DEVICE_ID" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
SCRIPT_EOF
chmod +x "$TMP_SCRIPT"

sudo -S "$TMP_SCRIPT" <<< "$SUDO_PW"
RC=$?
rm -f "$TMP_SCRIPT"
echo ""
echo "Done (exit=$RC)."
echo "Results: $REPO/experiments/output/bench_full.json"
exit $RC
