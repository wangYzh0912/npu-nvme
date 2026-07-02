#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTALL_ROOT=$(dirname "$SCRIPT_DIR")
ASCEND_ROOT=${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}

if [ -f "$ASCEND_ROOT/bin/setenv.bash" ]; then
    # shellcheck disable=SC1091
    source "$ASCEND_ROOT/bin/setenv.bash"
fi

export LD_LIBRARY_PATH="$INSTALL_ROOT/lib:$ASCEND_ROOT/lib64:${LD_LIBRARY_PATH:-}"

echo "[INFO] Executable: $INSTALL_ROOT/bin/npu_nvme_api_test"
echo "[INFO] LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

sudo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$INSTALL_ROOT/bin/npu_nvme_api_test" "$@"
