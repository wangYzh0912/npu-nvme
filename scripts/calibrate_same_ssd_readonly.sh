#!/usr/bin/env bash
# Temporarily expose 83:00.0 through kernel NVMe for a read-only A/B baseline.
set -euo pipefail
PCI=0000:83:00.0
SYS=/sys/bus/pci/devices/$PCI
PYTHON=${1:-/home/user7/miniconda3/envs/ms_2.5/bin/python}
OUTPUT=${2:-/home/user7/npu-nvme/results/ppt-evidence-20260829/P1/calibration.json}

[ "$(id -u)" -eq 0 ] || { echo "root EUID required" >&2; exit 2; }
[ -e "$SYS" ] || { echo "$PCI absent" >&2; exit 2; }
[ "$(readlink -f "$SYS/driver" | xargs basename)" = uio_pci_generic ] || {
  echo "$PCI must start on uio_pci_generic" >&2; exit 2; }
[ "$(findmnt -T /models -no SOURCE)" = /dev/nvme1n1 ] || {
  echo "protected /models identity mismatch" >&2; exit 2; }

restore_uio() {
  if [ "$(readlink -f "$SYS/driver" 2>/dev/null | xargs -r basename)" = nvme ]; then
    echo "$PCI" > /sys/bus/pci/drivers/nvme/unbind
  fi
  echo uio_pci_generic > "$SYS/driver_override"
  echo "$PCI" > /sys/bus/pci/drivers_probe
  echo > "$SYS/driver_override"
}
trap restore_uio EXIT INT TERM

echo "$PCI" > /sys/bus/pci/drivers/uio_pci_generic/unbind
echo nvme > "$SYS/driver_override"
echo "$PCI" > /sys/bus/pci/drivers_probe
echo > "$SYS/driver_override"
for _ in $(seq 1 50); do
  DEV83=$(find "$SYS/nvme" -type d -name 'nvme*n1' -printf '/dev/%f\n' -quit 2>/dev/null || true)
  [ -b "${DEV83:-}" ] && break
  sleep 0.1
done
[ -b "${DEV83:-}" ] || { echo "83 namespace not exposed" >&2; exit 2; }
[ "$DEV83" != /dev/nvme1n1 ] || { echo "refusing ambiguous device identity" >&2; exit 2; }
[ -z "$(findmnt -S "$DEV83" -no TARGET 2>/dev/null || true)" ] || {
  echo "$DEV83 unexpectedly mounted" >&2; exit 2; }

"$PYTHON" /home/user7/npu-nvme/experiments/benchmarks/p1_odirect_calibration.py \
  --devices "$DEV83" /dev/nvme1n1 --output "$OUTPUT"
