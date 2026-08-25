#!/usr/bin/env bash
# Controlled same-device experiment for 0000:83:00.0.
# This script is intentionally destructive only when --confirm-83-wipe is
# supplied.  0000:84:00.0 and /models are checked and never touched.
set -euo pipefail

PCI="0000:83:00.0"
MOUNTPOINT="/mnt/npu_nvme83_fs"
SYS="/sys/bus/pci/devices/${PCI}"

usage() {
    echo "usage: $0 fs|restore --confirm-83-wipe" >&2
    exit 2
}

[ "${1:-}" = fs ] || [ "${1:-}" = restore ] || usage
[ "${2:-}" = --confirm-83-wipe ] || usage

die() { echo "[same-device-83] ERROR: $*" >&2; exit 1; }

[ -e "$SYS" ] || die "PCI device $PCI is absent"
[ "$(findmnt -T /models -n -o SOURCE 2>/dev/null || true)" = /dev/nvme1n1 ] \
    || die "/models is not on the protected 84.0.0 device"

driver_name() {
    readlink -f "$SYS/driver" 2>/dev/null | awk -F/ '{print $NF}' || true
}

wait_for() {
    local path="$1" i
    for i in $(seq 1 50); do
        [ -e "$path" ] && return 0
        sleep 0.1
    done
    die "timeout waiting for $path"
}

find_namespace() {
    local nvme_dir
    nvme_dir=$(find "$SYS/nvme" -mindepth 1 -maxdepth 1 -type d \
        -regextype posix-extended -regex '.*/nvme[0-9]+' \
        -print -quit 2>/dev/null || true)
    [ -n "$nvme_dir" ] || die "no NVMe controller exposed for $PCI"
    basename "$nvme_dir"
}

bind_driver() {
    local wanted="$1" current
    current=$(driver_name)
    if [ "$current" != "$wanted" ]; then
        [ -z "$current" ] || echo "$PCI" > "/sys/bus/pci/drivers/$current/unbind"
        echo "$wanted" > "$SYS/driver_override"
        echo "$PCI" > /sys/bus/pci/drivers_probe
        echo > "$SYS/driver_override"
    fi
    [ "$(driver_name)" = "$wanted" ] || die "failed to bind $PCI to $wanted"
}

if [ "$1" = fs ]; then
    [ "$(findmnt -T "$MOUNTPOINT" -n -o TARGET 2>/dev/null || true)" != "$MOUNTPOINT" ] \
        || die "$MOUNTPOINT is already mounted"
    [ "$(driver_name)" = uio_pci_generic ] || die \
        "expected current SPDK/uio driver, got $(driver_name)"
    mkdir -p "$MOUNTPOINT"
    echo "[same-device-83] unbinding uio and binding kernel nvme"
    echo "$PCI" > /sys/bus/pci/drivers/uio_pci_generic/unbind
    echo nvme > "$SYS/driver_override"
    echo "$PCI" > /sys/bus/pci/drivers_probe
    echo > "$SYS/driver_override"
    [ "$(driver_name)" = nvme ] || die "kernel nvme bind failed"
    CTRL=$(find_namespace)
    DEV="/dev/${CTRL}n1"
    wait_for "$DEV"
    [ -z "$(findmnt -S "$DEV" -n -o TARGET 2>/dev/null || true)" ] \
        || die "$DEV is unexpectedly mounted"
    echo "[same-device-83] formatting $DEV as ext4"
    wipefs -a "$DEV"
    mkfs.ext4 -F -E lazy_itable_init=0,lazy_journal_init=0 "$DEV"
    mount -o noatime,nodiratime "$DEV" "$MOUNTPOINT"
    findmnt -T "$MOUNTPOINT"
    printf '%s\n' "pci=$PCI" "device=$DEV" "controller=$CTRL" \
        "filesystem=ext4" "mount=$MOUNTPOINT" > "$MOUNTPOINT/device_identity.txt"
    sync
    echo "[same-device-83] filesystem phase ready: $MOUNTPOINT"
else
    [ "$(driver_name)" = nvme ] || die \
        "expected kernel nvme driver for restore, got $(driver_name)"
    if [ "$(findmnt -T "$MOUNTPOINT" -n -o TARGET 2>/dev/null || true)" = "$MOUNTPOINT" ]; then
        DEV=$(findmnt -T "$MOUNTPOINT" -n -o SOURCE 2>/dev/null || true)
        [ -n "$DEV" ] || die "$MOUNTPOINT has no source device"
        [ "$DEV" != /dev/nvme1n1 ] || die "refusing to touch protected /models disk"
        umount "$MOUNTPOINT"
        wipefs -a "$DEV"
    else
        CTRL=$(find_namespace)
        DEV="/dev/${CTRL}n1"
        [ -z "$(findmnt -S "$DEV" -n -o TARGET 2>/dev/null || true)" ] \
            || die "$DEV is unexpectedly mounted"
    fi
    echo "$PCI" > /sys/bus/pci/drivers/nvme/unbind
    echo uio_pci_generic > "$SYS/driver_override"
    echo "$PCI" > /sys/bus/pci/drivers_probe
    echo > "$SYS/driver_override"
    [ "$(driver_name)" = uio_pci_generic ] || die "uio rebinding failed"
    echo "[same-device-83] restored $PCI to uio_pci_generic/SPDK"
fi
