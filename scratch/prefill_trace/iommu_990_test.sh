#!/usr/bin/env bash
# ROOT on box 2 (owner 10-11): does IOMMU translation cap the 990 Pro? Runtime, NON-persistent (a reboot restores
# DMA-FQ). Switches ONLY the 990 Pro's IOMMU group (21, alone: 0000:c5:00.0, /weights2) to passthrough ("identity");
# Thunderbolt/USB4 devices keep strict translation (kernel DMA protection is on). Coordinates with the box-1 watcher
# (scratch/prefill_trace/iommu_watch.sh, run as claude-code on box 1), which stops the hub + box 2's expertd before the
# switch and restarts both after, via flag files in /home/claude-code/iommu_test. Always remounts /weights2.
#   sudo bash iommu_990_test.sh            switch to identity + read tests (before / after)
#   sudo bash iommu_990_test.sh revert     switch back to DMA-FQ (also through the watcher)
set -u
MODE=${1:-test}
D=/home/claude-code/iommu_test
DEV=0000:c5:00.0
GRP=21
F2=/weights2/dsv4.1f/model-00048-of-00048.safetensors   # 990 Pro
F1=/weights/dsv4.1f/model-00048-of-00048.safetensors    # SN5000 (stays translated: the control)
say() { echo "$(date -u +%T) $*" | tee -a "$D/root.log"; }
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
[ "$(ls /sys/kernel/iommu_groups/$GRP/devices)" = "$DEV" ] || { echo "group $GRP is not exactly $DEV"; exit 1; }
blkdev() { ls -d /sys/bus/pci/devices/$DEV/nvme/nvme*/nvme*n1 2>/dev/null | head -1 | xargs -r basename; }
B=$(blkdev); SRC=$(findmnt -no SOURCE /weights2); OPTS=$(findmnt -no OPTIONS /weights2)
UUID=$(blkid -s UUID -o value "$SRC")
[ -n "$UUID" ] || { echo "no UUID for /weights2 ($SRC)"; exit 1; }
ddtest() {  # $1 file, $2 skip (16 MB blocks): 10 GB O_DIRECT sequential read
  dd if="$1" of=/dev/null bs=16M count=640 skip="$2" iflag=direct 2>&1 | tail -1
}
TARGET=identity; [ "$MODE" = revert ] && TARGET=DMA-FQ
say "start ($MODE): group $GRP type $(cat /sys/kernel/iommu_groups/$GRP/type) -> $TARGET; $B max_hw_sectors_kb $(cat /sys/block/$B/queue/max_hw_sectors_kb); /weights2 = $SRC ($OPTS) UUID $UUID"
rm -f "$D/go" "$D/done"
date -u +%FT%T > "$D/req"; chown claude-code "$D/req"
say "waiting for the box-1 watcher to stop the hub + expertd (up to 15 min)..."
for i in $(seq 1 900); do [ -f "$D/go" ] && break; sleep 1; done
[ -f "$D/go" ] || { say "no go from the watcher: nothing changed"; rm -f "$D/req"; exit 1; }
RESULT=FAIL
finish() {
  # Whatever happened: driver bound, /weights2 mounted, then tell the watcher.
  [ -n "$(blkdev)" ] || echo $DEV > /sys/bus/pci/drivers/nvme/bind 2>/dev/null
  for i in $(seq 1 30); do [ -n "$(blkdev)" ] && [ -e /dev/disk/by-uuid/$UUID ] && break; sleep 1; done
  mountpoint -q /weights2 || mount -o "$OPTS" UUID=$UUID /weights2
  mountpoint -q /weights2 && say "/weights2 mounted ($(findmnt -no SOURCE /weights2))" || say "!! /weights2 NOT mounted"
  echo "$RESULT" > "$D/done"; chown claude-code "$D/done"
  say "done: $RESULT; group $GRP type $(cat /sys/kernel/iommu_groups/$GRP/type)"
}
trap finish EXIT
say "BEFORE: 990 Pro $(ddtest $F2 100)"
say "BEFORE: 990 Pro $(ddtest $F2 1500)"
say "BEFORE: SN5000 $(ddtest $F1 2900)"
umount /weights2 || { say "umount failed (busy?): $(fuser -vm /weights2 2>&1 | tail -3)"; exit 1; }
echo $DEV > /sys/bus/pci/drivers/nvme/unbind || exit 1
echo $TARGET > /sys/kernel/iommu_groups/$GRP/type || { say "setting $TARGET failed"; exit 1; }
echo $DEV > /sys/bus/pci/drivers/nvme/bind || exit 1
for i in $(seq 1 30); do [ -n "$(blkdev)" ] && [ -e /dev/disk/by-uuid/$UUID ] && break; sleep 1; done
B=$(blkdev)
say "AFTER: group $GRP type $(cat /sys/kernel/iommu_groups/$GRP/type); $B max_hw_sectors_kb $(cat /sys/block/$B/queue/max_hw_sectors_kb) max_sectors_kb $(cat /sys/block/$B/queue/max_sectors_kb)"
mount -o "$OPTS" UUID=$UUID /weights2 || exit 1
say "AFTER: 990 Pro $(ddtest $F2 2300)"
say "AFTER: 990 Pro $(ddtest $F2 3700)"
say "AFTER: SN5000 $(ddtest $F1 4500)"
RESULT=OK
