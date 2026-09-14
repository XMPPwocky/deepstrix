#!/usr/bin/env bash
# Host-level tuning for the deepstrix two-box cluster.
# NON-PERSISTENT — every one of these resets on reboot. Re-run after any reboot.
# Run from box 1 (lumi-brain). Needs sudo here and on box 2.
#
# MEASURED 2026-09-14: together these took decode 4.07 -> 4.94 tok/s (+21%).
set -u
B2=${B2:-10.99.0.2}

apply_local() {
  echo "== $(hostname) =="
  # 1. Thunderbolt/USB4 runtime power management.
  #    Was `auto`. The link suspends between per-layer requests and the wake-up
  #    lands on the critical path 40x per token.
  #    MEASURED: at 500us idle, rtt 1255 -> 592 us (-53%), link 843 -> 200 (-76%).
  #    Small at the CURRENT 5ms duty cycle (-2%) but grows as per-layer work shrinks.
  for f in /sys/class/net/thunderbolt0/device/power/control; do
    [ -e "$f" ] && { echo on | sudo tee "$f" >/dev/null; echo "  tb power/control = $(cat $f)"; }
  done
  # 2. CPU C3 idle state.
  #    Exit latency 350 us (POLL/C1/C2/C3 = 0/1/18/350, governor `menu`). Paid on
  #    every link wake AND every host-side HIP stream sync -- sel_sync blocks the
  #    host once per layer, 22-23 ms/token, and a C3 wake taxes each one.
  #    MEASURED: rtt at 5ms idle 978 -> 643 us; raw ICMP idle 0.788 -> 0.203 ms.
  local n=0
  for c in /sys/devices/system/cpu/cpu*/cpuidle/state3/disable; do
    [ -e "$c" ] && { echo 1 | sudo tee "$c" >/dev/null; n=$((n+1)); }
  done
  echo "  C3 disabled on $n cpus"
}

apply_local
echo
echo "== $B2 (via ssh) =="
ssh -t "$B2" 'bash -s' <<'REMOTE'
set -u
for f in /sys/class/net/thunderbolt0/device/power/control; do
  [ -e "$f" ] && { echo on | sudo tee "$f" >/dev/null; echo "  tb power/control = $(cat $f)"; }
done
n=0
for c in /sys/devices/system/cpu/cpu*/cpuidle/state3/disable; do
  [ -e "$c" ] && { echo 1 | sudo tee "$c" >/dev/null; n=$((n+1)); }
done
echo "  C3 disabled on $n cpus"
REMOTE

echo
echo "verify:"
echo "  box1 tb=$(cat /sys/class/net/thunderbolt0/device/power/control 2>/dev/null) c3=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state3/disable 2>/dev/null)"
echo "  box2 tb=$(ssh $B2 'cat /sys/class/net/thunderbolt0/device/power/control' 2>/dev/null) c3=$(ssh $B2 'cat /sys/devices/system/cpu/cpu0/cpuidle/state3/disable' 2>/dev/null)"
