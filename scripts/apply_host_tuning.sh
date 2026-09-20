#!/usr/bin/env bash
# Host-level tuning for the deepstrix two-box cluster.
# PERSISTED in the NixOS flake since 2026-09-20 (modules/host-tuning.nix +
# modules/interconnect.nix, both hosts rebuilt and verified). This script is now
# the manual fallback / verifier only; it needs sudo here and on box 2.
#
# From your dev machine (keys live there; box 1 is the bastion):
#   ssh -A -t mimir@lumi-brain sudo --preserve-env=SSH_AUTH_SOCK /home/claude-code/deepstrix/scripts/apply_host_tuning.sh
# Or from box 1 as yourself: scripts/apply_host_tuning.sh   (sudo prompts on each box)
#
# MEASURED 2026-09-14: tb runtime PM + C3 off took decode 4.07 -> 4.94 tok/s (+21%).
# MEASURED 2026-09-19 (docs/v41/LINK_IDLE_LATENCY.md): governor performance, C2 off,
# POLL-only idle on the link CCX, expertd pinned to box 2's NHI CCX = ~-3 ms/token
# on a warm token; busy_read 5000 lets the server's per-phase SO_BUSY_POLL (3000 us
# in decode) be accepted -- without it the kernel refuses it and decode pays ~6
# ms/token of link again.
set -u
B2=${B2:-10.99.0.2}

apply_host() {                     # runs on EACH box; hostname decides the CCX
  local h; h=$(hostname)
  local S="sudo"; [ "$(id -u)" = 0 ] && S=""
  echo "== $h =="
  # 1. Thunderbolt/USB4 runtime power management: `auto` suspends the link
  #    between per-layer requests and the wake-up lands on the critical path.
  #    MEASURED: at 500us idle, rtt 1255 -> 592 us (-53%).
  for f in /sys/class/net/thunderbolt0/device/power/control; do
    [ -e "$f" ] && { echo on | $S tee "$f" >/dev/null; echo "  tb power/control = $(cat $f)"; }
  done
  # 2. CPU idle states. C3 (350 us exit) is paid on every link wake AND every
  #    host-side HIP stream sync (40x/token). C2 (18 us) still costs ~1 ms/token
  #    on the link; the link CCX runs POLL-only (no C1 either): a polling core is
  #    woken without an IPI, which is what the HIP-sync wake needs (-1.7 ms/token
  #    sel_sync measured with polling; C1-only kept none of it).
  local n=0
  for c in /sys/devices/system/cpu/cpu*/cpuidle/state3/disable /sys/devices/system/cpu/cpu*/cpuidle/state2/disable; do
    [ -e "$c" ] && { echo 1 | $S tee "$c" >/dev/null; n=$((n+1)); }
  done
  echo "  C2+C3 disabled ($n files)"
  #    Link CCX = the L3 that owns the NHI interrupts and the link threads:
  #    box 1 (lumi-brain): 0-7,16-23 (server engine/reader/writer + RX irq cpu 4)
  #    box 2 (lumi-brain2): 8-15,24-31 (NHI irqs on 27 and 31; expertd pinned here)
  local ccx; case "$h" in lumi-brain) ccx="0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23";; *) ccx="8 9 10 11 12 13 14 15 24 25 26 27 28 29 30 31";; esac
  for c in $ccx; do echo 1 | $S tee /sys/devices/system/cpu/cpu$c/cpuidle/state1/disable >/dev/null; done
  echo "  POLL-only idle on cpus: $(echo $ccx | tr ' ' ',')"
  # 3. cpufreq: idle cores dropped to ~1.8 GHz between calls and ran the wake
  #    path cold (~1 ms/token on the link).
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | $S tee "$g" >/dev/null; done
  echo "  governor = $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  # 4. Busy-poll cap: an unprivileged socket may not ask for SO_BUSY_POLL above
  #    net.core.busy_read. The server asks for 3000 us during decode
  #    (HetEngine::remote_set_phase_busy_poll); the daemon stays at 500.
  $S sysctl -q -w net.core.busy_read=5000 net.core.busy_poll=5000
  echo "  sysctl busy_read=$(sysctl -n net.core.busy_read) busy_poll=$(sysctl -n net.core.busy_poll)"
  # 2026-09-21: the SENDER's TSO/GSO defers the second segment of any link reply
  # over one 65,520-B MTU by ~1 ms on thunderbolt-net (docs/v41/LINK_IDLE_LATENCY.md);
  # off on both boxes (requests over one segment go hub -> box 2). ethtool is not in
  # the system profile; this store path was copied to box 2 with `nix copy`.
  ET=/nix/store/d8dzlcfskzpk8wxlypj8h1lj8mwisxdd-ethtool-7.1/bin/ethtool
  if [ -x "$ET" ]; then
    $S "$ET" -K thunderbolt0 tso off gso off 2>/dev/null || echo "  ethtool tso/gso off FAILED on $(hostname)"
    echo "  thunderbolt0 $("$ET" -k thunderbolt0 | grep -E '^tcp-segmentation-offload|^generic-segmentation-offload' | tr '\n' ' ')"
  else
    echo "  $ET missing on $(hostname): nix copy --to ssh://<box> $(dirname "$(dirname "$ET")")"
  fi
  # 5. box 2 only: keep the expert daemon on the NHI CCX (its reader<->compute
  #    handoff then never crosses an L3; ~-0.8 ms/token). The restart script
  #    launches it pinned; this re-pins a running one after a host-tuning re-run.
  local pid; pid=$(pgrep -o -f 'release/deepstrix-expertd .*--listen 0.0.0.0:7431' || true)
  if [ -n "$pid" ]; then $S taskset -apc 8-15,24-31 "$pid" >/dev/null; echo "  expertd pid $pid pinned to 8-15,24-31"; fi
}

apply_host
echo
echo "== $B2 (via ssh) =="
T=""; [ -t 0 ] && T="-t"
if [ "$(id -u)" = 0 ]; then
  # root on box 1 has no key for box 2: log in as the invoking user with the agent
  # forwarded from the dev machine (ssh -A + sudo --preserve-env=SSH_AUTH_SOCK).
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new); [ -n "${SUDO_USER:-}" ] && SSH_OPTS+=(-l "$SUDO_USER")
  [ -n "${SSH_AUTH_SOCK:-}" ] && SSH_OPTS+=(-o "IdentityAgent=$SSH_AUTH_SOCK")
  ssh $T "${SSH_OPTS[@]}" "$B2" "$(declare -f apply_host); apply_host"
else
  ssh $T "$B2" "$(declare -f apply_host); apply_host"
fi

echo
echo "verify:"
v() { echo "tb=$(cat /sys/class/net/thunderbolt0/device/power/control 2>/dev/null) c1[link ccx]=$(cat /sys/devices/system/cpu/cpu${1}/cpuidle/state1/disable) c2=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state2/disable) c3=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state3/disable) gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) busy_read=$(sysctl -n net.core.busy_read)"; }
echo "  box1: $(v 4)"
echo "  box2: $(ssh "$B2" "$(declare -f v); v 27" 2>/dev/null)"
