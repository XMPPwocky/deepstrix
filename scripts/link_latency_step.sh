#!/usr/bin/env bash
# One-command link-latency experiments on BOTH boxes (box 1 locally, box 2 via ssh).
# See docs/v41/LINK_IDLE_LATENCY.md ("2026-09-19") for why each step exists.
#
#   scripts/link_latency_step.sh <step> <on|off|status>
#   scripts/link_latency_step.sh status            # everything, both boxes
#   scripts/link_latency_step.sh measure [N]        # scoreboard: last N tokens of the live log
#
#   step 1   cpufreq governor performance (idle cores stop dropping to 1.8 GHz)
#   step 2   hold PM QoS /dev/cpu_dma_latency=0 (TEST ONLY: every core polls instead of idling)
#   step 2p  disable C2 (the permanent form of 2 if 2 helps; leaves C1 at 1 us)
#   step 2c  POLL-only idle on ONE CCX per box (the one with the link threads/IRQs): step 2's
#            no-IPI wakeups where they matter, at half the power. Others keep C1 (2p).
#   step 3   box 2 only: pin every expertd thread to the CCX that owns the NHI IRQs
#   step 4s  sysctl net.core.busy_read/busy_poll=5000 on both: lets an UNPRIVILEGED socket ask for
#            SO_BUSY_POLL above 500 us (the bench / daemon --busy-poll N). Harmless until a process uses it.
#
# From your dev machine (key lives there; box 1 is the bastion, agent forwarded):
#   ssh -A -t mimir@lumi-brain sudo --preserve-env=SSH_AUTH_SOCK /home/claude-code/deepstrix/scripts/link_latency_step.sh 1 on
# Prompts for your sudo password on box 1, then on box 2.
# All NON-PERSISTENT and reversible with `off`. Nothing here touches the server.
set -u
B2=${B2:-10.99.0.2}
CCX=${CCX:-8-15,24-31}            # box 2: L3 shared by CPUs 8-15,24-31; NHI irqs land on 27 and 31
CCX_B1=${CCX_B1:-0-7,16-23}       # box 1: the server's engine/reader/writer threads and the NHI RX irq (cpu 4) live here
QOS_PID=/run/cpu_dma_latency_hold.pid

apply_step() {                    # runs on EACH box; $1=step $2=on|off|status
  local step=$1 mode=$2 h; h=$(hostname)
  local S="sudo"; [ "$(id -u)" = 0 ] && S=""
  case "$step" in
    1)
      case $mode in
        on)  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | $S tee "$g" >/dev/null; done ;;
        off) for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo powersave   | $S tee "$g" >/dev/null; done ;;
      esac
      echo "  [$h] governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null) cpu27_khz=$(cat /sys/devices/system/cpu/cpu27/cpufreq/scaling_cur_freq)" ;;
    2)
      case $mode in
        on)  if [ -f $QOS_PID ] && [ -d /proc/"$(cat $QOS_PID)" ]; then echo "  [$h] qos holder already running"; else
               # bash-only holder (box 2 has no system python): keep fd 3 open on the device with 0 written.
               $S setsid bash -c 'exec 3<>/dev/cpu_dma_latency; printf "\x00\x00\x00\x00" >&3; echo $$ > '"$QOS_PID"'; exec sleep infinity' </dev/null >/dev/null 2>&1 &
               sleep 0.5; fi ;;
        off) # kill EVERY holder (fd 3 open on the device), not just the one in the pidfile
             $S bash -c 'for d in /proc/[0-9]*; do [ "$(readlink $d/fd/3 2>/dev/null)" = /dev/cpu_dma_latency ] && kill ${d#/proc/}; done; rm -f '"$QOS_PID" ;;
      esac
      if [ -f $QOS_PID ] && [ -d /proc/"$(cat $QOS_PID)" ]; then echo "  [$h] qos=HELD at 0 (pid $(cat $QOS_PID)) -> all cores poll"; else echo "  [$h] qos=not held"; fi
      # ground truth regardless of pidfile: a polling core never enters C1/C2
      local u0 u1; u0=$(cat /sys/devices/system/cpu/cpu5/cpuidle/state1/usage); sleep 1; u1=$(cat /sys/devices/system/cpu/cpu5/cpuidle/state1/usage)
      echo "  [$h] cpu5 C1 entries in the last second: $((u1-u0))  (0 = QoS in effect)" ;;
    2p)
      case $mode in
        on)  for c in /sys/devices/system/cpu/cpu*/cpuidle/state2/disable; do echo 1 | $S tee "$c" >/dev/null; done ;;
        off) for c in /sys/devices/system/cpu/cpu*/cpuidle/state2/disable; do echo 0 | $S tee "$c" >/dev/null; done ;;
      esac
      echo "  [$h] idle: C1 disable=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state1/disable) C2 disable=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state2/disable) C3 disable=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state3/disable)" ;;
    2c)
      local ccx=$CCX; [ "$h" = lumi-brain ] && ccx=$CCX_B1
      local cpus; cpus=$(python3 -c "
r='$ccx'.split(',');print(' '.join(str(i) for a in r for i in range(int(a.split('-')[0]),int(a.split('-')[-1])+1)))" 2>/dev/null || \
        for a in ${ccx//,/ }; do seq -s ' ' ${a%-*} ${a#*-}; done)
      case $mode in
        on)  for c in $cpus; do echo 1 | $S tee /sys/devices/system/cpu/cpu$c/cpuidle/state1/disable /sys/devices/system/cpu/cpu$c/cpuidle/state2/disable >/dev/null; done ;;
        off) for c in $cpus; do echo 0 | $S tee /sys/devices/system/cpu/cpu$c/cpuidle/state1/disable >/dev/null; done ;;
      esac
      local first=${cpus%% *} other=$([ "$h" = lumi-brain ] && echo 8 || echo 0)
      echo "  [$h] ccx $ccx: cpu$first C1 disable=$(cat /sys/devices/system/cpu/cpu$first/cpuidle/state1/disable) C2 disable=$(cat /sys/devices/system/cpu/cpu$first/cpuidle/state2/disable) | elsewhere cpu$other C1 disable=$(cat /sys/devices/system/cpu/cpu$other/cpuidle/state1/disable)" ;;
    4s)
      case $mode in
        on)  $S sysctl -q -w net.core.busy_read=5000 net.core.busy_poll=5000 ;;
        off) $S sysctl -q -w net.core.busy_read=500 net.core.busy_poll=500 ;;
      esac
      echo "  [$h] sysctl busy_read=$(sysctl -n net.core.busy_read) busy_poll=$(sysctl -n net.core.busy_poll)" ;;
    3)
      local pid; pid=$(pgrep -o -f '^(\./)?[^ ]*target-b2/release/deepstrix-expertd ' || true)   # the daemon's own argv[0], not a shell mentioning it
      if [ -z "$pid" ]; then echo "  [$h] no deepstrix-expertd here (box 2 only step)"; return 0; fi
      case $mode in
        on)  $S taskset -apc "$CCX" "$pid" >/dev/null ;;
        off) $S taskset -apc 0-31 "$pid" >/dev/null ;;
      esac
      echo "  [$h] expertd pid $pid affinity=$(taskset -pc "$pid" | sed 's/.*: //') threads on cpus: $(ps -L -o psr= -p "$pid" | sort -n | uniq | tr '\n' ' ')"
      echo "  [$h] NHI irqs on cpus: $(for i in $(grep -E 'thunderbolt' /proc/interrupts | awk -F: '{print $1}'); do cat /proc/irq/$i/effective_affinity_list; done | tr '\n' ' ')" ;;
    status)
      apply_step 1 status; apply_step 2 status; apply_step 2p status; apply_step 2c status; apply_step 3 status; apply_step 4s status
      echo "  [$h] tb power/control=$(cat /sys/class/net/thunderbolt0/device/power/control 2>/dev/null)" ;;
    *) echo "unknown step $step" >&2; return 2 ;;
  esac
}

remote() {                        # run apply_step on box 2 with the same definition
  local T=""; [ -t 0 ] && T="-t"
  local -a o=(-o StrictHostKeyChecking=accept-new)
  if [ "$(id -u)" = 0 ]; then
    # root on box 1 has no key for box 2. Log in there as the invoking user and use the
    # agent forwarded from the dev machine:  ssh -A -t you@lumi-brain sudo --preserve-env=SSH_AUTH_SOCK $0 ...
    [ -n "${SUDO_USER:-}" ] && o+=(-l "$SUDO_USER")
    if [ -n "${SSH_AUTH_SOCK:-}" ]; then o+=(-o "IdentityAgent=$SSH_AUTH_SOCK")
    else echo "  no forwarded agent (SSH_AUTH_SOCK empty): connect with  ssh -A  and run  sudo --preserve-env=SSH_AUTH_SOCK $0 $1 $2" >&2; return 1; fi
  fi
  ssh $T "${o[@]}" "$B2" "$(declare -f apply_step); QOS_PID=$QOS_PID; CCX=$CCX; CCX_B1=$CCX_B1; apply_step $1 $2"
}

case "${1:-}" in
  measure)
    N=${2:-600}
    if [ -r /home/claude-code/logs/v41-server.log ]; then python3 "$(dirname "$0")/token_summary.py" "$N"
    else sudo python3 "$(dirname "$0")/token_summary.py" "$N"; fi
    exit ;;
  status) STEP=status; MODE=status ;;
  1|2|2p|2c|3|4s) STEP=$1; MODE=${2:-status}; case $MODE in on|off|status) ;; *) echo "mode must be on|off|status" >&2; exit 2;; esac ;;
  *) sed -n 2,20p "$0"; exit 2 ;;
esac

echo "== box 1 =="
if [ "$STEP" = 3 ]; then echo "  (box 2 only)"; else apply_step "$STEP" "$MODE"; fi
echo "== box 2 ($B2) =="
remote "$STEP" "$MODE"
echo
echo "now wait ~2-3 min of agent traffic, then: $0 measure   (link/srv/sel_sync/total p50 over the last 600 tokens)"
