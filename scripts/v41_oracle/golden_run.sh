#!/usr/bin/env bash
# Golden-corpus capture on box 2, beside the production expertd, without
# disturbing it. Runs the CPU oracle (`oracle.py --golden`) over each case in
# turn. Safety:
#   * oom_score_adj=1000: under memory pressure the kernel kills THIS, never expertd.
#   * nice 19, pinned to 12 threads (0-5,16-21) outside expertd's 8-15,24-31,
#     leaving 6,7,22,23 for the OS / network interrupts.
#   * a watchdog kills the oracle if MemAvailable drops below MEM_FLOOR_KB
#     (expertd pins ~116 GB; only ~12 GB is free to begin with).
# I/O: the oracle reads ~3-7 GB of experts per layer over several minutes
# (buffered, reclaimable page cache) -- a few MB/s against drives doing GB/s.
# ionice is pointless here: the NVMe queues use the `none` scheduler.
#
# Usage (on box 2):  nohup golden_run.sh <out_root> <case>... &
#   case = short | agentic
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT=${1:?out root}; shift
PY=${PY:-/nix/store/b5bpi6zfajzzrwwpgba2q6li3nnya4bs-python3-3.14.7/bin/python3}
export PYTHONPATH=$(cat "$HOME/pypath.txt")
export V41_MODEL=${V41_MODEL:-/weights2/dsv4.1f}
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
MEM_FLOOR_KB=${MEM_FLOOR_KB:-3000000}
CPUS=${CPUS:-0-5,16-21}
mkdir -p "$OUT_ROOT"

run_case() {
  local name=$1; shift
  local out="$OUT_ROOT/$name"
  if [ -f "$out/manifest.json" ]; then echo "[$(date -u +%T)] $name: already complete, skipping"; return 0; fi
  mkdir -p "$out"
  echo "[$(date -u +%T)] $name: start ($*)"
  (
    echo 1000 > /proc/self/oom_score_adj
    exec nice -n 19 taskset -c "$CPUS" "$PY" "$HERE/oracle.py" --golden --out "$out" "$@"
  ) > "$out/oracle.log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$avail" -lt "$MEM_FLOOR_KB" ]; then
      echo "[$(date -u +%T)] $name: MemAvailable ${avail} kB < floor; killing oracle to protect expertd"
      kill -TERM "$pid"; sleep 5; kill -KILL "$pid" 2>/dev/null
      return 1
    fi
    sleep 5
  done
  wait "$pid"; local rc=$?
  echo "[$(date -u +%T)] $name: exit $rc"
  return $rc
}

for c in "$@"; do
  case $c in
    short)   run_case short --prompt "The capital of France is" ;;
    agentic) run_case agentic --prompt-ids "$(tr -d '[] \n' < "$HERE/agentic_tokens.json")" ;;
    smoke)   run_case smoke --layers 2 --prompt "The capital of France is" ;;
    *) echo "unknown case $c"; exit 2 ;;
  esac || { echo "[$(date -u +%T)] stopping after $c failed"; exit 1; }
done
echo "[$(date -u +%T)] all cases done"
