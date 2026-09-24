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
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONUNBUFFERED=1  # live per-layer progress in oracle.log
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
    # Routing-swap experiments on the agentic prompt (logits only):
    #   swap:EPS        swap every 6th/7th near-tie with gap < EPS, at generated (assistant) positions only
    #   swap:EPS:cold   only when the 6th is outside the box-1 hot-set proxy and the 7th inside
    swap:*)
      IFS=: read -r _ eps mode <<< "$c"
      extra=(); tag="swap_${eps}"
      if [ "${mode:-}" = cold ]; then extra=(--swap-cold-only "$HOME/b1_hotset_proxy.json"); tag="${tag}_cold"; fi
      # generated positions only (assistant spans), as substitution would run in production
      run_case "$tag" --no-layer-dumps --swap-eps "$eps" "${extra[@]}" --swap-positions "$HERE/agentic_generated_positions.json" \
        --prompt-ids "$(tr -d '[] \n' < "$HERE/agentic_tokens.json")" ;;
    # Ungated substitution policy runs (generated positions only):
    #   policy:upper  6th -> 7th whenever the 6th is outside the box-1 hot-set proxy
    #   policy:rate   same, but only a seeded 9% of eligible (~1 swap/token, the production rate)
    #   policy:drop   as rate, but drop the 6th and renormalise over 5 (fallback when the 7th is not resident)
    policy:*)
      pol=${c#policy:}; extra=(--swap-eps inf --swap-sixth-cold "$HOME/b1_hotset_proxy.json")
      case $pol in
        upper) ;;
        rate)  extra+=(--swap-frac 0.09 --swap-seed 1) ;;
        drop)  extra+=(--swap-frac 0.09 --swap-seed 1 --swap-mode drop) ;;
        # null controls: (n1) a random 0.1% of ALL generated token-layers swapped 6th->7th;
        # (n2) policy_rate's selection pattern, but round the 6th expert's output to bf16
        null_rand) extra=(--swap-eps inf --swap-frac 0.001 --swap-seed 2) ;;
        null_bf16) extra+=(--swap-frac 0.09 --swap-seed 1 --swap-mode bf16) ;;
        # validity controls:
        #   zero   policy_rate's exact flags with eps=0 -> 0 swaps; logits must equal the baseline bit for bit
        #   rank1  positive control: same 9% selection, but replace the RANK-1 pick with the 7th
        #   rate2  policy_rate again + per-layer site checks; logits must equal the first policy_rate run
        zero)  extra=(--swap-eps 0 --swap-sixth-cold "$HOME/b1_hotset_proxy.json" --swap-frac 0.09 --swap-seed 1) ;;
        rank1) extra+=(--swap-frac 0.09 --swap-seed 1 --swap-rank 1 --swap-check-sites 3) ;;
        rate2) extra+=(--swap-frac 0.09 --swap-seed 1 --swap-check-sites 3) ;;
        # any-rank cold substitution (the zero-blocking-miss policy): a seeded fraction of the
        # cold picks at ANY rank -> best unchosen expert. 78 cold picks/generated token on the
        # agentic transcript, so 0.032 ~ 2.5 swaps/token (production box-2 miss rate), 0.064 ~ 5.
        anyrank25) extra=(--swap-eps inf --swap-anyrank-cold "$HOME/b1_hotset_proxy.json" --swap-frac 0.032 --swap-seed 3 --swap-check-sites 2) ;;
        anyrank50) extra=(--swap-eps inf --swap-anyrank-cold "$HOME/b1_hotset_proxy.json" --swap-frac 0.064 --swap-seed 3 --swap-check-sites 2) ;;
        # the shippable weight rule: the substitute inherits the replaced pick's weight, no renormalization
        anyrank25_inherit) extra=(--swap-eps inf --swap-anyrank-cold "$HOME/b1_hotset_proxy.json" --swap-frac 0.032 --swap-seed 3 --swap-check-sites 2 --swap-weights inherit) ;;
        rate_inherit) extra+=(--swap-frac 0.09 --swap-seed 1 --swap-check-sites 3 --swap-weights inherit) ;;
        # matched-count null for anyrank25: its exact site stream, no swap, bf16-round each marked pick
        anyrank25_bf16null) extra=(--swap-eps inf --swap-anyrank-cold "$HOME/b1_hotset_proxy.json" --swap-frac 0.032 --swap-seed 3 --swap-mode bf16) ;;
        # production-faithful (turn-local): production re-prefills every turn, so perturb ONLY inside
        # the longest assistant turn (inputs 467..682, 217 predictions); everything before it stays exact
        anyrank25_turn3) extra=(--swap-eps inf --swap-anyrank-cold "$HOME/b1_hotset_proxy.json" --swap-frac 0.032 --swap-seed 3 --swap-check-sites 2) ;;
        anyrank25_bf16null_turn3) extra=(--swap-eps inf --swap-anyrank-cold "$HOME/b1_hotset_proxy.json" --swap-frac 0.032 --swap-seed 3 --swap-mode bf16) ;;
        *) echo "unknown policy $pol"; exit 2 ;;
      esac
      posf="$HERE/agentic_generated_positions.json"
      case $pol in *_turn3) posf="$HERE/agentic_turn3_positions.json" ;; esac
      run_case "policy_$pol" --no-layer-dumps "${extra[@]}" --swap-positions "$posf" \
        --prompt-ids "$(tr -d '[] \n' < "$HERE/agentic_tokens.json")" ;;
    *) echo "unknown case $c"; exit 2 ;;
  esac || { echo "[$(date -u +%T)] stopping after $c failed"; exit 1; }
done
echo "[$(date -u +%T)] all cases done"
