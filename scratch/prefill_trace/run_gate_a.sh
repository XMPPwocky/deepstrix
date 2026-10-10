#!/usr/bin/env bash
# Gate A: layer-outer bit-identity vs chunked prefill (docs/v41/P1_P2A_GPU_RUNBOOK.md sec 1).
# Uses the PRE-BUILT test binary in target-gate/ (cargo test --release --no-run), so the server-down window has no compile time.
#
#   run_gate_a.sh [CASES]          CASES default lazy,restored (~17-20 min); all four: fresh,multi,restored,lazy (~31 min, measured 10-01)
#
# REFUSES to run while the hub is up (it needs the dGPU VRAM and ~40 GB of pager RAM). Stop the hub first (hub launcher is a SUPERVISOR:
# kill the run_v41_server.sh bash first, see feedback_hub_launcher_is_supervised), keep box 2's expertd up (the test attaches it via
# V41_REMOTE_ADDR so paging is not box-1-disk-only: ~3.7 s/pass otherwise), then run this. Restart the hub afterwards.
set -u
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
CASES=${1:-lazy,restored}
if pgrep -f '^/home/claude-code/deepstrix/target-v41/release/deepstrix-server' > /dev/null; then
  echo "REFUSING: the production hub is running. Stop it first (and its supervisor), then re-run." >&2
  exit 3
fi
BIN=$(ls -t "$WT"/target-gate/release/deps/lm_prefill_bitexact-* 2>/dev/null | grep -v '\.d$' | head -1)
if [ -z "$BIN" ]; then echo "no pre-built test binary: cargo test --release -p v4flash-kernels --features v41 --test lm_prefill_bitexact --no-run (CARGO_TARGET_DIR=$WT/target-gate)" >&2; exit 4; fi
export KI_STATE=${KI_STATE:-/home/claude-code/.claude/jobs/b8eef48e/tmp/ki_state}
# shellcheck disable=SC1091
source "$WT/scratch/kernel_ideas/_infra/env.sh"
# the test's documented env (tests/lm_prefill_bitexact.rs header) + layer-outer ON
export HIP_VISIBLE_DEVICES=0,1 V41_PAGED_EXPERTS=1 V41_INDEX_K=1 V41_CANDIDATE_POOL=1
export V41_LM_PREFILL=1 V41_LM_LAYER_OUTER=1
export LM_CASES="$CASES"
# box 2 attached (else box-1-only paging ~3.7 s/pass) and a CONSTANT split so the chunked-vs-chunked null is reproducible
# (the 10-01 gate that passed ran exactly this way: reference_gpu_gate_windows + project_v41_layer_major_prefill)
export V41_REMOTE_ADDR=${V41_REMOTE_ADDR:-10.43.0.95:7431}
export V41_T2_CATCHALL=${V41_T2_CATCHALL:-2}
# keep V41_SUB unset for bit-exact gates, and the P1 split / reclaim OFF (invariant I1)
unset V41_SUB V41_PREFILL_SPLIT V41_PAGER_NO_DENSE
cd "$WT/crates/v4flash-kernels" || exit 5
echo "== Gate A: $BIN  cases=$CASES  $(date -u +%FT%TZ)"
"$BIN" --ignored --nocapture 2>&1 | tee "/home/claude-code/gate_a_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "== exit ${PIPESTATUS[0]}  $(date -u +%FT%TZ)"
