#!/usr/bin/env bash
# Integration check (2026-09-26, hub LIVE): the E_indexer in-tree kernels vs the kernels they
# replace, synthetic data only (< 90 MB device memory), defaults and every E knob = 0, plus the
# pre-existing synthetic indexer tests. ONE scheduler job per device:
#   bash ../_infra/gpu_run.sh --dev dgpu --mb 130 --label E_indexer/intree -- bash intree/run_tests.sh
#   bash ../_infra/gpu_run.sh --dev igpu --mb 300 --label E_indexer/intree_igpu -- bash intree/run_tests.sh
set -u
T=/home/claude-code/deepstrix/target-v41-ki/release/deps
bin() { ls -t "$T"/"$1"-* 2>/dev/null | grep -v '\.d$' | head -1; }
rc=0
filt='^test |test result|PASS|SKIP|panicked|FAIL|skipping|error|mismatch|identical|OK|max_abs'
run() {  # run <label> <env...> -- <test-binary-stem> <args...>
    local label=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local stem=$1; shift
    echo "### $label"
    local out; out=$(env "${envs[@]}" "$(bin "$stem")" "$@" --test-threads=1 --nocapture 2>&1); local r=$?
    echo "$out" | grep -E "$filt" | grep -v '^select class' | tail -40
    echo "### rc=$r"
    [ $r -eq 0 ] || rc=1
}
OFF="V41_IDX_SCORE_QREG=0 V41_IDX_TOPK_HYBRID=0 V41_CAND_THRESH_ILP=0 V41_IDX_GATHER_B128=0"
run "indexer_sweep_bitexact (defaults)" -- indexer_sweep_bitexact --ignored
run "indexer_sweep_bitexact (E knobs = 0)" $OFF -- indexer_sweep_bitexact --ignored
if [ "${1:-}" = igpu ]; then   # gfx1151: the non-WMMA kernels (the rest are gfx1201-only tests)
    run "candidate_blocks_oracle (defaults)" -- candidate_blocks_oracle --ignored
    echo "### overall rc=$rc"
    exit $rc
fi
# NOT in this list although synthetic: indexer_topk_select_oracle and indexer_score_gemm_oracle have
# B=512 x 49152 cases (~200 MB of device memory, over the 130 MB live-hub budget). Both passed once
# (defaults and E knobs = 0) on 2026-09-26 before that was noticed; do not rerun them beside a hub.
for t in candidate_blocks_oracle indexer_score_mw_oracle multistream_row_bases v41_indexer_selection_oracle; do
    run "$t (defaults)" -- $t --ignored
    run "$t (E knobs = 0)" $OFF -- $t --ignored
done
echo "### overall rc=$rc"
exit $rc
