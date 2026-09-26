#!/usr/bin/env bash
# Integration check (2026-09-26, hub LIVE): the F_mhc_glue in-tree kernels vs the kernels they
# replace, synthetic data only (< 60 MB device memory), defaults and every F knob = 0, plus the
# pre-existing synthetic tests that touch the same wrappers. Run as ONE scheduler job per device:
#   bash ../_infra/gpu_run.sh --dev dgpu --mb 120 --label F_mhc_glue/intree -- bash intree/run_tests.sh
#   bash ../_infra/gpu_run.sh --dev igpu --mb 200 --label F_mhc_glue/intree_igpu -- bash intree/run_tests.sh
set -u
T=/home/claude-code/deepstrix/target-v41-ki/release/deps
bin() { ls -t "$T"/"$1"-* 2>/dev/null | grep -v '\.d$' | head -1; }
rc=0
filt='^test |test result|PASS|SKIP|panicked|FAIL|bit-identical|skipping|error'
run() {  # run <label> <env...> -- <test-binary-stem> <args...>
    local label=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local stem=$1; shift
    echo "### $label"
    local out; out=$(env "${envs[@]}" "$(bin "$stem")" "$@" --test-threads=1 --nocapture 2>&1); local r=$?
    echo "$out" | grep -E "$filt"
    echo "### rc=$r"
    [ $r -eq 0 ] || rc=1
}
OFF="V41_RMS_FAST=0 V41_ROUTER_MV_H20=0 V41_TOPK_WFRED=0 V41_MHC_GEMM_NARROW=0"
run "mhc_glue_bitexact (defaults)" -- mhc_glue_bitexact --ignored
run "mhc_glue_bitexact (F knobs = 0)" $OFF -- mhc_glue_bitexact --ignored
run "mhc_arena_bitexact (defaults)" -- mhc_arena_bitexact --ignored
run "mhc_arena_bitexact (V41_RMS_FAST=0)" V41_RMS_FAST=0 -- mhc_arena_bitexact --ignored
run "router_topk_alts (defaults)" -- router_topk_alts
run "router_topk_alts (V41_TOPK_WFRED=0)" V41_TOPK_WFRED=0 -- router_topk_alts
echo "### overall rc=$rc"
exit $rc
