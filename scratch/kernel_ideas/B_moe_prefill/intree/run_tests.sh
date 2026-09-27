#!/usr/bin/env bash
# Integration check (2026-09-26, hub LIVE) for families A (dn2 small-b down) + B (int8 WMMA arm):
# synthetic MXFP4 experts on the iGPU (< 500 MB), defaults and every A/B knob = 0, plus the
# pre-existing small MXFP4 tests, plus the B harness `cmp` mode against a code object built from
# the IN-TREE kernels/mxfp4_moe_wmma.hip (intree/build_intree.sh) at 8 physical experts. ONE job:
#   bash ../_infra/gpu_run.sh --dev igpu --mb 600 --label B_moe_prefill/intree -- bash intree/run_tests.sh
# Not run: mxfp4_iq2s_oracle (materialises all 384 experts = 2.4 GB) and remote_experts_loopback
# (loads a checkpoint shard).
set -u
cd "$(dirname "$0")/.."
T=/home/claude-code/deepstrix/target-v41-ki/release/deps
bin() { ls -t "$T"/"$1"-* 2>/dev/null | grep -v '\.d$' | head -1; }
rc=0
filt='^test |test result|PASS|SKIP|panicked|FAIL|error|rel_rmse|bit-exact|smallb b=|max_abs'
run() {  # run <label> <env...> -- <test-binary-stem> <args...>
    local label=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local stem=$1; shift
    echo "### $label"
    local out; out=$(env "${envs[@]}" "$(bin "$stem")" "$@" --test-threads=1 --nocapture 2>&1); local r=$?
    echo "$out" | grep -E "$filt" | tail -40
    echo "### rc=$r"
    [ $r -eq 0 ] || rc=1
}
OFF="V41_MOE_DOWN_DN2=0 V41_MOE_WMMA_GATEUP=0 V41_MOE_WMMA_DOWN=0"
run "mxfp4_moe_sweep (defaults)" -- mxfp4_moe_sweep --ignored
run "mxfp4_moe_sweep (A/B knobs = 0)" $OFF -- mxfp4_moe_sweep --ignored
run "mxfp4_wi_devcount" -- mxfp4_wi_devcount --ignored
run "mxfp4_pair_oracle" -- mxfp4_pair_oracle --ignored
if [ -f intree/intree_wmma_gfx1151.hsaco ]; then
    for B in 512 128; do
        echo "### B harness cmp B=$B n_exp=8 (candidate = in-tree mxfp4_moe_wmma.hip)"
        env KB_CAND=intree/intree_wmma ./harness cmp $B 8 7 3 2>&1 | grep -E 'CMP|cmp|rel|bit|cand' | head -20
    done
fi
echo "### overall rc=$rc"
exit $rc
