#!/usr/bin/env bash
# Integration check (2026-09-26, hub LIVE) for C2_dense_prefill: synthetic data on the dGPU
# (< 110 MB), defaults and every C2 knob = 0, plus the pre-existing synthetic Q8_0 tests that go
# through the touched wrappers (matvec_batched's new z16 arm at b = 64, gemm_f16x). ONE job:
#   bash ../_infra/gpu_run.sh --dev dgpu --mb 120 --label C2_dense_prefill/intree -- bash intree/run_tests.sh
# Not run: q8_0_grouped_matvec (needs the GGUF).
set -u
T=/home/claude-code/deepstrix/target-v41-ki/release/deps
bin() { ls -t "$T"/"$1"-* 2>/dev/null | grep -v '\.d$' | head -1; }
rc=0
filt='^test |test result|PASS|SKIP|panicked|FAIL|error|rel_rmse|pass rule|overall|max_abs'
run() {  # run <label> <env...> -- <test-binary-stem> <args...>
    local label=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local stem=$1; shift
    echo "### $label"
    local out; out=$(env "${envs[@]}" "$(bin "$stem")" "$@" --test-threads=1 --nocapture 2>&1); local r=$?
    echo "$out" | grep -E "$filt" | tail -30
    echo "### rc=$r"
    [ $r -eq 0 ] || rc=1
}
OFF="V41_ENGRAM_I8X=0 V41_ENGRAM_CHUNK128=0 V41_GEMV_BPACK_Z16=0 V41_F16X_DB_BN64=0 V41_F16X_256=0 V41_REPLAY_F16X=0"
run "q8_0_sweep_c2_bitexact (defaults)" -- q8_0_sweep_c2_bitexact --ignored
run "q8_0_sweep_c2_bitexact (C2 knobs = 0)" $OFF -- q8_0_sweep_c2_bitexact --ignored
run "q8_0_tb_bitexact (defaults: C1 tB + C2 z16 share matvec_batched)" -- q8_0_tb_bitexact --ignored
run "q8_0_matvec_batched (defaults)" -- q8_0_matvec_batched --ignored
run "q8_0_matvec_wmma (matvec_batched at b=64 -> z16)" -- q8_0_matvec_wmma --ignored
run "q8_0_matvec_wmma (V41_GEMV_BPACK_Z16=0)" V41_GEMV_BPACK_Z16=0 -- q8_0_matvec_wmma --ignored
run "q8_0_gemm_f16x (defaults)" -- q8_0_gemm_f16x --ignored
echo "### overall rc=$rc"
exit $rc
