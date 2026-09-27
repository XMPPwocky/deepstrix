#!/usr/bin/env bash
# run_tests.sh "<label>|<env...>|<test-stem>" ... : run built v4flash-kernels integration tests
# (--ignored, one thread) from the sweep target dir; for a scheduler job, e.g.
#   bash ../_infra/gpu_run.sh --dev dgpu --mb 120 --label round2/a_intree -- bash round2/run_tests.sh "defaults||grid_pad_bitexact"
set -u
T=/home/claude-code/deepstrix/target-v41-ki/release/deps
bin() { ls -t "$T"/"$1"-* 2>/dev/null | grep -v '\.d$' | head -1; }
rc=0
filt='^test |test result|PASS|SKIP|panicked|FAIL|error|rel_rmse|overall|max_abs|bit_diff=[1-9]|diff=[1-9]|MISMATCH'
for spec in "$@"; do
    label=${spec%%|*}; rest=${spec#*|}; envs=${rest%%|*}; stem=${rest#*|}
    echo "### $label ($stem${envs:+, $envs})"
    # shellcheck disable=SC2086
    out=$(env $envs "$(bin "$stem")" --ignored --test-threads=1 --nocapture 2>&1); r=$?
    echo "$out" | grep -E "$filt" | tail -40
    echo "### rc=$r"
    [ $r -eq 0 ] || rc=1
done
echo "### overall rc=$rc"
exit $rc
