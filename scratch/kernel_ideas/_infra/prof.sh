#!/usr/bin/env bash
# prof.sh — rocprofv3 with the recipes that work on this box. ALWAYS run it under gpu_run.sh:
#   gpu_run.sh --dev igpu --mb 512 --label fam/idea -- bash _infra/prof.sh att igpu <kernel-regex> OUTDIR -- ./harness args
#   gpu_run.sh --dev dgpu --mb 128 --label fam/idea -- bash _infra/prof.sh att dgpu <kernel-regex> OUTDIR -- ./harness args
#   gpu_run.sh --dev igpu --mb 512 --label fam/idea -- bash _infra/prof.sh pmc "SQ_WAVES SQ_INSTS_VALU SQ_INSTS_LDS GRBM_GUI_ACTIVE" OUTDIR -- ./harness args
#   gpu_run.sh --dev igpu --mb 512 --label fam/idea -- bash _infra/prof.sh trace OUTDIR -- ./harness args
# ATT traces only matching dispatches #3-#4 (override: ATT_ITERS='[10-12]'); ATT inflates timings ~2-20x.
# ATT output: OUTDIR/stats_ui_output_*.csv -> python3 ~/scripts/att_top.py <csv> --by stall --top 25
# PMC output: OUTDIR/run_counter_collection.csv (Counter_Name/Counter_Value per dispatch; device-global!)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mode=$1; shift
case "$mode" in
    att)
        dev=$1 re=$2 out=$3; shift 3; [ "$1" = "--" ] && shift
        idx=1; [ "$dev" = dgpu ] && idx=0   # PHYSICAL agent index, independent of the device mask
        rm -rf "$out"
        "$ROCPROFV3" --att --att-gpu-index $idx --att-library-path "$ATT_DECODER_DIR" \
            --att-target-cu 1 --kernel-include-regex "$re" --kernel-iteration-range "${ATT_ITERS:-[3-4]}" \
            -d "$out" -o run -- "$@"
        rc=$?
        ls "$out"/stats_ui_output_*.csv 2>/dev/null || echo "prof.sh: no ATT stats produced (regex '$re' matched nothing?)"
        exit $rc ;;
    pmc)
        ctrs=$1 out=$2; shift 2; [ "$1" = "--" ] && shift
        rm -rf "$out"
        "$ROCPROFV3" --pmc $ctrs -d "$out" -o run --output-format csv -- "$@"
        exit $? ;;
    trace)
        out=$1; shift; [ "$1" = "--" ] && shift
        rm -rf "$out"
        "$ROCPROFV3" --kernel-trace -d "$out" -o run --output-format csv -- "$@"
        exit $? ;;
    *) echo "prof.sh: mode must be att|pmc|trace" >&2; exit 64 ;;
esac
