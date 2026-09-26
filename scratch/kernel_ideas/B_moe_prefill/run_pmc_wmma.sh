#!/usr/bin/env bash
# PMC counters for the WMMA candidates and the production kernels at B=1024 (96 experts).
# One rocprofv3 session at a time (sequential). Usage: bash run_pmc_wmma.sh [kernels...]
set -u
cd "$(dirname "$0")"
KS=${*:-gu down kwide kwide2}
for k in $KS; do
    bash ../_infra/prof.sh pmc "SQ_WAVES SQ_INSTS_VALU SQ_INSTS_LDS SQ_INSTS_SALU" prof_pmcA_$k -- ./harness prof_$k 1024 96 > results/pmcA_${k}_B1024.log 2>&1
    bash ../_infra/prof.sh pmc "SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_INSTS_VMEM MemUnitBusy" prof_pmcB_$k -- ./harness prof_$k 1024 96 > results/pmcB_${k}_B1024.log 2>&1
    echo "pmc $k done"
done
python3 pmc_summary.py prof_pmcA_* prof_pmcB_* 2>&1 | grep -A5 "mxfp4" > results/pmc_wmma_summary.txt
