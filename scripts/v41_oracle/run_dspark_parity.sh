#!/usr/bin/env bash
# DSpark drafter parity vs DeepSeek's reference drafter (plan DSPARK_ARENA_PLAN.md, M-A step A1).
#
# Needs the iGPU and ~8.7 GB of box-1 memory for the drafter, so the hub must be DOWN.
# Does not load the main model. Runs, back to back (RUNS overrides; "<ref>[:v4]"):
#   base      seeded, markov on, V4.1 ring quantizer (the reference's) -> target E 4.382 (bar ~4.2)
#   base:v4   same, legacy ring quantizer (V41_MTP_KV_QUANT=v4)       -> A/B for the quantizer fix
#   nomarkov  transformer-only drafts                                  -> reference 2.382 (markov head isolated)
# `noseed` is available but NOT like for like (see the test's doc).
#
# Inputs are built once from the oracle's gen2 dump:
#   python3 scripts/v41_oracle/parity_convert.py ~/.cache/deepstrix/v41/agentic/gen2
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-$HOME/logs/dspark_parity_$(date -u +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
DIR=${PARITY_DIR:-$HOME/.cache/deepstrix/v41/agentic/gen2/parity}
[ -f "$DIR/mh.bin" ] || python3 scripts/v41_oracle/parity_convert.py "$(dirname "$DIR")"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-target-v41}
for run in ${RUNS:-base base:v4 nomarkov}; do
  ref=${run%%:*}
  quant=v41
  [ "$run" != "$ref" ] && quant=${run#*:}
  tag="${ref}_${quant}"
  echo "=== PARITY_REF=$ref V41_MTP_KV_QUANT=$quant -> $OUT/$tag.log"
  HIP_VISIBLE_DEVICES=0,1 PARITY_REF=$ref V41_MTP_KV_QUANT=$quant PARITY_OUT="$OUT/ours_$tag.csv" \
    PARITY_SHOW=${PARITY_SHOW:-6} \
    nix develop -c cargo test -p v4flash-kernels --release --features v41 --test dspark_parity \
      -- --ignored --nocapture 2>&1 | tee "$OUT/$tag.log" \
      | grep -E '^parity:|^depth|^  d[1-5]|^E \(K=5\)|^draft agreement|^E\(ours\)|panicked|FAILED' || true
done
echo "logs in $OUT"
