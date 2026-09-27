#!/usr/bin/env bash
# DSpark drafter parity vs DeepSeek's reference drafter (plan DSPARK_ARENA_PLAN.md, M-A step A1).
#
# Needs the iGPU and ~8.7 GB of box-1 memory for the drafter, so the hub must be DOWN.
# Does not load the main model. Runs three references back to back:
#   base     seeded, markov on       -> the target, E 4.382 (bar: >= ~4.2)
#   noseed   window not seeded       -> reference 3.281 (our window-validity rule differs, see test doc)
#   nomarkov transformer-only drafts -> reference 2.382 (isolates the markov head)
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
for ref in ${REFS:-base nomarkov noseed}; do
  echo "=== PARITY_REF=$ref -> $OUT/$ref.log"
  HIP_VISIBLE_DEVICES=0,1 PARITY_REF=$ref PARITY_OUT="$OUT/ours_$ref.csv" PARITY_SHOW=${PARITY_SHOW:-6} \
    nix develop -c cargo test -p v4flash-kernels --release --features v41 --test dspark_parity \
      -- --ignored --nocapture 2>&1 | tee "$OUT/$ref.log" | grep -E '^parity:|^depth|^  d[1-5]|^E \(K=5\)'
done
echo "logs in $OUT"
