#!/usr/bin/env bash
# run.sh <shape> [harness opts...]   -> gpu_run.sh --dev dgpu with an honest --mb per shape
set -eu
cd "$(dirname "$0")"
SHAPE=$1; shift
case "$SHAPE" in
  qa) MB=120 ;; qb) MB=400 ;; kv) MB=112 ;; woa) MB=160 ;; wob) MB=155 ;; shg) MB=130 ;; shd) MB=130 ;;
  engram) MB=600 ;; cast_*) MB=120 ;; qb64) MB=155 ;; kv64) MB=110 ;; wob64) MB=155 ;; woa64) MB=150 ;;
  *) MB=160 ;;
esac
../_infra/gpu_run.sh --dev dgpu --mb $MB --label C2_dense_prefill/$SHAPE --timeout 300 -- ./harness_gfx1201 gfx1201 . "$SHAPE" "$@"
