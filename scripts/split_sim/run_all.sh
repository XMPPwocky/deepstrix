#!/usr/bin/env bash
# Hot-split simulator: full pipeline (docs/v41/HOT_SPLIT_SIM.md).
# Read-only on the live data; writes only under scripts/split_sim/data/.
# Usage: scripts/split_sim/run_all.sh [HUB_EVT] [PICK_TRACE] [SERVER_LOG]
set -euo pipefail
ulimit -v 4000000
cd "$(dirname "$0")"
EVT=${1:-/home/claude-code/logs/evtrace/hub-20261003-185254-7270-000.evt}
TRACE=${2:-/home/claude-code/logs/picks-sub-20261003-1852.trace}
LOG=${3:-/home/claude-code/logs/v41-server.log}
STEM=$(basename "$EVT" .evt)
N="nice -n 19"
mkdir -p data
# 1. evtrace -> TSV (hub_step, decode-sized hub_req), streamed
$N python3 extract_evt.py "$EVT" data
# 2. box-1 hot-set refresh times (only the refresh lines of the log are read)
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "hot set refreshed" | awk '{print $1, $9, $10, $11}' > data/hot_refresh.txt
# 3. align the pick trace with hub_step -> replay cache
$N python3 build_cache.py "$TRACE" "data/$STEM.hub_step.tsv" data/cache.pkl
# 4. validate alignment + today's ownership replica against hub_req
$N python3 check_ownership.py data/cache.pkl "data/$STEM.hub_req.tsv" data/hot_refresh.txt "$EVT" 420
# 5. fit the cost model
$N python3 fit_costs.py data/cache.pkl "data/$STEM.hub_req.tsv" data/hot_refresh.txt "$EVT" data/costs.json > data/costs.log
# 6. calibrate (a) with measured misses, (b) with the pool model at the calibrated miss_keep
$N python3 calibrate.py data/cache.pkl "data/$STEM.hub_req.tsv" data/hot_refresh.txt "$EVT" --costs data/costs.json --params data/final/params.json | tee data/calibration_measured_misses.txt
$N python3 calibrate.py data/cache.pkl "data/$STEM.hub_req.tsv" data/hot_refresh.txt "$EVT" --costs data/costs.json --params data/final/params.json --pool-model | grep "box-2 misses over"
$N python3 calibrate.py data/cache.pkl "data/$STEM.hub_req.tsv" data/hot_refresh.txt "$EVT" --costs data/costs.json --params data/final/params.json --pool-model --miss-keep "${MISS_KEEP:-0.31}" | tee data/calibration_pool_model.txt
# 7. policies (learn on the first half, evaluate on the second)
$N python3 compare.py data/cache.pkl data/hot_refresh.txt "$EVT" --costs data/costs.json --params data/final/params.json --miss-keep "${MISS_KEEP:-0.31}" --out data/compare.json | tee data/compare.txt
# 8. sensitivity (slow: ~2-3 min per row on 7k steps)
$N python3 sensitivity.py data/cache.pkl data/hot_refresh.txt "$EVT" --costs data/costs.json --params data/final/params.json --out data/sens.json | tee data/sens.txt
