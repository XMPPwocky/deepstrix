#!/usr/bin/env bash
# DSpark gate 4 (DSPARK_DESIGN.md "the one that matters"):
#
#   greedy decode WITH DSpark must be byte-identical to greedy decode WITHOUT it.
#   Speculative decoding is EXACT -- if the output differs, accept/reject is wrong.
#
# Needs no reference capture: the non-speculative run IS the oracle. Sibling of
# scripts/v41_determinism_gate.sh, which only compares a config to ITSELF and so
# cannot see this class of bug.
#
# Usage: scripts/v41_dspark_gate4.sh [max_tokens] [extra server env...]
# Exits non-zero on divergence. Discards a warm-up request per arm (the first
# request after a restart is cold -- see the discard-first-request rule).
set -u
PORT=${PORT:-18141}
MAXTOK=${1:-80}
OUT=${OUT:-$(mktemp -d)}
PROMPT=${PROMPT:-"Explain in detail how a lighthouse keeper in the 1800s would maintain the lamp, the lens, and the fog signal through a winter storm."}
# T2_CATCHALL=2 by default: mode 1 makes the box1/box2 split a function of
# request history, so its output is not reproducible and gate 4 is meaningless.
BASE_ENV=${BASE_ENV:-"V41_T2_CATCHALL=2"}

req () {
  python3 -c "
import json,sys
print(json.dumps({'model':'deepseek-v4.1-flash','max_tokens':$MAXTOK,'temperature':0,
 'reasoning_effort':'off','messages':[{'role':'user','content':sys.argv[1]}]}))" "$PROMPT"
}

arm () { # name, extra env
  local name=$1; shift
  pkill -x deepstrix-serve 2>/dev/null; sleep 8
  env $BASE_ENV "$@" setsid nohup bash ~/run_v41_server.sh > "$OUT/$name.log" 2>&1 < /dev/null 9>&- &
  for _ in $(seq 1 300); do
    curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 3
  done
  sleep 2
  req > "$OUT/req.json"
  # warm-up, discarded
  curl -s -m 3600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' --data @"$OUT/req.json" > /dev/null
  curl -s -m 3600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' --data @"$OUT/req.json" > "$OUT/$name.json"
  pkill -x -TERM deepstrix-serve; sleep 5
}

arm off
arm on V41_DSPARK=accept

python3 - "$OUT" <<'PY'
import hashlib, json, sys
d = sys.argv[1]
def text(n):
    return json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"]
try:
    a, b = text("off"), text("on")
except Exception as e:
    print(f"gate4: FAILED to read outputs: {e}"); sys.exit(2)
ha, hb = (hashlib.sha256(x.encode()).hexdigest()[:16] for x in (a, b))
print(f"gate4: dspark-off sha={ha}\ngate4: dspark-on  sha={hb}")
if a == b:
    print("gate4: PASS (byte-identical)"); sys.exit(0)
n = min(len(a), len(b))
i = next((k for k in range(n) if a[k] != b[k]), n)
print(f"gate4: *** FAIL *** first divergence at char {i}")
print(f"  off: {a[max(0,i-70):i+70]!r}")
print(f"  on : {b[max(0,i-70):i+70]!r}")
sys.exit(1)
PY
