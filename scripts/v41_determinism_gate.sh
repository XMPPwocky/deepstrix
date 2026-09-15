#!/usr/bin/env bash
# CORRECTNESS GATE for any change to expert residency, masking, paging or the
# box1/box2 split.
#
# WHY THIS EXISTS. Three separate "wins" on this engine were skipped work:
#   - the submit mask that silently dropped 77% of routed experts
#   - the small-B offload whose 5.3x came with acceptance halved
#   - the coalesced expert read (2026-09-15), which looked like decode
#     115-122 -> 66-70 ms/tok AND a hit-rate improvement, while producing
#     different output on every run
# tok/s cannot tell a speedup from a wrong answer. This can.
#
# HOW IT WORKS. `V41_T2_CATCHALL=2` makes the box1/box2 partition a CONSTANT, so
# residency cannot legitimately change any result; temperature 0 makes sampling
# deterministic. Therefore a change that is genuinely only a CACHE must leave the
# generated text bit-identical. If the sha moves, work is being skipped or read
# from a reused slot.
#
# Run it in the regime where the path under test actually executes. For anything
# touching the expert read/page-in path that means V41_B2_POOL_FLOOR=0 on box 2,
# where a decode request pages constantly — the coalescing bug hid for hours
# because it was validated at the default floor, where page-ins are rare.
#
#   scripts/v41_determinism_gate.sh [runs]        # default 3
set -u
RUNS=${1:-3}
ADDR=${ADDR:-127.0.0.1:18141}
PROMPT=${PROMPT:-"Explain how a CPU cache works, in about 120 words."}
MAXTOK=${MAXTOK:-120}

gen() {
  curl -s -N -m 900 "http://$ADDR/v1/chat/completions" -H 'Content-Type: application/json' -d "$(
    printf '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":%s}],"max_tokens":%d,"temperature":0,"stream":true}' \
      "$(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" "$MAXTOK"
  )" 2>/dev/null | python3 -c "
import json,sys,hashlib
out=[]
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('data: ') or line=='data: [DONE]': continue
    try: d=json.loads(line[6:])
    except: continue
    for c in d.get('choices',[]):
        dl=c.get('delta',{}); out.append(dl.get('content') or dl.get('reasoning_content') or '')
t=''.join(out)
print(hashlib.sha256(t.encode()).hexdigest()[:16], len(t))"
}

echo "determinism gate: $RUNS runs, temperature 0"
echo "  REQUIRED: server started with V41_T2_CATCHALL=2 (constant partition)"
echo
first=""; ok=1
for i in $(seq 1 "$RUNS"); do
  r=$(gen); sha=${r% *}; len=${r#* }
  [ -z "$first" ] && first=$sha
  [ "$sha" = "$first" ] || ok=0
  printf "  run %d: sha %s  len %s\n" "$i" "$sha" "$len"
done
echo
if [ "$len" = "0" ]; then
  echo "INCONCLUSIVE: empty output (model still inside its reasoning block, or an error). Raise MAXTOK."
  exit 2
fi
if [ "$ok" = "1" ]; then
  echo "PASS — output bit-identical across runs. The change is a cache, not a computation."
else
  echo "FAIL — output differs between identical runs. Work is being SKIPPED or read"
  echo "       from a reused slot. Any speedup measured here is not real."
  exit 1
fi
