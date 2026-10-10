#!/usr/bin/env bash
# Launch the hub supervisor from the env file (+ extra NAME=value args) and health-check it. Hub must be DOWN.
# Lines of the env file that are not NAME=value (comments, blanks) are SKIPPED: `env -i` runs a non-assignment arg as
# the COMMAND (10-10: a '# comment' line in the env file made the supervisor exec the comment; the hub never started).
set -u
ENVF=/home/claude-code/scratch-ms/hub_env_c_20261006.txt
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
URL=http://127.0.0.1:18080
say() { echo "$(date -u +%T) $*"; }
if pgrep -f "$HUBPAT" >/dev/null || pgrep -f "$SUPPAT" >/dev/null; then echo "hub/supervisor already running"; exit 1; fi
mapfile -t ENVV < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENVF" | grep -v -E '^(V41_LOG_ATTACHED|V41_PICK_TRACE|V41_PREFILL_LOGITS_DUMP)=')
ENVV+=("V41_PICK_TRACE=/home/claude-code/logs/picks-sub-$(date -u +%Y%m%d-%H%M).trace" "$@")
for a in "${ENVV[@]}"; do case "$a" in *=*) ;; *) echo "non-assignment env arg: $a"; exit 1;; esac; done
echo "$(date -u +%FT%T) DEPLOY src=target-v41 md5=$(md5sum < /home/claude-code/deepstrix/target-v41/release/deepstrix-server | cut -c1-8) launch_hub.sh $*" >> /home/claude-code/logs/evtrace/deploy_markers.log
(cd ~ && env -i "${ENVV[@]}" setsid nohup ~/run_v41_server.sh >/dev/null 2>&1 </dev/null &)
sleep 3
ok=0
for i in $(seq 1 400); do
  if sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | tail -400 | grep -q 'listening on'; then ok=1; break; fi
  if [ "$i" = 15 ] && ! pgrep -f "$SUPPAT" >/dev/null; then say "supervisor DIED within 15 s"; exit 1; fi
  sleep 1
done
ns=$(pgrep -fc "$SUPPAT"); nh=$(pgrep -fc "$HUBPAT")
say "hub listening=$ok after ${i}s; supervisors=$ns hubs=$nh"
[ "$ok" = 1 ] && [ "$ns" = 1 ] && [ "$nh" = 1 ] || exit 1
python3 - "$URL" <<'EOF'
import json, sys, urllib.request
url = sys.argv[1]
models = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=30).read())
chat = [m["id"] for m in models["data"] if "embedding" not in m["id"]][0]
req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps({"model": chat, "max_tokens": 16, "temperature": 0,
      "messages": [{"role": "user", "content": "Reply with the single word: ready"}]}).encode(), headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=600).read())
print(f"chat: {(r['choices'][0]['message'].get('content') or '')[:60]!r}, usage {r.get('usage')}")
sys.exit(0 if r["usage"]["completion_tokens"] > 0 else 1)
EOF
rc=$?
sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | grep -aE 'expert pager:|multistream scheduler ON|remote experts:|UNTESTED FIDELITY|ERROR|panicked' | tail -6 | cut -c1-220
exit $rc
