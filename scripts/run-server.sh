#!/usr/bin/env bash
# Restart loop for deepstrix-server.
#
# The server's hang watchdog (see engine_worker::run_watchdog) calls
# abort() when forward progress stalls past DEEPSTRIX_HANG_DEADLINE_MS.
# Without a supervisor that turns abort() back into a fresh process,
# the GPU stays wedged and chat is dead. This is a 12-line "good
# enough" supervisor for manual / tmux operation.
#
# Usage: scripts/run-server.sh -- --gguf <path> --addr 127.0.0.1:18080 ...
# Anything after `--` is forwarded to deepstrix-server.

set -u

BIN="${BIN:-./target/release/deepstrix-server}"
LOG="${LOG:-/tmp/deepstrix-server.log}"
RESTART_DELAY_S="${RESTART_DELAY_S:-2}"

if [[ "${1:-}" == "--" ]]; then
    shift
fi

while true; do
    echo "$(date -Is) launching $BIN $*" | tee -a "$LOG"
    "$BIN" "$@" >> "$LOG" 2>&1
    code=$?
    echo "$(date -Is) deepstrix-server exited with code $code; restarting in ${RESTART_DELAY_S}s" | tee -a "$LOG"
    sleep "$RESTART_DELAY_S"
done
