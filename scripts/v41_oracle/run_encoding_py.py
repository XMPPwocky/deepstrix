"""Batch driver for the checkpoint's reference encoder (encoding/encoding.py),
used by crates/deepstrix-server/tests/oracle_prompt.rs to cross-check
DeepSeek's two references (deepseek-recipe vs encoding.py) on generated cases.

    python3 -I scripts/v41_oracle/run_encoding_py.py /weights2/dsv4.1f-1017/encoding < cases.jsonl

`-I` keeps the interpreter from importing anything from this script's
directory or the cwd; the encoding dir is put on sys.path explicitly and only
`encoding` is imported from it (stdlib-only: typing, copy, json, re).

Each input line: {"messages": [...], "thinking_mode": "thinking"|"chat",
"reasoning_effort": null|"low"|"high"|"max"|int, "drop_thinking": bool}. Tools
are already attached to messages[0] (load_cases' convention). Each output
line: {"ok": prompt} or {"err": "<exception>"}.
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import encoding as enc  # noqa: E402

for line in sys.stdin:
    case = json.loads(line)
    try:
        prompt = enc.encode_messages(
            case["messages"],
            thinking_mode=case["thinking_mode"],
            reasoning_effort=case.get("reasoning_effort"),
            drop_thinking=case.get("drop_thinking", True),
        )
        out = {"ok": prompt}
    except Exception as e:  # noqa: BLE001 -- any reference failure is data
        out = {"err": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
sys.stdout.flush()
