#!/usr/bin/env python3
"""One non-streaming chat completion against the V4.1 server; prints wall + tok/s."""
import json, sys, time, urllib.request, pathlib

prompt_file = sys.argv[1]
max_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 64
nonce = sys.argv[3] if len(sys.argv) > 3 else ""
body = pathlib.Path(prompt_file).read_text() if prompt_file != "-" else "What is the capital of France?"
if nonce:
    body = f"[run {nonce}]\n" + body
req = {
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": body}],
    "max_tokens": max_tok,
    "temperature": 0.0,
    "stream": False,
}
t0 = time.time()
r = urllib.request.Request("http://127.0.0.1:18141/v1/chat/completions",
                           data=json.dumps(req).encode(),
                           headers={"Content-Type": "application/json"})
with urllib.request.urlopen(r, timeout=3600) as f:
    out = json.load(f)
dt = time.time() - t0
u = out.get("usage", {})
print(json.dumps({"wall_s": round(dt, 2), "usage": u,
                  "text": (out["choices"][0]["message"].get("content") or "")[:300]}, indent=1))
