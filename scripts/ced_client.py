#!/usr/bin/env python3
"""POST one chat completion to the local V4.1 server; print timing + text.
usage: ced_client.py <prompt-file|-> <max_tokens> [tag]   (temperature 0)"""
import json, sys, time, urllib.request
src, max_tokens = sys.argv[1], int(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else src
prompt = sys.stdin.read() if src == "-" else open(src).read()
body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0}
req = urllib.request.Request("http://127.0.0.1:18141/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=7200) as r:
    resp = json.load(r)
dt = time.time() - t0
ch = resp["choices"][0]
u = resp.get("usage", {})
print(f"[{tag}] wall={dt:.1f}s prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')} finish={ch.get('finish_reason')}")
m = ch['message']
print(f"[{tag}] text={json.dumps(m.get('content'))} reasoning={json.dumps(m.get('reasoning_content'))}")
