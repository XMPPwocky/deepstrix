#!/usr/bin/env python3
"""Stream one chat completion; print reasoning + content text and timing.
usage: ced_stream.py <prompt-file> <max_tokens> <tag>"""
import json, sys, time, urllib.request
src, max_tokens, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": open(src).read()}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
req = urllib.request.Request("http://127.0.0.1:18141/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); first = None; reason = []; content = []; n = 0
with urllib.request.urlopen(req, timeout=7200) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"): continue
        payload = line[5:].strip()
        if payload == "[DONE]": break
        try: ev = json.loads(payload)
        except Exception: continue
        for ch in ev.get("choices", []):
            d = ch.get("delta") or {}
            for key, sink in (("reasoning_content", reason), ("reasoning", reason), ("content", content)):
                v = d.get(key)
                if v:
                    if first is None: first = time.time() - t0
                    sink.append(v); n += 1
dt = time.time() - t0
dec = (dt - first) if first else 0
print(f"[{tag}] ttft={first:.1f}s total={dt:.1f}s decode_chunks={n} decode_tok_s={(n-1)/dec:.2f}" if first and dec > 0 else f"[{tag}] ttft={first} total={dt:.1f}s chunks={n}")
print(f"[{tag}] reasoning={json.dumps(''.join(reason))}")
print(f"[{tag}] content={json.dumps(''.join(content))}")
