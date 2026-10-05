#!/usr/bin/env python3
"""Embed-phase gates that need a running hub (docs/v41/EMBED_PHASE_DESIGN.md §10):
E3 (loan integrity: chat output unchanged by interleaved embed phases), E6 (fault
injection), E7 (phase cost, chat ITL with embedding traffic). Standard library only.

Run against a PRIVATE-port hub (its own knob file, its own loan image):

  window_checks.py e3 --url http://127.0.0.1:18099
  window_checks.py e6 --url ... --knobs /path/private-knobs.txt
  window_checks.py e7 --url ... --log /path/test-hub.log

Each prints PASS/FAIL lines and exits non-zero on a failure.
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

CHAT_MODEL = "deepseek-v4.1-flash"
PROMPT = "List the first ten prime numbers, then explain in two sentences why 1 is not prime."


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except json.JSONDecodeError:
            return e.code, {}


def chat(base, max_tokens):
    st, r = post(base + "/v1/chat/completions", {
        "model": CHAT_MODEL, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
    })
    if st != 200:
        return st, None
    m = r["choices"][0]["message"]
    return st, (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")


def chat_stream_itl(base, max_tokens):
    """Inter-token latencies (ms) of one streamed greedy completion."""
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({
        "model": CHAT_MODEL, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
    }).encode(), headers={"Content-Type": "application/json"})
    times = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            delta = d.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                times.append(time.monotonic())
    return [(b - a) * 1e3 for a, b in zip(times, times[1:])]


def embed(base, inputs, **kw):
    return post(base + "/v1/embeddings", {"input": inputs, **kw})


def cos(a, b):
    ab = sum(x * y for x, y in zip(a, b))
    return ab / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


class Hammer:
    """Background embedding traffic until stopped."""

    def __init__(self, base, inputs, pause_s=0.0):
        self.base, self.inputs, self.pause_s = base, inputs, pause_s
        self.stop = False
        self.ok = self.fail = 0
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop:
            st, _ = embed(self.base, self.inputs)
            if st == 200:
                self.ok += 1
            else:
                self.fail += 1
            time.sleep(self.pause_s)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join(timeout=600)


FAILS = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


def e3(base, n_tokens):
    # A/A first: identical greedy outputs with no embed phases, or E3 cannot be judged.
    _, a1 = chat(base, n_tokens)
    _, a2 = chat(base, n_tokens)
    check(a1 is not None and a1 == a2, f"E3(a) A/A: two greedy {n_tokens}-token completions identical")
    with Hammer(base, ["The loan is returned byte-exact after every phase. " * 40] * 8) as h:
        _, b = chat(base, n_tokens)
    check(h.ok > 0, f"E3(b) embed phases ran during the completion ({h.ok} ok, {h.fail} failed requests)")
    check(b == a1, "E3(b) completion with embed phases interleaved == without")
    _, c = chat(base, n_tokens)
    check(c == a1, "E3(b) completion after the phases == before")
    if b != a1 and a1 and b:
        i = next((k for k, (x, y) in enumerate(zip(a1, b)) if x != y), min(len(a1), len(b)))
        print(f"  first difference at char {i}: {a1[max(0, i - 40):i + 40]!r} vs {b[max(0, i - 40):i + 40]!r}")


def set_knob(path, name, value):
    lines = []
    if os.path.exists(path):
        lines = [l for l in open(path).read().splitlines() if not l.startswith(name + "=")]
    if value is not None:
        lines.append(f"{name}={value}")
    tmp = path + ".tmp"
    open(tmp, "w").write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    time.sleep(3)  # the knob watcher re-reads every second


def e6(base, knobs):
    text = ["Fault injection must fail only the embedding requests of the faulted phase."]
    st, base_r = embed(base, text)
    check(st == 200, "E6 baseline embedding")
    ref = base_r["data"][0]["embedding"] if st == 200 else None
    for layer in (0, 17, 35):
        set_knob(knobs, "V41_EMBED_FAULT_LAYER", layer)
        st, r = embed(base, text)
        check(st == 500 and "injected fault" in json.dumps(r), f"E6 fault at layer {layer}: the request fails with the injected error (status {st})")
        set_knob(knobs, "V41_EMBED_FAULT_LAYER", None)
        st, r = embed(base, text)
        ok = st == 200 and ref is not None and cos(r["data"][0]["embedding"], ref) > 0.99999
        check(ok, f"E6 after the layer-{layer} fault: embeddings unchanged (loan returned)")
    st, out = chat(base, 32)
    check(st == 200 and out, "E6 chat still answers after the faults")


def ms_embed_lines(log, since_pos):
    with open(log, errors="replace") as f:
        f.seek(since_pos)
        txt = f.read()
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    out = []
    for line in txt.splitlines():
        if "ms.embed" not in line:
            continue
        kv = dict(re.findall(r"(\w+)=([0-9.]+|true|false)", line))
        out.append(kv)
    return out


def e7(base, log):
    pos = os.path.getsize(log)
    for _ in range(3):
        embed(base, ["hi"])
    small = ms_embed_lines(log, pos)
    pos = os.path.getsize(log)
    big_text = "Embedding throughput is measured over many medium-length passages of ordinary text. " * 28
    st, r = embed(base, [big_text] * 32)
    big = ms_embed_lines(log, pos)
    for name, rows in (("1-token input", small), ("32 x ~500-token inputs", big)):
        for kv in rows:
            print(f"  {name}: tokens {kv.get('tokens')} total_ms {kv.get('total_ms')} fwd_ms {kv.get('fwd_ms')} "
                  f"read_ms {kv.get('read_ms')} wait_read_ms {kv.get('wait_read_ms')} return_ms {kv.get('return_ms')} "
                  f"verify_ms {kv.get('verify_ms')} pinned_ms {kv.get('pinned_ms')} rows_ms {kv.get('rows_ms')}")
    check(bool(small) and bool(big), "E7 ms.embed lines recorded")
    alone = chat_stream_itl(base, 160)
    with Hammer(base, ["Interleaved embedding traffic during a streamed completion. " * 20] * 4, pause_s=0.2) as h:
        loaded = chat_stream_itl(base, 160)
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
    print(f"  chat ITL alone : p50 {q(alone, .5):.0f} ms  p99 {q(alone, .99):.0f} ms  ({len(alone)} gaps)")
    print(f"  chat ITL loaded: p50 {q(loaded, .5):.0f} ms  p99 {q(loaded, .99):.0f} ms  ({len(loaded)} gaps, {h.ok} embed requests)")
    check(bool(alone) and bool(loaded), "E7 chat ITL measured with and without embedding traffic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["e3", "e6", "e7"])
    ap.add_argument("--url", default="http://127.0.0.1:18099")
    ap.add_argument("--knobs")
    ap.add_argument("--log")
    ap.add_argument("--tokens", type=int, default=96)
    a = ap.parse_args()
    if a.check == "e3":
        e3(a.url, a.tokens)
    elif a.check == "e6":
        e6(a.url, a.knobs)
    else:
        e7(a.url, a.log)
    print(f"{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
