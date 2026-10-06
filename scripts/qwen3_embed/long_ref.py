#!/usr/bin/env python3
"""Long-input check for the embed phase (inputs past the 7K the window corpus
covers, up to V41_EMBED_MAX_INPUT_TOKENS). Standard library only.

Real text (repo docs) is cut to exact token counts with llama.cpp's tokenizer
(identical to HF on the 22-case corpus).

  make : llama.cpp's embeddings (same Q8_0 GGUF) as the reference
           llama-server -m Q8_0.gguf --embedding --pooling last -fa on \\
             -c 16384 -b 16384 -ub 16384 -t 8 --port 18199 &
           long_ref.py make --out long_ref.json
  check: embed the same texts on a hub, compare (token count and cosine)
           long_ref.py check --url http://127.0.0.1:18080 --ref long_ref.json

llama.cpp's embedding at 11K+ tokens needs ~8 GB of host RAM (measured
2026-10-06: MemAvailable 9.7 -> 1.7 GB in 5 s), which box 1 does not have
while the hub runs. Without a hub-down window, use the CPU oracle instead:

  texts: llama.cpp TOKENIZER only (any small context works)
           long_ref.py texts --out long_texts.json
  hub  : the hub's embeddings for those texts, one oracle reference per case
           long_ref.py hub --ref long_texts.json --out hub_long
         then, per case,
           QWEN3_EMBED_REF=hub_long.<kind>.json QWEN3_EMBED_E1_MAX_TOKENS=16384 \\
             cargo test -p v4flash-core --test qwen3_embed_ref -- --ignored e0 e1
         (E0: our tokenizer == llama.cpp's ids; E1: the CPU oracle == the
         hub's GPU forward)
"""

import argparse
import json
import math
import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCES = ["docs/v41/KV_PREFIX_STORE_DESIGN.md", "docs/v41/KNOWN_BUGS.md"]
TARGETS = [11412, 16384]  # the client's failing input size; the new limit (EOS included)


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(llama, text):
    return post(llama + "/tokenize", {"content": text, "add_special": True})["tokens"]


def cut(llama, text, n):
    """Longest prefix of `text` with at most `n` tokens (EOS included)."""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenize(llama, text[:mid])) <= n:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def write(path, model, cases):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"model": model, "eos_appended": True, "cases": cases}, f)
    os.replace(tmp, path)


def texts(a, embed=False):
    cases = []
    for src, n in zip(SOURCES, TARGETS):
        text = cut(a.llama, open(os.path.join(ROOT, src)).read(), n)
        ids = tokenize(a.llama, text)
        c = {"kind": f"long-{len(ids)}", "source": src, "text": text, "ids": ids}
        if embed:
            c["embedding"] = post(a.llama + "/v1/embeddings", {"input": text})["data"][0]["embedding"]
        cases.append(c)
        print(f"{src}: {len(ids)} tokens", file=sys.stderr, flush=True)
    model = "llama.cpp llama-server --embedding --pooling last (same Q8_0 GGUF)" if embed else "llama.cpp tokenizer"
    write(a.out, model, cases)


def make(a):
    texts(a, embed=True)


def hub(a):
    """The hub's embeddings for the texts, one CPU-oracle reference per case."""
    for c in json.load(open(a.ref))["cases"]:
        r = post(a.url + "/v1/embeddings", {"model": a.model, "input": c["text"]})
        e = r["data"][0]["embedding"]
        n = r["usage"]["prompt_tokens"]
        norm = math.sqrt(sum(x * x for x in e))
        finite = all(math.isfinite(x) for x in e)
        print(f"{c['kind']}: hub tokens {n} (llama.cpp {len(c['ids'])}), dim {len(e)}, norm {norm:.6f}, finite {finite}", flush=True)
        write(f"{a.out}.{c['kind']}.json", f"deepstrix hub {a.url}", [{**c, "embedding": e}])


def check(a):
    ref = json.load(open(a.ref))
    fails = 0
    for c in ref["cases"]:
        r = post(a.url + "/v1/embeddings", {"model": a.model, "input": c["text"]})
        e = r["data"][0]["embedding"]
        n, cs = r["usage"]["prompt_tokens"], cos(e, c["embedding"])
        ok = n == len(c["ids"]) and cs >= 0.998
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {c['kind']}: tokens {n} (ref {len(c['ids'])}), cos vs llama.cpp {cs:.6f}", flush=True)
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["texts", "make", "hub", "check"])
    ap.add_argument("--llama", default="http://127.0.0.1:18199")
    ap.add_argument("--out", default="long_ref.json")
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--ref", default="long_ref.json")
    ap.add_argument("--model", default="qwen3-embedding-4b")
    a = ap.parse_args()
    {"texts": texts, "make": make, "hub": hub, "check": check}[a.mode](a)


if __name__ == "__main__":
    main()
