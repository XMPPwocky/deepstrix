#!/usr/bin/env python3
"""Gate E4 smoke test for /v1/embeddings (docs/v41/EMBED_PHASE_DESIGN.md §10).

Run against a PRIVATE-port hub started with --embed-gguf (and its own
V41_EMBED_LOAN_IMAGE). Standard library only.

  python3 scripts/qwen3_embed/e2e_smoke.py --url http://127.0.0.1:18099 \\
      --ref ~/.cache/deepstrix/goldens/qwen3emb/ref.json

Checks: /v1/models lists the chat model first and the embedding model; every
input shape works; text input and the same text's token ids give the same
vector; `dimensions` truncates to a unit vector; base64 decodes to the float
result; empty inputs are 400s; and the served embeddings match the HF
reference (min cosine >= 0.998: the whole pipeline, tokenizer included).
"""

import argparse
import base64
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def cos(a, b):
    ab = sum(x * y for x, y in zip(a, b))
    return ab / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def norm(a):
    return math.sqrt(sum(x * x for x in a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18099")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    emb_url = args.url + "/v1/embeddings"
    ref = json.load(open(args.ref))
    fails = []

    def check(ok, what):
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    models = get(args.url + "/v1/models")["data"]
    check(len(models) >= 2 and models[0].get("type") == "llm" and any(m.get("type") == "embeddings" for m in models),
          f"/v1/models: chat first, embedding model listed ({[m.get('id') for m in models]})")
    emb_model = next(m["id"] for m in models if m.get("type") == "embeddings")

    # The reference corpus, batched: the served vectors vs HF.
    cases = ref["cases"]
    worst = (1.0, None)
    t0 = time.time()
    for i in range(0, len(cases), args.batch):
        chunk = cases[i:i + args.batch]
        st, r = post(emb_url, {"input": [c["text"] for c in chunk], "model": emb_model})
        check(st == 200 and len(r.get("data", [])) == len(chunk), f"batch {i // args.batch}: status {st}")
        if st != 200:
            continue
        for c, d in zip(chunk, r["data"]):
            check(all(math.isfinite(x) for x in d["embedding"]) and abs(norm(d["embedding"]) - 1) < 1e-3, f"{c['kind']}: finite unit vector")
            if c["embedding"]:
                cs = cos(d["embedding"], c["embedding"])
                if cs < worst[0]:
                    worst = (cs, c["kind"])
        want_tokens = sum(len(c["ids"]) for c in chunk)
        check(r["usage"]["prompt_tokens"] == want_tokens, f"usage.prompt_tokens {r['usage']['prompt_tokens']} == ref ids {want_tokens} (EOS counted)")
    if worst[1] is None:
        print(f"corpus served in {time.time() - t0:.1f} s; the reference has no embeddings (--tokens-only dump): HF cosine not checked")
    else:
        print(f"corpus served in {time.time() - t0:.1f} s; min cosine vs HF {worst[0]:.6f} ({worst[1]})")
        check(worst[0] >= 0.998, f"min cosine vs reference {worst[0]:.6f} >= 0.998")

    text = cases[0]["text"]
    st, a = post(emb_url, {"input": text})
    v = a["data"][0]["embedding"]
    check(st == 200 and len(v) == ref_dim(ref) and abs(norm(v) - 1) < 1e-3, f"single string: dim {len(v)}, |v| {norm(v):.5f}")
    ids = cases[0]["ids"][:-1]  # the handler appends EOS itself
    st, b = post(emb_url, {"input": ids})
    check(st == 200 and cos(b["data"][0]["embedding"], v) > 0.99999, "token-id input == text input")
    st, b = post(emb_url, {"input": [ids, ids[:5]]})
    check(st == 200 and len(b["data"]) == 2, "array of token arrays")
    st, d = post(emb_url, {"input": text, "dimensions": 256})
    dv = d["data"][0]["embedding"] if st == 200 else []
    check(len(dv) == 256 and abs(norm(dv) - 1) < 1e-3 and cos(dv, v[:256]) > 0.9999, f"dimensions=256: len {len(dv)}, unit, prefix of the full vector")
    st, e = post(emb_url, {"input": text, "encoding_format": "base64"})
    raw = base64.b64decode(e["data"][0]["embedding"]) if st == 200 else b""
    fv = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    check(len(fv) == len(v) and max(abs(x - y) for x, y in zip(fv, v)) < 1e-6, "base64 == float")
    for bad in [{"input": ""}, {"input": []}, {"input": [[]]}, {"input": "x", "dimensions": 8}, {"input": "x", "encoding_format": "f16"}]:
        st, _ = post(emb_url, bad)
        check(st == 400, f"400 for {json.dumps(bad)} (got {st})")

    print(f"\n{len(fails)} failure(s)")
    sys.exit(1 if fails else 0)


def ref_dim(ref):
    return len(ref["cases"][0]["embedding"]) or 2560


if __name__ == "__main__":
    main()
