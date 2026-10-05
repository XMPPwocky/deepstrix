#!/usr/bin/env python3
"""A same-weights external reference for the embed-phase gates: llama.cpp's
`llama-server --embedding --pooling last` on the SAME Q8_0 GGUF
(docs/v41/EMBED_PHASE_DESIGN.md §10, E1 with llama.cpp in place of HF bf16;
the owner waived the HF comparison 2026-10-05).

Reads the tokens-only HF dump (ref_embed.py --tokens-only: the HF ids, the
gates' tokenizer truth), asks llama.cpp for each text's ids (/tokenize, with
special tokens) and embedding (/v1/embeddings), and writes a reference in the
format the gates read (QWEN3_EMBED_REF / e2e_smoke --ref):

  {"model": "llama.cpp ...", "eos_appended": true, "cases": [{kind, text, ids (HF), embedding (llama.cpp)}]}

  llama-server -m Q8_0.gguf --embedding --pooling last -c 10240 -b 10240 -ub 10240 --port 18199 &
  llama_ref.py --url http://127.0.0.1:18199 --tokens ref_tokens.json --out ref_llama.json
"""

import argparse
import json
import sys
import urllib.request


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18199")
    ap.add_argument("--tokens", required=True, help="ref_embed.py --tokens-only output (HF ids)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = json.load(open(a.tokens))
    out = []
    same_ids = 0
    for c in ref["cases"]:
        lid = post(a.url + "/tokenize", {"content": c["text"], "add_special": True}).get("tokens", [])
        if lid == c["ids"]:
            same_ids += 1
        else:
            print(f"  {c['kind']}: llama.cpp ids differ from HF ({len(lid)} vs {len(c['ids'])})", file=sys.stderr)
        r = post(a.url + "/v1/embeddings", {"input": c["text"]})
        emb = r["data"][0]["embedding"]
        out.append({"kind": c["kind"], "text": c["text"], "ids": c["ids"], "embedding": emb})
        print(f"{c['kind']:12s} {len(c['ids']):6d} tokens  dim {len(emb)}", file=sys.stderr)
    print(f"llama.cpp tokenization identical to HF on {same_ids} / {len(ref['cases'])} cases", file=sys.stderr)
    # Atomic: a gate may pick the file up the moment it exists.
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"model": "llama.cpp llama-server --embedding --pooling last (same Q8_0 GGUF)", "dtype": "q8_0",
                   "eos_appended": ref.get("eos_appended", True), "eos_id": ref.get("eos_id"), "cases": out}, f)
    import os
    os.replace(tmp, a.out)
    print(f"wrote {len(out)} cases to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
