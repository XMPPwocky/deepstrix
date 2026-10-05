#!/usr/bin/env python3
"""Reference token ids + embeddings for the embed-phase gates
(docs/v41/EMBED_PHASE_DESIGN.md §10: E0 tokenizer, E1 CPU oracle, E2 GPU).

Runs the official HF model on the CPU, exactly as its model card does:
left-padding tokenizer, last-token pooling, L2 normalize. Writes one JSON
file the Rust gates read (QWEN3_EMBED_REF):

  {"model": ..., "dtype": ..., "cases": [{"kind", "text", "ids", "embedding"}]}

`ids` are the reference tokenizer's ids for the text INCLUDING whatever it
appends (the gate checks that this is <|endoftext|>, design §2).

Usage (CPU; ~16 GB RAM in f32, ~8 GB in bf16):

  nix-shell -p 'python3.withPackages (p: [p.torch p.transformers])' --run \\
    'python3 scripts/qwen3_embed/ref_embed.py --model /path/to/Qwen3-Embedding-4B \\
       --out ~/.cache/deepstrix/goldens/qwen3emb/ref.json'

Needs transformers >= 4.51 (Qwen3). `--model` is the HF snapshot dir (or the
hub id if this box has the cache). Nothing here downloads unless given a hub id.
"""

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF Qwen3-Embedding-4B dir (or hub id)")
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(__file__), "corpus.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--tokens-only", action="store_true",
                    help="only the tokenizer (gate E0): needs tokenizer.json etc., not the weights; no embeddings written")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    with open(args.corpus) as f:
        corpus = json.load(f)
    texts = [c["text"] * c.get("repeat", 1) for c in corpus["cases"]]

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    if args.tokens_only:
        cases = []
        for c, text in zip(corpus["cases"], texts):
            ids = tok([text], padding=False, truncation=True, max_length=args.max_length)["input_ids"][0]
            cases.append({"kind": c["kind"], "text": text, "ids": ids, "embedding": []})
        appended = all(c["ids"][-1] == eos for c in cases)
        print(f"<|endoftext|> = {eos}; appended to every input: {appended}", file=sys.stderr)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "dtype": None, "eos_appended": appended, "eos_id": eos, "cases": cases}, f)
        print(f"wrote {len(cases)} tokenized cases (no embeddings) to {args.out}", file=sys.stderr)
        return

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel

    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()

    out_cases = []
    t0 = time.time()
    for c, text in zip(corpus["cases"], texts):
        # One input per forward: no padding at all, so pooling = the last row.
        batch = tok([text], padding=False, truncation=True, max_length=args.max_length, return_tensors="pt")
        ids = batch["input_ids"][0].tolist()
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        emb = F.normalize(hidden[:, -1].float(), p=2, dim=1)[0]
        out_cases.append({"kind": c["kind"], "text": text, "ids": ids, "embedding": emb.tolist()})
        print(f"{c['kind']:12s} {len(ids):6d} tokens  last id {ids[-1]}  ({time.time() - t0:.1f} s)", file=sys.stderr)

    appended = all(c["ids"][-1] == eos for c in out_cases)
    print(f"<|endoftext|> = {eos}; appended to every input: {appended}", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "dtype": args.dtype, "eos_appended": appended, "eos_id": eos, "cases": out_cases}, f)
    print(f"wrote {len(out_cases)} cases to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
