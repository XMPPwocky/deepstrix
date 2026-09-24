"""Export a golden-corpus case (oracle.py --golden output) into the compact
fixture the engine's golden gate reads (tests/v41_golden_gate.rs).

One little-endian binary per tensor kind, plus fixture.json with shapes,
dtypes and sha256, so a Rust test reads each with a single fs::read:

  tokens.json                     token ids (verbatim)
  spans.json                      assistant spans [(think_idx, eos_idx)] = generated regions
  logits_all.f32                  [T, V]        reference logits at every position
  topk_ids.i32                    [L, T, 6]     reference routed experts, rank order
  router_w.f32                    [L, T, 6]     reference routing weights (final, x route_scale)
  router_sel.f32                  [L, T, 384]   biased selection scores (margins = flip risk)
  compress_idxs_LNN.i32           [T, topk]     compressed positions attended (compressed layers)
  residual_LNN.f32                [T, 4, 5120]  per-layer residual stream (only with --residuals)
  embed_hc.f32                    [T, 4, 5120]  layer-0 input (only with --residuals)

Usage (box 2, the oracle's torch env):
  python3 export_golden.py ~/goldens/agentic ~/goldens-fixtures/agentic [--residuals]
"""
import argparse
import hashlib
import json
import os

import torch

THINK, EOS = 128821, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--residuals", action="store_true", help="also export per-layer residuals (~82 MB/layer at T=1006)")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    src_man = json.load(open(os.path.join(a.src, "manifest.json")))
    ids = json.load(open(os.path.join(a.src, "tokens.json")))
    T, L = len(ids), src_man["layers"]
    files = {}

    def put(name, t, dtype):
        t = t.detach().contiguous()
        t = t.to(torch.int32) if dtype == "i32" else t.float()
        p = os.path.join(a.dst, name)
        with open(p, "wb") as f:
            f.write(t.numpy().tobytes())
        files[name] = {"dtype": dtype, "shape": list(t.shape), "bytes": os.path.getsize(p),
                       "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}

    json.dump(ids, open(os.path.join(a.dst, "tokens.json"), "w"))
    spans, i = [], 0
    while i < T:
        if ids[i] == THINK and EOS in ids[i + 1:]:
            j = ids.index(EOS, i + 1)
            spans.append([i, j])
            i = j
        i += 1
    json.dump(spans, open(os.path.join(a.dst, "spans.json"), "w"))

    put("logits_all.f32", torch.load(os.path.join(a.src, "logits_all.pt")), "f32")
    ld = lambda n: torch.load(os.path.join(a.src, n))
    put("topk_ids.i32", torch.stack([ld(f"layer_{l:02d}_topk_ids.pt").reshape(T, -1) for l in range(L)]), "i32")
    put("router_w.f32", torch.stack([ld(f"layer_{l:02d}_router_w.pt").reshape(T, -1) for l in range(L)]), "f32")
    put("router_sel.f32", torch.stack([ld(f"layer_{l:02d}_router_sel.pt").reshape(T, -1) for l in range(L)]), "f32")
    for l in range(L):
        p = os.path.join(a.src, f"layer_{l:02d}_compress_idxs.pt")
        if os.path.exists(p):
            put(f"compress_idxs_L{l:02d}.i32", torch.load(p).reshape(T, -1), "i32")
    if a.residuals:
        put("embed_hc.f32", ld("embed_hc.pt").reshape(T, -1), "f32")
        for l in range(L):
            put(f"residual_L{l:02d}.f32", ld(f"layer_{l:02d}_residual.pt").reshape(T, -1), "f32")

    fx = {"tokens": T, "layers": L, "vocab": files["logits_all.f32"]["shape"][1], "spans": spans,
          "source": {k: src_man.get(k) for k in ("model_dir", "config_sha256", "reference_model_py_sha256", "oracle_rev")},
          "source_manifest_sha256": hashlib.sha256(open(os.path.join(a.src, "manifest.json"), "rb").read()).hexdigest(),
          "files": files}
    json.dump(fx, open(os.path.join(a.dst, "fixture.json"), "w"), indent=1)
    tot = sum(f["bytes"] for f in files.values())
    print(f"{a.dst}: T={T} L={L} {len(files)} files {tot/1e9:.2f} GB, {len(spans)} generated spans")


if __name__ == "__main__":
    main()
