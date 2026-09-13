"""Sparse (reference CSA2) vs dense (deepstrix today) attention, measured side by side.

Runs DeepSeek's own `inference/model.py` one Block at a time, exactly like `oracle.py`,
but with batch row 0 = the reference indexer's top-`index_topk` selection and batch row 1
= the whole compressed store (see `dense_index.py`). Both rows carry the *same* tokens, so
every weight, every dequantised expert and every engram lookup is shared between the two
runs; the only difference in the whole forward is which compressed rows attention reads.

Outputs, under --out:
  sparse/, dense/   one dump each in `oracle.py --dump-argmax` layout (tokens.json,
                    head_input.pt, main_argmax.pt, layer_{36,37,38}_residual.pt), so
                    `dspark_accept.py` runs on either without modification
  compare.json      per-position top-1 agreement + KL(sparse || dense) + entropies
  rows.json         how many compressed rows each index-source layer scored

  nix-shell -p python3Packages.{torch,numpy,pillow,sympy,tokenizers} --run \\
    "python3 indexer_ab.py --tokens p.json --out ~/.cache/deepstrix/v41/dense_ab"
"""
import argparse
import dataclasses
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import oracle  # noqa: E402  (sets sys.path for the reference package)
import model as ref  # noqa: E402
import dense_index  # noqa: E402
from engram import EngramLayout, NgramHashState  # noqa: E402
from lazy import LazyEngram, LazyMoE, _assign, load_module  # noqa: E402
from loader import Checkpoint  # noqa: E402

DSPARK_RESIDUAL_LAYERS = (36, 37, 38)  # inputs of dspark_target_layer_ids = 37/38/39


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=None, help="json list of token ids (BOS included)")
    ap.add_argument("--prompt-text-file", default=None)
    ap.add_argument("--limit", type=int, default=0, help="truncate the prompt to this many tokens")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="sparse,dense")
    ap.add_argument("--all-residuals", action="store_true", help="dump every layer (control runs)")
    a = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    ref.world_size, ref.rank, ref.default_dtype = 1, 0, torch.float8_e4m3fn
    rows = a.rows.split(",")
    B = len(rows)
    outs = [os.path.join(a.out, f"row{i}_{r}") for i, r in enumerate(rows)]
    for d in outs:
        os.makedirs(d, exist_ok=True)

    ckpt = Checkpoint(oracle.MODEL)
    tok = oracle._TokShim(os.path.join(oracle.MODEL, "tokenizer.json"))
    if a.prompt_text_file:
        ids = [0] + tok.backend_tokenizer.encode(open(a.prompt_text_file).read(), add_special_tokens=False).ids
    else:
        ids = json.load(open(a.tokens))
    if a.limit:
        ids = ids[: a.limit]
    T = len(ids)
    for d in outs:
        json.dump(ids, open(os.path.join(d, "tokens.json"), "w"))

    ref_cfg = json.load(open(os.path.join(oracle.MODEL, "inference", "config.json")))
    args = oracle.make_args(ref_cfg, max_seq_len=((T + 127) // 128) * 128 + 128)
    args = dataclasses.replace(args, max_batch_size=B)
    n_layers = min(a.layers, args.n_layers)
    args.n_layers = n_layers
    print(f"T={T} tokens, {n_layers} layers, batch rows {rows}", flush=True)

    dense_index.install(rows)

    layout = EngramLayout.from_args(args)
    t0 = time.time()
    hs = NgramHashState(args, layout, tok)
    input_ids = torch.tensor([ids] * B, dtype=torch.long)
    hashes = hs(input_ids, 0, None)
    print(f"engram hashes ready ({time.time() - t0:.1f}s)", flush=True)

    emb = ckpt.get("embed.weight")
    h = emb[torch.tensor(ids)].unsqueeze(0).repeat(B, 1, 1).clone()   # [B,T,dim]
    h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)                  # [B,T,hc,dim]
    pre_mix = ref.make_identity_pre_mix(h, args.hc_mult)

    last_block = None
    small = dataclasses.replace(args, n_routed_experts=1, n_activated_experts=1)
    for L in range(n_layers):
        t0 = time.time()
        pfx = f"layers.{L}."
        block = ref.Block(L, small, None)
        missing = load_module(block.attn, ckpt, pfx + "attn.")
        missing += load_module(block.attn_norm, ckpt, pfx + "attn_norm.")
        missing += load_module(block.ffn_norm, ckpt, pfx + "ffn_norm.")
        for nm in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale"):
            _assign(getattr(block, nm), ckpt.get(pfx + nm), pfx + nm)
        block.ffn = LazyMoE(L, args, ckpt, pfx + "ffn.")
        missing += load_module(block.ffn.gate, ckpt, pfx + "ffn.gate.")
        missing += load_module(block.ffn.shared_experts, ckpt, pfx + "ffn.shared_experts.")
        if L in layout.layer_ids:
            block.engram = LazyEngram(args, L, layout, ckpt, pfx + "engram.")
            missing += load_module(block.engram, ckpt, pfx + "engram.", skip=("embed",))
        if missing:
            raise SystemExit(f"L{L}: missing tensors: {missing[:5]} ...")
        t_load = time.time() - t0

        dense_index._LAYER[0] = L
        with torch.inference_mode():
            if block.engram is not None:
                h = block.engram(h, hashes[:, :, block.engram.layer_hash_index, :], None)
            h, pre_mix = block(h, 0, pre_mix, None)
        if a.all_residuals or L in DSPARK_RESIDUAL_LAYERS:
            for b, d in enumerate(outs):
                torch.save(h[b : b + 1].float(), os.path.join(d, f"layer_{L:02d}_residual.pt"))
        drift = "  ".join(f"|r0-r{b}|={(h[0].float() - h[b].float()).abs().max().item():.3e}"
                          for b in range(1, B))
        rms = h[0].float().pow(2).mean().sqrt().item()
        print(f"L{L:02d} ratio={args.compress_ratios[L]} experts={len(block.ffn.touched):3d} "
              f"rms={rms:.4f} {drift}  load {t_load:.1f}s fwd {time.time()-t0-t_load:.1f}s",
              flush=True)
        last_block = block
        del block

    # ---- head: per-position logits for both rows, compared on the fly
    with torch.inference_mode():
        x = last_block.hc_pre(h, pre_mix)
        del h
        norm = ref.RMSNorm(args.dim, args.norm_eps)
        load_module(norm, ckpt, "norm.")
        head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        _assign(head.weight, ckpt.get("head.weight"), "head.weight")
        head_in = norm(x).float()                                    # [B,T,dim]
        for b, d in enumerate(outs):
            torch.save(head_in[b : b + 1], os.path.join(d, "head_input.pt"))
        am = [[] for _ in range(B)]
        keys = ("top1_agree", "kl_sd", "kl_ds", "ent_sparse", "ent_dense",
                "p_sparse_top1_under_dense", "top5_agree", "tv")
        recs = [{k: [] for k in keys} for _ in range(B)]
        # per-row teacher-forced quality: NLL and top-1 hit on the ACTUAL next token.
        # Unlike pairwise agreement these are unbiased under arithmetic noise, so the
        # duplicate control row measures zero effect by construction.
        nll = [[] for _ in range(B)]
        hit = [[] for _ in range(B)]
        nxt = torch.tensor(ids[1:] + [ids[-1]])
        for c0 in range(0, T, 32):
            lg = [head(head_in[b:b + 1, c0:c0 + 32], full_logits=True).float()[0] for b in range(B)]
            tgt = nxt[c0:c0 + lg[0].shape[0]]
            for b in range(B):
                am[b].append(lg[b].argmax(-1))
                lpb = torch.log_softmax(lg[b], -1)
                nll[b] += (-lpb.gather(-1, tgt.unsqueeze(-1))[:, 0]).tolist()
                hit[b] += (lg[b].argmax(-1) == tgt).int().tolist()
            lp0 = torch.log_softmax(lg[0], -1)
            p0 = lp0.exp()
            t1_0 = lg[0].argmax(-1)
            k5_0 = lg[0].topk(5, -1).indices
            for b in range(1, B):
                rec = recs[b]
                lp1 = torch.log_softmax(lg[b], -1)
                p1 = lp1.exp()
                t1_1 = lg[b].argmax(-1)
                rec["top1_agree"] += (t1_0 == t1_1).int().tolist()
                rec["kl_sd"] += (p0 * (lp0 - lp1)).sum(-1).tolist()
                rec["kl_ds"] += (p1 * (lp1 - lp0)).sum(-1).tolist()
                rec["ent_sparse"] += (-(p0 * lp0).sum(-1)).tolist()
                rec["ent_dense"] += (-(p1 * lp1).sum(-1)).tolist()
                rec["tv"] += (0.5 * (p0 - p1).abs().sum(-1)).tolist()
                rec["p_sparse_top1_under_dense"] += p1.gather(-1, t1_0.unsqueeze(-1))[:, 0].tolist()
                k5 = lg[b].topk(5, -1).indices
                rec["top5_agree"] += [len(set(u.tolist()) & set(v.tolist())) for u, v in zip(k5_0, k5)]
        for b, d in enumerate(outs):
            torch.save(torch.cat(am[b]).to(torch.int32), os.path.join(d, "main_argmax.pt"))
    for b in range(B):
        json.dump({"nll": nll[b], "hit": hit[b], "mode": rows[b], "tokens": ids},
                  open(os.path.join(outs[b], "teacher.json"), "w"))
    for b in range(1, B):
        recs[b]["tokens"] = ids
        recs[b]["rows"] = [rows[0], rows[b]]
        name = "compare.json" if b == 1 else f"compare_row{b}.json"
        json.dump(recs[b], open(os.path.join(a.out, name), "w"))
    json.dump([list(s) for s in dense_index.STATS], open(os.path.join(a.out, "rows.json"), "w"))
    torch.save({L: (v[0][0], v[0][1]) for L, v in dense_index.MASS.items()},
               os.path.join(a.out, "attn_mass.pt"))
    for b in range(1, B):
        agree = recs[b]["top1_agree"]
        kl = recs[b]["kl_sd"]
        first = next((i for i, v in enumerate(agree) if not v), None)
        print(f"\nrow0({rows[0]}) vs row{b}({rows[b]}): top-1 agreement {sum(agree)/len(agree):.4f}; "
              f"first flip at position {first}; mean KL {sum(kl)/len(kl):.5f}")
    print("\nteacher-forced, whole prompt: " + "  ".join(
        f"row{b}({rows[b]}) NLL {sum(nll[b][:-1])/(T-1):.4f} acc {sum(hit[b][:-1])/(T-1):.4f}" for b in range(B)))


if __name__ == "__main__":
    main()
