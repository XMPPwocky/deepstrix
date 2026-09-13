"""Greedy generation with the CPU oracle, all 40 Blocks kept RESIDENT (non-routed
weights ≈ 13 GB bf16; routed experts stay lazy) so decode steps run incrementally
with the reference's own KV/compressor/indexer caches instead of re-prefilling.
Produces the model's OWN continuation of a prompt — the on-distribution text
DSpark acceptance must be measured on — and dumps, for the generated span,
the residual after every layer + per-position greedy ids (the same layout as
oracle.py --dump-argmax, so dspark_accept.py runs unchanged on the output).

  nix-shell -p python3Packages.{torch,numpy,pillow,sympy,tokenizers} --run \\
    "python3 oracle_generate.py --prompt-ids-file ~/.cache/deepstrix/v41/agentic/tokens.json --prefix 256 --gen 96 --out ~/.cache/deepstrix/v41/agentic/gen"
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oracle  # noqa: E402
import model as ref  # noqa: E402
from engram import EngramLayout, NgramHashState  # noqa: E402
from lazy import LazyEngram, LazyMoE, _assign, load_module  # noqa: E402
from loader import Checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-ids-file", required=True)
    ap.add_argument("--prompt-text", default=None, help="tokenise this text instead (BOS added)")
    ap.add_argument("--prompt-text-file", default=None, help="like --prompt-text, read from a file")
    ap.add_argument("--prefix", type=int, default=256)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dspark", action="store_true",
                    help="also run the DSpark drafter in-process exactly like the reference loop "
                         "(forward_spec after every main step) and report position-1 greedy acceptance")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok0 = oracle._TokShim(os.path.join(oracle.MODEL, "tokenizer.json"))
    if a.prompt_text_file:
        a.prompt_text = open(a.prompt_text_file).read().strip()
    if a.prompt_text:
        ids = [0] + tok0.backend_tokenizer.encode(a.prompt_text, add_special_tokens=False).ids
        a.prefix = len(ids)
    else:
        ids = json.load(open(a.prompt_ids_file))[: a.prefix]
    total = a.prefix + a.gen
    ref_cfg = json.load(open(os.path.join(oracle.MODEL, "inference", "config.json")))
    args = oracle.make_args(ref_cfg, max_seq_len=((total + 127) // 128) * 128 + 128)
    torch.set_default_dtype(torch.bfloat16)
    ref.default_dtype = torch.bfloat16
    ckpt = Checkpoint(oracle.MODEL)
    tok = oracle._TokShim(os.path.join(oracle.MODEL, "tokenizer.json"))
    layout = EngramLayout.from_args(args)
    hs = NgramHashState(args, layout, tok)

    t0 = time.time()
    small = dataclasses.replace(args, n_routed_experts=1, n_activated_experts=1)
    blocks = []
    for L in range(args.n_layers):
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
        if layout is not None and L in layout.layer_ids:
            block.engram = LazyEngram(args, L, layout, ckpt, pfx + "engram.")
            missing += load_module(block.engram, ckpt, pfx + "engram.", skip=("embed",))
        if missing:
            raise SystemExit(f"L{L}: missing {missing[:5]}")
        blocks.append(block)
    # Bounded LRU cache of dequantised routed experts, bf16 (lossless for MXFP4
    # values: E2M1 × 2^e has 3 significant bits), converted to f32 at use.
    # Without it every touched expert is re-dequantised every step (~105 ms each,
    # 240 per token). V41_EXPERT_CACHE_GB (default 50) ≈ 700 experts.
    from collections import OrderedDict
    budget = float(os.environ.get("V41_EXPERT_CACHE_GB", "50")) * (1 << 30)
    cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
    used = [0]
    def make_cached(orig):
        def cached(e, which, _o=orig):
            k = (id(_o), e, which)
            if k in cache:
                cache.move_to_end(k)
                return cache[k].float()
            w = _o(e, which)
            wb = w.to(torch.bfloat16)
            cache[k] = wb
            used[0] += wb.numel() * 2
            while used[0] > budget and cache:
                old = cache.pop(next(iter(cache)))  # FIFO eviction on a plain dict
                used[0] -= old.numel() * 2
            return w if w.dtype == torch.float32 else w.float()
        return cached
    for block in blocks:
        block.ffn._w = make_cached(block.ffn._w)
    embed = ref.ParallelEmbedding(args.vocab_size, args.dim)
    _assign(embed.weight, ckpt.get("embed.weight"), "embed.weight")
    norm = ref.RMSNorm(args.dim, args.norm_eps)
    load_module(norm, ckpt, "norm.")
    head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
    _assign(head.weight, ckpt.get("head.weight"), "head.weight")
    print(f"40 blocks resident in {time.time() - t0:.0f}s", flush=True)

    residuals = [[] for _ in range(args.n_layers)]  # per layer: list of [1, s, hc, dim]
    argmax = []
    head_in = []

    def run(input_ids, start_pos):
        """One forward over input_ids at start_pos; returns greedy next tokens [s]."""
        x = torch.tensor([input_ids])
        hashes = hs(x, start_pos, None)
        h = embed(x).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        pre_mix = ref.make_identity_pre_mix(h, args.hc_mult)
        main_hiddens = []
        with torch.inference_mode():
            for L, block in enumerate(blocks):
                if block.engram is not None:
                    h = block.engram(h, hashes[:, :, block.engram.layer_hash_index, :], None)
                if L in args.dspark_target_layer_ids:
                    main_hiddens.append(h.mean(dim=2))  # attention input of the target layer, as model.py
                h, pre_mix = block(h, start_pos, pre_mix, None)
                residuals[L].append(h.float().cpu())
            xh = blocks[-1].hc_pre(h, pre_mix)
            head_in.append(norm(xh).float().cpu())  # [1, s, dim]: main logits = head(head_in) offline
            lg = head(norm(xh), full_logits=True).float()[0]  # [s, V]
        am = lg.argmax(-1).tolist()
        argmax.extend(am)
        run.last_main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return am

    drafter = None
    if a.dspark:
        d_args = dataclasses.replace(args, temperature=0.0)
        small = dataclasses.replace(d_args, dspark_n_routed_experts=1, dspark_n_activated_experts=1)
        stages = []
        for s_ in range(d_args.n_mtp_layers):
            blk = ref.DSparkBlock(d_args.n_layers + s_, small)
            # skip is a PREFIX match: "ffn" would also drop ffn_norm.weight (the 2026-09-13 acceptance bug)
            missing = load_module(blk, ckpt, f"mtp.{s_}.", skip=("ffn.", "embed.", "head."))
            blk.ffn = LazyMoE(d_args.n_layers + s_, d_args, ckpt, f"mtp.{s_}.ffn.")
            missing += load_module(blk.ffn.gate, ckpt, f"mtp.{s_}.ffn.gate.")
            missing += load_module(blk.ffn.shared_experts, ckpt, f"mtp.{s_}.ffn.shared_experts.")
            if missing:
                raise SystemExit(f"mtp.{s_}: missing {missing[:5]}")
            # bf16 LRU with a budget (V41_DSPARK_CACHE_GB, default 10): an uncapped fp32
            # cache of 128 x 3 x 3 matrices is ~54 GB and OOM-kills the box.
            from collections import OrderedDict as _OD
            cache = _OD()
            d_budget = float(os.environ.get("V41_DSPARK_CACHE_GB", "10")) * (1 << 30)
            d_used = [0]
            orig = blk.ffn._w
            def cached(e, which, _o=orig, _c=cache):
                k = (e, which)
                if k in _c:
                    _c.move_to_end(k)
                    return _c[k].float()
                w = _o(e, which).to(torch.bfloat16)
                _c[k] = w
                d_used[0] += w.numel() * 2
                while d_used[0] > d_budget and _c:
                    old = _c.pop(next(iter(_c)))
                    d_used[0] -= old.numel() * 2
                return w.float()
            blk.ffn._w = cached
            stages.append(blk)
        stages[0].embed = embed
        stages[-1].head = head
        def forward_spec(input_ids, main_hidden, start_pos):
            h_, main_x = stages[0].forward_embed(main_hidden, input_ids)
            pm = ref.make_identity_pre_mix(h_, d_args.hc_mult)
            for layer in stages:
                h_, pm = layer(h_, start_pos, pm, main_x)
            if start_pos == 0:
                return None
            return stages[-1].forward_head(h_, pm, input_ids)
        drafter = forward_spec
        print("dspark drafter loaded (in-process)", flush=True)
    t1 = time.time()
    nxt = run(ids, 0)[-1]
    print(f"prefill {a.prefix} tokens in {time.time() - t1:.0f}s; first token {tok.backend_tokenizer.decode([nxt])!r}", flush=True)
    pending = None  # (drafts, confidence) issued at the previous step, verified when the next main token is known
    hits = [0] * 5
    n_ver = 0
    if drafter is not None:
        with torch.inference_mode():
            drafter(torch.tensor([[nxt]]), run.last_main_hidden, 0)  # seed the window with the prefix, like the reference
            out_ids, _lg, _conf = drafter(torch.tensor([[nxt]]), run.last_main_hidden[:, -1:], a.prefix - 1)
        pending = out_ids[0, 1:].tolist()
    gen = []
    for k in range(a.gen):
        ts = time.time()
        gen.append(nxt)
        ids.append(nxt)
        pos_k = a.prefix + k
        nxt = run([nxt], pos_k)[-1]
        if drafter is not None:
            # the main model just produced the token for position pos_k+1: verify the previous drafts
            if pending is not None:
                n_ver += 1
                ok = True
                for j in range(5):
                    ok = ok and (pending[j] == nxt if j == 0 else False)  # only draft 1 is verifiable one step at a time
                hits[0] += pending[0] == nxt
            with torch.inference_mode():
                out_ids, _lg, _conf = drafter(torch.tensor([[nxt]]), run.last_main_hidden, pos_k)
            pending = out_ids[0, 1:].tolist()
            if n_ver and n_ver % 10 == 0:
                print(f"  dspark in-process: draft-1 greedy acceptance {hits[0] / n_ver:.3f} over {n_ver} steps", flush=True)
        if k % 8 == 0:
            print(f"  gen {k + 1}/{a.gen} ({time.time() - ts:.0f}s/step): {tok.backend_tokenizer.decode(gen[-8:])!r}", flush=True)
    json.dump(ids, open(os.path.join(a.out, "tokens.json"), "w"))  # processed positions only
    json.dump(ids + [nxt], open(os.path.join(a.out, "tokens_plus_next.json"), "w"))
    for L in range(args.n_layers):
        torch.save(torch.cat(residuals[L], dim=1), os.path.join(a.out, f"layer_{L:02d}_residual.pt"))
    torch.save(torch.tensor(argmax, dtype=torch.int32), os.path.join(a.out, "main_argmax.pt"))
    torch.save(torch.cat(head_in, dim=1), os.path.join(a.out, "head_input.pt"))
    open(os.path.join(a.out, "generated.txt"), "w").write(tok.backend_tokenizer.decode(gen))
    print("generated:", repr(tok.backend_tokenizer.decode(gen))[:600])


if __name__ == "__main__":
    main()
