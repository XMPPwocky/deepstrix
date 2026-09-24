"""Layer-streaming CPU oracle for DeepSeek-V4.1-Flash.

Runs DeepSeek's own `inference/model.py` (unmodified) one `Block` at a time
against the mmap'd HF checkpoint, with the tilelang kernels swapped for the
CPU implementations in ./kernel.py and the two things that do not fit in RAM
(384 expert modules per layer, the 98 GB Engram table) replaced by lazy
equivalents in ./lazy.py. Dumps every layer's residual stream and the final
logits. This is the reference every GPU kernel in the port gets compared to.

Usage:
  nix-shell -p python3Packages.torch python3Packages.numpy python3Packages.pillow \
            python3Packages.sympy python3Packages.tokenizers \
    --run "python3 oracle.py --layers 2 --prompt 'The quick brown fox' --out /tmp/o"
"""
import argparse
import dataclasses
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("V41_MODEL", os.path.expanduser("~/.cache/deepstrix/models/dsv4.1f"))
sys.path.insert(0, HERE)  # our kernel.py shadows the tilelang one
sys.path.insert(1, os.path.join(MODEL, "inference"))

import torch  # noqa: E402

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cpu")

import model as ref  # noqa: E402  (DeepSeek's, unmodified)
from engram import EngramLayout, NgramHashState  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from lazy import LazyEngram, LazyMoE, load_module  # noqa: E402
from loader import Checkpoint  # noqa: E402


class _TokShim:
    """What NgramHashState.build_compressed_token_map needs from a HF tokenizer."""

    def __init__(self, path):
        self.backend_tokenizer = Tokenizer.from_file(path)

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size()


def make_args(cfg: dict, max_seq_len: int) -> ref.ModelArgs:
    fields = {f.name for f in dataclasses.fields(ref.ModelArgs)}
    kw = {k: v for k, v in cfg.items() if k in fields}
    kw["max_batch_size"] = 1
    kw["max_seq_len"] = max_seq_len
    return ref.ModelArgs(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layers", type=int, default=40, help="run this many layers (all=40)")
    ap.add_argument("--out", default=os.path.join(HERE, "dump"))
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--prompt-ids", default=None, help="comma-separated token ids, used verbatim instead of --prompt")
    ap.add_argument("--dump-argmax", action="store_true", help="also dump the greedy token at every position")
    ap.add_argument("--golden", action="store_true",
                    help="golden-corpus capture: per-layer router selection scores + weights, the compressed "
                         "positions each layer attends to, the candidate-block mask, full logits at every "
                         "position, and a manifest with sha256 of every file")
    ap.add_argument("--swap-eps", type=float, default=None,
                    help="routing-swap experiment: wherever the biased 6th-vs-7th selection gap is < EPS, "
                         "route to the 7th expert instead of the 6th (weights recomputed as ref.Gate does). "
                         "Implies --golden's router tap; compare logits_all against an unswapped run.")
    ap.add_argument("--swap-cold-only", default=None,
                    help="JSON [layer][ids] hot set: only swap when the 6th pick is OUTSIDE it and the 7th inside")
    ap.add_argument("--swap-positions", default=None,
                    help="JSON list of token positions where swapping is allowed (e.g. only the generated "
                         "tokens, so prompt KV stays exact as it would in production); default: all")
    ap.add_argument("--swap-sixth-cold", default=None,
                    help="JSON [layer][ids] hot set: eligible only when the 6th pick is OUTSIDE it (7th unconstrained)")
    ap.add_argument("--swap-frac", type=float, default=1.0,
                    help="apply the swap to only this fraction of eligible token-layers (deterministic, --swap-seed)")
    ap.add_argument("--swap-seed", type=int, default=0)
    ap.add_argument("--swap-mode", choices=("seventh", "drop"), default="seventh",
                    help="seventh: route to the 7th instead; drop: drop the 6th and renormalise over the other 5")
    ap.add_argument("--no-layer-dumps", action="store_true", help="skip per-layer residual/routing files")
    a = ap.parse_args()
    if a.swap_eps is not None:
        a.golden = True
    hot_sets = None
    if a.swap_cold_only:
        hot_sets = [set(x) for x in __import__("json").load(open(a.swap_cold_only))]
    sixth_cold_sets = None
    if a.swap_sixth_cold:
        sixth_cold_sets = [set(x) for x in __import__("json").load(open(a.swap_sixth_cold))]
    swap_counts = []
    swap_pos = None
    if a.swap_positions:
        swap_pos = __import__("json").load(open(a.swap_positions))
    os.makedirs(a.out, exist_ok=True)

    ckpt = Checkpoint(MODEL)
    tok = _TokShim(os.path.join(MODEL, "tokenizer.json"))
    if a.prompt_ids:
        ids = [int(x) for x in a.prompt_ids.split(",")]  # verbatim token ids (incl. BOS)
    else:
        ids = [ckpt.config["bos_token_id"]] + tok.backend_tokenizer.encode(a.prompt).ids
    os.makedirs(a.out, exist_ok=True)
    __import__("json").dump(ids, open(os.path.join(a.out, "tokens.json"), "w"))
    T = len(ids)
    print(f"prompt {T} tokens: {ids}")

    # Reference globals normally set by Transformer.__init__
    ref.world_size, ref.rank, ref.default_dtype = 1, 0, torch.float8_e4m3fn
    # ModelArgs field names match inference/config.json, NOT the HF config.json
    # (whose keys are num_attention_heads etc.). Using the wrong one silently
    # builds the 16-head default model and fails on the first shape mismatch.
    import json
    ref_cfg = json.load(open(os.path.join(MODEL, "inference", "config.json")))
    args = make_args(ref_cfg, max_seq_len=((T + 127) // 128) * 128 + 128)
    args.n_layers = a.layers if a.layers < args.n_layers else args.n_layers
    n_layers = a.layers

    layout = None if a.no_engram else EngramLayout.from_args(args)
    hashes = None
    if layout is not None:
        t0 = time.time()
        hs = NgramHashState(args, layout, tok)
        input_ids = torch.tensor([ids], dtype=torch.long)
        hashes = hs(input_ids, 0, None)  # [1,T,n_engram_layers,24]
        print(f"engram hashes ready ({time.time()-t0:.1f}s)")

    # ---- embed -> hc copies
    emb = ckpt.get("embed.weight")  # bf16 [V, dim]
    h = emb[torch.tensor(ids)].unsqueeze(0).clone()  # [1,T,dim]
    h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)  # [1,T,hc,dim]
    pre_mix = ref.make_identity_pre_mix(h, args.hc_mult)
    torch.save(h.float(), os.path.join(a.out, "embed_hc.pt"))

    # ---- layers, one Block at a time
    last_block = None
    for L in range(n_layers):
        t0 = time.time()
        pfx = f"layers.{L}."
        # Build the reference Block with a 1-expert MoE (we replace ffn below);
        # attention/mHC/norms are the real ones.
        small = dataclasses.replace(args, n_routed_experts=1, n_activated_experts=1)
        block = ref.Block(L, small, None)
        missing = load_module(block.attn, ckpt, pfx + "attn.")
        missing += load_module(block.attn_norm, ckpt, pfx + "attn_norm.")
        missing += load_module(block.ffn_norm, ckpt, pfx + "ffn_norm.")
        for nm in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale"):
            from lazy import _assign
            _assign(getattr(block, nm), ckpt.get(pfx + nm), pfx + nm)
        block.ffn = LazyMoE(L, args, ckpt, pfx + "ffn.")
        missing += load_module(block.ffn.gate, ckpt, pfx + "ffn.gate.")
        missing += load_module(block.ffn.shared_experts, ckpt, pfx + "ffn.shared_experts.")
        if layout is not None and L in layout.layer_ids:
            block.engram = LazyEngram(args, L, layout, ckpt, pfx + "engram.")
            missing += load_module(block.engram, ckpt, pfx + "engram.", skip=("embed",))
        if missing:
            raise SystemExit(f"L{L}: missing tensors: {missing[:5]} ...")
        t_load = time.time() - t0

        # Stage taps (V41_ORACLE_STAGES=1): wrap hc_pre / hc_post / norms /
        # attn / ffn / gate so the engine can be compared sub-step by sub-step.
        stages = {}
        if os.environ.get("V41_ORACLE_STAGES") == "1":
            def tap(name, fn):
                calls = []
                def wrapped(*args, **kw):
                    out = fn(*args, **kw)
                    # clone: several reference ops modify their outputs in place later
                    # (rope / fp4_act_quant on the compressor latent), and the tap must
                    # capture the value at THIS point
                    calls.append(out.clone() if torch.is_tensor(out) else out)
                    stages[name] = calls
                    return out
                return wrapped
            block.hc_pre = tap("hc_pre", block.hc_pre)
            block.hc_post = tap("hc_post", block.hc_post)
            block.attn_norm.forward = tap("attn_norm", block.attn_norm.forward)
            block.attn.forward = tap("attn", block.attn.forward)
            block.ffn_norm.forward = tap("ffn_norm", block.ffn_norm.forward)
            block.ffn.gate.forward = tap("gate", block.ffn.gate.forward)
            block.ffn.forward = tap("ffn", block.ffn.forward)
            # attention internals (module outputs; wo_b also records its input = post-wo_a "low")
            for nm in ("wq_a", "q_norm", "wq_b", "wkv", "kv_norm"):
                m_ = getattr(block.attn, nm, None)
                if m_ is not None:
                    m_.forward = tap("attn_" + nm, m_.forward)
            if getattr(block.attn, "compressor", None) is not None:
                block.attn.compressor.forward = tap("compressor", block.attn.compressor.forward)  # latent pre-RoPE
            if hasattr(block.attn, "wo_b"):
                def tap_io(name, fn):
                    def wrapped(*args, **kw):
                        out = fn(*args, **kw)
                        stages.setdefault(name + "_in", []).append(args[0])
                        stages.setdefault(name, []).append(out)
                        return out
                    return wrapped
                block.attn.wo_b.forward = tap_io("attn_wo_b", block.attn.wo_b.forward)
        golden = {}
        if a.golden:
            gate = block.ffn.gate
            gate_fwd = gate.forward

            def gate_tap(x, image_mask=None, _gate=gate, _fwd=gate_fwd):
                w, idx = _fwd(x, image_mask)
                # The SELECTION scores top-k ranks, recomputed exactly as ref.Gate.forward
                # does (model.py): score_func(x W^T / temp) + bias. Their gap at rank k vs
                # k+1 is the margin a routing flip has to cross.
                sc = ref.linear(x.float(), _gate.weight.float()) / _gate.gate_temp
                if _gate.score_func == "softmax":
                    sc = sc.softmax(dim=-1)
                elif _gate.score_func == "sigmoid":
                    sc = sc.sigmoid()
                else:
                    sc = torch.nn.functional.softplus(sc).sqrt()
                bias = _gate.bias
                if image_mask is not None and _gate.bias_vl is not None:
                    bias = torch.where(image_mask.unsqueeze(-1), _gate.bias_vl, bias)
                sel = sc + bias
                assert torch.equal(sel.topk(_gate.topk, dim=-1)[1], idx), "recomputed selection != reference"
                golden["router_sel"] = sel.float().clone()
                golden["router_w"] = w.float().clone()
                if a.swap_eps is not None:
                    top = sel.topk(_gate.topk + 1, dim=-1)
                    k = _gate.topk
                    gap = top.values[:, k - 1] - top.values[:, k]
                    swap = gap < a.swap_eps
                    if swap_pos is not None:
                        allowed = torch.zeros_like(swap)
                        allowed[torch.tensor(swap_pos, dtype=torch.long)] = True
                        swap = swap & allowed
                    if hot_sets is not None:
                        hs = hot_sets[L]
                        sixth, seventh = top.indices[:, k - 1].tolist(), top.indices[:, k].tolist()
                        cold_ok = torch.tensor([(s6 not in hs) and (s7 in hs) for s6, s7 in zip(sixth, seventh)])
                        swap = swap & cold_ok
                    if sixth_cold_sets is not None:
                        hs6 = sixth_cold_sets[L]
                        swap = swap & torch.tensor([e not in hs6 for e in top.indices[:, k - 1].tolist()])
                    if a.swap_frac < 1.0:
                        # Deterministic per (seed, layer, position): a seeded generator over the rows.
                        g = torch.Generator().manual_seed(a.swap_seed * 1000003 + L)
                        swap = swap & (torch.rand(swap.shape[0], generator=g) < a.swap_frac)
                    idx = top.indices[:, :k].clone()
                    if a.swap_mode == "seventh":
                        idx[swap, k - 1] = top.indices[swap, k]
                    # Weights exactly as ref.Gate.forward: unbiased scores of the chosen set,
                    # renormalised, times route_scale. "drop" zeroes the 6th before renormalising.
                    w = sc.gather(1, idx)
                    if a.swap_mode == "drop":
                        w[swap, k - 1] = 0.0
                    if _gate.norm_topk_prob and k > 1:
                        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
                    w = w * _gate.route_scale
                    swap_counts.append(int(swap.sum()))
                return w, idx
            gate.forward = gate_tap
            if args.compress_ratios[L]:
                ctk = block.attn._compress_topk_idxs

                def ctk_tap(*a_, _f=ctk, **k_):
                    r = _f(*a_, **k_)
                    golden["compress_idxs"] = r.clone()
                    return r
                block.attn._compress_topk_idxs = ctk_tap
        with torch.inference_mode():
            if block.engram is not None:
                h = block.engram(h, hashes[:, :, block.engram.layer_hash_index, :], None)
            h, pre_mix = block(h, 0, pre_mix, None)
        if os.environ.get("V41_ORACLE_STAGES") == "1":
            # The mHC pre-mix this block hands to the NEXT block's attention collapse
            # (single-pass mHC): lets the engine run layer L+1 in isolation.
            stages["pre_mix"] = [pre_mix.detach().float().clone()]
            # The caches as the reference leaves them: window rows (fp8-fake-quantised) and,
            # on kv-source layers, the compressed rows (post-RoPE, E2M1×E4M3/16 fake-quantised).
            T_ = h.shape[1]
            stages["window_kv"] = [block.attn.window_kv_cache[:1, :min(T_, block.attn.window_size)].clone()]
            if getattr(block.attn, "compress_kv_cache", None) is not None:
                n_c = T_ // args.compress_ratios[L]
                stages["compress_kv"] = [block.attn.compress_kv_cache[:1, :n_c].clone()]
        for name, calls in stages.items():
            for ci, out in enumerate(calls):
                if out is None:
                    continue
                if isinstance(out, tuple):
                    for oi, o in enumerate(out):
                        torch.save(o.detach().float().cpu(), os.path.join(a.out, f"layer_{L:02d}_stage_{name}{ci}_{oi}.pt"))
                else:
                    torch.save(out.detach().float().cpu(), os.path.join(a.out, f"layer_{L:02d}_stage_{name}{ci}.pt"))
        if a.no_layer_dumps:
            golden.clear()
        else:
            torch.save(h.float(), os.path.join(a.out, f"layer_{L:02d}_residual.pt"))
        if not a.no_layer_dumps and getattr(block.ffn, "last_indices", None) is not None:
            torch.save(block.ffn.last_indices.to(torch.int32), os.path.join(a.out, f"layer_{L:02d}_topk_ids.pt"))
        for name, t in golden.items():
            torch.save(t.cpu() if name != "compress_idxs" else t.to(torch.int32).cpu(),
                       os.path.join(a.out, f"layer_{L:02d}_{name}.pt"))
        idxr = getattr(block.attn, "indexer", None)
        if a.golden and idxr is not None and getattr(idxr, "is_candidate_source", False):
            torch.save(ref.shared_attn.candidates.clone().cpu(), os.path.join(a.out, f"layer_{L:02d}_candidates.pt"))
        touched = len(block.ffn.touched)
        rms = h.float().pow(2).mean().sqrt().item()
        nan = torch.isnan(h).any().item()
        print(f"L{L:02d} ratio={args.compress_ratios[L]} experts_touched={touched:3d} "
              f"rms={rms:.4f} nan={nan}  load {t_load:.1f}s fwd {time.time()-t0-t_load:.1f}s")
        last_block = block
        del block

    # ---- head
    with torch.inference_mode():
        x = last_block.hc_pre(h, pre_mix)
        norm = ref.RMSNorm(args.dim, args.norm_eps)
        load_module(norm, ckpt, "norm.")
        head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        from lazy import _assign
        _assign(head.weight, ckpt.get("head.weight"), "head.weight")
        logits = head(norm(x))  # [1, V] last position
        if a.dump_argmax:
            # Greedy next token at EVERY position (teacher-forced references for
            # DSpark acceptance); chunked so T x 129280 f32 never materialises at once.
            am, top5 = [], []
            for c0 in range(0, x.shape[1], 32):
                lg = head(norm(x[:, c0:c0 + 32]), full_logits=True).float()[0]  # [c, V]
                am.append(lg.argmax(-1)); top5.append(lg.topk(5, dim=-1).indices)
            torch.save(torch.cat(am).to(torch.int32), os.path.join(a.out, "main_argmax.pt"))
            torch.save(torch.cat(top5).to(torch.int32), os.path.join(a.out, "main_top5.pt"))
            print("dumped main_argmax/top5 for", x.shape[1], "positions")
        if a.golden:
            # f32 [T, V], chunked through the head so the activations stay small.
            chunks = []
            for c0 in range(0, x.shape[1], 32):
                chunks.append(head(norm(x[:, c0:c0 + 32]), full_logits=True).float()[0].cpu())
            torch.save(torch.cat(chunks), os.path.join(a.out, "logits_all.pt"))
    torch.save(logits.float(), os.path.join(a.out, "logits_last.pt"))
    if a.golden:
        import hashlib, subprocess
        files = {}
        for fn in sorted(os.listdir(a.out)):
            if fn == "manifest.json":
                continue
            h_ = hashlib.sha256()
            with open(os.path.join(a.out, fn), "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h_.update(blk)
            files[fn] = {"sha256": h_.hexdigest(), "bytes": os.path.getsize(os.path.join(a.out, fn))}
        cfg_sha = hashlib.sha256(open(os.path.join(MODEL, "config.json"), "rb").read()).hexdigest()
        model_py_sha = hashlib.sha256(open(os.path.join(MODEL, "inference", "model.py"), "rb").read()).hexdigest()
        try:
            rev = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        except Exception:
            rev = ""
        json.dump({"tokens": len(ids), "layers": n_layers, "engram": layout is not None,
                   "swap_eps": a.swap_eps, "swap_cold_only": a.swap_cold_only, "swap_positions": a.swap_positions,
                   "swap_sixth_cold": a.swap_sixth_cold, "swap_frac": a.swap_frac, "swap_seed": a.swap_seed,
                   "swap_mode": a.swap_mode,
                   "swap_counts_per_layer": swap_counts,
                   "model_dir": MODEL, "config_sha256": cfg_sha, "reference_model_py_sha256": model_py_sha,
                   "oracle_rev": rev or os.environ.get("V41_ORACLE_REV", ""), "files": files},
                  open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
        print(f"manifest: {len(files)} files")
    top = torch.topk(logits[0].float(), 8)
    print("top-8 next tokens:")
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {v:8.3f}  {i:6d}  {tok.backend_tokenizer.decode([i])!r}")


if __name__ == "__main__":
    main()
