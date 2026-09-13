"""Teacher-forced DSpark acceptance on a real transcript, through DeepSeek's
UNMODIFIED reference drafter (model.py DSparkBlock / forward_spec) with the
same lazy weight streaming as oracle.py.

Inputs: a full oracle run over the transcript with `--dump-argmax`
(residuals after every layer → main_hidden = cat(mean over hc copies of the
attention INPUT of layers 37/38/39 = the residual after layers 36/37/38); main_argmax[j] = the main model's greedy
token for position j+1) and the transcript's token ids.

Per step i (the main model has just processed position i): the drafter takes
the "sampled" token tok[i+1] and main_hidden[i], writes its window slot for
position i, and emits drafts d_1..d_5 for positions i+2..i+6. Greedy
acceptance of d_k = (d_k == main_argmax[i+k]); teacher match = (d_k == tok[i+1+k]).
Expected accepted tokens per step for K drafts = 1 + Σ_{k≤K} Π_{j≤k} p_j (prefix acceptance).

History (2026-09-13): the first version of this file loaded the stages with
`skip=("ffn", "embed", "head")`, a PREFIX match that also dropped `ffn_norm.weight`
(left at RMSNorm's init of 1.0 while the trained gains are 0.16 / 0.20 / 0.24) —
every drafter FFN saw an input 5-6x too large. That version measured 0.44 at
draft position 1. `--legacy-skip-ffn-norm` (or the `legacy` experiment) reproduces
it as a control. See docs/v41/DSPARK_ACCEPTANCE_INVESTIGATION.md.

Experiments (`--experiments a;b;c`, all run in one process on one weight load;
each writes <dump>/dspark_accept_<name>.json incl. per-step records):
  base        fixed loading, prefill-seeded ring, markov bias on, hidden/token shift 0
  legacy      base but ffn_norm gains forced to 1.0 (the old harness)
  noseed      base but the window ring left at zeros (no prefix seeding)
  incseed     base but the ring seeded by stepping the drafter's attention one
              position at a time over the last 128 prefix positions (decode path)
  nomarkov    base but the markov logit bias zeroed
  hs=k / ts=k alignment probes: main hidden of position i+k / sampled token tok[i+1+k]
  start=p     first step index (default = --seed; p = seed-1 mirrors oracle_generate --dspark)
Tokens may be combined with ',' e.g. "hs=1,ts=1" or "legacy,noseed".

  nix-shell -p python3Packages.{torch,numpy,pillow,sympy,tokenizers} --run \\
    "python3 dspark_accept.py ~/.cache/deepstrix/v41/agentic/gen2 ~/.cache/deepstrix/v41/agentic/gen2/tokens.json --seed 256 --experiments 'legacy;base'"
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from collections import OrderedDict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oracle  # noqa: E402  (sets sys.path for the reference package)
import model as ref  # noqa: E402
from lazy import LazyMoE, _assign, load_module  # noqa: E402
from loader import Checkpoint  # noqa: E402


def audit_stage(blk, ckpt, pfx):
    """Which checkpoint tensors under `pfx` no parameter consumed, and which directly
    assignable parameters differ from the checkpoint (i.e. were left at init)."""
    consumed, problems = set(), []
    for name, p in blk.named_parameters():
        if name.startswith(("ffn.experts.", "embed.", "head.")):  # streamed / tied to the top level
            continue
        full = pfx + name
        if not ckpt.has(full):
            problems.append(f"{full}: parameter has no checkpoint tensor")
            continue
        consumed.add(full)
        scale = full[: -len("weight")] + "scale" if full.endswith("weight") else None
        src = ckpt.get(full)
        if scale and ckpt.has(scale):
            consumed.add(scale)  # fp8 + scale, dequantised into a bf16 parameter
            if not torch.isfinite(p.data.float()).all():
                problems.append(f"{full}: non-finite after dequant")
            continue
        ref_val = src.float() if p.dtype == torch.float32 and src.dtype == torch.bfloat16 else src
        if ref_val.dtype == p.dtype and ref_val.numel() == p.numel():
            if not torch.equal(p.data.reshape(-1), ref_val.reshape(-1)):
                problems.append(f"{full}: parameter differs from checkpoint (left at init?)")
        else:
            problems.append(f"{full}: dtype/shape mismatch {src.dtype}{tuple(src.shape)} vs {p.dtype}{tuple(p.shape)}")
    unconsumed = [n for n in ckpt.names(pfx) if ".experts." not in n and n not in consumed]
    return problems, unconsumed


def load_drafter(args, ckpt, legacy_skip_ffn_norm=False, cache_gb=6.0):
    """The three DSpark stages with LazyMoE experts (bf16 LRU cache, lossless for MXFP4)."""
    small = dataclasses.replace(args, dspark_n_routed_experts=1, dspark_n_activated_experts=1)
    # The old skip=("ffn", ...) is a prefix match and also drops "ffn_norm.weight".
    skip = ("ffn", "embed", "head") if legacy_skip_ffn_norm else ("ffn.", "embed.", "head.")
    stages = []
    budget = cache_gb * (1 << 30)
    for s in range(args.n_mtp_layers):
        pfx = f"mtp.{s}."
        blk = ref.DSparkBlock(args.n_layers + s, small)
        missing = load_module(blk, ckpt, pfx, skip=skip)
        blk.ffn = LazyMoE(args.n_layers + s, args, ckpt, pfx + "ffn.")
        missing += load_module(blk.ffn.gate, ckpt, pfx + "ffn.gate.")
        missing += load_module(blk.ffn.shared_experts, ckpt, pfx + "ffn.shared_experts.")
        if missing:
            raise SystemExit(f"mtp.{s}: missing tensors: {missing[:8]}")
        cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        used = [0]
        orig = blk.ffn._w

        def cached(e, which, _o=orig, _c=cache, _u=used):
            k = (e, which)
            if k in _c:
                _c.move_to_end(k)
                return _c[k].float()
            w = _o(e, which).to(torch.bfloat16)  # E2M1 x 2^e is exact in bf16
            _c[k] = w
            _u[0] += w.numel() * 2
            while _u[0] > budget and _c:
                old = _c.pop(next(iter(_c)))
                _u[0] -= old.numel() * 2
            return w.float()

        blk.ffn._w = cached
        stages.append(blk)
    emb = ref.ParallelEmbedding(args.vocab_size, args.dim)
    _assign(emb.weight, ckpt.get("embed.weight"), "embed.weight")
    head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
    _assign(head.weight, ckpt.get("head.weight"), "head.weight")
    stages[0].embed = emb
    stages[-1].head = head
    return stages, head


def parse_experiment(spec, seed):
    e = dict(name=spec, hs=0, ts=0, seed=True, incseed=False, markov=True, ffn_norm=True, start=seed)
    for tok in spec.split(","):
        tok = tok.strip()
        if tok in ("", "base"):
            continue
        elif tok == "legacy":
            e["ffn_norm"] = False
        elif tok == "noseed":
            e["seed"] = False
        elif tok == "incseed":
            e["incseed"] = True
        elif tok == "nomarkov":
            e["markov"] = False
        elif tok.startswith("hs="):
            e["hs"] = int(tok[3:])
        elif tok.startswith("ts="):
            e["ts"] = int(tok[3:])
        elif tok.startswith("start="):
            e["start"] = int(tok[6:])
        else:
            raise SystemExit(f"unknown experiment token {tok!r}")
    e["name"] = spec.replace(",", "_").replace("=", "")
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("tokens")
    ap.add_argument("--seed", type=int, default=256, help="prefix length used to seed the drafter's window")
    ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--out", default=None, help="JSON path (single experiment) — default <dump>/dspark_accept_<name>.json")
    ap.add_argument("--show", type=int, default=0, help="print decoded drafts for the first N steps")
    ap.add_argument("--experiments", default="base", help="';'-separated experiment specs, see module docstring")
    ap.add_argument("--variants", default=None, help="(old spelling) ';'-separated 'hidden_shift,token_shift' pairs")
    ap.add_argument("--legacy-skip-ffn-norm", action="store_true",
                    help="load the stages exactly as the first harness did (ffn_norm.weight left at 1.0)")
    ap.add_argument("--cache-gb", type=float, default=float(os.environ.get("V41_DSPARK_CACHE_GB", "6")))
    ap.add_argument("--check-ring", action="store_true",
                    help="verify prefill seeding == incremental (decode-path) seeding of the window ring")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    ids = json.load(open(a.tokens))
    T = len(ids)
    ref_cfg = json.load(open(os.path.join(oracle.MODEL, "inference", "config.json")))
    args = oracle.make_args(ref_cfg, max_seq_len=((T + 127) // 128) * 128 + 128)
    args = dataclasses.replace(args, temperature=0.0)  # greedy drafts
    ref.default_dtype = torch.bfloat16
    torch.set_default_dtype(torch.bfloat16)
    ckpt = Checkpoint(oracle.MODEL)

    # main_hidden [1, T, 3*dim]: the drafter reads the attention INPUT of its target
    # layers (model.py: "the MTP head reads the attention input of its target layers,
    # not their output") = the residual after layer l-1, mean over hc copies.
    mh0 = torch.cat([torch.load(os.path.join(a.dump, f"layer_{l - 1:02d}_residual.pt")).mean(dim=2)
                     for l in args.dspark_target_layer_ids], dim=-1).to(torch.bfloat16)
    assert mh0.shape[1] == T, (mh0.shape, T)
    main_argmax = torch.load(os.path.join(a.dump, "main_argmax.pt")).tolist()
    tok = torch.tensor([ids])
    tokz = oracle._TokShim(os.path.join(oracle.MODEL, "tokenizer.json"))

    t0 = time.time()
    stages, head = load_drafter(args, ckpt, a.legacy_skip_ffn_norm, a.cache_gb)
    for s, blk in enumerate(stages):
        problems, unconsumed = audit_stage(blk, ckpt, f"mtp.{s}.")
        print(f"load audit mtp.{s}: {len(problems)} parameter problems, {len(unconsumed)} unconsumed checkpoint tensors")
        for p in problems:
            print("   PROBLEM", p)
        for u in unconsumed:
            print("   UNCONSUMED", u)
        w = blk.ffn_norm.weight.float()
        print(f"   ffn_norm gain mean {w.mean():.3f} (checkpoint {ckpt.get(f'mtp.{s}.ffn_norm.weight').float().mean():.3f})")
    print(f"drafter loaded in {time.time() - t0:.0f}s; T={T}, seed prefix {a.seed}", flush=True)
    real_ffn_norm = [blk.ffn_norm.weight.data.clone() for blk in stages]
    real_markov_fwd = stages[-1].markov_head.forward

    def forward_spec(input_ids, main_hidden, start_pos):
        h, main_x = stages[0].forward_embed(main_hidden, input_ids)
        pre_mix = ref.make_identity_pre_mix(h, args.hc_mult)
        for layer in stages:
            h, pre_mix = layer(h, start_pos, pre_mix, main_x)
        if start_pos == 0:
            return None
        return stages[-1].forward_head(h, pre_mix, input_ids)

    def reset_ring():
        for blk in stages:
            blk.attn.window_kv_cache.zero_()

    def seed_incremental(mh, upto):
        """Decode-path seeding: write positions max(0, upto-128)..upto-1 one at a time through
        DSparkAttention.forward (start_pos > 0), the way the reference decode loop does."""
        main_x = stages[0].main_norm(stages[0].main_proj(mh[:, :upto]))
        dummy = torch.zeros(1, args.dspark_block_size, args.dim)
        for p in range(max(1, upto - args.window_size), upto):
            for blk in stages:
                blk.attn(dummy, p, main_x[:, p:p + 1])

    if a.check_ring:
        with torch.inference_mode():
            reset_ring()
            forward_spec(tok[:, a.seed], mh0[:, :a.seed], 0)
            ring_prefill = [blk.attn.window_kv_cache.clone() for blk in stages]
            reset_ring()
            seed_incremental(mh0, a.seed)
            ring_inc = [blk.attn.window_kv_cache.clone() for blk in stages]
        for s, (rp, ri) in enumerate(zip(ring_prefill, ring_inc)):
            d = (rp.float() - ri.float()).abs()
            print(f"ring check mtp.{s}: prefill vs incremental max|Δ| {d.max():.3e}, rows differing {(d.amax(-1) > 0).sum().item()}/{rp.shape[1]}, "
                  f"ring rms {rp.float().pow(2).mean().sqrt():.3f}, zero rows {(rp.float().abs().amax(-1) == 0).sum().item()}")

    # Speculative-sampling acceptance needs the main model's full distribution: recompute it
    # from the dumped head input (norm(hc_pre(h))) with the tied head. acceptance = Σ_x min(p_main, q_draft).
    head_in_path = os.path.join(a.dump, "head_input.pt")
    head_in = torch.load(head_in_path)[0].to(torch.bfloat16) if os.path.exists(head_in_path) else None

    K = args.dspark_block_size
    specs = a.experiments.split(";")
    if a.variants:
        specs = [f"hs={p.split(',')[0]},ts={p.split(',')[1]}" for p in a.variants.split(";")]
    experiments = [parse_experiment(s, a.seed) for s in specs]
    summary = {}
    for ex in experiments:
        hs, ts = ex["hs"], ex["ts"]
        # shifted views: H(p) = mh0[p+hs] (edge-clamped), sampled token for step i = tok0[i+1+ts]
        idx = torch.arange(mh0.shape[1]).add(hs).clamp(0, mh0.shape[1] - 1)
        mh = mh0[:, idx]
        for blk, w in zip(stages, real_ffn_norm):
            blk.ffn_norm.weight.data = w.clone() if ex["ffn_norm"] else torch.ones_like(w)
        if ex["markov"]:
            stages[-1].markov_head.forward = real_markov_fwd
        else:
            def _nomarkov(token_ids, _f=real_markov_fwd):
                lb, emb = _f(token_ids)
                return torch.zeros_like(lb), emb
            stages[-1].markov_head.forward = _nomarkov
        print(f"\n##### experiment {ex['name']}: {ex} #####", flush=True)
        steps = []
        with torch.inference_mode():
            reset_ring()
            if ex["incseed"]:
                seed_incremental(mh, ex["start"])
            elif ex["seed"]:
                forward_spec(tok[:, ex["start"]], mh[:, :ex["start"]], 0)
            acc = [0] * K
            teach = [0] * K
            prefix_ok = [0] * K
            n = 0
            t1 = time.time()
            for i in range(ex["start"], min(T - K - 2 - abs(ts), ex["start"] + a.max_steps)):
                out_ids, _logits, conf = forward_spec(tok[:, i + 1 + ts], mh[:, i:i + 1], i)
                drafts = out_ids[0, 1:].tolist()
                top5 = _logits[0, 0].float().topk(5).indices.tolist()
                rec = dict(i=i, drafts=drafts, greedy=[main_argmax[i + 1 + k] for k in range(K)],
                           text=[ids[i + 2 + k] for k in range(K)], conf=[float(conf[0, k]) for k in range(K)],
                           top5_hit=main_argmax[i + 1] in top5)
                if head_in is not None:
                    # main distribution for position i+2+k = head output at position i+1+k; the text is
                    # the greedy continuation, so for k>0 this is the main's distribution given the
                    # greedy prefix (= given drafts 1..k accepted, when they were)
                    rs, ent = [], []
                    for k in range(K):
                        pm = torch.softmax(head(head_in[i + 1 + k:i + 2 + k], full_logits=True).float()[0], -1)
                        qd = torch.softmax(_logits[0, k].float(), -1)
                        rs.append(torch.minimum(pm, qd).sum().item())
                        ent.append(-(pm * torch.log(pm + 1e-30)).sum().item())
                    rec["rs"] = rs
                    rec["ent"] = ent
                if n < a.show:
                    dec = lambda t: tokz.backend_tokenizer.decode(list(t))
                    print(f"  pos {i}: ctx ...{dec(ids[i - 6:i + 2])!r}\n"
                          f"      true next5 {dec(ids[i + 2:i + 7])!r} | main greedy {dec([main_argmax[i + 1]])!r}"
                          f" | drafts {dec(drafts)!r} | main in drafter top5: {rec['top5_hit']} {[dec([t]) for t in top5]}", flush=True)
                ok_prefix = True
                hits = []
                for k in range(K):
                    hit = drafts[k] == main_argmax[i + 1 + k]
                    hits.append(hit)
                    acc[k] += hit
                    teach[k] += drafts[k] == ids[i + 2 + k]
                    ok_prefix = ok_prefix and hit
                    prefix_ok[k] += ok_prefix
                rec["hit"] = hits
                steps.append(rec)
                n += 1
                if n % 10 == 0:
                    print(f"  step {n} (pos {i}): pos-1 acc {acc[0] / n:.3f}, {(time.time() - t1) / n:.2f} s/step", flush=True)
        print(f"\nsteps={n} (positions {ex['start']}..{ex['start'] + n - 1}), {(time.time() - t1) / max(n, 1):.2f} s/step")
        print("draft pos  greedy-acc  teacher-match  prefix-acc(all ≤k)  conditional p(k | ≤k-1 ok)")
        for k in range(K):
            cond = prefix_ok[k] / prefix_ok[k - 1] if k and prefix_ok[k - 1] else (acc[0] / n if not k else float("nan"))
            print(f"   {k + 1}       {acc[k] / n:.3f}       {teach[k] / n:.3f}          {prefix_ok[k] / n:.3f}            {cond:.3f}")
        exp_tokens = [1 + sum(prefix_ok[j] / n for j in range(k + 1)) for k in range(K)]
        print("expected accepted tokens/step for K=1..5:", " ".join(f"{x:.2f}" for x in exp_tokens))
        p1 = acc[0] / n
        se = (p1 * (1 - p1) / max(n, 1)) ** 0.5
        print(f"position-1: {p1:.3f} ± {se:.3f} (1σ binomial); main greedy in drafter top-5 {sum(r['top5_hit'] for r in steps) / n:.3f}")
        # trajectory: acceptance by quarter of the run (does it climb?)
        q = max(1, n // 4)
        print("pos-1 acceptance by quarter:", " ".join(f"{sum(r['hit'][0] for r in steps[j:j + q]) / max(1, len(steps[j:j + q])):.2f}" for j in range(0, n, q)))

        def auc(pairs):
            pos = sorted(c for c, h in pairs if h)
            neg = sorted(c for c, h in pairs if not h)
            if not pos or not neg:
                return float("nan")
            import bisect
            s = sum(bisect.bisect_left(neg, c) + 0.5 * (bisect.bisect_right(neg, c) - bisect.bisect_left(neg, c)) for c in pos)
            return s / (len(pos) * len(neg))
        conf_auc = [auc([(r["conf"][k], r["hit"][k]) for r in steps]) for k in range(K)]
        print("confidence AUC per draft position:", " ".join(f"{x:.3f}" for x in conf_auc))
        res = dict(experiment=ex, steps=n, greedy_acc=[x / n for x in acc], teacher_match=[x / n for x in teach],
                   prefix_acc=[x / n for x in prefix_ok], expected_tokens=exp_tokens, conf_auc=conf_auc,
                   records=steps)
        if head_in is not None:
            rs1 = [r["rs"][0] for r in steps]
            lo = [r["rs"][0] for r in steps if r["ent"][0] < 1.0]
            hi = [r["rs"][0] for r in steps if r["ent"][0] >= 1.0]
            print(f"rejection-sampling acceptance (T=1), draft position 1: mean {sum(rs1) / n:.3f} | main entropy < 1 nat: "
                  f"{len(lo)} steps acc {sum(lo) / max(len(lo), 1):.3f} | ≥ 1 nat: {len(hi)} steps acc {sum(hi) / max(len(hi), 1):.3f}"
                  f" | mean main entropy {sum(r['ent'][0] for r in steps) / n:.2f} nats")
            # RS chain: E[accepted] = Σ_k mean_i Π_{j≤k} rs_ij (the text is the greedy continuation,
            # so later positions are the main's distributions given the greedy prefix)
            chain = []
            for r in steps:
                p, c = 1.0, []
                for k in range(K):
                    p *= r["rs"][k]
                    c.append(p)
                chain.append(c)
            rs_prefix = [sum(c[k] for c in chain) / n for k in range(K)]
            rs_exp = [1 + sum(rs_prefix[:k + 1]) for k in range(K)]
            print("RS per-position mean acceptance:", " ".join(f"{sum(r['rs'][k] for r in steps) / n:.3f}" for k in range(K)))
            print("RS expected accepted tokens/step for K=1..5:", " ".join(f"{x:.2f}" for x in rs_exp))
            res.update(rs_mean=[sum(r["rs"][k] for r in steps) / n for k in range(K)], rs_prefix=rs_prefix, rs_expected_tokens=rs_exp)
        out = a.out if (a.out and len(experiments) == 1) else os.path.join(a.dump, f"dspark_accept_{ex['name']}.json")
        json.dump(res, open(out, "w"), indent=1)
        summary[ex["name"]] = dict(steps=n, pos1=p1, greedy_acc=res["greedy_acc"], expected_tokens=exp_tokens)
        print("wrote", out, flush=True)
    print("\n===== summary =====")
    for name, s in summary.items():
        print(f"{name:24s} steps {s['steps']:4d}  greedy-acc " + " ".join(f"{x:.3f}" for x in s["greedy_acc"])
              + "  E[tok/step] K=1..5 " + " ".join(f"{x:.2f}" for x in s["expected_tokens"]))


if __name__ == "__main__":
    main()
