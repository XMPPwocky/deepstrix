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

  nix-shell -p python3Packages.{torch,numpy,pillow,sympy,tokenizers} --run \\
    "python3 dspark_accept.py ~/.cache/deepstrix/v41/agentic/main ~/.cache/deepstrix/v41/agentic/tokens.json --seed 256"
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
import oracle  # noqa: E402  (sets sys.path for the reference package)
import model as ref  # noqa: E402
from lazy import LazyMoE, _assign, load_module  # noqa: E402
from loader import Checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("tokens")
    ap.add_argument("--seed", type=int, default=256, help="prefix length used to seed the drafter's window")
    ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=0, help="print decoded drafts for the first N steps")
    ap.add_argument("--variants", default="0,0", help="alignment experiment: ';'-separated 'hidden_shift,token_shift' pairs; "
                    "hidden_shift k uses the main hidden of position i+k for step i (and seeds the window the same way), "
                    "token_shift k feeds tok[i+1+k] as the sampled token")
    a = ap.parse_args()
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
    mh = torch.cat([torch.load(os.path.join(a.dump, f"layer_{l - 1:02d}_residual.pt")).mean(dim=2)
                    for l in args.dspark_target_layer_ids], dim=-1).to(torch.bfloat16)
    assert mh.shape[1] == T, (mh.shape, T)
    main_argmax = torch.load(os.path.join(a.dump, "main_argmax.pt")).tolist()
    tok = torch.tensor([ids])
    tokz = oracle._TokShim(os.path.join(oracle.MODEL, "tokenizer.json"))

    # Reference DSpark stages with a 1-expert MoE placeholder (replaced by LazyMoE).
    t0 = time.time()
    small = dataclasses.replace(args, dspark_n_routed_experts=1, dspark_n_activated_experts=1)
    stages = []
    for s in range(args.n_mtp_layers):
        blk = ref.DSparkBlock(args.n_layers + s, small)
        missing = load_module(blk, ckpt, f"mtp.{s}.", skip=("ffn", "embed", "head"))
        blk.ffn = LazyMoE(args.n_layers + s, args, ckpt, f"mtp.{s}.ffn.")
        missing += load_module(blk.ffn.gate, ckpt, f"mtp.{s}.ffn.gate.")
        missing += load_module(blk.ffn.shared_experts, ckpt, f"mtp.{s}.ffn.shared_experts.")
        if missing:
            raise SystemExit(f"mtp.{s}: missing tensors: {missing[:8]}")
        stages.append(blk)
    emb = ref.ParallelEmbedding(args.vocab_size, args.dim)
    _assign(emb.weight, ckpt.get("embed.weight"), "embed.weight")
    head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
    _assign(head.weight, ckpt.get("head.weight"), "head.weight")
    stages[0].embed = emb
    stages[-1].head = head
    # Cache dequantised drafter experts (128 × 3 stages × 3 matrices, bf16 ≈ 13 GB) so
    # the sweep is not dequant-bound after the first pass over the expert set.
    for blk in stages:
        cache = {}
        orig = blk.ffn._w
        def cached(e, which, _o=orig, _c=cache):
            k = (e, which)
            if k not in _c:
                _c[k] = _o(e, which).float()
            return _c[k]
        blk.ffn._w = cached
    print(f"drafter loaded in {time.time() - t0:.0f}s; T={T}, seed prefix {a.seed}", flush=True)

    def forward_spec(input_ids, main_hidden, start_pos):
        h, main_x = stages[0].forward_embed(main_hidden, input_ids)
        pre_mix = ref.make_identity_pre_mix(h, args.hc_mult)
        for layer in stages:
            h, pre_mix = layer(h, start_pos, pre_mix, main_x)
        if start_pos == 0:
            return None
        return stages[-1].forward_head(h, pre_mix, input_ids)

    # Speculative-sampling acceptance for the first draft position needs the
    # main model's full distribution: recompute it from the dumped head input
    # (norm(hc_pre(h))) with the tied head. acceptance_1 = Σ_x min(p_main, q_draft).
    head_in_path = os.path.join(a.dump, "head_input.pt")
    head_in = torch.load(head_in_path)[0].to(torch.bfloat16) if os.path.exists(head_in_path) else None
    ov = []  # per step: (overlap acceptance, entropy of main)

    K = args.dspark_block_size
    mh0, tok0_ = mh, tok
    variants = [tuple(int(v) for v in pair.split(",")) for pair in a.variants.split(";")]
    for (hs, ts) in variants:
      # shifted views: H(p) = mh0[p+hs] (edge-clamped), sampled token for step i = tok0[i+1+ts]
      idx = torch.arange(mh0.shape[1]).add(hs).clamp(0, mh0.shape[1] - 1)
      mh = mh0[:, idx]
      tok = tok0_
      if len(variants) > 1:
          print(f"\n##### variant hidden_shift={hs} token_shift={ts} #####", flush=True)
      ov.clear()
      with torch.inference_mode():
        # Seed the drafter's window with the prefix (positions 0..seed-1).
        forward_spec(tok[:, a.seed], mh[:, :a.seed], 0)
        acc = [0] * K
        teach = [0] * K
        prefix_ok = [0] * K
        conf_acc = [[] for _ in range(K)]
        n = 0
        on_dist = 0      # positions where main greedy == transcript token (text is "on-distribution")
        acc1_on = 0      # position-1 acceptance restricted to those
        top5_hit = 0     # main greedy within the drafter's top-5 at position 1
        t1 = time.time()
        for i in range(a.seed, min(T - K - 2 - abs(ts), a.seed + a.max_steps)):
            out_ids, _logits, conf = forward_spec(tok[:, i + 1 + ts], mh[:, i:i + 1], i)
            drafts = out_ids[0, 1:].tolist()
            top5 = _logits[0, 0].float().topk(5).indices.tolist()
            if head_in is not None:
                with torch.inference_mode():
                    # main distribution for position i+2 = head output at position i+1 ([1, V] → [V])
                    pm = torch.softmax(head(head_in[i + 1:i + 2], full_logits=True).float()[0], -1)
                qd = torch.softmax(_logits[0, 0].float(), -1)
                ov.append((torch.minimum(pm, qd).sum().item(), -(pm * torch.log(pm + 1e-30)).sum().item()))
            top5_hit += main_argmax[i + 1] in top5
            if main_argmax[i + 1] == ids[i + 2]:
                on_dist += 1
                acc1_on += drafts[0] == main_argmax[i + 1]
            if n < a.show:
                dec = lambda t: tokz.backend_tokenizer.decode(list(t))
                top5 = _logits[0, 0].float().topk(5).indices.tolist()
                print(f"  pos {i}: ctx ...{dec(ids[i - 6:i + 2])!r}\n"
                      f"      true next5 {dec(ids[i + 2:i + 7])!r} | main greedy {dec([main_argmax[i + 1]])!r}"
                      f" | drafts {dec(drafts)!r} | main in drafter top5: {main_argmax[i + 1] in top5} {[dec([t]) for t in top5]}", flush=True)
            ok_prefix = True
            for k in range(K):
                d = drafts[k]
                g = main_argmax[i + 1 + k]
                hit = d == g
                acc[k] += hit
                teach[k] += d == ids[i + 2 + k]
                ok_prefix = ok_prefix and hit
                prefix_ok[k] += ok_prefix
                conf_acc[k].append((float(conf[0, k]), hit))
            n += 1
            if n % 50 == 0:
                print(f"  step {n}: pos-1 acc {acc[0] / n:.3f}, {(time.time() - t1) / n:.2f} s/step", flush=True)
      print(f"\nsteps={n} (positions {a.seed}..{a.seed + n - 1}), {(time.time() - t1) / max(n, 1):.2f} s/step")
      print("draft pos  greedy-acc  teacher-match  prefix-acc(all ≤k)")
      for k in range(K):
          print(f"   {k + 1}       {acc[k] / n:.3f}       {teach[k] / n:.3f}          {prefix_ok[k] / n:.3f}")
      exp_tokens = [1 + sum(prefix_ok[j] / n for j in range(k + 1)) for k in range(K)]
      print("expected accepted tokens/step for K=1..5:", " ".join(f"{x:.2f}" for x in exp_tokens))
      print(f"position-1: main greedy in drafter top-5 {top5_hit / n:.3f}; on-distribution positions "
            f"(main greedy == text) {on_dist / n:.3f} of steps, acceptance there {acc1_on / max(on_dist, 1):.3f}")
  
      # Confidence calibration: AUC of confidence vs. acceptance per draft position.
      def auc(pairs):
          pos = sorted(c for c, h in pairs if h)
          neg = sorted(c for c, h in pairs if not h)
          if not pos or not neg:
              return float("nan")
          import bisect
          s = sum(bisect.bisect_left(neg, c) + 0.5 * (bisect.bisect_right(neg, c) - bisect.bisect_left(neg, c)) for c in pos)
          return s / (len(pos) * len(neg))
      if ov:
          accs = [o[0] for o in ov]
          ents = [o[1] for o in ov]
          lo = [o[0] for o in ov if o[1] < 1.0]
          hi = [o[0] for o in ov if o[1] >= 1.0]
          print(f"rejection-sampling acceptance, draft position 1 (T=1): mean {sum(accs) / len(accs):.3f} "
                f"| main entropy < 1 nat: {len(lo)} steps acc {sum(lo) / max(len(lo), 1):.3f} | ≥ 1 nat: {len(hi)} steps acc {sum(hi) / max(len(hi), 1):.3f}"
                f" | mean main entropy {sum(ents) / len(ents):.2f} nats")
      print("confidence AUC per draft position:", " ".join(f"{auc(conf_acc[k]):.3f}" for k in range(K)))
      res = dict(steps=n, greedy_acc=[x / n for x in acc], teacher_match=[x / n for x in teach],
                 prefix_acc=[x / n for x in prefix_ok], expected_tokens=exp_tokens,
                 conf_auc=[auc(conf_acc[k]) for k in range(K)],
                 rs_accept_pos1=[o[0] for o in ov], main_entropy=[o[1] for o in ov])
      json.dump(res, open(a.out or os.path.join(a.dump, "dspark_accept.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
