"""Localise the compressed-KV mismatch at layer 2: replay the reference chain
(latent pre-RoPE tap → RoPE at compressed positions → fp4_act_quant E4M3/16)
against the dumped cache, then diff the engine's rows block by block.

  nix-shell -p python3Packages.{torch,numpy,sympy,tokenizers,pillow} --run \\
    "python3 compress_kv_probe.py ~/.cache/deepstrix/v41/oracle_stages3_bins <engine dump dir> 2"
"""
import json, os, sys
import numpy as np
import torch
import oracle  # noqa
import model as ref  # noqa
import kernel as k  # noqa (the oracle's CPU shim)

bins, eng, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
man = json.load(open(os.path.join(bins, "manifest.json")))
def rows(tag):
    ts = sorted([t for t in man["tensors"] if t["tag"] == tag and t["layer"] == L], key=lambda t: t["token"])
    return np.stack([np.fromfile(os.path.join(bins, t["path"]), dtype=np.float32) for t in ts])
lat = torch.from_numpy(rows("stage_compressor0"))       # [n, 512] latent pre-RoPE (bf16 values)
ckv = torch.from_numpy(rows("stage_compress_kv0"))      # [n, 512] cache rows
wkv = torch.from_numpy(rows("stage_window_kv0"))        # [T, 512]
n = lat.shape[0]
cfg = json.load(open(os.path.join(oracle.MODEL, "inference", "config.json")))
args = oracle.make_args(cfg, max_seq_len=1024)
ratio = args.compress_ratios[L]
freqs = ref.precompute_freqs_cis(args.rope_head_dim, args.max_seq_len, args.original_seq_len, args.compress_rope_theta,
                                 args.rope_factor, args.beta_fast, args.beta_slow)
def replay(x_in, dtype):
    x = x_in.to(dtype).clone()
    y = x.unsqueeze(0)  # [1, n, 512]
    ref.apply_rotary_emb(y[..., -args.rope_head_dim:], freqs[: n * ratio : ratio])
    k.fp4_act_quant(y, 16, True, scale_dtype=torch.float8_e4m3fn)
    return y[0].float()
for dtype in (torch.bfloat16, torch.float32):
    r = replay(lat, dtype)
    eq = (r == ckv).float().mean().item()
    print(f"replay latent→rope→fp4 in {dtype}: bit-equal {eq*100:.1f}%  max|Δ| {(r-ckv).abs().max():.3e}")
p = os.path.join(eng, f"engine_compress_kv_L{L:02d}.bin")
if os.path.exists(p):
    e = torch.from_numpy(np.fromfile(p, dtype=np.float32)).view(-1, 512)[:n]
    d = (e != ckv)
    print(f"engine rows vs cache: bit-equal {(1-d.float().mean().item())*100:.1f}%")
    blk = d.view(n, 32, 16).any(-1)
    print(f"  blocks with any mismatch: {blk.sum().item()}/{n*32}; mismatched elements per bad block: "
          f"{[int(x) for x in d.view(n,32,16).sum(-1)[blk][:12].tolist()]}")
    # per bad block: compare the engine scale (from its values) vs the ref scale
    for r_ in range(n):
        for b in range(32):
            if blk[r_, b]:
                er, cr = e[r_, b*16:(b+1)*16], ckv[r_, b*16:(b+1)*16]
                lr = lat[r_, b*16:(b+1)*16]
                print(f"  row {r_} blk {b}: ref amax {cr.abs().max():.4f} eng amax {er.abs().max():.4f} latent amax {lr.abs().max():.4f} | ref {cr[:6].tolist()} | eng {er[:6].tolist()}")
                break
    # rope part vs nope part
    print(f"  mismatches in nope dims: {d[:, :448].sum().item()}, in rope dims: {d[:, 448:].sum().item()}")
p = os.path.join(eng, f"engine_window_kv_L{L:02d}.bin")
if os.path.exists(p):
    e = torch.from_numpy(np.fromfile(p, dtype=np.float32)).view(-1, 512)[: wkv.shape[0]]
    q = e.unsqueeze(0).clone(); k.act_quant(q, 32, "ue8m0", torch.float8_e8m0fnu, True)
    print(f"window: engine f16 rows vs cache bit-equal {(e==wkv).float().mean()*100:.1f}%; after fp8 fake-quant of the engine rows: {(q[0]==wkv).float().mean()*100:.1f}% (max|Δ| {(q[0]-wkv).abs().max():.3e})")

# Engine's last pooled row (pre-norm, pre-RoPE) → apply the reference norm → compare
# with the tapped latent (now cloned = pre-RoPE), then replay rope+fp4 and see whether
# the engine's own flips reproduce (i.e. the latent, not the quantiser, is the cause).
p = os.path.join(eng, f"engine_pooled_last_L{L:02d}.bin")
if os.path.exists(p) and lat.shape[0] >= 1:
    from loader import Checkpoint
    ck = Checkpoint(oracle.MODEL)
    w = ck.get(f"layers.{L}.attn.compressor.norm.weight").float()
    pooled = torch.from_numpy(np.fromfile(p, dtype=np.float32))
    def rms(x, w, eps): return x * torch.rsqrt(x.pow(2).mean() + eps) * w
    r_last = n - 1
    eng_lat_f32 = rms(pooled, w, args.norm_eps)
    eng_lat_bf = rms(pooled.to(torch.bfloat16).float(), w, args.norm_eps).to(torch.bfloat16).float()  # reference dtype path
    ref_lat = lat[r_last]
    for name, v in (("engine f32 latent", eng_lat_f32), ("engine latent w/ bf16 casts", eng_lat_bf)):
        d = (v - ref_lat).abs()
        print(f"{name} vs tapped latent row {r_last}: max|Δ| {d.max():.3e} (Δ/scale {d.max()/ref_lat.abs().max():.3e}), rel rms {(d.pow(2).mean().sqrt()/ref_lat.pow(2).mean().sqrt()):.3e}, bit-equal {(v==ref_lat).float().mean()*100:.1f}%")
    # replay rope+quant from each latent variant and compare to the cache row
    for name, v in (("tapped latent", ref_lat), ("engine f32 latent", eng_lat_f32), ("engine bf16-path latent", eng_lat_bf)):
        y = v.to(torch.bfloat16).clone().view(1, 1, 512)
        ref.apply_rotary_emb(y[..., -args.rope_head_dim:], freqs[r_last * ratio : r_last * ratio + 1])
        k.fp4_act_quant(y, 16, True, scale_dtype=torch.float8_e4m3fn)
        y = y[0, 0].float()
        print(f"  rope+fp4({name}) vs cache row {r_last}: bit-equal {(y==ckv[r_last]).float().mean()*100:.1f}%")
