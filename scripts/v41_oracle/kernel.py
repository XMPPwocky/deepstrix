"""CPU replacements for DeepSeek's tilelang `kernel.py`, so the shipped
`inference/model.py` runs unmodified on this box as a reference oracle.

Every function here mirrors the tilelang kernel's *numerics* (block sizes,
scale rounding, clamps, the finite -1e30 softmax floor, the attn_sink term),
not its performance. Dequantise -> f32 -> torch op is the whole strategy.

Layouts (from inference/kernel.py + convert.py):
  fp8 weight  : [out, in] e4m3, scale [out/32, in/32] e8m0  (32x32 blocks)
  fp4 weight  : [out, in/2] packed e2m1 (low nibble = element 2i),
                scale [out, in/32] e8m0
  activations : fp8 per 32 along K with ue8m0 (power-of-2) scales
"""
import os
import torch
import torch.nn.functional as F

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# ---------------------------------------------------------------- dequant


_Q8_WEIGHTS = os.environ.get("V41_ORACLE_Q8") == "1"


def q8_0_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """ggml Q8_0 quantise+dequantise along the input dim (blocks of 32):
    d = amax/127 (f32), q = round-half-away(x/d), x' = q * f16(d). This is
    exactly what the engine holds for every fp8 projection (see
    v4flash-core hf_v41.rs), so a reference run with this on is the
    per-layer noise floor for the engine's oracle gates."""
    out, inn = w.shape
    b = w.float().view(out, inn // 32, 32)
    d = b.abs().amax(dim=-1, keepdim=True) / 127.0
    inv = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    x = b * inv
    q = torch.sign(x) * torch.floor(torch.abs(x) + 0.5)
    d16 = d.to(torch.float16).to(torch.float32)
    return (q * d16).view(out, inn)


def dequant_fp8(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """[out,in] e4m3 x [out/32,in/32] e8m0 -> f32 (Q8_0-roundtripped when V41_ORACLE_Q8=1)."""
    out, inn = w.shape
    wf = w.float().view(out // block, block, inn // block, block)
    sf = s.float().view(out // block, 1, inn // block, 1)
    wf = (wf * sf).view(out, inn)
    return q8_0_roundtrip(wf) if _Q8_WEIGHTS else wf


# 256-entry byte -> (low-nibble value, high-nibble value) table: one uint8
# gather instead of two int64 index gathers. Measured 5x faster (35 vs 177 ms
# per 2304x5120 expert matrix), bit-equal. torch 2.12 has no CPU
# float4_e2m1fn_x2 -> f32 conversion (copy_kernel NotImplementedError).
_BYTE_LUT = torch.stack(
    [FP4_TABLE[torch.arange(256) & 0x0F], FP4_TABLE[torch.arange(256) >> 4]], dim=-1
)


def dequant_fp4(w_packed: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """[out,in/2] packed e2m1 (as uint8/int8/float4_e2m1fn_x2) x [out,in/32] e8m0 -> f32."""
    u = w_packed.view(torch.uint8) if w_packed.dtype != torch.uint8 else w_packed
    out, half = u.shape
    wf = _BYTE_LUT[u.long()].view(out, half * 2)  # element 2i = low nibble
    inn = half * 2
    sf = s.float().view(out, inn // block, 1).expand(out, inn // block, block).reshape(out, inn)
    return wf * sf


# ---------------------------------------------------------- act quant (fake)


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(x)) EXACTLY — matches fast_round_scale (ue8m0, kernel.py:22-37).
    (Was exp2(ceil(log2(x))): f32 log2 rounds 2^k * (1 + 2^-23) to k once
    |k| >= 8, one scale too small. Identical on every bf16 input, which is all
    the reference ever feeds it; exact for f32 too now.) frexp: x = m * 2^e,
    m in [0.5, 1), so ceil(log2 x) = e - 1 exactly when m == 0.5, else e."""
    m, e = torch.frexp(x)
    e = torch.where(m == 0.5, e - 1, e)
    return torch.ldexp(torch.ones_like(x), e)


def _fake_fp8(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).float()


# V41_ORACLE_ATTN_O_FP8 -- the engine's V41_ATTN_O_FP8 switch, for the reference:
#   unset / "ref": follow model.py (the 1017 model.py rounds the attention output `o`
#                  through act_quant before wo_a, model.py:787 and :1073; older ones do not);
#   "0":           skip that rounding (compare against an engine run with the knob off);
#   "1":           require it: `attn_o_fp8_check()` raises when model.py never asked for it
#                  (a pre-1017 model.py cannot be forced: the call is model.py's, not ours).
# The `o` calls are the only IN-PLACE act_quant calls on a 4-D tensor
# (o.view(bsz, seqlen, n_groups, -1)); the window-KV calls are 3-D [bsz, seqlen, 512].
ATTN_O_FP8 = os.environ.get("V41_ORACLE_ATTN_O_FP8", "ref")
assert ATTN_O_FP8 in ("ref", "0", "1"), f"V41_ORACLE_ATTN_O_FP8={ATTN_O_FP8!r}: expected ref, 0 or 1"
ATTN_O_FP8_CALLS = 0


def attn_o_fp8_check():
    """Call at the end of a run: under V41_ORACLE_ATTN_O_FP8=1 the reference must
    have rounded `o` at least once."""
    if ATTN_O_FP8 == "1" and ATTN_O_FP8_CALLS == 0:
        raise RuntimeError("V41_ORACLE_ATTN_O_FP8=1 but model.py never rounded the attention "
                           "output (pre-1017 model.py? point V41_MODEL at the 1017 checkpoint)")
    if ATTN_O_FP8 != "ref":
        print(f"attn-o fp8: mode {ATTN_O_FP8}, {ATTN_O_FP8_CALLS} act_quant(o) calls "
              f"({'applied' if ATTN_O_FP8 == '1' else 'skipped'})")


def _attn_o_fp8_atexit():
    # every driver (oracle.py, oracle_generate.py, the dspark scripts) gets the check
    try:
        attn_o_fp8_check()
    except RuntimeError as e:
        print(f"ERROR: {e}", flush=True)
        os._exit(3)


if ATTN_O_FP8 != "ref":
    import atexit
    atexit.register(_attn_o_fp8_atexit)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise FP8 fake-quant along the last dim. With scale_fmt set the
    scale is a power of two (2^ceil(log2(amax/448))), amax floored at 1e-4.
    Returns (y, s) like the kernel; if inplace, writes the dequantised value
    back into x and returns x."""
    global ATTN_O_FP8_CALLS
    if inplace and x.dim() == 4:  # the attention output `o` (see ATTN_O_FP8)
        ATTN_O_FP8_CALLS += 1
        if ATTN_O_FP8 == "0":
            return x
    xf = x.float()
    n = xf.shape[-1]
    assert n % block_size == 0
    # V41_ORACLE_NOACTQ=1: leave fp8-Linear inputs unrounded (floor experiment: what
    # part of the engine's gap is the reference's fp8 activation rounding, which the
    # engine's int8 activations do not reproduce). In-place callers (window KV) unchanged.
    if not inplace and __import__("os").environ.get("V41_ORACLE_NOACTQ") == "1":
        return xf.to(x.dtype), torch.ones(*xf.shape[:-1], n // block_size, dtype=torch.float32)
    g = xf.view(*xf.shape[:-1], n // block_size, block_size)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil(amax / 448.0) if scale_fmt is not None else amax / 448.0
    q = _fake_fp8((g / s).clamp(-448.0, 448.0))
    if inplace:
        x.copy_((q * s).view_as(xf).to(x.dtype))
        return x
    y = q.view_as(xf).to(torch.float8_e4m3fn)
    return y, s.squeeze(-1).to(scale_dtype)


def _fake_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round to the e2m1 grid {0,.5,1,1.5,2,3,4,6} with round-to-nearest-even
    on ties, sign preserved. Values are pre-clamped to |x|<=6."""
    a = x.abs()
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    # nearest grid point; ties -> even index (RNE on the mantissa)
    d = (a.unsqueeze(-1) - grid).abs()
    idx = d.argmin(dim=-1)
    # tie handling: if two neighbours are equidistant, argmin picks the lower
    # index; RNE wants the even mantissa. e2m1 even-mantissa points are
    # 0,1,2,4 (idx 0,2,4,6). Fix the odd-index ties upward when the upper
    # neighbour is even and equally distant.
    up = (idx + 1).clamp_max(7)
    tie = (d.gather(-1, idx.unsqueeze(-1)).squeeze(-1) == d.gather(-1, up.unsqueeze(-1)).squeeze(-1)) & (idx % 2 == 1)
    idx = torch.where(tie, up, idx)
    return torch.copysign(grid[idx], x)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """FP4 fake-quant along the last dim. E8M0 scales (indexer q/k, block 32):
    s = 2^ceil(log2(amax/6)), amax >= 6*2^-126. E4M3 scales (compressed KV,
    block 16): s = e4m3(amax/6), amax >= 6*2^-9. Values clamp to +-6."""
    xf = x.float()
    n = xf.shape[-1]
    assert n % block_size == 0
    g = xf.view(*xf.shape[:-1], n // block_size, block_size)
    amax = g.abs().amax(dim=-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = _fake_fp8(amax.clamp_min(6.0 * 2.0**-9) / 6.0)
    else:
        s = _pow2_ceil(amax.clamp_min(6.0 * 2.0**-126) / 6.0)
    q = _fake_fp4((g / s).clamp(-6.0, 6.0))
    y = (q * s).view_as(xf)
    if inplace:
        x.copy_(y.to(x.dtype))
        return x
    return y.to(x.dtype), s.squeeze(-1)


# ------------------------------------------------------------------- gemm


def fp8_gemm(x_q, x_s, w, w_s, scale_dtype=None, block_size=32):
    """x: fp8 [.., K] with per-32 scales [.., K/32]; w: fp8 [N,K] with
    [N/32,K/32] e8m0. Dequantise both, f32 matmul."""
    xf = x_q.float().view(*x_q.shape[:-1], -1, block_size) * x_s.float().unsqueeze(-1)
    xf = xf.view(*x_q.shape)
    wf = dequant_fp8(w, w_s, block_size)
    # tilelang kernels allocate the output in torch.get_default_dtype() (bf16)
    return (xf @ wf.t()).to(torch.get_default_dtype())


def fp4_gemm(x_q, x_s, w_packed, w_s, scale_dtype=None, act_block_size=32):
    """x: fp8 activations (per-32 scales); w: packed fp4 [N,K/2] with [N,K/32]
    e8m0 scales. Dequantise both, f32 matmul."""
    xf = x_q.float().view(*x_q.shape[:-1], -1, act_block_size) * x_s.float().unsqueeze(-1)
    xf = xf.view(*x_q.shape)
    wf = dequant_fp4(w_packed, w_s, 32)
    return (xf @ wf.t()).to(torch.get_default_dtype())


# ---------------------------------------------------------- sparse attention


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b,m,h,d]; kv [b,n,d] (K==V latent); attn_sink [h]; topk_idxs [b,m,topk]
    int32 with -1 = empty. Online-softmax semantics: scores start at a finite
    -1e30 floor, sink adds exp(sink-max) to the denominator, rows with no valid
    index give zeros. Computed densely over the gathered set."""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    safe = idx.clamp_min(0)
    # gather [b,m,topk,d]
    g = kv.float().unsqueeze(1).expand(b, m, kv.shape[1], d)
    kvg = torch.gather(g, 2, safe.unsqueeze(-1).expand(b, m, idx.shape[-1], d))
    s = torch.einsum("bmhd,bmtd->bmht", q.float(), kvg) * softmax_scale
    s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = s.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    e = torch.exp(s - mx)
    e = e.masked_fill(~valid.unsqueeze(2), 0.0)
    denom = e.sum(dim=-1, keepdim=True) + torch.exp(attn_sink.float().view(1, 1, h, 1) - mx)
    o = torch.einsum("bmht,bmtd->bmhd", e / denom, kvg)
    return o.to(q.dtype)


# ------------------------------------------------------------ mHC sinkhorn


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """mixes [b,s,(2+hc)*hc] f32 -> pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc].
    Mirrors hc_split_sinkhorn_kernel exactly: sigmoid(+eps), 2*sigmoid,
    row-softmax(+eps) then column-normalise, then (iters-1) x (row, col)."""
    m = mixes.float()
    hc = hc_mult
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(*m.shape[:-1], hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
