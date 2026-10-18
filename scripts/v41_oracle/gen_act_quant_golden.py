"""Golden vectors for the CPU emulation of the 1017 reference's act_quant
(crates/v4flash-kernels/src/act_quant_fp8.rs, test `matches_torch_golden_vectors`).

The reference kernel (inference/kernel.py act_quant_kernel, round_scale=True,
inplace=True) re-implemented in torch ops, step for step:
  kernel.py:72-74  x bf16 -> f32 compute, amax = reduce_absmax over the 32-block
  kernel.py:76     amax = max(amax, f32(1e-4))
  kernel.py:78     s = fast_round_scale(amax, f32(1/448)):
  kernel.py:22-27    t = amax * f32(1/448); e = exp_field(t) - 127 + (mantissa(t) != 0)
  kernel.py:30-33    s = f32 bits (e + 127) << 23
  kernel.py:83-86  y = bf16(f32(e4m3(clamp(x / s, -448, 448))) * s)
  kernel.py:91     S = e8m0(s)
and cross-checked against the oracle's shim (./kernel.py act_quant), which the
CPU reference runs. Writes, little-endian, into OUT (default the test dir):
  x_bf16.bin (u16 bits), y_bf16.bin (u16 bits), codes_e4m3.bin (u8), scales_e8m0.bin (u8).

Run:  nix-shell -p python3Packages.torch --run \
        "python3 scripts/v41_oracle/gen_act_quant_golden.py [OUT]"
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel as shim  # noqa: E402  (the oracle's CPU act_quant)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "..", "crates", "v4flash-kernels", "tests", "data", "act_quant_fp8")

F32 = torch.float32


def act_quant_ref(x: torch.Tensor):
    """x: [nblk, 32] bf16 -> (y bf16 [nblk, 32], codes e4m3 [nblk, 32], scales e8m0 [nblk])."""
    assert x.dtype == torch.bfloat16 and x.shape[-1] == 32
    xf = x.to(F32)
    amax = xf.abs().amax(dim=-1)
    amax = torch.maximum(amax, torch.tensor(1e-4, dtype=F32))
    t = amax * torch.tensor(1.0 / 448.0, dtype=F32)
    bits = t.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    s = ((e + 127) << 23).to(torch.int32).view(F32)
    q = torch.clamp(xf / s[:, None], -448.0, 448.0)
    codes = q.to(torch.float8_e4m3fn)
    y = (codes.to(F32) * s[:, None]).to(torch.bfloat16)
    scales = s.to(torch.float8_e8m0fnu)
    return y, codes, scales


def bf16(v):
    return torch.tensor(v, dtype=F32).to(torch.bfloat16)


def blocks():
    g = torch.Generator().manual_seed(1017)
    out = []
    # random gaussian blocks over 13 decades, a few sign mixes
    for dec in range(-9, 5):
        for _ in range(16):
            out.append((torch.randn(32, generator=g) * 10.0 ** dec).to(torch.bfloat16))
    # heavy-tailed: one outlier per block
    for _ in range(48):
        b = torch.randn(32, generator=g) * 0.05
        b[torch.randint(0, 32, (1,), generator=g)] = torch.randn(1, generator=g) * 40
        out.append(b.to(torch.bfloat16))
    z = torch.zeros(32)
    out.append(z.to(torch.bfloat16))                                   # all zero
    out.append((-z).to(torch.bfloat16))                                # all -0
    out.append(torch.full((32,), 5e-5).to(torch.bfloat16))             # below the 1e-4 floor
    sub = torch.arange(32, dtype=F32) * 2.0 ** -31                     # floor scale 2^-22: subnormal grid
    out.append(sub.to(torch.bfloat16))
    out.append((sub * 0.5 + 2.0 ** -32).to(torch.bfloat16))           # odd half-steps: ties to even
    # amax exactly at, one bf16 ulp above and below 448 * 2^k
    for k in (-20, -7, -1, 0, 1, 6, 30):
        for d in (0, 1, -1):
            a = bf16(448.0 * 2.0 ** k)
            a = (a.view(torch.int16) + d).view(torch.bfloat16)
            b = (torch.randn(32, generator=g) * 448.0 * 2.0 ** k * 0.3).to(torch.bfloat16)
            b[0] = a
            b[1] = -bf16(1.0 * 2.0 ** k)
            out.append(b)
    # every e4m3 midpoint at scale 1 (amax 448 pins s = 1), signed
    vals = torch.arange(0, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).to(F32)
    mids = (vals[:-1] + vals[1:]) / 2
    seq = torch.cat([mids, -mids, vals, -vals])
    for i in range(0, len(seq), 31):
        b = torch.zeros(32)
        chunk = seq[i:i + 31]
        b[:len(chunk)] = chunk
        b[31] = 448.0
        out.append(b.to(torch.bfloat16))
    # midpoints at a small scale (2^-12) and at a large one (2^20)
    for k in (-12, 20):
        for i in range(0, len(mids), 31):
            b = torch.zeros(32)
            chunk = mids[i:i + 31] * 2.0 ** k
            b[:len(chunk)] = chunk
            b[31] = -448.0 * 2.0 ** k
            out.append(b.to(torch.bfloat16))
    # mixed magnitudes inside one block
    for _ in range(16):
        b = torch.randn(32, generator=g) * torch.pow(10.0, torch.randint(-8, 3, (32,), generator=g).to(F32))
        out.append(b.to(torch.bfloat16))
    # near bf16 max
    b = torch.randn(32, generator=g) * 1e37
    b[0] = 3.0e38
    out.append(b.to(torch.bfloat16))
    return torch.stack(out)


def main():
    x = blocks()
    assert not torch.isnan(x.to(F32)).any() and torch.isfinite(x.to(F32)).all()
    y, codes, scales = act_quant_ref(x)
    # the oracle shim must agree bit for bit (it is what the CPU reference runs)
    xs = x.clone().view(1, -1, 32)                      # 3-D: the shim's window-KV-shaped in-place call
    shim.act_quant(xs, 32, "ue8m0", torch.float8_e8m0fnu, True)
    assert torch.equal(xs.view(-1, 32).view(torch.int16), y.view(torch.int16)), "shim act_quant (in place) differs"
    yq, sq = shim.act_quant(x.clone().view(1, -1, 32), 32, "ue8m0", torch.float8_e8m0fnu, False)
    assert torch.equal(yq.view(-1, 32).view(torch.uint8), codes.view(torch.uint8)), "shim act_quant codes differ"
    assert torch.equal(sq.view(-1).view(torch.uint8), scales.view(torch.uint8)), "shim act_quant scales differ"
    # the bf16 output is the exact product (no rounding at the bf16 store)
    exact = codes.to(torch.float64) * scales.to(F32).to(torch.float64)[:, None]
    assert torch.equal(y.to(torch.float64), exact), "bf16 store rounded"
    os.makedirs(OUT, exist_ok=True)
    for name, t in (("x_bf16.bin", x.view(torch.int16)), ("y_bf16.bin", y.view(torch.int16)),
                    ("codes_e4m3.bin", codes.view(torch.uint8)), ("scales_e8m0.bin", scales.view(torch.uint8))):
        b = bytes(t.contiguous().view(torch.uint8).flatten().tolist())
        open(os.path.join(OUT, name), "wb").write(b)
    # f32 inputs (the engine's attention output): the reference's act_quant input is their bf16
    # rounding (torch's RNE cast) -- random f32 over many decades plus the two cases where skipping
    # the bf16 point changes the result: just past an e4m3 midpoint, and amax in (448, 449] * 2^k
    g = torch.Generator().manual_seed(18)
    xf = [torch.randn(32, generator=g) * 10.0 ** d for d in range(-8, 4) for _ in range(8)]
    vals = torch.arange(0, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).to(F32)
    mids = (vals[:-1] + vals[1:]) / 2
    for k in (-6, 0, 5):
        for i in range(0, len(mids), 31):
            b = torch.zeros(32)
            ch = mids[i:i + 31] * 2.0 ** k
            nudge = torch.where(torch.arange(len(ch)) % 2 == 0, 1.0, -1.0) * ch.abs() * 2.0 ** -12
            b[:len(ch)] = ch + nudge
            b[31] = 448.0 * 2.0 ** k
            xf.append(b)
    for k in (-12, -3, 0, 2, 9):
        b = torch.randn(32, generator=g) * 2.0 ** k
        b[0] = 448.5 * 2.0 ** k
        xf.append(b)
        b = torch.randn(32, generator=g) * 2.0 ** k
        b[5] = -448.0 * 2.0 ** k * (1 + 2.0 ** -20)
        xf.append(b)
    xf = torch.stack(xf).to(F32)
    yf, codes_f, scales_f = act_quant_ref(xf.to(torch.bfloat16))
    for name, t in (("x_f32.bin", xf.view(torch.int32)), ("codes_bf16in_e4m3.bin", codes_f.view(torch.uint8)),
                    ("scales_bf16in_e8m0.bin", scales_f.view(torch.uint8))):
        b = bytes(t.contiguous().view(torch.uint8).flatten().tolist())
        open(os.path.join(OUT, name), "wb").write(b)
    print(f"{x.shape[0]} bf16 blocks ({x.numel()} values), {xf.shape[0]} f32 blocks -> {OUT}")


if __name__ == "__main__":
    main()
