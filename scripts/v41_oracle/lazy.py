"""The two parts of the reference model that cannot be instantiated on this
box, replaced with lazy equivalents that reproduce the reference math:

  * LazyMoE   — routed experts dequantised one at a time from the mmap'd
                shard (the reference builds 384 Expert modules = 6.7 GiB).
  * LazyEngram — the reference Engram with its 98 GB embedding table replaced
                by an mmap row-gather.

Plus `load_block` which fills a reference `Block` from the checkpoint.
"""
import torch
import torch.nn.functional as F

import model as ref  # DeepSeek's inference/model.py, unmodified
from kernel import act_quant, dequant_fp4, dequant_fp8
from loader import Checkpoint


# ----------------------------------------------------------------- MoE


class LazyMoE(torch.nn.Module):
    """Same forward contract as ref.MoE, with routed experts streamed."""

    def __init__(self, layer_id: int, args: ref.ModelArgs, ckpt: Checkpoint, prefix: str):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed, self.n_act = args.get_moe_config(layer_id)
        self.swiglu_limit = args.swiglu_limit
        self.ckpt = ckpt
        self.prefix = prefix  # e.g. "layers.3.ffn."
        self.gate = ref.Gate(layer_id, args)
        self.shared_experts = ref.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
        self.touched: dict[int, int] = {}  # expert -> tokens (stats)
        # oracle.py --expert-rt: an expert_rt.ExpertRT serving round-tripped weights for
        # the experts it covers (shared across layers). None = the checkpoint's weights.
        self.rt = None
        # oracle.py --imatrix-capture: {"x2": [E, dim], "h2": [E, inter], "n": [E]} sums of
        # squares of each expert's two GEMM inputs over its routed rows. None = off.
        self.imx = None

    def _w(self, e: int, which: str) -> torch.Tensor:
        p = f"{self.prefix}experts.{e}.{which}."
        return dequant_fp4(self.ckpt.get(p + "weight"), self.ckpt.get(p + "scale"))

    def _expert(self, e: int, x: torch.Tensor, w: torch.Tensor, W: dict | None = None,
                capture: bool = False) -> torch.Tensor:
        """ref.Expert.forward with dequant-on-demand weights. x [n,dim] bf16.
        W: replacement f32 weights {"w1","w3","w2"} (--expert-rt); capture: accumulate
        the imatrix sums for this expert (--imatrix-capture)."""
        dtype = x.dtype
        wt = (lambda which: W[which]) if W is not None else (lambda which: self._w(e, which))
        xq, xs = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        xf = (xq.float().view(*xq.shape[:-1], -1, 32) * xs.float().unsqueeze(-1)).view(*xq.shape)
        gate = xf @ wt("w1").t()
        up = xf @ wt("w3").t()
        if self.swiglu_limit > 0:
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
            gate = gate.clamp(max=self.swiglu_limit)
        h = (w * (F.silu(gate) * up)).to(dtype)
        hq, hs = act_quant(h, 32, "ue8m0", torch.float8_e8m0fnu)
        hf = (hq.float().view(*hq.shape[:-1], -1, 32) * hs.float().unsqueeze(-1)).view(*hq.shape)
        if capture:
            # the two GEMM inputs exactly as the experts see them (FP8-act-quantised; the
            # routing weight is already folded into h, so hf is w2's real input)
            self.imx["x2"][e] += xf.pow(2).sum(0)
            self.imx["h2"][e] += hf.pow(2).sum(0)
            self.imx["n"][e] += xf.shape[0]
        return hf @ wt("w2").t()

    def forward(self, x: torch.Tensor, image_mask=None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        self.last_indices = indices.detach().clone()  # [n, k] routed expert ids (dumped by oracle.py)
        y = torch.zeros_like(x, dtype=torch.float32)
        # Null-control hook (oracle.py --swap-mode bf16): rows whose rank-k expert's
        # contribution is rounded to bf16 instead of being swapped. None = off.
        rnd = getattr(self, "bf16_round", None)
        order = torch.unique(indices).tolist()
        if self.rt is not None:
            self.rt.begin_layer(self.layer_id, order)
        for e in order:
            idx, top = torch.where(indices == e)
            self.touched[e] = self.touched.get(e, 0) + idx.numel()
            W = None
            if self.rt is not None:
                self.rt.stats["calls"] += idx.numel()
                if self.rt.covers(self.layer_id, e):
                    W = self.rt.get(self.layer_id, e)
                    self.rt.stats["rt_calls"] += idx.numel()
            out = self._expert(e, x[idx], weights[idx, top, None], W=W, capture=self.imx is not None)
            if rnd is not None:
                rows, experts = rnd
                if rows.dim() == 2:  # [n, k] site mask over the picks (any-rank null)
                    m = (rows[idx] & (experts[idx] == e)).any(dim=1)
                else:  # [n] rows, one marked expert per row
                    m = rows[idx] & (experts[idx] == e)
                if m.any():
                    out = out.clone()
                    out[m] = out[m].to(torch.bfloat16).float()
            y[idx] += out
        y += self.shared_experts(x)
        # Swap-takes-effect check (oracle.py --swap-check-sites): recompute the sampled
        # rows' routed output with the ORIGINAL routing and record the difference.
        chk = getattr(self, "check", None)
        if chk is not None:
            for r in chk["rows"]:
                def routed(ids, ws):
                    acc = torch.zeros(self.dim, dtype=torch.float32)
                    for j, e in enumerate(ids.tolist()):
                        acc += self._expert(e, x[r:r + 1], ws[j:j + 1, None])[0].float()
                    return acc
                y_ref = routed(chk["idx_ref"][r], chk["w_ref"][r])
                y_new = routed(chk["idx_new"][r], chk["w_new"][r])
                chk["out"].append({
                    "layer": chk["layer"], "row": r,
                    "ids_ref": chk["idx_ref"][r].tolist(), "ids_used": chk["idx_new"][r].tolist(),
                    "w_ref": [round(v, 5) for v in chk["w_ref"][r].tolist()],
                    "w_used": [round(v, 5) for v in chk["w_new"][r].tolist()],
                    "ffn_norm": y_ref.norm().item(), "dffn_norm": (y_new - y_ref).norm().item(),
                })
            self.check = None
        return y.type_as(x).view(shape)


# --------------------------------------------------------------- Engram


class _MmapEngramTable(torch.nn.Module):
    """ref.ParallelEngramEmbedding.forward over an mmap'd fp8 table."""

    def __init__(self, ckpt: Checkpoint, name: str):
        super().__init__()
        self.w = ckpt.get(name + ".weight")  # [rows, 256] e4m3
        self.s = ckpt.get(name + ".scale")  # [rows, 8]  e8m0
        self.block = 32

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1).long()
        v = self.w[flat].float().unflatten(-1, (-1, self.block))
        s = self.s[flat].float().unsqueeze(-1)
        return (v * s).flatten(-2).to(torch.bfloat16).view(*indices.shape, -1)


class LazyEngram(ref.Engram):
    def __init__(self, args, layer_id, layout, ckpt: Checkpoint, prefix: str):
        torch.nn.Module.__init__(self)
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.clamp_value = 1e-6
        self.embed = _MmapEngramTable(ckpt, prefix + "embed")
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = ref.Linear(n_hash_cols * layout.head_dim, args.dim * (args.hc_mult + 1))
        self.eps = args.norm_eps
        self.q_weight = torch.nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = torch.nn.Parameter(torch.ones(args.hc_mult, args.dim))


# ------------------------------------------------------------ loading


def _assign(param: torch.Tensor, src: torch.Tensor, name: str):
    # The reference promotes a few bf16 checkpoint tensors to f32 modules
    # (ratio>1 compressor wkv/wgate, the output head). Allow exactly that.
    if param.dtype == torch.float32 and src.dtype == torch.bfloat16 and src.numel() == param.numel():
        param.data = src.float().reshape(param.shape)
        return
    if src.numel() * src.element_size() != param.numel() * param.element_size():
        raise ValueError(f"{name}: bytes {src.numel()*src.element_size()} != param {param.numel()*param.element_size()} (src {src.dtype}{tuple(src.shape)} vs {param.dtype}{tuple(param.shape)})")
    # packed fp4 arrives as int8 and must become the reference's float4_e2m1fn_x2
    if param.dtype == torch.float4_e2m1fn_x2:
        src = src.view(torch.float4_e2m1fn_x2)
    elif src.dtype != param.dtype:
        raise ValueError(f"{name}: dtype {src.dtype} vs param {param.dtype}")
    param.data = src.reshape(param.shape).clone()


def load_module(mod: torch.nn.Module, ckpt: Checkpoint, prefix: str, skip=()):
    """Fill every parameter of `mod` from `prefix + name`. Returns missing names.

    One conversion mirrors convert.py: a tensor stored fp8 + block scale whose
    module parameter is bf16 (wo_a) is dequantised here, block size inferred
    from the weight/scale shapes (convert.py accepts 32x32 or 128x128)."""
    missing = []
    for name, p in mod.named_parameters():
        if any(name.startswith(s) for s in skip):
            continue
        full = prefix + name
        if not ckpt.has(full):
            missing.append(full)
            continue
        src = ckpt.get(full)
        scale_name = full[: -len("weight")] + "scale" if full.endswith("weight") else None
        if (p.dtype == torch.bfloat16 and src.dtype == torch.float8_e4m3fn
                and scale_name and ckpt.has(scale_name)):
            sc = ckpt.get(scale_name)
            bo, bi = src.shape[0] // sc.shape[0], src.shape[1] // sc.shape[1]
            assert bo == bi and bo in (32, 128), (full, src.shape, sc.shape)
            p.data = dequant_fp8(src, sc, bo).to(torch.bfloat16)
            continue
        _assign(p, src, full)
    return missing
