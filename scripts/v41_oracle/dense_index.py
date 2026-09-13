"""Force DeepSeek's CSA2 indexer selection OFF, i.e. make every layer attend over its
*whole* compressed store instead of the indexer's top-`index_topk` — which is what
deepstrix's V4.1 attention does today (it never ported the indexer).

Nothing in `inference/model.py` is edited. Two module globals of the reference are
rebound instead:

  * `Attention._compress_topk_idxs` — the one method that decides which compressed
    rows a query may read. The patched version still runs the real `Indexer` (so the
    index-key cache, the candidate blocks and the shared-selection plumbing behave
    exactly as shipped) and then, per batch row, keeps either the indexer's top-k
    (mode "sparse" = reference) or every causally visible row (mode "dense" = ours).

  * `sparse_attn` — the CPU shim's gather formulation materialises [b, m, topk, d],
    which is >7 GB once topk is the whole store, and gathering through an expanded
    source is also ~50x below the machine's GEMM rate. The replacement scores q
    against the KV with one wide GEMM per query chunk and masks: the compressed half
    against the whole store, the window half against the (chunk + window)-row band the
    chunk can reach. Agrees with the shim's gather output to ~1e-4 relative (1 bf16
    ULP); the difference is f32 accumulation order.

Both modes are expressed in the SAME width (one slot per compressed row, -1 where the
row is not selected), so the only difference between batch row 0 and row 1 is which
slots are -1. sparse_attn treats every slot independently, so that representation is
equivalent to the reference's compacted top-k list.

The driver is `indexer_ab.py`; `install(row_modes)` must be called before any Block is
built, since it rebinds unbound methods on `ref.Attention`.

Usage:
    import dense_index
    dense_index.install(row_modes=["sparse", "dense", "sparse"])   # row 2 = noise control
"""
import os

import torch

import model as ref

# Which selection each batch row gets: "sparse" = the reference indexer's top-k,
# "dense" = every causally visible compressed row.
ROW_MODES: list[str] = ["sparse"]

# Per index-source call: (layer_id, compress_ratio, store_rows, mean_visible, mean_selected_sparse)
STATS: list[tuple] = []

# Attention-mass probe: for every layer, how much of each query's attention probability
# the dense rows put on compressed rows the indexer would have DROPPED. Deterministic and
# chaos-free: it is read off the dense row's own softmax. MASS[layer] = list per batch row
# of (dropped_mass[m], compressed_mass[m]) averaged over heads.
_PROBE: dict | None = None
MASS: dict[int, list] = {}
_LAYER = [-1]

_orig_compress_topk_idxs = ref.Attention._compress_topk_idxs
_orig_window_kv = ref.Attention._window_kv
_orig_sparse_attn = ref.sparse_attn

# How the current layer split `kv` / `topk_idxs` between the sliding window and the
# compressed store, recorded by the `_window_kv` wrapper: (kv rows, idx columns).
# `sparse_attn` needs it because the two halves want different formulations — the
# window is a narrow per-query gather (128 slots out of `seqlen` rows in prefill),
# the compressed half is a wide mask over the whole store.
_WIN = [0, 0]


def _visible_lens(seqlen: int, ratio: int, start_pos: int) -> torch.Tensor:
    """[seqlen, 1] — how many compressed rows each query in this chunk may read.
    Copied from Indexer.forward so the dense mask is exactly the reference's own
    reachability rule."""
    if start_pos == 0:
        return (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
    return torch.full((seqlen, 1), (start_pos + seqlen) // ratio)


def _window_kv(self, x, freqs_cis, start_pos):
    kv, idxs = _orig_window_kv(self, x, freqs_cis, start_pos)
    _WIN[0], _WIN[1] = kv.shape[1], idxs.shape[-1]
    return kv, idxs


def _compress_topk_idxs(self, x, qr, latent, start_pos, offset, compress_len):
    if not self.is_index_source:
        return ref.shared_attn.topk_idxs
    bsz, seqlen, _ = x.size()
    if compress_len == 0:
        idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        ref.shared_attn.topk_idxs = idxs
        return idxs

    assert self.indexer is not None
    if self.indexer.freqs_cis is None:
        self.indexer.freqs_cis = self.freqs_cis
    # the real indexer, unmodified: publishes index_k / candidates and returns the top-k
    narrow = self.indexer(x, qr, latent, start_pos, offset)  # [b, s, topk], -1 = unused

    lens = _visible_lens(seqlen, self.compress_ratio, start_pos)          # [s, 1]
    visible = torch.arange(compress_len).unsqueeze(0) < lens              # [s, n]

    # the indexer's picks as a [b, s, n] mask (scatter into a dummy column for -1)
    pos = torch.where(narrow >= 0, (narrow - offset).long(), compress_len)
    sel = torch.zeros(bsz, seqlen, compress_len + 1, dtype=torch.bool)
    sel.scatter_(-1, pos, True)
    sel = sel[..., :compress_len]

    dense = visible.unsqueeze(0).expand(bsz, -1, -1)
    is_dense = torch.tensor([m == "dense" for m in ROW_MODES[:bsz]]).view(bsz, 1, 1)
    keep = torch.where(is_dense, dense, sel)

    global _PROBE
    _PROBE = {"sel": sel, "layer": self.layer_id}
    STATS.append((
        self.layer_id, self.compress_ratio, compress_len,
        visible.sum(-1).float().mean().item(),
        sel[0].sum(-1).float().mean().item(),
    ))

    slot = (torch.arange(compress_len, dtype=torch.int32) + offset)
    idxs = torch.where(keep, slot, torch.tensor(-1, dtype=torch.int32))
    ref.shared_attn.topk_idxs = idxs
    return idxs


def sparse_attn_masked(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Same semantics as the shim's gather `sparse_attn`, evaluated in query chunks with
    the two KV sources handled the way they are shaped:

      * sliding window — `_WIN[1]` (=128) slots per query out of `_WIN[0]` rows. The
        slots a chunk of queries can reach span at most chunk+window rows, so only that
        band is scored;
      * compressed store — up to the whole store per query: scored densely against the
        store and masked, so nothing of size [b, m, store, d] is ever materialised.

    One joint softmax over both halves, so this is the reference computation; only the
    f32 accumulation order differs from the pure-gather shim (see the module docstring).
    """
    b, m, h, d = q.shape
    n = kv.shape[1]
    wk, wi = _WIN
    if not (0 < wi <= topk_idxs.shape[-1] and 0 < wk <= n):
        wk, wi = n, topk_idxs.shape[-1]        # unknown split: treat everything as window
    kvw = kv[:, :wk].float()
    kvc = kv[:, wk:].float()
    nc = kvc.shape[1]
    idxw = topk_idxs[..., :wi].long()
    idxc = topk_idxs[..., wi:].long()
    qf = q.float()
    sink = attn_sink.float().view(1, 1, h, 1)
    out = torch.empty(b, m, h, d, dtype=q.dtype)
    budget = int(os.environ.get("V41_ATTN_CHUNK_ELEMS", str(64 << 20)))
    chunk = max(1, min(m, budget // max(1, b * h * (nc + 2 * wi))))
    dropped = []
    for c0 in range(0, m, chunk):
        c1 = min(m, c0 + chunk)
        c = c1 - c0
        qc = qf[:, c0:c1]
        iw = idxw[:, c0:c1]
        # The window slots a chunk of queries can reach span at most chunk+window rows
        # (a band in prefill, the whole ring in decode), so slice the window KV to that
        # range and score it with one wide GEMM + mask. A per-query gather here is a
        # batched 64x512x128 matmul, which torch runs ~100x below the GEMM rate.
        ok = iw >= 0
        lo = int(iw.masked_fill(~ok, 1 << 30).min()) if bool(ok.any()) else 0
        hi = int(iw.max()) + 1
        hi = max(hi, lo + 1)
        kvg = kvw[:, lo:hi]                                            # [b, W, d]
        W = hi - lo
        mw = torch.zeros(b, c, W + 1, dtype=torch.bool)
        mw.scatter_(-1, torch.where(ok, iw - lo, torch.tensor(W, dtype=torch.long)), True)
        vw = mw[..., :W].unsqueeze(2)                                  # [b,c,1,W]
        sw = torch.einsum("bmhd,bnd->bmhn", qc, kvg) * softmax_scale
        sw = sw.masked_fill(~vw, float("-inf"))
        if nc:
            ic = idxc[:, c0:c1]
            pos = torch.where(ic >= 0, ic - wk, torch.tensor(nc, dtype=torch.long))
            mc = torch.zeros(b, c, nc + 1, dtype=torch.bool)
            mc.scatter_(-1, pos, True)
            mc = mc[..., :nc].unsqueeze(2)                             # [b,c,1,nc]
            sc = torch.einsum("bmhd,bnd->bmhn", qc, kvc) * softmax_scale
            sc = sc.masked_fill(~mc, float("-inf"))
            mx = torch.maximum(sw.amax(-1, keepdim=True), sc.amax(-1, keepdim=True)).clamp_min(-1e30)
            ew = torch.exp(sw - mx).masked_fill(~vw, 0.0)
            ec = torch.exp(sc - mx).masked_fill(~mc, 0.0)
            den = ew.sum(-1, keepdim=True) + ec.sum(-1, keepdim=True) + torch.exp(sink - mx)
            pw, pc = ew / den, ec / den
            if _PROBE is not None and _PROBE["sel"].shape[-1] == nc and _PROBE["sel"].shape[1] == m:
                kept = _PROBE["sel"][:, c0:c1].unsqueeze(2)
                dropped.append((pc.masked_fill(kept, 0.0).sum(-1).mean(2), pc.sum(-1).mean(2)))
            out[:, c0:c1] = (torch.einsum("bmhn,bnd->bmhd", pw, kvg)
                             + torch.einsum("bmhn,bnd->bmhd", pc, kvc)).to(q.dtype)
        else:
            mx = sw.amax(-1, keepdim=True).clamp_min(-1e30)
            ew = torch.exp(sw - mx).masked_fill(~vw, 0.0)
            den = ew.sum(-1, keepdim=True) + torch.exp(sink - mx)
            out[:, c0:c1] = torch.einsum("bmhn,bnd->bmhd", ew / den, kvg).to(q.dtype)
    if dropped:
        MASS.setdefault(_LAYER[0], []).append(
            (torch.cat([x[0] for x in dropped], 1), torch.cat([x[1] for x in dropped], 1)))
    return out


def install(row_modes):
    global ROW_MODES
    ROW_MODES = list(row_modes)
    ref.Attention._compress_topk_idxs = _compress_topk_idxs
    ref.Attention._window_kv = _window_kv
    ref.sparse_attn = sparse_attn_masked


def uninstall():
    ref.Attention._compress_topk_idxs = _orig_compress_topk_idxs
    ref.Attention._window_kv = _orig_window_kv
    ref.sparse_attn = _orig_sparse_attn


def sparse_attn_gather_chunked(q, kv, attn_sink, topk_idxs, softmax_scale):
    """The shim's own gather formulation, evaluated in query chunks so the gathered
    [b, m, topk, d] never materialises in full. Query rows are independent, and the
    per-row GEMM shape is unchanged, so this is bit-identical to `kernel.sparse_attn`
    (checked in --selftest)."""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    t = idx.shape[-1]
    out = torch.empty(b, m, h, d, dtype=q.dtype)
    kvf = kv.float()
    sink = attn_sink.float().view(1, 1, h, 1)
    budget = int(os.environ.get("V41_ATTN_CHUNK_ELEMS", str(64 << 20)))
    chunk = max(1, min(m, budget // max(1, b * t * d)))
    for c0 in range(0, m, chunk):
        c1 = min(m, c0 + chunk)
        ii = idx[:, c0:c1]
        valid = ii >= 0
        safe = ii.clamp_min(0)
        g = kvf.unsqueeze(1).expand(b, c1 - c0, kv.shape[1], d)
        kvg = torch.gather(g, 2, safe.unsqueeze(-1).expand(b, c1 - c0, t, d))
        s = torch.einsum("bmhd,bmtd->bmht", q[:, c0:c1].float(), kvg) * softmax_scale
        s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
        mx = s.amax(dim=-1, keepdim=True).clamp_min(-1e30)
        e = torch.exp(s - mx).masked_fill(~valid.unsqueeze(2), 0.0)
        denom = e.sum(dim=-1, keepdim=True) + torch.exp(sink - mx)
        out[:, c0:c1] = torch.einsum("bmht,bmtd->bmhd", e / denom, kvg).to(q.dtype)
    return out
