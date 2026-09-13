# The missing CSA2 indexer: dense vs sparse compressed attention

**Status: measured 2026-09-13 on the CPU oracle (DeepSeek's unmodified `inference/model.py`).**
Scripts: `scripts/v41_oracle/dense_index.py` (the switch), `indexer_ab.py` (the A/B driver),
`indexer_ab_report.py` (tables), `indexer_cost.py` (arithmetic). Nothing here touches the GPU.

## 0. What the engine does today, and why nobody noticed

V4.1's Compressed Sparse Attention 2 gives every query two KV sources: a 128-row sliding window
and a compressed store. Which compressed rows a query may read is decided by the **indexer**
(ARCH_SPEC §1.4): eight index-source layers (2, 8, 14, 20, 24, 28, 32, 36) score the store with a
small FP4 side-attention and keep `index_topk = 512` rows; every other layer reuses the most
recent source's selection; layer 20 additionally builds the hierarchical candidate pool (§1.5)
that layers 24/28/32/36 search inside.

deepstrix never ported any of it. `forward_layer.rs:1050` gates the sparse path on
`ratio == 4 && n_index_comp > INDEXER_TOP_K`; V4.1's ratios are 1 and 2, so the gate never fires
and V4.1 attention **scores densely over the entire compressed store** at every layer ≥ 2.

The reason this survived parity validation is structural, not accidental:

* the store has `T / compress_ratio` rows — `T/2` for layers 2–19, **`T` for layers 20–39**;
* `topk = min(index_topk, store_rows)`, and rows a query cannot reach score `-inf`, so when a
  query's reachable set is ≤ 512 rows the top-512 **selects all of them**;
* therefore dense ≡ sparse, bit for bit, for every query at context ≤ 512 (ratio-1 layers) or
  ≤ 1024 (ratio-2 layers);
* every V4.1 parity run so far was at 6 and 129 tokens. Both are inside that regime. The
  validation could not have seen the difference.

## 1. Method

`dense_index.py` rebinds two names in the reference **without editing `model.py`**:

* `Attention._compress_topk_idxs` — the single method that decides which compressed rows a query
  reads. The replacement still runs the real `Indexer` (so the index-key cache, the candidate
  pool and the shared-selection plumbing behave exactly as shipped) and then, **per batch row**,
  keeps either the indexer's top-512 (`sparse`, = reference) or every causally reachable row
  (`dense`, = deepstrix). Reachability uses the reference's own rule, `p < (i+1)//ratio`.
* `sparse_attn` — the CPU shim gathers `[b, m, topk, d]`, which is >7 GB once `topk` is the whole
  store. The replacement scores q against the full KV per query chunk and masks. It is *not*
  bit-identical to the gather formulation (different GEMM shapes ⇒ different f32 accumulation
  order); measured fidelity is below.

Both modes are expressed in the same width — one slot per compressed row, `-1` where the row is
not selected — so the only difference between the batch rows is which slots are `-1`. Feeding the
reference the compacted top-512 list or the padded full-width list is **bit-identical**
(`dense_index.py --selftest`), because `sparse_attn` treats every slot independently.

**One run, three rows.** The A/B is a single 40-layer prefill over one prompt with
`--rows sparse,dense,sparse`: same tokens, same weights, same dequantised experts, same engram
lookups for all three. Row 1 is the engine's behaviour. **Row 2 is a duplicate of row 0** and is
the control: it is mathematically identical to row 0, so everything that separates it from row 0
is arithmetic noise.

**Why the control is necessary.** Torch's CPU GEMM is not row-position deterministic: the same
input row at a different offset in the same matmul differs by ~2e-6 relative (measured). V4.1
amplifies that. bf16 rounding turns it into 1-ULP flips, and by layer 3 it flips *router* top-k
decisions, which are discrete. So two *equally correct* runs of this model diverge; "top-1
agreement" without a control is meaningless. The unbiased metrics — teacher-forced NLL and the
attention-mass probe — are the ones to read.

**Harness fidelity.** At T=6, 40 layers, row 0 against a stock `oracle.py` gather-path dump:
layers 0–2 bit-identical, worst relative residual drift over all layers **1.07e-3**
(sub-bf16-ULP), per-layer RMS identical to 7 significant digits. The masked attention path is
faithful; it perturbs the model at the same order as the control row does.

**Prompt.** 3072 tokens of the DeepSeek-V4.1-Flash tech report (title + abstract + §1 onwards),
natural technical prose with long-range structure. One prefill gives every context length from 1
to 3072 at once, because position *i* attends only over its own causal prefix — so the
quality-vs-context curve comes out of a single pass.

## 2. Cost — what dense scoring buys the hardware nothing for

`scripts/v41_oracle/indexer_cost.py`, pure arithmetic from `inference/config.json`
(`index_topk=512`, `window=128`, `head_dim=512`, `n_heads=64`; 2 layers ratio 0, 18 ratio 2,
20 ratio 1). A compressed row is 288 B (E2M1 + one E4M3 scale per 16); the `+ indexer` column is
the FP4 side-attention the sparse path pays *instead* (32 heads × 128, score only), on the 8
index-source layers, over their full store.

| context | store rows/layer (ratio 2 / ratio 1) | compressed rows read per token, sparse | dense | dense/sparse | compressed-KV bytes/token sparse → dense | attn MACs/token sparse (+indexer) → dense |
|---|---|---|---|---|---|---|
| 512 | 256 / 512 | 14,848 | 14,848 | **1.0×** | 7.2 → 7.2 MB | 0.97 (+0.01) → 0.97 G |
| 1,024 | 512 / 1,024 | 19,456 | 29,696 | 1.5× | 8.6 → 11.5 MB | 1.28 (+0.03) → 1.95 G |
| 2,048 | 1,024 / 2,048 | 19,456 | 59,392 | 3.1× | 8.6 → 20.1 MB | 1.28 (+0.06) → 3.89 G |
| 3,072 | 1,536 / 3,072 | 19,456 | 89,088 | 4.6× | 8.6 → 28.6 MB | 1.28 (+0.08) → 5.84 G |
| 4,096 | 2,048 / 4,096 | 19,456 | 118,784 | 6.1× | 8.6 → 37.2 MB | 1.28 (+0.11) → 7.79 G |
| 32,768 | 16,384 / 32,768 | 19,456 | 950,272 | 48.8× | 8.6 → 276.6 MB | 1.28 (+0.87) → 62.3 G |
| 100,000 | 50,000 / 100,000 | 19,456 | 2,900,000 | **149×** | 8.6 → **838 MB** | 1.28 (+2.66) → **190 G** |
| 1,000,000 | 500,000 / 1,000,000 | 19,456 | 29,000,000 | 1,490× | 8.6 → 8,355 MB | 1.28 (+26.6) → 1,901 G |

Read the rows-per-token column: **the sparse path is flat in context from 1K onwards** (512 rows
per layer plus the 128-row window, forever), which is exactly the property the whole CSA2 design
exists to buy. Dense throws it away and goes linear.

The 100K line is the one that matters. A decode step would have to stream **838 MB of compressed
KV per token**; at the dGPU's ~640 GB/s that is 1.3 s/token before anything else runs — ~0.8 tok/s
against a 30 tok/s target. The sparse path reads 8.6 MB, ~13 µs. That is not an optimisation
opportunity, it is the difference between the model working at long context and not.

This agrees with the independent kernel-level pricing in `DECODE_M8_PLAN.md` (§214, 2026-09-13):
dense compressed attention costs **+4–5 ms/token at 8K and +10–18 ms/token at 32K**, growing
linearly, and it is why that plan's "flat 0.55 ms/layer chain" does not exist in the tree.

Note also what the indexer itself costs: 0.9 G MAC/token at 32K, 2.7 G at 100K, 27 G at 1M — the
only part of V4.1 decode attention that grows with context, FP4, on 8 layers. Everything else is
constant. PLAN §7's "~20 GFLOP FP4 at 1M" is the same number.
