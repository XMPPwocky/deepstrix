# DeepSeek-V4.1-Flash — architecture spec from the shipped reference (rev 0, 2026-09-11)

Source of truth: `/persist/lumi/models/dsv4.1f-full/inference/{model,engram,kernel,convert}.py`
and `encoding/`. This is DeepSeek's own readable reference implementation (tilelang kernels,
torch glue). Everything below is read off that code, not the paper. Where the paper's
production behaviour differs (bounded replay), it is called out.

Shapes: dim 5120 · 40 backbone layers + 3 DSpark layers · hc_mult 4 · 64 heads · head_dim 512
(nope 448 + rope 64) · q_lora 1280 · o_groups 8 · o_lora 1024 · 384+1 experts, top-6, inter 2304
· window 128 · index 32 heads × 128, top-512 · candidate 2048 blocks × 8.

---

## 1. Forward pass (backbone)

```
h = embed(ids)                                  # bf16 [B,L,5120]
h = h.repeat(hc_mult) -> [B,L,4,5120]           # residual stream = 4 copies
pre_mix = one-hot(copy 0)                       # initial mix
for l in 0..39:
    if l in {1,14}: h = Engram_l(h, hashes)     # applied BEFORE the block, on all 4 copies
    if l in dspark_target_layer_ids {37,38,39}: main_hiddens.append(h.mean(copies))
    h, pre_mix = Block_l(h, pre_mix)
h = hc_pre(h, pre_mix); logits = head(norm(h))  # head weight kept fp32 in the reference
```

### 1.1 Block (Single-Pass mHC, two sub-blocks)
```
residual = x
attn_pre, attn_post, attn_comb = hc_mixes(x, hc_attn_*)     # coefficients from THIS input
x = hc_pre(x, pre_mix)          # collapse 4 copies with the PREVIOUS sub-block's pre  (shift!)
x = attn(attn_norm(x))
x = hc_post(x, residual, attn_post, attn_comb)
residual = x
ffn_pre, ffn_post, ffn_comb = hc_mixes(x, hc_ffn_*)
x = hc_pre(x, attn_pre)         # FFN uses the pre computed by this block's attention mixes
x = ffn(ffn_norm(x))
x = hc_post(x, residual, ffn_post, ffn_comb)
return x, ffn_pre               # next block's attention uses ffn_pre
```
`hc_mixes`: flatten copies → [B,L,20480] fp32, rsqrt(mean(x²)+norm_eps) per token, `mixes =
x @ hc_fn.T * rsqrt` with hc_fn [24, 20480] fp32. Then `hc_split_sinkhorn`:
- pre[j]  = sigmoid(m[j]·s0 + b[j]) + eps                       (4)
- post[j] = 2·sigmoid(m[4+j]·s1 + b[4+j])                        (4)
- comb    = softmax_row(m[8:24]·s2 + b[8:24]) + eps; then col-normalise (+eps); then 19×(row,col)
  normalise → doubly stochastic 4×4.  hc_eps = 1e-6, 20 iters.
- hc_pre: y = Σ_j pre[j]·x[j]   (fp32, cast back)
- hc_post: y[j] = post[j]·x + Σ_k comb[j,k]·residual[k]
Per layer params: hc_attn_fn, hc_ffn_fn [24,20480] fp32 (2 × 1.97 MB), base [24], scale [3].
Residual stream dtype is bf16 (40 KB/token). Norms: RMSNorm in fp32, `norm_eps = 1e-20`.

### 1.2 Attention (per layer; all layers have SWA, layers ≥2 add compressed KV)
```
qr = q_norm(wq_a(x))                     # 5120→1280 fp8, RMSNorm(1280)
q  = wq_b(qr) -> [64, 512]; rope(q[..., -64:])          # 1280→32768 fp8
kv_w = kv_norm(wkv(x)) -> [512]; rope(kv_w[..., -64:])  # 5120→512 fp8; window K=V latent
kv_w = fp8_fakequant(kv_w, block 32, ue8m0)             # window KV stored FP8 (whole 512 incl rope)
window ring: 128 slots; prefill attends causal window per query, decode attends whole ring
if compress_ratio > 0:
    latent = compressor(x) if kv_source else None       # see 1.3; pre-RoPE
    idxs = indexer(x, qr, latent) if index_source else shared.topk_idxs
    if latent: rope(latent[..., -64:], compressed positions); fp4_fakequant(latent, block 16, E4M3)
               compress_kv_cache[pos//ratio] = latent; shared.compress_kv = cache
    kv = cat(window_kv, compress_kv); idxs = cat(window_idxs, compress_idxs + window_len)
o = sparse_attn(q, kv, attn_sink, idxs, 512^-0.5)       # online softmax; sink adds exp(sink-max) to denom
rope^-1(o[..., -64:])                                    # un-rotate output tail (V == K latent)
o -> [8 groups, 4096]; o_g = einsum(o_g, wo_a_g [1024,4096])  # wo_a block-diag bf16 (fp8+scale in HF)
x = wo_b(o.flatten)                                      # 8192→5120 fp8
```
- `attn_sink`: [64] fp32 learned per-head sink logit.
- K == V == the 512-d latent (MLA fully absorbed; no separate V). Output per head is 512-d.
- RoPE: YaRN (factor 16, orig 65536, β 32/1) with `compress_rope_theta` 160000 for layers with
  compress_ratio > 0; plain theta 10000 and **no YaRN** for SWA-only layers (0, 1, DSpark).
  Compressed position of group j = j·ratio (first token of the group).

### 1.3 Compressor (kv_source layers 2, 8, 14 at ratio 2; layer 20 at ratio 1)
- ratio 1: `norm(wkv(x))` bf16, 5120→512. That's it.
- ratio 2: fp32 `kv = wkv(x)`, `score = wgate(x)` (both 5120→512 **fp32** weights); per group of
  2 tokens: `latent = Σ_t softmax_t(score)·kv` **per channel**; then RMSNorm(512). No overlap, no
  abs-pos. Decode: partial group parked in `kv_state/score_state`, emits every 2nd token.
- Storage: E2M1, one **E4M3** scale per 16 channels (amax floored at 6·2⁻⁹) → 512 → 288 B/entry.

### 1.4 Indexer (index_source layers 2, 8, 14, 20, 24, 28, 32, 36)
```
if owns_k (kv_source): k = k_norm(wk(latent)) [128]; rope(k[..., -64:]); fp4_fakequant(k, 32, E8M0)
                       index_k_cache[...] = k; shared.index_k = cache
q = wq_b(qr) -> [32, 128]; rope(q[..., -64:]); fp4_fakequant(q, 32, E8M0)     # 1280→4096 fp8
w = weights_proj(x) * (128^-0.5 * 32^-0.5)      # 5120→32 bf16
score[t] = Σ_h relu(q_h · k_t) * w_h            # over all reachable compressed positions
if candidate_source (layer 20): shared.candidates = select_candidate_blocks(score)   # §1.5
elif uses_candidates (layers 24..36): score[~candidates] = -inf
idxs = topk(score, 512) sorted by position; unreachable → -1
```
Reachability: compressed position p visible to query at position i iff p < (i+1)//ratio.
Index K storage: E2M1 + E8M0 per 32 → 128 → 68 B/entry.

### 1.5 Hierarchical candidate pool (built at layer 20, used by 24, 28, 32, 36)
Block score = max over 8 consecutive positions (pad −inf); the block holding the query's newest
position is pinned (+inf); top-2048 blocks; mask = union of selected blocks (16,384 positions).
Reuse layers (everything else) take `shared.topk_idxs` from the most recent index source.

### 1.6 Gate / MoE
```
s = sqrt(softplus(x_f32 @ W_gate^T))            # gate weight [384,5120] (bf16 in ckpt), fp32 math
idx = topk(s + bias, 6)                          # bias = e_score_correction_bias fp32 [384]; bias_vl for image tokens
w = s[idx]; w /= (w.sum + 1e-20); w *= 1.5       # norm_topk_prob, route_scale
y = Σ_e  w2_e( w_e * silu(clamp(w1_e x, max=10)) * clamp(w3_e x, -10, 10) )   # weight applied BEFORE w2
y += shared_expert(x)                            # same SwiGLU+clamps, fp8 weights
```
Routed experts: w1/w3 [2304,5120], w2 [5120,2304], E2M1 packed + E8M0 scale per 32 along K.
Activations for expert GEMMs are FP8 (block 32, ue8m0) — the fp4_gemm is fp4-weight × fp8-act.
Accumulation fp32.

### 1.7 Engram (layers 1 and 14; applied to the residual copies before the block)
Hash ids (token-only, computed once per position):
```
c = token_map[id]                                # compressed vocab (99,092): NFKC→NFD→strip accents→lower→ws collapse→strip
tokens[s] = c[pos-s] for s in 0..3 (pad_id if pos<s or crosses an image span)
prod[s] = tokens[s] * mult[layer][s]             # odd int64 multipliers from rng(10007*layer_id), bound (2^63-1)/V/2
rolling = prod[0]; for n in 1..3: rolling ^= prod[n]; hash[n-1] = rolling % prime[layer][n-1][head] + offset
```
→ 3 n-gram orders × 8 heads = 24 rows per position per module (each head has its own prime
≈16M; primes distinct across all (layer, order, head), drawn from 16,000,000 upward).
Table: [~384M rows, 256] FP8 + [rows, 8] E8M0 scales (block 32). Row gather → 24×256 = 6144 →
`wkv` 6144 → 5120×5 (fp8): key [4 copies × 5120], value [5120].
Gate per (token, copy): `dot = Σ_d h·(q_w·k_w)·key · rstd(h)·rstd(key) / √5120`;
`gate = sigmoid(copysign(sqrt(max(|dot|,1e-6)), dot))`; `h += gate·value`. Image tokens: gate=0.

### 1.8 DSpark (mtp.0..2, SWA-only, 128 experts top-3, block 5)
- Input: `main_x = main_norm(main_proj(cat(mean-over-copies of h at layers 37,38,39)))` (15360→5120).
- Draft tokens: [last_token, noise×4] embedded (tied backbone embed), expanded to 4 copies.
- DSparkAttention: window KV from `main_x` (the backbone stream), the 5 draft positions attend
  to the window + each other (block causal via idx matrix). Prefill only seeds the window cache.
- Head: tied backbone head on norm(hc_pre(h)); per position i: logits += MarkovHead(prev token)
  (embed 129280×256 → head 256→129280), sample, feed forward; ConfidenceHead(cat(h, markov_emb))
  → per-position confidence. The verification loop is NOT in the reference.

### 1.9 Vision
ViT 32 layers (dim 1024, 16 heads, inter 2816, patch 14, 2D rope), aligner w1 9216→5120 (3×3
unshuffle), w2 5120→5120; learned image_start/end/newline embeddings; image tokens carry
`image_token_id` 129264 in input_ids and use `bias_vl` in the gate; no Engram on image spans.

---

## 2. On-disk format (HF safetensors, 48 shards, 475 GiB)
- Names: `model.layers.N.self_attn.{wq_a,wq_b,wkv,wo_a,wo_b,q_norm,kv_norm,attn_sink,compressor.*,indexer.*}`,
  `model.layers.N.mlp.{experts.E.w{1,2,3}, shared_experts.w{1,2,3}, gate.weight, gate.e_score_correction_bias[_vl]}`,
  `model.layers.N.{hc_attn_fn,hc_ffn_fn,hc_*_base,hc_*_scale,attn_norm,ffn_norm}`,
  `model.layers.{1,14}.engram.{embed.weight,embed.scale?,wkv,q_weight,k_weight}`, `mtp.N.*`
  (DSpark; embed/head tied), `vision.*`, `aligner.*`, `embed.weight`, `head.weight`, `norm`.
- Expert weights: **int8 tensors of packed E2M1 nibbles, [out, in/2], low nibble = element 2i**;
  `weight_scale_inv` [out, in/32] E8M0. FP4 table: idx 0-7 = {0,.5,1,1.5,2,3,4,6}, bit 3 = sign.
- FP8 linears: `weight` E4M3 [out,in] + `weight_scale_inv` [out/32, in/32] E8M0 (32×32 blocks).
  wo_a may be (128,128)-blocked; the reference dequantizes it to bf16.
- Engram tables: rows FP8 + per-row E8M0 scales (block 32) — 189 GiB in shards 47/48.
- Shard 1 = vision+aligner (bf16), shard 2 = embed + image_* (bf16), 3–42 = layers, 43–46 misc.

## 3. Numerics we must match for oracle parity
- Window KV: FP8 fake-quant of the whole 512 (incl. rope tail), block 32, scale = 2^ceil(log2(amax/448)), amax ≥ 1e-4.
- Compressed KV: E2M1, block 16, E4M3 scale = fp8(amax/6), amax ≥ 6·2⁻⁹, quantised **after** RoPE.
- Index q/k: E2M1, block 32, E8M0 scale = 2^ceil(log2(amax/6)), amax ≥ 6·2⁻¹²⁶.
- Expert activations: FP8 block 32 ue8m0 (same rule as window KV). Gate math fp32.
- Sparse attention: online softmax with finite init (−1e30); sink term exp(sink − max) in the
  denominator; rows with no valid index → zero output.
- Top-k: `topk(sorted=False)` then position-sort; ties are implementation-defined → oracle
  compares attention outputs, not index sets, near ties.
- mHC math entirely fp32; residual bf16.

## 4. Reference vs production behaviour
- The reference runs **all 40 layers over the whole prompt** (exact). Production V4.1 uses CED +
  Decoder SWA Bounded Replay (decoder over the last 128 tokens only) and Encoder SWA Bounded
  Replay on cache hits — both *approximate* by the paper's own statement. deepstrix should have
  an exact mode (oracle parity) and a bounded-replay mode (speed); compare like with like.
- The reference's DSpark verification loop is absent; only the drafter forward exists.

## 5. Prompt format (encoding/README.md)
`<｜begin▁of▁sentence｜>[<｜System｜>Reasoning Effort: N (range 1-100, ...)\n\n]{system}<｜User｜>…<｜Assistant｜><think>…</think>…<｜end▁of▁sentence｜>`
- Effort is numeric 1–100 (low=50, high=75 default, max=100), rendered only at index 0 and only
  in thinking mode. DSML tags gained a leading space: `<｜DSML｜ calls>`, `<｜DSML｜ invoke>`,
  `<｜DSML｜ parameter>`. Mid-conversation `<｜System｜>` supported. Tool results are merged into
  the preceding user message as `<tool_result>` blocks, ordered by the assistant's tool_calls.
- `encoding/test_encoding.py` + `tests/` are golden vectors: port them into the server's renderer
  tests (cf. the V4 render_prompt divergence in docs/TOOL_PROMPT_FIDELITY.md).

## 6. Per-layer weight bytes (decode-relevant, fp8 unless noted)
| tensor | shape | bytes |
|---|---|---|
| wq_a | 1280×5120 | 6.6 MB |
| wq_b | 32768×1280 | 41.9 MB |
| wkv | 512×5120 | 2.6 MB |
| wo_a | 8×1024×4096 | 33.5 MB (67 MB if kept bf16) |
| wo_b | 5120×8192 | 41.9 MB |
| hc fn ×2 (fp32) | 2×24×20480 | 3.9 MB |
| gate | 384×5120 | 2–4 MB |
| shared expert | 3×2304×5120 | 35.4 MB |
| compressor (source layers) | 2×512×5120 fp32 | 21 MB |
| indexer (source layers) | wq_b 4096×1280 + proj | 5.4 MB |
| **per layer, non-routed** | | **~168 MB** (avg) |
| ×40 + head (662 MB fp8 / 1.3 GB bf16) | | **~7.4–8 GB** |
This is ~2× the first-pass estimate in PLAN.md §6: the dGPU attention+shared leg is ~11-12 ms at
640 GB/s, not 6.5. wo_a must be kept fp8 (block-scaled) rather than the reference's bf16.

## Checkpoint storage formats (HF safetensors, 48 shards, 96 085 tensors) — measured 2026-09-12

| role | HF tensors | storage | engine presentation (`V41HfWeights`) |
|---|---|---|---|
| routed experts w1/w3/w2 (384/layer) | `layers.L.ffn.experts.E.wN.weight` I8 `[out, in/2]` (elem 2i = low nibble) + `.scale` F8_E8M0 `[out, in/32]` | MXFP4 native, 0.53125 B/elem | MXFP4 ggml 17-B blocks, stacked `[n_expert, out, in]`; scale byte verbatim |
| attn wq_a/wq_b/wkv/wo_a/wo_b, shared experts, indexer wq_b, engram wkv | `.weight` F8_E4M3 `[out, in]` + `.scale` F8_E8M0 `[out/32, in/32]` | fp8 with **32×32 e8m0 block scales** (not V3's 128×128 f32) | Q8_0 (dequant f32 → ggml quantiser) |
| router gate | `ffn.gate.weight` BF16 `[384, 5120]`, `.bias`/`.bias_vl` F32 | | raw BF16 / F32 |
| norms, hc_*, attn_sink, compressor norm, indexer proj/k_norm | BF16 or F32 | | F32 |
| compressor wkv/wgate, indexer wk | BF16 | | F16 |
| Engram table (layers 1, 14) | `engram.embed.weight` F8_E4M3 `[384 006 168, 256]` + `.scale` F8_E8M0 `[…, 8]` | 98.3 GB + 3.1 GB per layer | not presented; gathered per token via `raw()` |
| DSpark | `mtp.0.*` (128-expert gate, own attn/hc) | | not presented yet |

Layer 0–1 have no compressor/indexer/engram tensors; layer 20 (kv_source Reindex) has indexer
k/norm but no compressor gate; e8m0 byte e means 2^(e−127) (torch `float8_e8m0fnu`).

**Per-layer tensor sets by CSA2 role (measured from the index, 2026-09-12).** Every layer has the
base set (attn wq_a/wq_b/wkv/wo_a/wo_b, q_norm/kv_norm, attn_sink, norms, hc_*, gate, shared +
routed experts) — Reuse-mode layers (3–7, 9–13, 15–19, 21–23, …) still carry `attn.wkv`, so the
loader presents it; whether the reference uses it there is a model.py question (§1). On top:

| layer role | extra tensors |
|---|---|
| Engram (1, 14) | `engram.embed(+scale)`, `engram.wkv`, `engram.q_weight`, `engram.k_weight` |
| KV source, ratio 2 (2, 8, 14) | `attn.compressor.{norm,wgate,wkv}`, `attn.indexer.{wq_b,weights_proj,wk,k_norm}` |
| KV source, ratio 1 (20) | same minus `compressor.wgate` |
| index source only (24, 28, 32, 36) | `attn.indexer.{wq_b,weights_proj}` (keys come from the KV source layer) |
| Reuse (all others) | none |
