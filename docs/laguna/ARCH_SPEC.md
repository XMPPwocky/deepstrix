# Laguna-S-2.1 — Implementation-Grade Architecture Spec

Authoritative contract for the deepstrix Laguna port (weight loader + kernels).
Sources: llama.cpp PR **ggml-org/llama.cpp#25165** (`src/models/laguna.cpp`,
`conversion/laguna.py`, `gguf-py/gguf/{constants,tensor_mapping}.py`,
`src/llama-vocab.cpp`) + authoritative `poolside/Laguna-S-2.1/config.json`, cross-checked
against the base config and vLLM S-2.1 recipe. Weights NOT downloaded — per-tensor quant
(§9) is a Phase-0 GGUF-header check.

## 0. Model shape (config.json)
hidden 3072 · 48 layers · head_dim 128 · **Q heads 48 (full) / 72 (SWA)** · KV heads 8 ·
vocab 100352 · intermediate_size (dense) 12288 · num_experts 256 · **top-k 10** ·
moe_intermediate 1024 · shared_expert_intermediate 1024 · **moe_routed_scaling 2.5** ·
norm_topk_prob true · rms_eps 1e-6 · max_pos 1048576 · tie_word_embeddings false ·
gating "per-head".
Derived per-layer: `n_embd_q = 128·n_head` = **6144 full / 9216 SWA**; `n_embd_k =
n_embd_v = 1024`. `wo` maps `n_embd_q → 3072`. GQA ratio 6 (full) / 9 (SWA).

## 1. GGUF metadata keys (`laguna.*`)
- `attention.layer_norm_rms_epsilon` = 1e-6
- `leading_dense_block_count` (n_layer_dense_lead) = **1** (layer 0 dense)
- `expert_feed_forward_length` = 1024 · `expert_shared_feed_forward_length` = 1024
- `expert_gating_func` = **SIGMOID** · `expert_weights_scale` = **2.5** · `expert_weights_norm` = **true**
- `expert_shared_count` = 1 (optional, default 1) · `expert_count`/`expert_used_count` = 256/**10**
- `attention.sliding_window` = **512** · `attention.sliding_window_pattern` = period **4, dense_first=true** (full at il%4==0)
- `rope.freq_base` = 500000 (full) · `rope.freq_base_swa` = 10000 (SWA)
- `rope.dimension_count` = **64** (full, partial 0.5) · `rope.dimension_count_swa` = **128** (SWA, full)
- `attention.head_count` = **per-layer array** [48,72,72,72,…] (read as array!) · `head_count_kv` = 8
- YaRN keys (full layers): scaling.type=yarn, **factor 32.0**, original_context_length 8192,
  **attn_factor 1.0**, beta_fast 32, beta_slow 1 (verified from GGUF; effective mscale = 1.3466, §4).

## 2. Tensor names (loader contract)
Global: `token_embd.weight` [3072,100352] · `output_norm.weight` [3072] ·
`output.weight` [3072,100352] (NOT_REQUIRED → tied fallback; S-2.1 expects real one).
Per layer: `blk.{i}.attn_norm.weight` [3072] · `attn_q.weight` [3072, 6144/9216] ·
`attn_k.weight` [3072,1024] · `attn_v.weight` [3072,1024] · `attn_output.weight`
[6144/9216, 3072] · `attn_q_norm.weight` [128] · `attn_k_norm.weight` [128] ·
`attn_gate.weight` [3072, n_head(48/72)] (per-head) · `ffn_norm.weight` [3072].
MoE layers (i≥1): `ffn_gate_inp.weight` [3072,256] · `exp_probs_b.bias` [256] ·
`ffn_gate_exps.weight` [3072,1024,256] 3D · `ffn_up_exps.weight` [3072,1024,256] 3D ·
`ffn_down_exps.weight` [1024,3072,256] 3D · `ffn_{gate,up,down}_shexp.weight` (2D, 1024).
Dense layer (i=0): `ffn_{gate,up}.weight` [3072,12288] · `ffn_down.weight` [12288,3072].
No attn q/k/v/o biases.

## 3. Attention gate — **softplus, per-head, before o_proj**
`gate = softplus(W_g · rmsnorm(x_t))` (W_g = `attn_gate`, input = pre-attn normed hidden,
same as q/k/v input). Per-head: broadcast one scalar per head over head_dim, multiply the
attention output, THEN apply `wo`. softplus = log(1+exp(x)) in fp32. **Not sigmoid** (the
router is sigmoid — do not conflate).

## 4. RoPE — NEOX, per-layer-type
Full layers (il%4==0): θ=500000, n_rot=**64** (partial rotary 0.5), **YaRN** — from the
actual GGUF: **factor=32.0** (=262144/8192, this is the 262K quant not 1M), orig_ctx 8192,
beta_fast 32, beta_slow 1, **yarn_attn_factor=1.0**. **CORRECTED BY SPIKE — mscale IS
applied**: `mscale = attn_factor·(1 + 0.1·ln(factor))` = 1·(1+0.1·ln 32) = **1.3466**; the
oracle scales position q/k by exactly this. The `laguna.cpp` "pre-divide cancels" comment is
FALSE for this GGUF (mscale=1.0 was the first divergence). Feed factor=32/attn_factor=1.0 into
`mscale_eff`/`corr_dims` AND multiply q/k by mscale. ext_factor≠0. SWA layers: θ=10000, n_rot=**128** (full rotary), **plain NEOX no YaRN**
(freq_scale 1, ext_factor 0, attn_factor 1). Long context carried by YaRN on full layers only.

## 5. Sliding-window attention
window 512, LLAMA_SWA_TYPE_STANDARD (causal + previous-512-inclusive). Pattern period 4
dense_first → il%4==0 FULL, others SWA. Full/SWA differ in rope AND head count (48/72).

## 6. MoE
top-k **10**, 256 routed + **1 shared** (always-on, parallel sum, SiLU). Router **sigmoid**
+ score-correction bias `exp_probs_b` (selection only; routing weights are bias-free
sigmoid). Top-k weights **sum-normalized** then **×2.5** (routed_scaling). No router
softcap. `moe_apply_router_weight_on_input=false`. DeepSeek/afmoe-style — check ds4 router reuse.

## 7. Activation
**SwiGLU**: `down(SiLU(gate·x) ⊙ (up·x))`, parallel gate/up. Experts, shared, dense FFN all SiLU.

## 8. Normalization
RMSNorm eps 1e-6, **pre-norm**, no post-attn/post-ffn norms. **QK-norm**: RMSNorm on Q,K at
head_dim(128), after proj, before RoPE (Qwen3-style). Shared expert shares `ffn_norm`.
`output_norm` before LM head.

## 9. Per-tensor quant (Q4_K_M) — RESOLVED from actual GGUF header (Phase 0)
Histogram: 287 F32, 240 F16, 239 Q4_K, 48 Q6_K. file_type=15 (Q4_K_M). Verified map:
- **Attention q/k/v/output/gate = F16** (NOT quantized!) → reuse existing `f16_matvec`.
- `attn_q_norm`/`attn_k_norm` = F32; `ffn_gate_inp` (router) = F32; `exp_probs_b` = F32;
  all `*_norm` = F32.
- **LM head `output.weight` = Q6_K**; **`token_embd.weight` = Q4_K** (dequant row per token).
- **Dense layer 0**: `ffn_gate`,`ffn_up` = Q4_K; **`ffn_down` = Q6_K**.
- **MoE experts + shared**: `ffn_gate_exps`,`ffn_up_exps`,`ffn_gate_shexp`,`ffn_up_shexp`
  = Q4_K; **`ffn_down_exps`,`ffn_down_shexp` = Q6_K** (all `_exps` are 3D-stacked ×256).
Kernel impact: dense Q4_K (L0 gate/up, embed) + dense Q6_K (LM head, L0 down) — BUILT.
Attention = f16_matvec (reuse). MoE: Q4_K gate/up + Q6_K down — spike loops dense matvecs
per selected expert; Phase-3 perf = batched Q4_K (reuse ds4) + batched Q6_K (new) expert GEMM.
**CORRECTED BY SPIKE — quant is MIXED per-tensor, not uniform**: the histogram above is
aggregate; individual `ffn_down`/`ffn_down_exps` tensors are Q4_K on SOME layers and Q6_K on
others (spike crashed at layer 6 assuming Q6_K). The loader/forward MUST dispatch each matvec
on the tensor's ACTUAL `dtype`, never assume by name. (Also: the oracle's MoE down-proj
L2-normalize→down→restore→scale dance is a mathematical identity — `down·swiglu` direct is exact.)

## 10. Unusual / must-handle
1. Softplus per-head gate pre-o_proj. 2. **Per-layer head count 48/72** → q/o width + GQA
ratio vary per layer; read head_count array. 3. Mixed rotary (partial 64 full-layer +YaRN;
full 128 SWA plain). 4. Router score-correction bias for selection. 5. Always-on shared
expert. 6. One leading dense FFN (layer 0, 12288). 7. Tokenizer: custom BPE pre-tokenizer
`LLAMA_VOCAB_PRE_TYPE_LAGUNA` (regex), eos=[2,24] (tok 24 `</assistant>` = eot/control),
clean_spaces=false, pad=9. 8. Tied-embedding fallback in loader. 9. ABSENT (don't add): no
MuP/embed scale, no logit softcap, no attention sinks, no per-layer output scalars.

## Discrepancies resolved
- **top-k = 10** (config.json + vLLM S recipe + loader). "top-8" was the XS variant / card prose.
- **shared expert = 1** (size 1024, always built).
- **dense 12288** = real, layer 0 only (n_layer_dense_lead=1).
- gate=softplus (attn) vs router=sigmoid (MoE) — distinct.

## UNRESOLVED — verify in Phase 0 (GGUF header)
1. Per-tensor quant types (§9). 2. Real `output.weight` present vs tied. 3. Exact YaRN
attn_factor written to GGUF (vs mscale pre-division). 4. Confirm `sliding_window=512` +
swa rope keys carried in the S-2.1 GGUF.
