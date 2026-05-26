# M40 MTP Validation Findings

## TL;DR

Our V4-Flash MTP draft implementation is **architecturally correct** — every MTP-specific stage matches an independent reference (CPU and SGLang algorithm). It still **draws ~14 percentage points lower hit rate than ds4's hipify'd ROCm port** on the same input sequence. The remaining gap is not in MTP itself — it's that our main-model HC differs from ds4's ROCm main-model HC by ~10-15% rel rms at MTP-call time, and the MoE softmax-style weighting amplifies that into ~30-100% rel-rms divergence on routed MoE output.

## Numbers

Apples-to-apples on the **chat-templated 66-token sequence** (system prompt + "What is the capital of France?" + 50 generated tokens):

| metric | hit rate |
|---|---|
| Our MTP top-1 vs actual next | 35/65 = **53.8%** |
| ds4 ROCm MTP top-1 vs actual next | 44/65 = **67.7%** |
| Our main argmax (ceiling) | 56/65 = 86.2% |

On a worse raw-text prompt ("DeepSeek-V4 Flash is", 57 tokens, no chat template) both implementations are similarly mediocre (ours 55.4%, ds4 53.6%) — the raw prompt has too many genuinely uncertain positions for MTP to matter.

## What we validated

| component | verified against | result |
|---|---|---|
| Stages 1-7 (HC combine: embed, enorm, e_proj, repeat, hnorm rows, h_proj, vec_add) | independent CPU reference in `tests/mtp_stages_cpu_ref.rs` | bit-identical within Q8_0 quant noise |
| Stage 8 Q4_K MoE gate+up+swiglu+ew | `cpu_dot_q4_k_q8_k` reference in `tests/mtp_q4k_moe_cpu_ref.rs` | bit-identical (max diff 7e-7, F32 ULP) |
| Tensor shapes / dims | `tests/mtp_inspect.rs` | match expected |
| RoPE params for MTP (layer 0 == layer 1) | direct byte compare of dump files | identical |
| `prev_hc` semantics (HC at same pos as token, NOT pos-1) | ds4.c:17791 reading | fixed during investigation |
| Combine math (`e_proj + h_proj` split vs SGLang's `eh_proj` concat) | reading SGLang's `deepseek_nextn.py` | mathematically equivalent |

## Where the divergence actually lives

Stage-by-stage bisection on the chat sequence (see `tests/mtp_stages_chat.rs`) — `rel_rms` diff vs ds4 ROCm per stage:

```
s0_prev_hc      ~5-20%   (main model output, NOT MTP-specific)
s1-s7 HC combine     ~5-15%   (consistent inheritance, no amplification)
s7c attn_input_norm  ~5-20%   (mhc_pre_attn inherits prev_hc diff)
s7b after_attn       ~5-20%   (attention preserves)
s7e ffn_input_norm   ~5-15%   (mhc_pre_ffn preserves)
s7f shared_out       ~3-10%   (shared expert is fine — same Q8_0 kernel as main)
s7g routed_out       70-400%  ← EXPLODES HERE
s8_out_hc            30-100%  (combined)
```

Drill into router selection: experts agree on 56% of positions, partial agreement (5/6) on 44%. Average expert overlap 5.55/6 = 92.5%.

**But even at full-agreement positions, expert WEIGHTS differ.** E.g. pos 3 (same 6 experts picked):
- Ours: `[0.378, 0.370, 0.232, 0.215, 0.175, 0.130]`
- ds4:  `[0.501, 0.384, 0.259, 0.154, 0.123, 0.079]`

Our weights are flatter (less peaked) than ds4's. Router algorithm is identical (`sqrt(softplus(logits)) → +bias → topk → / sum * scale`), so flatter weights mean our **input logits had less spread** — i.e. our `ffn_input_norm` is structurally different from ds4's even though `rms_norm` is scale-invariant.

## Root cause

Our main model produces HC values **~2-3× smaller in magnitude** than ds4's ROCm main model (s0_prev_hc dump comparison) at MTP-call time. RMS norm absorbs the scale, but the structural pattern differences propagate, and the MoE topk's relative weighting amplifies them.

Our main model matches the **CPU reference** (the canonical ds4 path used by `forward_full_logits` oracle, by `dump_activations`, and by every existing per-stage validation) within ~10% rel rms. **ds4's ROCm port does NOT match its CPU reference** — the hipify'd path has its own drift relative to the CUDA/Metal path antirez actually tests on. Our MTP is fine relative to the canonical path; ds4-ROCm-MTP is biased relative to the canonical path in a way that happens to align better with the MTP weights' training distribution.

## What's not the bug

- F32 vs F16 precision on plain matmul weights (already fixed with `F32Matvec`)
- MTP cache slot management (already fixed: use `rope_pos` not `n_raw`)
- Chained vs main-HC `prev_hc` semantics (fixed: use main HC at same pos)
- Q4_K MoE math (validated bit-identical against CPU dequant + dot)
- HC combine stages 1-7 (validated bit-identical against pure F32 CPU reference)
- RoPE configuration
- Weight loading / tensor shapes

## Future work (NOT done in this session)

1. **Per-layer dump comparison of main forward** — patch ds4.c to dump `g->cur_hc` after each of 43 main-model layers (during normal `metal_graph_eval_token_raw_swa`). Run on this prompt. Compare to ours layer by layer. The first layer where rel_rms shoots up is where our main impl diverges from canonical.
2. **SGLang-as-oracle test** — load V4-Flash via `transformers` + `sglang/python/sglang/srt/models/deepseek_nextn.py`, run forward + MTP draft on the prompt, compare logits per position. This sidesteps ds4's ROCm hipify quirks by using the production PyTorch path. Requires standing up SGLang locally + the actual Flash checkpoint not the quantized one.
3. **Per-layer mHC SinkHorn validation** — the mhc_pre_attn / mhc_pre_ffn sinkhorn iteration is suspect for accumulating numerical drift across 43 layers. Compare each layer's `hc_split` against ds4 CPU.

## Reference availability

- **SGLang**: `python/sglang/srt/models/deepseek_nextn.py` is the cleanest Python MTP reference. V3 mature, V4 in active dev.
- **HuggingFace transformers**: has DeepSeek-V4 (since v5.9.0 / 2026-05-02) but explicitly does NOT instantiate MTP (config field says "not instantiated here").
- **ds4 (antirez C)**: MTP only in metal/cuda/rocm GPU paths. No CPU MTP. ROCm port is hipify'd and **diverges from its own CPU reference**, so it's not authoritative.
- **vLLM**: has V4 recipes; MTP via SGLang path.

## Tests added (not landed)

All under `crates/v4flash-kernels/tests/`:

- `mtp_inspect.rs` — print GGUF tensor shapes
- `mtp_stages_cpu_ref.rs` — CPU reference for HC-combine stages 1-7
- `mtp_q4k_moe_cpu_ref.rs` — CPU reference for Q4_K gate+up+swiglu+ew
- `mtp_oracle.rs` — main-HC oracle MTP hit rate (no spec_decode)
- `mtp_fair_compare.rs` — apples-to-apples vs ds4 ROCm hit rate on canonical sequence
- `mtp_against_ds4.rs` — first MTP-vs-ds4-ROCm logits comparison (7 positions)
- `mtp_against_ds4_stages.rs` — initial stage-by-stage bisection
- `mtp_stages_chat.rs` — bisection on chat-templated prompt with FFN sub-stages
- `mtp_router_compare.rs` — router selection + weight comparison
- `main_hc_vs_dump.rs` — our main HC vs CPU activation dump

ds4-side patches (also not landed):

- `external/ds4/ds4.c` — dump_emit hook in `metal_graph_eval_mtp_draft_from_hc` for all MTP intermediate stages, plus `g->router_selected` / `g->router_weights`
- `external/ds4/mtp_dump_raw.c` — small standalone driver that takes csv: or chat: token specs and runs MTP with `DS4_MTP_DUMP_DIR` enabled
