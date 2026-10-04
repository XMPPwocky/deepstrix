# Credits

deepstrix is a from-scratch Rust+HIP inference engine. It started as a
reimplementation of DeepSeek V4-Flash for one hybrid dGPU+iGPU box (AMD RX
9070 XT + Strix Halo); the live system is DeepSeek V4.1 on two boxes
(box 1 = hub, `deepstrix-server`; box 2 = `deepstrix-expertd`, linked over
USB4). V4-Flash was dropped as a direction on 2026-09-24. (corrected
2026-10-04) It owes a substantial intellectual debt to two upstream
projects:

## antirez/ds4

[antirez/ds4](https://github.com/antirez/ds4) is the reference C
implementation of DeepSeek V4-Flash by Salvatore Sanfilippo. It was the
correctness oracle of the V4-Flash engine: the V4-Flash kernel tests
compare against ds4 CPU dumps with max-abs tolerances (up to 5e-2, e.g.
`tests/attention_setup_chain.rs`, `tests/shared_expert.rs`), not
bit-exactly (corrected 2026-10-04). Many of
the algorithmic structures (compressor pipeline, indexer top-K, MoE
routing, FP8 KV quantization, the `cuda_block_q8_K` and
`cuda_block_iq2_xxs` layouts) are direct ports of ds4's logic.

## ejpir/ds4-hip

[ejpir/ds4-hip](https://github.com/ejpir/ds4-hip) (branch
`rocm-upstream-shape-cyberneurova`) is a HIP/ROCm port of ds4 by
ejpir / e2pir that achieves ~200 tok/s prefill on Strix Halo alone.
The deepstrix iq2 MoE kernels — particularly the per-expert tile8
GEMV with quarter-wave dot product and register-staged dequant
(`dev_dot_iq2_xxs_q8_K_block8_deq_lut` pattern) — are based on the
algorithmic shape pioneered in their `rocm/ds4_rocm_moe.cuh`.

A detailed analysis of how the fork hits its prefill numbers and what
shaped our adaptation is in
[EJPIR_DS4HIP_PREFILL_ANALYSIS.md](EJPIR_DS4HIP_PREFILL_ANALYSIS.md).

## DeepSeek V4.1 reference code

The V4.1 engine is validated against DeepSeek's own, unmodified
`inference/model.py`, run layer by layer on CPU by `scripts/v41_oracle/`
(`oracle.py`). Gates use tolerances and KL, not bit-exactness:
`tests/v41_layer0_parity.rs` (max|d|/max|ref| <= 5.4e-2 at layer 0) and
`tests/v41_golden_gate.rs` (KL(ref || engine), top-1 agreement and routing
flips over a golden transcript). (added 2026-10-04)

## Other references

Specific upstream techniques and citations are noted at their call sites
in source. Notable inheritances:

- llama.cpp's quantization formats (iq2_xxs, q2_k, q8_0, q8_k, fp8_e4m3fn)
  via ds4's `cuda_block_*` adaptations.
- DeepSeek's V4-Flash architecture spec for the compressor/indexer/MoE
  shapes and chat-template tokens.
- Laguna port: llama.cpp PR ggml-org/llama.cpp#25165 and poolside's
  `Laguna-S-2.1/config.json` (see `laguna/ARCH_SPEC.md`). (added 2026-10-04)
- unsloth's UD GGUF mixes (UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL), which the
  V4-Flash weight contract loads unmodified (`weight_contract.rs`). (added
  2026-10-04)
