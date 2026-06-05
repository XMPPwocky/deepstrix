# Credits

deepstrix is a from-scratch Rust+HIP reimplementation of DeepSeek V4-Flash
inference for hybrid dGPU+iGPU systems (specifically AMD RX 9070 XT +
Strix Halo). It owes a substantial intellectual debt to two upstream
projects:

## antirez/ds4

[antirez/ds4](https://github.com/antirez/ds4) is the reference C
implementation of DeepSeek V4-Flash by Salvatore Sanfilippo. It is the
canonical correctness oracle: every kernel in deepstrix is validated
bit-exactly against the corresponding ds4 CPU computation, and many of
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

## Other references

Specific upstream techniques and citations are noted at their call sites
in source. Notable inheritances:

- llama.cpp's quantization formats (iq2_xxs, q2_k, q8_0, q8_k, fp8_e4m3fn)
  via ds4's `cuda_block_*` adaptations.
- DeepSeek's V4-Flash architecture spec for the compressor/indexer/MoE
  shapes and chat-template tokens.
