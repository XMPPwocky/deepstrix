// qe_attn_g4_n16 built in WGP mode (no -mcumode): the WGP arm of qwen3_attn_g4.hip,
// 2 WGs per WGP sharing its 128 KB LDS (96 KB used). Same source and symbol as the
// primary cell; only the descriptor's workgroup_processor_mode differs.
// docs/v41/EMBED_PHASE_DESIGN.md §19.3.
#include "qwen3_attn_g4.hip"
