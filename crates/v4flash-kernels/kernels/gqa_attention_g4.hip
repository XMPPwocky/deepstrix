// gqa_attention.hip compiled a second time with HGP_G = 4: the full-GQA-group packed
// prefill kernel (`gqa_attn_prefill_flash_wmma_fa2_hg_packed`) for kv_group 4
// (Qwen3-Embedding-4B: 32 query heads over 8 KV heads, head_dim 128). The source is
// shared, not copied: HGP_G is #ifndef-guarded and the kernel body is generic in it,
// so gqa_attention.hip itself (and its module, HGP_G = 6) is unchanged.
//
// At G = 4 the work splits evenly (8 score tasks over 8 waves, 256 softmax slots over
// 256 threads) and LDS is ~46.6 KB (G = 6: 61.6 KB): one WG per CU, as at G = 6.
// docs/v41/EMBED_PHASE_DESIGN.md §18.
// A -DHGP_G from DEEPSTRIX_KERNEL_CFLAGS reaches both translation units; this one
// is 4 whatever it says (Rust's `for_arch_group4` assumes 4).
#undef HGP_G
#define HGP_G 4u
#include "gqa_attention.hip"
