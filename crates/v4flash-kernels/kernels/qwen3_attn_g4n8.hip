// qe_attn_g4_n8: the 8-wave, 2-stage, 2-WGs-per-CU cell of qwen3_attn_g4.hip (the
// same source, a compile-time cell; never an independent kernel). Built with
// -mcumode (build.rs CU_MODE_STEMS). docs/v41/EMBED_PHASE_DESIGN.md §19.3.
#define QA_CELL_NW8 1
#include "qwen3_attn_g4.hip"
