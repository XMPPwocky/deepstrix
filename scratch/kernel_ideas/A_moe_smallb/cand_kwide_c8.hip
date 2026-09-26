// A_moe_smallb/kwide_c8 — the UNMODIFIED production kwide gate/up kernel compiled with MXFP4_KW_MAX_CHUNK=8
// (LDS staging 16 KB -> 4 KB, per-lane accumulator arrays 2x32 -> 2x8 floats). Launch with chunk_size <= 8.
// Cheapest possible knob: does occupancy limit kwide at small b? The symbol is renamed for the harness only.
#define MXFP4_KW_MAX_CHUNK 8u
#define mxfp4_pair_matvec_fused_swiglu_kwide cand_gate_up
#include "mxfp4_pair_matvec.hip"
