// gu3 (kwide + next-pair prefetch + work-item loop) at MXFP4_KW_MAX_CHUNK=8, + member-outer down
#define MXFP4_KW_MAX_CHUNK 8u
#include "cand_gu3.hip"
#define DN_ROWS 2u
#include "cand_dn_v2.hip"
