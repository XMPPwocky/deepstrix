// kwide (MXFP4_KW_MAX_CHUNK=8) with work-item loop + member-outer down
#define MXFP4_KW_MAX_CHUNK 8u
#include "cand_wl_gu.hip"
#define DN_ROWS 2u
#include "cand_dn_v2.hip"
