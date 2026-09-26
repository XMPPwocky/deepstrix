// production kwide at MXFP4_KW_MAX_CHUNK=8 AND MXFP4_KW_UNROLL=8 (member loop fully unrolled), + work-item loop + dn2
#define MXFP4_KW_MAX_CHUNK 8u
#define MXFP4_KW_UNROLL 8
#include "cand_wl_gu.hip"
#define DN_ROWS 2u
#include "cand_dn_v2.hip"
