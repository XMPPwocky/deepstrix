// wl_c8dn2 with the down member loop unrolled by 2
#define MXFP4_KW_MAX_CHUNK 8u
#include "cand_wl_gu.hip"
#define DN_ROWS 2u
#define DN_MUNR 2
#include "cand_dn_v2.hip"
