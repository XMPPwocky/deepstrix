// Reviewer probe: the engineer's candidate body (verbatim from ../cand_cand.hip) instantiated at
// BLOCK=256 with U=8 (ILP only, production geometry) to separate the "bigger WG" lever from the
// "loads in flight" lever, because the 1024-thread WG loses 1.13-2.04x at the prefill shape (b=512/1024).
#include "../../cand_cand.hip"

extern "C" __global__ __launch_bounds__(256)
void candidate_threshold_ilp256(unsigned int* __restrict__ threshold, const float* __restrict__ block_score,
                                const unsigned int* __restrict__ n_per, unsigned int nb_stride, unsigned int cb_size, unsigned int topk) {
    cand_threshold_body<256u, 8u>(threshold, block_score, n_per, nb_stride, cb_size, topk);
}
extern "C" __global__ __launch_bounds__(256)
void candidate_threshold_ilp256_u4(unsigned int* __restrict__ threshold, const float* __restrict__ block_score,
                                   const unsigned int* __restrict__ n_per, unsigned int nb_stride, unsigned int cb_size, unsigned int topk) {
    cand_threshold_body<256u, 4u>(threshold, block_score, n_per, nb_stride, cb_size, topk);
}
