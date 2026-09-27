// (d) decode launch-count fusions: production chains (in-tree code objects, the wrappers' grids
// incl. the V41_GRID_PAD pads) vs the fused / dead-node-free versions, graph mode (production
// decode stages are graph-captured), warm activations, every output compared bit for bit.
//   ./chain_harness b1 b2 ...
#include "kbench.h"
#include <cmath>
#include <cstring>
#include <string>

static int ROUNDS = 40;

struct Rope { float theta_scale, freq_scale, ext_factor, mscale, corr_low, corr_high; };

template <typename T>
static size_t ndiff(const T* a, const T* b, size_t n) {
    auto ha = kb::d2h(a, n), hb = kb::d2h(b, n);
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += memcmp(&ha[i], &hb[i], sizeof(T)) != 0;
    return d;
}

int main(int argc, char** argv) {
    kb::init();
    if (getenv("CH_ROUNDS")) ROUNDS = atoi(getenv("CH_ROUNDS"));
    std::vector<unsigned> bs;
    for (int i = 1; i < argc; ++i) bs.push_back((unsigned)atoi(argv[i]));
    if (bs.empty()) bs = {1, 2, 3, 4, 5, 8};
    kb::Module m_rms("rms_norm_gfx1201.hsaco"), m_rope("rope_tail_gfx1201.hsaco"), m_fp4("fp4_kv_quant_gfx1201.hsaco"),
        m_q8("q8_0_matvec_gfx1201.hsaco"), m_q8k("q8_k_quantize_gfx1201.hsaco"), m_c("cand_fuse_gfx1201.hsaco");
    hipFunction_t f_rms = m_rms.fn("rms_norm_weighted_batched_fast"), f_rope = m_rope.fn("rope_tail_batched"),
                  f_fp8 = m_fp4.fn("fp8_act_quant_inplace"), f_qw = m_q8.fn("q8_0_quantize_f32_wave"),
                  f_cast = m_q8k.fn("f32_to_f16_cast_2d");
    kb::Module m_hcp("hc_post_gfx1201.hsaco"), m_vadd("vec_add_gfx1201.hsaco");
    hipFunction_t f_hcp = m_hcp.fn("hc_post_from_split_batched"), f_hcpa = m_hcp.fn("hc_post_from_split_batched_add"),
                  f_vadd = m_vadd.fn("vec_add_inplace");
    float* cb_bo = kb::dalloc<float>(8 * 5120); float* cb_rem = kb::dalloc<float>(8 * 5120);
    float* cb_rh = kb::dalloc<float>(8 * 4 * 5120); float* cb_sp = kb::dalloc<float>(8 * 44); float* cb_out = kb::dalloc<float>(8 * 4 * 5120);
    kb::fill_f32(cb_bo, 8 * 5120, 31); kb::fill_f32(cb_rem, 8 * 5120, 32, -1e-3f, 1e-3f); kb::fill_f32(cb_rh, 8 * 4 * 5120, 33); kb::fill_f32(cb_sp, 8 * 44, 34);
    // CH_INTREE=1: the fused kernels from the IN-TREE code objects (rope_tail.hip / rms_norm.hip)
    const bool intree = getenv("CH_INTREE") != nullptr;
    hipFunction_t f_kv = intree ? m_rope.fn("kv_rms_rope_fp8") : m_c.fn("kv_rms_rope_fp8"),
                  f_rq = intree ? m_rms.fn("rms_quant_q8_1280_batched") : m_c.fn("rms_quant_q8_1280"),
                  f_rc = intree ? m_rope.fn("rope_tail_batched_copy") : m_c.fn("rope_tail_batched_copy"),
                  f_riq = intree ? m_rope.fn("rope_inv_quant_q8") : m_c.fn("rope_inv_quant_q8");
    const unsigned BM = 8;
    const float eps = 1e-6f;
    // buffers (max 8 rows)
    float* kv_raw = kb::dalloc<float>(BM * 512); float* kv_a = kb::dalloc<float>(BM * 512); float* kv_b = kb::dalloc<float>(BM * 512);
    float* w512 = kb::dalloc<float>(512); float* w1280 = kb::dalloc<float>(1280); float* w5120 = kb::dalloc<float>(5120);
    float* qr = kb::dalloc<float>(BM * 1280); float* qrn_a = kb::dalloc<float>(BM * 1280); float* qrn_b = kb::dalloc<float>(BM * 1280);
    int8_t* xq_a = kb::dalloc<int8_t>(BM * 32768); int8_t* xq_b = kb::dalloc<int8_t>(BM * 32768);
    float* xs_a = kb::dalloc<float>(BM * 1024); float* xs_b = kb::dalloc<float>(BM * 1024);
    uint16_t* h16 = kb::dalloc<uint16_t>(BM * 32768 + 64);
    float* q = kb::dalloc<float>(BM * 32768); float* qn_a = kb::dalloc<float>(BM * 32768); float* qn_b = kb::dalloc<float>(BM * 32768);
    float* heads_src = kb::dalloc<float>(BM * 32768); float* heads_a = kb::dalloc<float>(BM * 32768); float* heads_b = kb::dalloc<float>(BM * 32768);
    float* ain = kb::dalloc<float>(BM * 5120); float* low = kb::dalloc<float>(BM * 8192);
    int* pos = kb::dalloc<int>(BM);
    kb::fill_f32(kv_raw, BM * 512, 1, -3.f, 3.f); kb::fill_f32(w512, 512, 2, 0.2f, 1.8f); kb::fill_f32(w1280, 1280, 3, 0.2f, 1.8f);
    kb::fill_f32(w5120, 5120, 4, 0.2f, 1.8f); kb::fill_f32(qr, BM * 1280, 5, -2.f, 2.f); kb::fill_f32(q, BM * 32768, 6, -2.f, 2.f);
    kb::fill_f32(heads_src, BM * 32768, 7, -2.f, 2.f); kb::fill_f32(ain, BM * 5120, 8, -2.f, 2.f); kb::fill_f32(low, BM * 8192, 9, -2.f, 2.f);
    { std::vector<int> hp(BM); for (unsigned i = 0; i < BM; ++i) hp[i] = 70000 + 13 * i; KB_CHECK(hipMemcpy(pos, hp.data(), BM * 4, hipMemcpyHostToDevice)); }
    const Rope ropes[2] = {{powf(10000.f, -2.f / 64.f), 1.f, 0.f, 1.f, 0.f, 0.f},              // plain
                           {powf(160000.f, -2.f / 64.f), 0.0625f, 1.f, 1.2772588f, 11.3f, 23.7f}};  // YaRN-like
    const unsigned pad = getenv("CH_NOPAD") ? 0u : 1u;  // production V41_GRID_PAD
    int bad = 0;
    auto rep = [&](const std::string& nm, size_t d) { printf("CMP %s: %s (%zu)\n", nm.c_str(), d ? "MISMATCH" : "bit-exact", d); if (d) bad = 1; };
    for (unsigned b : bs) {
        // ---------------- production launches
        auto rms = [&](hipStream_t s, float* o, const float* x, const float* w, unsigned n) {
            kb::launch(f_rms, dim3(b), dim3(256), 0, s, o, x, w, n, eps); };
        auto rope = [&](hipStream_t s, float* x, unsigned nh, unsigned hd, const Rope& r, int inv) {
            kb::launch(f_rope, dim3(nh + pad, 1, b), dim3(32), 0, s, x, (const int*)pos, nh, hd, 64u, r.theta_scale, r.freq_scale,
                       r.ext_factor, r.mscale, r.corr_low, r.corr_high, inv); };
        auto fp8 = [&](hipStream_t s, float* x) { kb::launch(f_fp8, dim3(b + pad), dim3(512), 0, s, x, b, 512u); };
        auto quant = [&](hipStream_t s, int8_t* xq, float* xs, const float* x, unsigned n) {
            const unsigned blocks = n / 32 * b; kb::launch(f_qw, dim3((blocks + 7) / 8 + 1), dim3(256), 0, s, xq, xs, x, blocks); };
        auto cast = [&](hipStream_t s, const float* x, unsigned cols) {
            const unsigned threads = b * (cols / 8); kb::launch(f_cast, dim3((threads + 255) / 256 + pad), dim3(256), 0, s, h16, x, b, cols, cols); };
        for (int ri = 0; ri < 2; ++ri) {
            const Rope& r = ropes[ri];
            const std::string rt = ri ? " yarn" : " plain";
            // (d1) kv chain
            rms(0, kv_a, kv_raw, w512, 512); rope(0, kv_a, 1, 512, r, 0); fp8(0, kv_a);
            kb::launch(f_kv, dim3(b), dim3(256), 0, 0, kv_b, (const float*)kv_raw, (const float*)w512, eps, (const int*)pos,
                       r.theta_scale, r.freq_scale, r.ext_factor, r.mscale, r.corr_low, r.corr_high);
            KB_CHECK(hipDeviceSynchronize());
            rep("kv_rms_rope_fp8 b=" + std::to_string(b) + rt, ndiff(kv_a, kv_b, b * 512));
            // (d3) q copy + rope
            KB_CHECK(hipMemcpy(qn_a, q, (size_t)b * 32768 * 4, hipMemcpyDeviceToDevice)); rope(0, qn_a, 64, 512, r, 0);
            kb::launch(f_rc, dim3(64 + pad, 1, b), dim3(128), 0, 0, qn_b, (const float*)q, (const int*)pos, 64u, 512u, 64u, r.theta_scale,
                       r.freq_scale, r.ext_factor, r.mscale, r.corr_low, r.corr_high, 0);
            KB_CHECK(hipDeviceSynchronize());
            rep("rope_tail_batched_copy b=" + std::to_string(b) + rt, ndiff(qn_a, qn_b, (size_t)b * 32768));
            // (d4) inverse rope + quantize of heads
            KB_CHECK(hipMemcpy(heads_a, heads_src, (size_t)b * 32768 * 4, hipMemcpyDeviceToDevice));
            KB_CHECK(hipMemcpy(heads_b, heads_src, (size_t)b * 32768 * 4, hipMemcpyDeviceToDevice));
            rope(0, heads_a, 64, 512, r, 1); quant(0, xq_a, xs_a, heads_a, 32768);
            kb::launch(f_riq, dim3(64 + pad, b), dim3(128), 0, 0, heads_b, xq_b, xs_b, (const int*)pos, r.theta_scale, r.freq_scale,
                       r.ext_factor, r.mscale, r.corr_low, r.corr_high);
            KB_CHECK(hipDeviceSynchronize());
            rep("rope_inv_quant heads b=" + std::to_string(b) + rt, ndiff(heads_a, heads_b, (size_t)b * 32768));
            rep("rope_inv_quant xq b=" + std::to_string(b) + rt, ndiff(xq_a, xq_b, (size_t)b * 32768));
            rep("rope_inv_quant xscale b=" + std::to_string(b) + rt, ndiff(xs_a, xs_b, (size_t)b * 1024));
        }
        // (d2) q_a norm + quantize
        rms(0, qrn_a, qr, w1280, 1280); quant(0, xq_a, xs_a, qrn_a, 1280);
        kb::launch(f_rq, dim3(b), dim3(256), 0, 0, qrn_b, xq_b, xs_b, (const float*)qr, (const float*)w1280, eps);
        KB_CHECK(hipDeviceSynchronize());
        rep("rms_quant qr_normed b=" + std::to_string(b), ndiff(qrn_a, qrn_b, (size_t)b * 1280));
        rep("rms_quant xq b=" + std::to_string(b), ndiff(xq_a, xq_b, (size_t)b * 1280));
        rep("rms_quant xscale b=" + std::to_string(b), ndiff(xs_a, xs_b, (size_t)b * 40));

        // ---------------- timing (graph, warm): one variant per chain shape
        const Rope& r = ropes[1];
        std::vector<kb::Variant> vs = {
            {"kv: rms + rope + fp8 (3 nodes)", [&](hipStream_t s) { rms(s, kv_a, kv_raw, w512, 512); rope(s, kv_a, 1, 512, r, 0); fp8(s, kv_a); }},
            {"kv: fused (1 node)", [&](hipStream_t s) { kb::launch(f_kv, dim3(b), dim3(256), 0, s, kv_b, (const float*)kv_raw, (const float*)w512, eps, (const int*)pos,
                                                                  r.theta_scale, r.freq_scale, r.ext_factor, r.mscale, r.corr_low, r.corr_high); }},
            {"qa: rms + cast_qr + quant (3 nodes)", [&](hipStream_t s) { rms(s, qrn_a, qr, w1280, 1280); cast(s, qrn_a, 1280); quant(s, xq_a, xs_a, qrn_a, 1280); }},
            {"qa: rms + quant (2 nodes)", [&](hipStream_t s) { rms(s, qrn_a, qr, w1280, 1280); quant(s, xq_a, xs_a, qrn_a, 1280); }},
            {"qa: fused (1 node)", [&](hipStream_t s) { kb::launch(f_rq, dim3(b), dim3(256), 0, s, qrn_b, xq_b, xs_b, (const float*)qr, (const float*)w1280, eps); }},
            {"q: memcpy + rope (2 nodes)", [&](hipStream_t s) { KB_CHECK(hipMemcpyAsync(qn_a, q, (size_t)b * 32768 * 4, hipMemcpyDeviceToDevice, s)); rope(s, qn_a, 64, 512, r, 0); }},
            {"q: rope_copy (1 node)", [&](hipStream_t s) { kb::launch(f_rc, dim3(64 + pad, 1, b), dim3(128), 0, s, qn_b, (const float*)q, (const int*)pos, 64u, 512u, 64u, r.theta_scale,
                                                                     r.freq_scale, r.ext_factor, r.mscale, r.corr_low, r.corr_high, 0); }},
            {"oproj: rope_inv + cast_heads + quant (3)", [&](hipStream_t s) { rope(s, heads_a, 64, 512, r, 1); cast(s, heads_a, 32768); quant(s, xq_a, xs_a, heads_a, 32768); }},
            {"oproj: rope_inv + quant (2 nodes)", [&](hipStream_t s) { rope(s, heads_a, 64, 512, r, 1); quant(s, xq_a, xs_a, heads_a, 32768); }},
            {"oproj: fused (1 node)", [&](hipStream_t s) { kb::launch(f_riq, dim3(64 + pad, b), dim3(128), 0, s, heads_b, xq_b, xs_b, (const int*)pos, r.theta_scale, r.freq_scale,
                                                                   r.ext_factor, r.mscale, r.corr_low, r.corr_high); }},
            {"dead: cast_input + quant + dup quant (3)", [&](hipStream_t s) { cast(s, ain, 5120); quant(s, xq_a, xs_a, ain, 5120); quant(s, xq_a, xs_a, ain, 5120); }},
            {"dead-free: quant (1 node)", [&](hipStream_t s) { quant(s, xq_a, xs_a, ain, 5120); }},
            {"dead: cast_low + quant (2 nodes)", [&](hipStream_t s) { cast(s, low, 8192); quant(s, xq_b, xs_b, low, 8192); }},
            {"dead-free: quant low (1 node)", [&](hipStream_t s) { quant(s, xq_b, xs_b, low, 8192); }},
            {"combine: vec_add + hc_post (2 nodes)", [&](hipStream_t s) {
                kb::launch(f_vadd, dim3((b * 5120 + 255) / 256), dim3(256), 0, s, cb_bo, (const float*)cb_rem, b * 5120u);
                kb::launch(f_hcp, dim3(20, 4, b), dim3(256), 0, s, cb_out, (const float*)cb_bo, (const float*)cb_rh, (const float*)cb_sp, 24u, 5120u, 4u); }},
            {"combine: hc_post_add (1 node)", [&](hipStream_t s) {
                kb::launch(f_hcpa, dim3(20, 4, b), dim3(256), 0, s, cb_out, (const float*)cb_bo, (const float*)cb_rem, (const float*)cb_rh, (const float*)cb_sp, 24u, 5120u, 4u); }},
        };
        kb::AbOpts o; o.graph = true; o.inner = 10; o.rounds = ROUNDS; o.warm_ms = 60;
        std::string tag = "decode chains b=" + std::to_string(b) + " warm graph inner=10";
        o.tag = tag.c_str();
        kb::ab(vs, o);
    }
    printf(bad ? "RESULT: MISMATCH\n" : "RESULT: all fused outputs bit-exact\n");
    return bad;
}
