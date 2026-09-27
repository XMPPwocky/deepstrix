// (c) z16 chunking harness: production f16_matvec_batched (and _h20 for the router) from the
// in-tree code object vs the grid.z = ceil(B/NB) candidates (cand_z16.hip), cold weights (rotating
// copies > MALL, graph of `copies` calls), bit-exact check of every candidate at every (shape, b).
//   ./harness <shape> b1 b2 ...        shape: idxq comp1 proj router tail
#include "kbench.h"
#include <cstring>
#include <string>

static int ROUNDS = 30;

struct Shape { const char* name; unsigned n, k; };

int main(int argc, char** argv) {
    kb::init();
    if (getenv("Z_ROUNDS")) ROUNDS = atoi(getenv("Z_ROUNDS"));
    const std::string sname = argc > 1 ? argv[1] : "idxq";
    std::vector<unsigned> bs;
    for (int i = 2; i < argc; ++i) bs.push_back((unsigned)atoi(argv[i]));
    Shape sh{"idxq", 4096, 1280};
    if (sname == "comp1") sh = {"comp1", 512, 5120};
    if (sname == "proj") sh = {"proj", 32, 5120};
    if (sname == "router") sh = {"router", 384, 5120};
    if (sname == "tail") sh = {"tail", 1000, 1000};   // k % 256 != 0: hoisted kernels fall back
    kb::Module base("f16_matvec_gfx1201.hsaco");
    kb::Module cand("cand_z16_gfx1201.hsaco");
    hipFunction_t f_base = base.fn("f16_matvec_batched"), f_h20 = base.fn("f16_matvec_batched_h20");
    struct C { std::string name; hipFunction_t f; unsigned nb; };
    std::vector<C> cands;
    const bool intree = getenv("Z_INTREE") != nullptr;  // "h8" = the IN-TREE f16_matvec_batched_z16_n<NB>
    for (unsigned nb : {1u, 2u, 4u, 8u, 16u}) {
        cands.push_back({"plain_n" + std::to_string(nb), cand.fn(("f16_mv_z_plain_n" + std::to_string(nb)).c_str()), nb});
        cands.push_back({"h8_n" + std::to_string(nb),
                         intree ? base.fn(("f16_matvec_batched_z16_n" + std::to_string(nb)).c_str())
                                : cand.fn(("f16_mv_z_h8_n" + std::to_string(nb)).c_str()), nb});
    }
    for (unsigned nb : {1u, 4u, 16u}) cands.push_back({"h4_n" + std::to_string(nb), cand.fn(("f16_mv_z_h4_n" + std::to_string(nb)).c_str()), nb});
    cands.push_back({"h16_n4", cand.fn("f16_mv_z_h16_n4"), 4});

    const size_t wel = (size_t)sh.n * sh.k;
    const int copies = std::max(2, (int)((80ull << 20) / (wel * 2)) + 1);
    const int ncop = std::min(copies, 64);  // graph size cap (proj: 64 x 0.33 MB = 21 MB -> warm-ish, noted)
    std::vector<uint16_t*> W;
    for (int c = 0; c < ncop; ++c) { W.push_back(kb::dalloc<uint16_t>(wel)); kb::fill_f16(W.back(), wel, 3 + c, -0.05f, 0.05f); }
    unsigned bmax = 1; for (unsigned b : bs) bmax = std::max(bmax, b);
    float* x = kb::dalloc<float>((size_t)bmax * sh.k); kb::fill_f32(x, (size_t)bmax * sh.k, 5, -1.f, 1.f);
    float* o0 = kb::dalloc<float>((size_t)bmax * sh.n);
    float* o1 = kb::dalloc<float>((size_t)bmax * sh.n);
    const unsigned gx = (sh.n + 7) / 8;
    fprintf(stderr, "[z16] %s %ux%u, %d weight copies (%.1f MB)\n", sh.name, sh.n, sh.k, ncop, ncop * wel * 2 / 1e6);
    int bad = 0;
    for (unsigned b : bs) {
        // correctness vs the production kernel on copy 0
        KB_CHECK(hipMemset(o0, 0xAA, (size_t)bmax * sh.n * 4));
        kb::launch(f_base, dim3(gx, 1, b), dim3(256), 0, 0, o0, (const uint16_t*)W[0], (const float*)x, sh.k, sh.n);
        KB_CHECK(hipDeviceSynchronize());
        auto ref = kb::d2h(o0, (size_t)b * sh.n);
        auto check = [&](const std::string& nm, std::function<void(float*)> run) {
            KB_CHECK(hipMemset(o1, 0x55, (size_t)bmax * sh.n * 4));
            run(o1);
            KB_CHECK(hipDeviceSynchronize());
            auto got = kb::d2h(o1, (size_t)b * sh.n);
            size_t d = 0; for (size_t i = 0; i < ref.size(); ++i) d += memcmp(&ref[i], &got[i], 4) != 0;
            printf("CMP %s %s b=%u bit_diff=%zu of %zu%s\n", sh.name, nm.c_str(), b, d, ref.size(), d ? "  MISMATCH" : "");
            if (d) bad = 1;
        };
        check("h20", [&](float* o) { kb::launch(f_h20, dim3(gx, 1, b), dim3(256), 0, 0, o, (const uint16_t*)W[0], (const float*)x, sh.k, sh.n); });
        for (auto& c : cands)
            check(c.name, [&](float* o) { kb::launch(c.f, dim3(gx, 1, (b + c.nb - 1) / c.nb), dim3(256), 0, 0, o, (const uint16_t*)W[0], (const float*)x, sh.k, sh.n, b); });
        if (sname == "tail") continue;
        // timing: base, (+1 pad), h20, and the candidates whose NB is 16 or the smallest >= b
        unsigned tight = 16; for (unsigned nb : {1u, 2u, 4u, 8u, 16u}) if (nb >= b) { tight = nb; break; }
        std::vector<kb::Variant> vs;
        int* rot = new int[64]();
        int slot = 0;
        auto rw = [&, ncop](int s) { int* r = rot + s; return [r, ncop]() { return (*r)++ % ncop; }; };
        { auto nx = rw(slot++); vs.push_back({"base f16_matvec_batched", [=](hipStream_t st) { kb::launch(f_base, dim3(gx, 1, b), dim3(256), 0, st, o0, (const uint16_t*)W[nx()], (const float*)x, sh.k, sh.n); }, (double)wel * 2}); }
        { auto nx = rw(slot++); vs.push_back({"base +1 WG", [=](hipStream_t st) { kb::launch(f_base, dim3(gx + 1, 1, b), dim3(256), 0, st, o0, (const uint16_t*)W[nx()], (const float*)x, sh.k, sh.n); }, (double)wel * 2}); }
        if (sname == "router") { auto nx = rw(slot++); vs.push_back({"h20 (router prod)", [=](hipStream_t st) { kb::launch(f_h20, dim3(gx, 1, b), dim3(256), 0, st, o0, (const uint16_t*)W[nx()], (const float*)x, sh.k, sh.n); }, (double)wel * 2}); }
        const bool confirm = getenv("Z_CONFIRM") != nullptr;
        for (auto& c : cands) {
            if (c.nb != 16 && c.nb != tight) continue;
            if (confirm && c.name != "h8_n" + std::to_string(tight)) continue;
            auto nx = rw(slot++);
            hipFunction_t f = c.f; unsigned nb = c.nb;
            vs.push_back({"cand " + c.name, [=](hipStream_t st) { kb::launch(f, dim3(gx, 1, (b + nb - 1) / nb), dim3(256), 0, st, o1, (const uint16_t*)W[nx()], (const float*)x, sh.k, sh.n, b); }, (double)wel * 2});
            if (sname == "idxq" || sname == "comp1") {
                auto ny = rw(slot++);
                vs.push_back({"cand " + c.name + " +1", [=](hipStream_t st) { kb::launch(f, dim3(gx + 1, 1, (b + nb - 1) / nb), dim3(256), 0, st, o1, (const uint16_t*)W[ny()], (const float*)x, sh.k, sh.n, b); }, (double)wel * 2});
            }
        }
        kb::AbOpts o; o.graph = true; o.inner = ncop; o.rounds = ROUNDS; o.warm_ms = 60;
        std::string tag = std::string(sh.name) + " " + std::to_string(sh.n) + "x" + std::to_string(sh.k) + " b=" + std::to_string(b) +
                          " COLD W (" + std::to_string(ncop) + " copies) graph";
        o.tag = tag.c_str();
        kb::ab(vs, o);
    }
    printf(bad ? "RESULT: MISMATCH\n" : "RESULT: all candidates bit-exact\n");
    return bad;
}
