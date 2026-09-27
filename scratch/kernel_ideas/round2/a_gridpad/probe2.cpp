// probe2: map the slow window. (1) many WG counts around 2^k at block 32/64/256/512, all in ONE
// interleaved kb::ab per block size (valid = grid, i.e. no idle WG); (2) kernel-length dependence:
// probe_spin at 2048x32 / 1024x256 vs +1 with growing per-thread work.
// usage: ./probe2 hsaco
#include "kbench.h"
#include <string>
int main(int argc, char** argv) {
    kb::init();
    kb::Module m(argv[1]);
    hipFunction_t frw = m.fn("probe_rw");
    hipFunction_t fsp = m.fn("probe_spin");
    const size_t cap = (size_t)8 << 20;
    float* out = kb::dalloc<float>(cap);
    float* in = kb::dalloc<float>(cap);
    kb::fill_f32(in, cap, 1);
    const int rounds = getenv("PR_ROUNDS") ? atoi(getenv("PR_ROUNDS")) : 40;
    struct Set { unsigned bs; std::vector<unsigned> g; };
    const Set sets[] = {
        {32, {1024, 1025, 1280, 1536, 1792, 1920, 2000, 2040, 2046, 2047, 2048, 2049, 2050, 2056, 2304, 2560, 3072, 3584, 4095, 4096, 4097}},
        {64, {512, 513, 768, 896, 1000, 1023, 1024, 1025, 1100, 1536, 2047, 2048, 2049, 3072, 4096, 4097, 6144, 8192}},
        {256, {128, 129, 160, 192, 224, 240, 250, 255, 256, 257, 320, 384, 448, 500, 511, 512, 513, 640, 768, 1000, 1023, 1024, 1025, 1280, 1536, 2048}},
        {512, {64, 100, 127, 128, 129, 192, 255, 256, 257, 384, 512, 513, 768, 1024, 1025}},
    };
    for (const Set& st : sets) {
        std::vector<kb::Variant> vs;
        for (unsigned g : st.g) {
            unsigned bs = st.bs;
            vs.push_back({"G=" + std::to_string(g), [=](hipStream_t s) {
                kb::launch(frw, dim3(g), dim3(bs), 0, s, out, (const float*)in, g);
            }});
        }
        kb::AbOpts o; o.graph = true; o.inner = 10; o.rounds = rounds; o.warm_ms = 40;
        std::string tag = "window block=" + std::to_string(st.bs);
        o.tag = tag.c_str();
        kb::ab(vs, o);
    }
    // kernel length dependence
    struct L { unsigned g, bs; };
    for (L l : {L{2048, 32}, L{1024, 256}}) {
        for (unsigned it : {0u, 64u, 256u, 1024u, 4096u}) {
            std::vector<kb::Variant> vs;
            for (unsigned pad : {0u, 1u}) {
                unsigned g = l.g + pad, bs = l.bs, valid = l.g;
                vs.push_back({"G=" + std::to_string(l.g) + "+" + std::to_string(pad), [=](hipStream_t s) {
                    kb::launch(fsp, dim3(g), dim3(bs), 0, s, out, (const float*)in, valid, it);
                }});
            }
            kb::AbOpts o; o.graph = true; o.inner = 10; o.rounds = rounds; o.warm_ms = 40;
            std::string tag = "spin G=" + std::to_string(l.g) + " block=" + std::to_string(l.bs) + " iters=" + std::to_string(it);
            o.tag = tag.c_str();
            kb::ab(vs, o);
        }
    }
    return 0;
}
