// probe_harness: grid G vs G+1 / G+7 / G-1 for synthetic kernels at several block sizes, graph
// mode (production decode stages are graph-captured), warm (small data).
// usage: ./probe_harness hsaco [nop|rw] [3d]
#include "kbench.h"
#include <string>
int main(int argc, char** argv) {
    kb::init();
    kb::Module m(argv[1]);
    const std::string kind = argc > 2 ? argv[2] : "rw";
    hipFunction_t f = m.fn(kind == "nop" ? "probe_nop" : "probe_rw");
    const bool rw = kind != "nop";
    const unsigned blocks_sz[] = {32, 64, 128, 256, 512, 1024};
    const unsigned grids[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    const size_t cap = (size_t)8 << 20;  // 8M floats = 32 MB per buffer
    float* out = kb::dalloc<float>(cap);
    float* in = kb::dalloc<float>(cap);
    kb::fill_f32(in, cap, 1);
    const int rounds = getenv("PR_ROUNDS") ? atoi(getenv("PR_ROUNDS")) : 40;
    const bool only3d = getenv("PR_ONLY3D") != nullptr;
    auto L = [=](hipStream_t s, dim3 g, unsigned bs, unsigned valid) {
        if (rw) kb::launch(f, g, dim3(bs), 0, s, out, (const float*)in, valid);
        else kb::launch(f, g, dim3(bs), 0, s, out, valid);
    };
    if (!only3d)
    for (unsigned bs : blocks_sz) {
        for (unsigned G : grids) {
            if (rw && (size_t)(G + 7) * bs > cap) continue;
            std::vector<kb::Variant> vs;
            for (unsigned o : {0u, 1u, 7u}) {
                unsigned g = G + o;
                vs.push_back({"grid=" + std::to_string(G) + "+" + std::to_string(o),
                              [=](hipStream_t s) { L(s, dim3(g), bs, G); }});
            }
            unsigned gm = G - 1;
            vs.push_back({"grid=" + std::to_string(gm) + " (valid-1)", [=](hipStream_t s) { L(s, dim3(gm), bs, gm); }});
            kb::AbOpts o; o.graph = true; o.inner = 10; o.rounds = rounds; o.warm_ms = 40;
            std::string tag = kind + " block=" + std::to_string(bs) + " G=" + std::to_string(G) +
                              " waves=" + std::to_string((size_t)G * ((bs + 31) / 32));
            o.tag = tag.c_str();
            kb::ab(vs, o);
        }
    }
    if (argc > 3) {
        struct S { unsigned x, y, z, bs; };
        const S sh[] = {{512, 2, 1, 256}, {512, 4, 1, 256}, {512, 8, 1, 256}, {512, 1, 4, 256},
                        {64, 1, 16, 32},  {64, 1, 32, 32},  {64, 1, 64, 32},  {64, 1, 512, 32},
                        {512, 4, 1, 64},  {512, 8, 1, 64},  {4096, 1, 1, 256}, {4096, 1, 4, 256}};
        for (const S& q : sh) {
            unsigned tot = q.x * q.y * q.z;
            if (rw && (size_t)(q.x + 1) * (q.y + 1) * (q.z + 1) * q.bs > cap) continue;
            std::vector<kb::Variant> vs;
            vs.push_back({"exact", [=](hipStream_t s) { L(s, dim3(q.x, q.y, q.z), q.bs, tot); }});
            vs.push_back({"x+1", [=](hipStream_t s) { L(s, dim3(q.x + 1, q.y, q.z), q.bs, tot); }});
            if (q.z > 1) vs.push_back({"z+1", [=](hipStream_t s) { L(s, dim3(q.x, q.y, q.z + 1), q.bs, tot); }});
            if (q.y > 1) vs.push_back({"y+1", [=](hipStream_t s) { L(s, dim3(q.x, q.y + 1, q.z), q.bs, tot); }});
            kb::AbOpts o; o.graph = true; o.inner = 10; o.rounds = rounds; o.warm_ms = 40;
            std::string tag = kind + " 3d " + std::to_string(q.x) + "x" + std::to_string(q.y) + "x" +
                              std::to_string(q.z) + " block=" + std::to_string(q.bs);
            o.tag = tag.c_str();
            kb::ab(vs, o);
        }
    }
    return 0;
}
