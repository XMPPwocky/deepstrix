// gcheck.cpp -- reviewer's independent correctness check for E_indexer/gather_b128.
// Compares gather_u4_r1 / gather_u4_r4 against the PRODUCTION indexer_gather_batched (and the
// production kernel against a CPU reference) on shapes the engineer did not test:
//   * sentinel rows (selected < 0), scattered and trailing (production's n_comp_per < top_k case)
//   * comp_base_per == nullptr (prefill path) and non-null with per-row bases (arena path)
//   * odd b, odd top_k, top_k=1, all-sentinel rows, b=5 (max one-lane decode), b=8, b=512, b=1024
//   * an odd-sized store (100003 rows) filled with ARBITRARY 16-bit patterns (NaN/Inf/denormal
//     encodings included) so any canonicalisation on the copy path would show up
//   * destinations pre-filled with 0xA5A5 so rows that must be left untouched are verified too
#include "kbench.h"

#include <cstring>
#include <random>
#include <string>

static const unsigned HD = 512;  // N_HEAD_DIM: main comp_kv row = 1 KB f16

__global__ void k_fill_bits(uint16_t* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = (uint16_t)x;
    }
}

struct Case { const char* name; unsigned b, top_k; double sent_frac; bool use_base; bool local; bool tail; };

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    kb::Module base(dir + "/base_indexer_gather_gfx1201.hsaco");
    kb::Module cand(dir + "/cand_gather_gfx1201.hsaco");
    hipFunction_t f_base = base.fn("indexer_gather_batched");
    struct C { const char* n; unsigned rpw, blk; hipFunction_t f; };
    C cs[] = {{"gather_u4_r1", 1, 64, cand.fn("gather_u4_r1")},
              {"gather_u4_r4", 4, 256, cand.fn("gather_u4_r4")}};

    const unsigned rows = 100003;  // odd row count, ~100 MB
    uint16_t* comp = kb::dalloc<uint16_t>((size_t)rows * HD);
    hipLaunchKernelGGL(k_fill_bits, dim3(4096), dim3(256), 0, 0, comp, (size_t)rows * HD, 0xC0FFEEu);
    KB_CHECK(hipDeviceSynchronize());
    std::vector<uint16_t> hcomp = kb::d2h(comp, (size_t)rows * HD);

    const unsigned B_MAX = 1024, K_MAX = 512;
    const size_t dst_n = (size_t)B_MAX * K_MAX * HD;
    uint16_t* dst = kb::dalloc<uint16_t>(dst_n);
    uint16_t* dst2 = kb::dalloc<uint16_t>(dst_n);
    int* sel = kb::dalloc<int>((size_t)B_MAX * K_MAX);
    int* base_per = kb::dalloc<int>(B_MAX);

    Case cases[] = {
        {"b1 k512 random, base=null", 1, 512, 0.0, false, false, false},
        {"b1 k512 random, 30% sentinels, base", 1, 512, 0.3, true, false, false},
        {"b4 k512 random, 10% sentinels, base", 4, 512, 0.1, true, false, false},
        {"b5 k512 random, base (max one-lane rows)", 5, 512, 0.0, true, false, false},
        {"b8 k512 random, 2% sentinels, base", 8, 512, 0.02, true, false, false},
        {"b7 k333 random, 5% sentinels, base (odd)", 7, 333, 0.05, true, false, false},
        {"b2 k1 random, base", 2, 1, 0.0, true, false, false},
        {"b2 k512 ALL sentinels, base", 2, 512, 1.0, true, false, false},
        {"b3 k512 trailing sentinels (n_comp<top_k), null", 3, 512, 0.5, false, false, true},
        {"b512 k512 local, base (prefill lane)", 512, 512, 0.0, true, true, false},
        {"b1024 k512 local, base=null (max prefill chunk)", 1024, 512, 0.0, false, true, false},
    };
    std::mt19937 rng(12345);
    int fails = 0, checks = 0;
    for (const Case& c : cases) {
        const size_t n = (size_t)c.b * c.top_k;
        std::vector<int> hsel(n);
        std::vector<int> hbase(c.b, 0);
        if (c.use_base) for (unsigned b = 0; b < c.b; b++) hbase[b] = (int)(rng() % (rows / 2));
        for (unsigned b = 0; b < c.b; b++) {
            const unsigned lim = rows - (unsigned)hbase[b];
            const unsigned centre = rng() % lim;
            for (unsigned k = 0; k < c.top_k; k++) {
                bool sentinel;
                if (c.sent_frac >= 1.0) sentinel = true;
                else if (c.tail) sentinel = k >= (unsigned)(c.top_k * (1.0 - c.sent_frac));
                else sentinel = (rng() % 1000) < (unsigned)(c.sent_frac * 1000);
                if (sentinel) { hsel[(size_t)b * c.top_k + k] = -1; continue; }
                const unsigned r = c.local ? (centre + (k * 3u + rng() % 3u)) % lim : rng() % lim;
                hsel[(size_t)b * c.top_k + k] = (int)r;
            }
        }
        KB_CHECK(hipMemcpy(sel, hsel.data(), n * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(base_per, hbase.data(), c.b * 4, hipMemcpyHostToDevice));
        const int* bp = c.use_base ? base_per : nullptr;
        const size_t dn = n * HD;
        KB_CHECK(hipMemset(dst, 0xA5, dn * 2));
        // production launch: grid (top_k, b, ceil(head_dim/256)) x 256
        kb::launch(f_base, dim3(c.top_k, c.b, 2), dim3(256), 0, 0, dst, (const uint16_t*)comp, (const int*)sel, c.top_k, HD, bp);
        KB_CHECK(hipDeviceSynchronize());
        std::vector<uint16_t> got_base = kb::d2h(dst, dn);
        size_t base_bad = 0;
        for (unsigned b = 0; b < c.b; b++)
            for (unsigned k = 0; k < c.top_k; k++) {
                const int s = hsel[(size_t)b * c.top_k + k];
                const uint16_t* row = got_base.data() + ((size_t)b * c.top_k + k) * HD;
                for (unsigned d = 0; d < HD; d++) {
                    const uint16_t exp = s < 0 ? 0xA5A5 : hcomp[((size_t)hbase[b] + (unsigned)s) * HD + d];
                    base_bad += row[d] != exp;
                }
            }
        printf("CASE %-50s base-vs-CPU diff=%zu %s\n", c.name, base_bad, base_bad ? "BASE-MISMATCH" : "ok");
        checks++; if (base_bad) fails++;
        for (auto& k : cs) {
            KB_CHECK(hipMemset(dst2, 0xA5, dn * 2));
            const unsigned gx = (c.top_k + k.rpw - 1) / k.rpw;
            kb::launch(k.f, dim3(gx, c.b), dim3(k.blk), 0, 0, dst2, (const uint16_t*)comp, (const int*)sel, c.top_k, HD, bp);
            KB_CHECK(hipDeviceSynchronize());
            std::vector<uint16_t> got = kb::d2h(dst2, dn);
            size_t diff = 0;
            for (size_t i = 0; i < dn; i++) diff += got[i] != got_base[i];
            printf("CMP  %-50s %-14s bitexact=%s diff=%zu\n", c.name, k.n, diff ? "no" : "YES", diff);
            checks++; if (diff) fails++;
        }
    }
    printf("GCHECK checks=%d fails=%d %s\n", checks, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
