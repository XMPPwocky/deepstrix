// Reviewer's extra correctness probe for cand_rms (rms_norm_weighted_batched_fast).
// Shapes / inputs the engineer did not test:
//   * b = 3, 7 (odd), 8 (max production rows), 1 with the 1-row head kernel
//   * adversarial x: all-zero, all-equal, large magnitude (1e18), tiny (1e-20), mixed with sign
//   * generic-path n: 256, 768, 100 (n < block: production loop bounds), 5376 (5120+256, NPT would
//     be 21 -> generic), 128 (indexer head dim N_INDEXER_HEAD_DIM)
//   * eps = 1e-20 (production RMS_EPS) and 1e-6
//   * out-of-range write guard: out2 has a 4096-float sentinel tail after b*n that must stay intact
#include "kbench.h"
#include <cmath>
#include <string>
#include <vector>

static int fails = 0;
static void chk(const char* tag, const kb::Cmp& c) {
    kb::print_cmp(tag, c);
    if (c.n_bit_diff != 0) fails++;  // bit-identical (incl. identical inf/nan) is the bar
}

int main(int, char** argv) {
    kb::init();
    std::string dir = getenv("HARNESS_DIR") ? getenv("HARNESS_DIR") : ".";
    kb::Module base(dir + "/base_rms_norm_gfx1201.hsaco");
    kb::Module cand(dir + "/cand_rms_gfx1201.hsaco");
    hipFunction_t fb = base.fn("rms_norm_weighted_batched");
    hipFunction_t fb1 = base.fn("rms_norm_weighted");
    hipFunction_t fc = cand.fn("rms_norm_weighted_batched_fast");
    const unsigned NMAX = 5376, BMAX = 8, GUARD = 4096;
    float* x = kb::dalloc<float>((size_t)BMAX * NMAX);
    float* w = kb::dalloc<float>(NMAX);
    float* o1 = kb::dalloc<float>((size_t)BMAX * NMAX + GUARD);
    float* o2 = kb::dalloc<float>((size_t)BMAX * NMAX + GUARD);
    std::vector<float> hx((size_t)BMAX * NMAX), hw(NMAX);

    struct Case { const char* name; float lo, hi; int mode; };
    // mode: 0 random in [lo,hi]; 1 all-zero; 2 all-equal (lo); 3 alternating +-lo
    Case cases[] = {
        {"rand[-2,2]", -2.f, 2.f, 0}, {"rand[-1e18,1e18]", -1e18f, 1e18f, 0},
        {"rand[-1e-20,1e-20]", -1e-20f, 1e-20f, 0}, {"zero", 0, 0, 1},
        {"all-equal 0.75", 0.75f, 0, 2}, {"alt +-3", 3.f, 0, 3}, {"rand[0,1e-3]", 0.f, 1e-3f, 0},
    };
    const unsigned ns[] = {5120u, 1280u, 512u, 128u, 256u, 768u, 100u, 5376u, 4864u};
    const unsigned bs[] = {1u, 3u, 7u, 8u};
    const float epss[] = {1.0e-20f, 1.0e-6f};
    uint32_t seed = 1000;
    int ncmp = 0;
    for (const Case& c : cases) {
        // host-side fill so the adversarial patterns are exact
        for (size_t i = 0; i < hx.size(); ++i) {
            uint32_t r = (uint32_t)i * 2654435761u ^ seed; r ^= r >> 16; r *= 0x7feb352du; r ^= r >> 15;
            float u = (r >> 8) * (1.0f / 16777216.0f);
            switch (c.mode) {
                case 0: hx[i] = c.lo + (c.hi - c.lo) * u; break;
                case 1: hx[i] = 0.f; break;
                case 2: hx[i] = c.lo; break;
                default: hx[i] = (i & 1) ? -c.lo : c.lo; break;
            }
        }
        for (size_t i = 0; i < hw.size(); ++i) {
            uint32_t r = (uint32_t)i * 2246822519u ^ (seed + 7); r ^= r >> 16; r *= 0x7feb352du; r ^= r >> 15;
            hw[i] = -1.5f + 3.0f * ((r >> 8) * (1.0f / 16777216.0f));  // includes negative weights
        }
        seed += 17;
        KB_CHECK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        KB_CHECK(hipMemcpy(w, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
        for (float eps : epss) for (unsigned n : ns) for (unsigned b : bs) {
            KB_CHECK(hipMemset(o1, 0x7f, ((size_t)BMAX * NMAX + GUARD) * 4));
            KB_CHECK(hipMemset(o2, 0x7f, ((size_t)BMAX * NMAX + GUARD) * 4));
            kb::launch(fb, dim3(b), dim3(256), 0, 0, o1, (const float*)x, (const float*)w, n, eps);
            kb::launch(fc, dim3(b), dim3(256), 0, 0, o2, (const float*)x, (const float*)w, n, eps);
            KB_CHECK(hipDeviceSynchronize());
            // compare the whole buffer incl. the untouched region and the guard tail
            auto a = kb::d2h(o1, (size_t)BMAX * NMAX + GUARD), g = kb::d2h(o2, (size_t)BMAX * NMAX + GUARD);
            char tag[128]; snprintf(tag, sizeof tag, "%s n=%u b=%u eps=%g", c.name, n, b, eps);
            chk(tag, kb::compare_f32(a.data(), g.data(), a.size()));
            ncmp++;
        }
        // 1-row head kernel vs candidate at grid (1), n = 5120
        KB_CHECK(hipMemset(o1, 0x7f, ((size_t)BMAX * NMAX + GUARD) * 4));
        KB_CHECK(hipMemset(o2, 0x7f, ((size_t)BMAX * NMAX + GUARD) * 4));
        kb::launch(fb1, dim3(1), dim3(256), 0, 0, o1, (const float*)x, (const float*)w, 5120u, 1.0e-20f);
        kb::launch(fc, dim3(1), dim3(256), 0, 0, o2, (const float*)x, (const float*)w, 5120u, 1.0e-20f);
        KB_CHECK(hipDeviceSynchronize());
        auto a = kb::d2h(o1, (size_t)BMAX * NMAX + GUARD), g = kb::d2h(o2, (size_t)BMAX * NMAX + GUARD);
        char tag[128]; snprintf(tag, sizeof tag, "%s head1 n=5120", c.name);
        chk(tag, kb::compare_f32(a.data(), g.data(), a.size()));
        ncmp++;
    }
    printf("EXTRA cmps=%d fails=%d\n", ncmp, fails);
    return fails ? 1 : 0;
}
