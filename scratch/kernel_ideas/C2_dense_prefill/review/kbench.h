// kbench.h — shared microbenchmark helpers for the 2026-09-26 kernel-ideas sweep.
//
// Methodology it enforces (see _infra/README.md):
//   * baseline = the PRODUCTION code object: compile the unmodified in-tree .hip with
//     $KFLAGS_V41 --genco to a .hsaco and load it with kb::Module; candidates the same way.
//   * INTERLEAVED A/B: every round runs every variant (rotating order), so production load
//     and DPM clock swings hit all variants alike. Report medians, never a single run.
//   * clock warm-up spin before timing (DPM `auto` idles the GPUs down between bursts).
//   * optional cache flush between timed blocks (64 MB MALL on the dGPU, 32 MB on the iGPU):
//     a weight-streaming kernel in production sees cold weights; a small-activation kernel
//     sees warm activations. Pick the regime production is in and SAY which.
//   * optional graph mode: decode stages run graph-captured in production; latency-bound
//     kernels must be timed the same way (graph launch cost differs from direct launch).
#pragma once
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define KB_CHECK(x)                                                                          \
    do {                                                                                     \
        hipError_t e_ = (x);                                                                 \
        if (e_ != hipSuccess) {                                                              \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,          \
                    hipGetErrorString(e_));                                                  \
            exit(2);                                                                         \
        }                                                                                    \
    } while (0)

namespace kb {

// ---------------------------------------------------------------- device / modules
inline hipDeviceProp_t init() {
    // gpu_run.sh exposes exactly one GPU, so it is always device 0.
    KB_CHECK(hipSetDevice(0));
    hipDeviceProp_t p;
    KB_CHECK(hipGetDeviceProperties(&p, 0));
    fprintf(stderr, "[kb] device 0 = %s (%s), %d CUs, %.0f MHz\n", p.name, p.gcnArchName,
            p.multiProcessorCount, p.clockRate / 1000.0);
    return p;
}

struct Module {
    hipModule_t m{};
    explicit Module(const std::string& path) { KB_CHECK(hipModuleLoad(&m, path.c_str())); }
    hipFunction_t fn(const char* name) const {
        hipFunction_t f;
        KB_CHECK(hipModuleGetFunction(&f, m, name));
        return f;
    }
};

// kb::launch(f, grid, block, shmem, stream, arg0, arg1, ...). Argument TYPES must match the
// kernel signature exactly (unsigned int vs int, float vs double, pointer vs value): they are
// passed by address with their own sizeof.
template <typename... A>
inline void launch(hipFunction_t f, dim3 g, dim3 b, unsigned shmem, hipStream_t s, A... args) {
    void* ptrs[] = {(void*)&args..., nullptr};
    KB_CHECK(hipModuleLaunchKernel(f, g.x, g.y, g.z, b.x, b.y, b.z, shmem, s, ptrs, nullptr));
}

template <typename T>
inline T* dalloc(size_t n) {
    T* p = nullptr;
    KB_CHECK(hipMalloc(&p, n * sizeof(T)));
    return p;
}
template <typename T>
inline std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    KB_CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

// ---------------------------------------------------------------- random fills (device-side)
__global__ void kb_fill_u32(uint32_t* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = x;
    }
}
__global__ void kb_fill_f32(float* p, size_t n, uint32_t seed, float lo, float hi) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = lo + (hi - lo) * ((x >> 8) * (1.0f / 16777216.0f));
    }
}
__global__ void kb_fill_f16(_Float16* p, size_t n, uint32_t seed, float lo, float hi) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = (_Float16)(lo + (hi - lo) * ((x >> 8) * (1.0f / 16777216.0f)));
    }
}
// Random BYTES (quantized weights: nibbles, scales...). Caller must then fix up any fields
// whose random value would be pathological (e.g. NaN/inf fp16 scales, E8M0 scale 0xFF).
inline void fill_bytes(void* p, size_t bytes, uint32_t seed) {
    size_t words = bytes / 4;
    if (words) hipLaunchKernelGGL(kb_fill_u32, dim3(1024), dim3(256), 0, 0, (uint32_t*)p, words, seed);
    size_t tail = bytes - words * 4;
    if (tail) {
        uint8_t t[4] = {0x5a, 0xa5, 0x3c, 0xc3};
        KB_CHECK(hipMemcpy((uint8_t*)p + words * 4, t, tail, hipMemcpyHostToDevice));
    }
    KB_CHECK(hipDeviceSynchronize());
}
inline void fill_f32(float* p, size_t n, uint32_t seed, float lo = -1.f, float hi = 1.f) {
    hipLaunchKernelGGL(kb_fill_f32, dim3(1024), dim3(256), 0, 0, p, n, seed, lo, hi);
    KB_CHECK(hipDeviceSynchronize());
}
inline void fill_f16(void* p, size_t n, uint32_t seed, float lo = -1.f, float hi = 1.f) {
    hipLaunchKernelGGL(kb_fill_f16, dim3(1024), dim3(256), 0, 0, (_Float16*)p, n, seed, lo, hi);
    KB_CHECK(hipDeviceSynchronize());
}

// ---------------------------------------------------------------- clocks / caches
__global__ void kb_spin(uint64_t ticks) {
    uint64_t t0 = wall_clock64();  // constant 100 MHz on AMD
    volatile float acc = 0.f;
    while (wall_clock64() - t0 < ticks) acc = acc * 1.0001f + 1.f;
}
// Busy every CU for `ms` so DPM ramps clocks before timing.
inline void warm(int ms = 300, hipStream_t s = 0) {
    hipDeviceProp_t p;
    KB_CHECK(hipGetDeviceProperties(&p, 0));
    hipLaunchKernelGGL(kb_spin, dim3(p.multiProcessorCount * 4), dim3(64), 0, s,
                       (uint64_t)ms * 100000ull);
    KB_CHECK(hipStreamSynchronize(s));
}

struct Flusher {  // overwrite `bytes` of memory to evict L2 + MALL between timed blocks
    void* buf = nullptr;
    size_t bytes = 0;
    int round = 0;
    explicit Flusher(size_t b) : bytes(b) { KB_CHECK(hipMalloc(&buf, b)); }
    void operator()(hipStream_t s) { KB_CHECK(hipMemsetAsync(buf, (round++) & 0xff, bytes, s)); }
};

// ---------------------------------------------------------------- stats / timing
struct Stats {
    double med = 0, p10 = 0, p90 = 0, mean = 0, min = 0;
    int n = 0;
};
inline Stats stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v[std::min(v.size() - 1, (size_t)(f * (v.size() - 1) + 0.5))]; };
    s.med = q(0.5); s.p10 = q(0.1); s.p90 = q(0.9); s.min = v.front(); s.n = (int)v.size();
    for (double x : v) s.mean += x;
    s.mean /= v.size();
    return s;
}

struct Variant {
    std::string name;
    std::function<void(hipStream_t)> run;  // ONE call of the kernel (or kernel chain) under test
    double bytes = 0;                      // DRAM bytes one call must move (roofline), 0 = n/a
    double flops = 0;                      // flops one call performs, 0 = n/a
};

struct AbOpts {
    int rounds = 60;      // interleaved rounds; each round times every variant once
    int inner = 10;       // calls per timed block (per-call time = block / inner)
    int warm_ms = 300;    // clock warm-up spin before timing
    bool graph = false;   // time a captured graph of `inner` calls (production decode path)
    std::function<void(hipStream_t)> between;  // untimed, before every timed block (e.g. Flusher)
    const char* tag = "";  // printed with the results (shape / regime description)
};

// Interleaved A/B. Variant 0 is the BASELINE; ratios are med(v)/med(v0) (<1 = faster).
inline std::vector<Stats> ab(std::vector<Variant>& vs, AbOpts o = {}) {
    hipStream_t s;
    KB_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipEvent_t e0, e1;
    KB_CHECK(hipEventCreate(&e0));
    KB_CHECK(hipEventCreate(&e1));
    std::vector<hipGraphExec_t> gx(vs.size(), nullptr);
    if (o.graph) {
        for (size_t v = 0; v < vs.size(); ++v) {
            hipGraph_t g;
            KB_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
            for (int i = 0; i < o.inner; ++i) vs[v].run(s);
            KB_CHECK(hipStreamEndCapture(s, &g));
            KB_CHECK(hipGraphInstantiate(&gx[v], g, nullptr, nullptr, 0));
        }
    }
    auto block = [&](size_t v) {
        if (o.graph) KB_CHECK(hipGraphLaunch(gx[v], s));
        else for (int i = 0; i < o.inner; ++i) vs[v].run(s);
    };
    warm(o.warm_ms, s);
    for (int w = 0; w < 3; ++w)  // untimed warm rounds (code-object load, TLB, first-touch)
        for (size_t v = 0; v < vs.size(); ++v) block(v);
    KB_CHECK(hipStreamSynchronize(s));
    std::vector<std::vector<double>> t(vs.size());
    for (int r = 0; r < o.rounds; ++r) {
        for (size_t k = 0; k < vs.size(); ++k) {
            size_t v = (k + r) % vs.size();  // rotate order every round
            if (o.between) o.between(s);
            KB_CHECK(hipEventRecord(e0, s));
            block(v);
            KB_CHECK(hipEventRecord(e1, s));
            KB_CHECK(hipEventSynchronize(e1));
            float ms = 0;
            KB_CHECK(hipEventElapsedTime(&ms, e0, e1));
            t[v].push_back(ms * 1000.0 / o.inner);
        }
    }
    std::vector<Stats> out;
    for (auto& x : t) out.push_back(stats(x));
    printf("\n== %s  (rounds=%d inner=%d graph=%d flush=%d)\n", o.tag, o.rounds, o.inner,
           (int)o.graph, (int)(bool)o.between);
    printf("%-40s %10s %10s %10s %8s %9s %9s\n", "variant", "med_us", "p10_us", "p90_us",
           "vs_base", "GB/s", "GFLOP/s");
    for (size_t v = 0; v < vs.size(); ++v) {
        const Stats& st = out[v];
        double gbs = vs[v].bytes > 0 ? vs[v].bytes / (st.med * 1e3) : 0;
        double gfl = vs[v].flops > 0 ? vs[v].flops / (st.med * 1e3) : 0;
        printf("%-40s %10.2f %10.2f %10.2f %8.3f %9.1f %9.1f\n", vs[v].name.c_str(), st.med,
               st.p10, st.p90, st.med / out[0].med, gbs, gfl);
    }
    for (size_t v = 0; v < vs.size(); ++v)
        printf("KBJSON {\"tag\":\"%s\",\"variant\":\"%s\",\"med_us\":%.3f,\"p10_us\":%.3f,"
               "\"p90_us\":%.3f,\"ratio\":%.4f,\"gbps\":%.2f,\"graph\":%d,\"flush\":%d}\n",
               o.tag, vs[v].name.c_str(), out[v].med, out[v].p10, out[v].p90,
               out[v].med / out[0].med, vs[v].bytes > 0 ? vs[v].bytes / (out[v].med * 1e3) : 0.0,
               (int)o.graph, (int)(bool)o.between);
    for (auto g : gx) if (g) KB_CHECK(hipGraphExecDestroy(g));
    KB_CHECK(hipEventDestroy(e0));
    KB_CHECK(hipEventDestroy(e1));
    KB_CHECK(hipStreamDestroy(s));
    return out;
}

// ---------------------------------------------------------------- correctness
struct Cmp {
    size_t n = 0, n_bit_diff = 0, worst = 0, n_nonfinite = 0;
    double max_abs = 0, max_rel = 0, rmse = 0, ref_rms = 0;
};
inline Cmp compare_f32(const float* ref, const float* got, size_t n) {
    Cmp c;
    c.n = n;
    double se = 0, rs = 0;
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(&ref[i], &got[i], 4) != 0) c.n_bit_diff++;
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) { c.n_nonfinite++; continue; }
        double d = std::fabs((double)got[i] - ref[i]);
        double rel = d / std::max(1e-6, std::fabs((double)ref[i]));
        if (d > c.max_abs) { c.max_abs = d; c.worst = i; }
        c.max_rel = std::max(c.max_rel, rel);
        se += d * d;
        rs += (double)ref[i] * ref[i];
    }
    c.rmse = std::sqrt(se / std::max<size_t>(1, n));
    c.ref_rms = std::sqrt(rs / std::max<size_t>(1, n));
    return c;
}
inline float h2f(uint16_t h) { _Float16 x; memcpy(&x, &h, 2); return (float)x; }
inline std::vector<float> f16_to_f32(const std::vector<uint16_t>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = h2f(v[i]);
    return o;
}
inline void print_cmp(const char* tag, const Cmp& c) {
    printf("CMP %-34s n=%zu bitexact=%s bit_diff=%zu nonfinite=%zu max_abs=%.3e max_rel=%.3e "
           "rmse=%.3e rel_rmse=%.3e worst_idx=%zu\n",
           tag, c.n, c.n_bit_diff == 0 ? "YES" : "no", c.n_bit_diff, c.n_nonfinite, c.max_abs,
           c.max_rel, c.rmse, c.rmse / std::max(1e-30, c.ref_rms), c.worst);
    printf("KBJSON {\"cmp\":\"%s\",\"bitexact\":%d,\"bit_diff\":%zu,\"nonfinite\":%zu,"
           "\"max_abs\":%.4e,\"rel_rmse\":%.4e}\n",
           tag, (int)(c.n_bit_diff == 0), c.n_bit_diff, c.n_nonfinite, c.max_abs,
           c.rmse / std::max(1e-30, c.ref_rms));
}

}  // namespace kb
