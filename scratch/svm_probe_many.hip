// Many-allocation probe: hipMallocManaged in a loop until failure.
// Usage:  svm_probe_many <chunk_gib> [device] [touch=0|1]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CK(expr) do { hipError_t _e=(expr); if(_e!=hipSuccess){ \
    fprintf(stderr,"FAIL %s:%d %s -> %d (%s)\n",__FILE__,__LINE__,#expr,_e,hipGetErrorString(_e)); return 1; }}while(0)

__global__ void touch_kernel(uint32_t* p, size_t n_u32, uint32_t seed) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
    for(size_t k=i;k<n_u32;k+=st) p[k]=(uint32_t)(k^seed);
}

int main(int argc, char** argv){
    double chunk_gib = (argc>1)?atof(argv[1]):2.0;
    int device     = (argc>2)?atoi(argv[2]):0;
    int do_touch   = (argc>3)?atoi(argv[3]):0;
    size_t bytes = (size_t)(chunk_gib * (1024.0*1024.0*1024.0));
    size_t n_u32 = bytes/4;

    CK(hipSetDevice(device));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, device));
    fprintf(stderr,"device %d: %s (arch=%s)\n", device, prop.name, prop.gcnArchName);
    fprintf(stderr,"chunk=%.2f GiB  touch=%d\n", chunk_gib, do_touch);

    std::vector<uint32_t*> ptrs;
    double total = 0.0;
    while (true) {
        uint32_t* p = nullptr;
        hipError_t e = hipMallocManaged((void**)&p, bytes);
        if (e != hipSuccess) {
            fprintf(stderr,"  alloc #%zu failed at total=%.2f GiB: %d (%s)\n",
                    ptrs.size(), total, e, hipGetErrorString(e));
            break;
        }
        hipMemAdvise(p, bytes, hipMemAdviseSetCoarseGrain, device);
        (void)hipGetLastError();
        if (do_touch) {
            touch_kernel<<<4096,256,0,0>>>(p, n_u32, (uint32_t)(0xC0FFEE00u + ptrs.size()));
            hipError_t ke = hipDeviceSynchronize();
            if (ke != hipSuccess) {
                fprintf(stderr,"  touch #%zu FAULT at total=%.2f GiB: %d (%s)\n",
                        ptrs.size(), total+chunk_gib, ke, hipGetErrorString(ke));
                break;
            }
        }
        ptrs.push_back(p);
        total += chunk_gib;
        fprintf(stderr,"  alloc #%zu ok  total=%.2f GiB\n", ptrs.size(), total);
        if (ptrs.size() >= 80) break;
    }
    fprintf(stderr,"final total=%.2f GiB across %zu allocs\n", total, ptrs.size());
    for (auto p : ptrs) hipFree(p);
    return 0;
}
