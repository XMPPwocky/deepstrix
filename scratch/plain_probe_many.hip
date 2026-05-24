// Many-allocation probe: plain hipMalloc (no managed/SVM).
// Usage:  plain_probe_many <chunk_gib> [device]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CK(expr) do { hipError_t _e=(expr); if(_e!=hipSuccess){ \
    fprintf(stderr,"FAIL %s:%d %s -> %d (%s)\n",__FILE__,__LINE__,#expr,_e,hipGetErrorString(_e)); return 1; }}while(0)

int main(int argc, char** argv){
    double chunk_gib = (argc>1)?atof(argv[1]):2.0;
    int device     = (argc>2)?atoi(argv[2]):0;
    size_t bytes = (size_t)(chunk_gib * (1024.0*1024.0*1024.0));

    CK(hipSetDevice(device));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, device));
    fprintf(stderr,"device %d: %s (arch=%s)  totalGlobalMem=%.2f GiB\n",
            device, prop.name, prop.gcnArchName,
            (double)prop.totalGlobalMem/(1024.0*1024.0*1024.0));
    fprintf(stderr,"chunk=%.2f GiB  allocator=hipMalloc\n", chunk_gib);

    std::vector<void*> ptrs;
    double total = 0.0;
    while (true) {
        void* p = nullptr;
        hipError_t e = hipMalloc(&p, bytes);
        if (e != hipSuccess) {
            fprintf(stderr,"  alloc #%zu failed at total=%.2f GiB: %d (%s)\n",
                    ptrs.size(), total, e, hipGetErrorString(e));
            break;
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
