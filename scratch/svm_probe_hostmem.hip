// Probe hipHostMalloc with various flags; touch from a kernel via mapped device ptr.
//   flags: 0, hipHostMallocDefault, hipHostMallocMapped, ...Coherent, ...NonCoherent
// Usage:  svm_probe_hostmem <chunk_gib> [device] [flags_hex]
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
    unsigned flags = (argc>3)?(unsigned)strtoul(argv[3],nullptr,0):0u;
    size_t bytes = (size_t)(chunk_gib * (1024.0*1024.0*1024.0));
    size_t n_u32 = bytes/4;

    CK(hipSetDevice(device));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, device));
    fprintf(stderr,"device %d: %s (arch=%s)\n", device, prop.name, prop.gcnArchName);
    fprintf(stderr,"chunk=%.2f GiB  flags=0x%x\n", chunk_gib, flags);

    std::vector<void*> hosts;
    std::vector<uint32_t*> devs;
    double total = 0.0;
    while (true) {
        void* h = nullptr;
        hipError_t e = hipHostMalloc(&h, bytes, flags);
        if (e != hipSuccess) {
            fprintf(stderr,"  hipHostMalloc #%zu failed at total=%.2f GiB: %d (%s)\n",
                    hosts.size(), total, e, hipGetErrorString(e));
            break;
        }
        // Get device pointer for kernel access
        uint32_t* d = nullptr;
        hipError_t ge = hipHostGetDevicePointer((void**)&d, h, 0);
        if (ge != hipSuccess) {
            fprintf(stderr,"  hipHostGetDevicePointer #%zu failed: %d (%s)\n",
                    hosts.size(), ge, hipGetErrorString(ge));
            // try touching via host pointer cast
            d = (uint32_t*)h;
        }
        // touch from kernel
        touch_kernel<<<4096,256,0,0>>>(d, n_u32, (uint32_t)(0xC0FFEE00u + hosts.size()));
        hipError_t ke = hipDeviceSynchronize();
        if (ke != hipSuccess) {
            fprintf(stderr,"  touch #%zu FAULT at total=%.2f GiB: %d (%s)\n",
                    hosts.size(), total+chunk_gib, ke, hipGetErrorString(ke));
            break;
        }
        hosts.push_back(h);
        devs.push_back(d);
        total += chunk_gib;
        fprintf(stderr,"  alloc #%zu ok  total=%.2f GiB  (h=%p d=%p)\n",
                hosts.size(), total, h, (void*)d);
        if (hosts.size() >= 60) break;
    }
    fprintf(stderr,"final total=%.2f GiB across %zu allocs\n", total, hosts.size());
    for (auto h : hosts) hipHostFree(h);
    return 0;
}
