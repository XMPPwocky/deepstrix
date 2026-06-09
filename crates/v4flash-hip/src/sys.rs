//! Hand-rolled `extern "C"` declarations for the subset of the HIP runtime
//! used by Phase 0. Types and functions transcribed from
//! `<rocm>/include/hip/hip_runtime_api.h` for HIP 7.2.3.
//!
//! NOTE: `hipDeviceProp_t` is large and layout-sensitive. We validate at
//! runtime by reading the `name` field and comparing against
//! `hipDeviceGetName`, which gives us a sanity gate against silent layout
//! drift across ROCm versions.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_int, c_uint, c_void};

pub type hipError_t = c_int;
pub type hipDeviceptr_t = *mut c_void;
pub type hipStream_t = *mut c_void;
pub type hipEvent_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipGraph_t = *mut c_void;
pub type hipGraphExec_t = *mut c_void;
pub type hipGraphNode_t = *mut c_void;

pub const HIP_SUCCESS: hipError_t = 0;

// Stream priorities: smaller = higher priority. Phase 0 will check
// hipDeviceGetStreamPriorityRange to learn actual bounds.
pub const HIP_STREAM_DEFAULT: c_uint = 0;

// Event creation flags
pub const HIP_EVENT_DEFAULT: c_uint = 0;
pub const HIP_EVENT_BLOCKING_SYNC: c_uint = 1;
pub const HIP_EVENT_DISABLE_TIMING: c_uint = 2;
pub const HIP_EVENT_INTERPROCESS: c_uint = 4;

// Stream capture modes
pub const HIP_STREAM_CAPTURE_MODE_GLOBAL: c_uint = 0;
pub const HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_uint = 1;
pub const HIP_STREAM_CAPTURE_MODE_RELAXED: c_uint = 2;

// hipKernelNodeParams — mirrors HIP's struct used by both
// hipGraphAddKernelNode and hipGraphExecKernelNodeSetParams.
#[repr(C)]
#[derive(Clone)]
pub struct hipKernelNodeParams {
    pub blockDim: hipDim3,
    pub extra: *mut *mut c_void,
    pub func: *mut c_void,
    pub gridDim: hipDim3,
    pub kernelParams: *mut *mut c_void,
    pub sharedMemBytes: c_uint,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct hipDim3 {
    pub x: c_uint,
    pub y: c_uint,
    pub z: c_uint,
}

// Peer-access flags (unused at present — 0)
pub const HIP_PEER_ACCESS_DEFAULT: c_uint = 0;

// hipMemcpy directions
pub const HIP_MEMCPY_HOST_TO_HOST: c_int = 0;
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
pub const HIP_MEMCPY_DEFAULT: c_int = 4;

// hipUUID — 16 raw bytes
#[repr(C)]
#[derive(Copy, Clone)]
pub struct hipUUID {
    pub bytes: [c_char; 16],
}

// hipDeviceArch_t — bitfield struct, packs into a single u32 in C.
// We treat it as an opaque u32 here.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct hipDeviceArch_t {
    pub flags: u32,
}

// hipDeviceProp_t (R0600 layout, HIP 6.x/7.x).
// Carefully mirrors include/hip/hip_runtime_api.h:111-254.
// Size sanity-checked at runtime against hipDeviceGetName behavior.
#[repr(C)]
pub struct hipDeviceProp_t {
    pub name: [c_char; 256],
    pub uuid: hipUUID,
    pub luid: [c_char; 8],
    pub luidDeviceNodeMask: c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: c_int,
    pub warpSize: c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: c_int,
    pub maxThreadsDim: [c_int; 3],
    pub maxGridSize: [c_int; 3],
    pub clockRate: c_int,
    pub totalConstMem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: c_int,
    pub multiProcessorCount: c_int,
    pub kernelExecTimeoutEnabled: c_int,
    pub integrated: c_int,
    pub canMapHostMemory: c_int,
    pub computeMode: c_int,
    pub maxTexture1D: c_int,
    pub maxTexture1DMipmap: c_int,
    pub maxTexture1DLinear: c_int,
    pub maxTexture2D: [c_int; 2],
    pub maxTexture2DMipmap: [c_int; 2],
    pub maxTexture2DLinear: [c_int; 3],
    pub maxTexture2DGather: [c_int; 2],
    pub maxTexture3D: [c_int; 3],
    pub maxTexture3DAlt: [c_int; 3],
    pub maxTextureCubemap: c_int,
    pub maxTexture1DLayered: [c_int; 2],
    pub maxTexture2DLayered: [c_int; 3],
    pub maxTextureCubemapLayered: [c_int; 2],
    pub maxSurface1D: c_int,
    pub maxSurface2D: [c_int; 2],
    pub maxSurface3D: [c_int; 3],
    pub maxSurface1DLayered: [c_int; 2],
    pub maxSurface2DLayered: [c_int; 3],
    pub maxSurfaceCubemap: c_int,
    pub maxSurfaceCubemapLayered: [c_int; 2],
    pub surfaceAlignment: usize,
    pub concurrentKernels: c_int,
    pub ECCEnabled: c_int,
    pub pciBusID: c_int,
    pub pciDeviceID: c_int,
    pub pciDomainID: c_int,
    pub tccDriver: c_int,
    pub asyncEngineCount: c_int,
    pub unifiedAddressing: c_int,
    pub memoryClockRate: c_int,
    pub memoryBusWidth: c_int,
    pub l2CacheSize: c_int,
    pub persistingL2CacheMaxSize: c_int,
    pub maxThreadsPerMultiProcessor: c_int,
    pub streamPrioritiesSupported: c_int,
    pub globalL1CacheSupported: c_int,
    pub localL1CacheSupported: c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: c_int,
    pub managedMemory: c_int,
    pub isMultiGpuBoard: c_int,
    pub multiGpuBoardGroupID: c_int,
    pub hostNativeAtomicSupported: c_int,
    pub singleToDoublePrecisionPerfRatio: c_int,
    pub pageableMemoryAccess: c_int,
    pub concurrentManagedAccess: c_int,
    pub computePreemptionSupported: c_int,
    pub canUseHostPointerForRegisteredMem: c_int,
    pub cooperativeLaunch: c_int,
    pub cooperativeMultiDeviceLaunch: c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: c_int,
    pub directManagedMemAccessFromHost: c_int,
    pub maxBlocksPerMultiProcessor: c_int,
    pub accessPolicyMaxWindowSize: c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: c_int,
    pub sparseHipArraySupported: c_int,
    pub hostRegisterReadOnlySupported: c_int,
    pub timelineSemaphoreInteropSupported: c_int,
    pub memoryPoolsSupported: c_int,
    pub gpuDirectRDMASupported: c_int,
    pub gpuDirectRDMAFlushWritesOptions: c_uint,
    pub gpuDirectRDMAWritesOrdering: c_int,
    pub memoryPoolSupportedHandleTypes: c_uint,
    pub deferredMappingHipArraySupported: c_int,
    pub ipcEventSupported: c_int,
    pub clusterLaunch: c_int,
    pub unifiedFunctionPointers: c_int,
    pub reserved: [c_int; 63],
    pub hipReserved: [c_int; 32],
    pub gcnArchName: [c_char; 256],
    pub maxSharedMemoryPerMultiProcessor: usize,
    pub clockInstructionRate: c_int,
    pub arch: hipDeviceArch_t,
    pub hdpMemFlushCntl: *mut c_uint,
    pub hdpRegFlushCntl: *mut c_uint,
    pub cooperativeMultiDeviceUnmatchedFunc: c_int,
    pub cooperativeMultiDeviceUnmatchedGridDim: c_int,
    pub cooperativeMultiDeviceUnmatchedBlockDim: c_int,
    pub cooperativeMultiDeviceUnmatchedSharedMem: c_int,
    pub isLargeBar: c_int,
    pub asicRevision: c_int,
}

// The header `#define`s the public name to the versioned symbol. We must
// call the versioned symbol explicitly from Rust since cpp #define isn't
// in play.
unsafe extern "C" {
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipGetDevice(deviceId: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(deviceId: c_int) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipGetErrorString(err: hipError_t) -> *const c_char;
    pub fn hipGetErrorName(err: hipError_t) -> *const c_char;
    pub fn hipGetLastError() -> hipError_t;

    // Use the versioned symbol explicitly. The header #defines
    // hipGetDeviceProperties → hipGetDevicePropertiesR0600.
    pub fn hipGetDevicePropertiesR0600(
        prop: *mut hipDeviceProp_t,
        deviceId: c_int,
    ) -> hipError_t;
    pub fn hipDeviceGetName(name: *mut c_char, len: c_int, deviceId: c_int) -> hipError_t;
    pub fn hipDeviceGetAttribute(
        pi: *mut c_int,
        attr: c_int,
        deviceId: c_int,
    ) -> hipError_t;
    pub fn hipDeviceCanAccessPeer(
        canAccess: *mut c_int,
        deviceId: c_int,
        peerDeviceId: c_int,
    ) -> hipError_t;
    pub fn hipDeviceEnablePeerAccess(peerDeviceId: c_int, flags: c_uint) -> hipError_t;
    pub fn hipDeviceDisablePeerAccess(peerDeviceId: c_int) -> hipError_t;

    pub fn hipMalloc(ptr: *mut hipDeviceptr_t, size: usize) -> hipError_t;
    pub fn hipFree(ptr: hipDeviceptr_t) -> hipError_t;
    pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t;
    pub fn hipHostFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(
        dst: hipDeviceptr_t,
        src: *const c_void,
        size: usize,
        kind: c_int,
    ) -> hipError_t;
    pub fn hipMemcpyAsync(
        dst: hipDeviceptr_t,
        src: *const c_void,
        size: usize,
        kind: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipMemcpyPeerAsync(
        dst: hipDeviceptr_t,
        dstDevice: c_int,
        src: hipDeviceptr_t,
        srcDevice: c_int,
        size: usize,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipMemset(dst: hipDeviceptr_t, value: c_int, size: usize) -> hipError_t;

    pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    pub fn hipStreamCreateWithPriority(
        stream: *mut hipStream_t,
        flags: c_uint,
        priority: c_int,
    ) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamWaitEvent(
        stream: hipStream_t,
        event: hipEvent_t,
        flags: c_uint,
    ) -> hipError_t;
    /// Enqueue a wait until `*ptr` (32-bit) satisfies the condition in
    /// `flags` (0 = GTE, 1 = EQ, 2 = AND, 3 = NOR) against `value`.
    /// Unlike hipStreamWaitEvent (which snapshots the event state at CALL
    /// time), the comparison happens at EXECUTION time — the primitive
    /// that makes pre-issued cross-device pipelines sound.
    pub fn hipStreamWaitValue32(
        stream: hipStream_t,
        ptr: *mut c_void,
        value: u32,
        flags: c_uint,
        mask: u32,
    ) -> hipError_t;
    pub fn hipStreamWriteValue32(
        stream: hipStream_t,
        ptr: *mut c_void,
        value: u32,
        flags: c_uint,
    ) -> hipError_t;
    pub fn hipDeviceGetStreamPriorityRange(
        leastPriority: *mut c_int,
        greatestPriority: *mut c_int,
    ) -> hipError_t;

    pub fn hipEventCreate(event: *mut hipEvent_t) -> hipError_t;
    pub fn hipEventCreateWithFlags(event: *mut hipEvent_t, flags: c_uint) -> hipError_t;
    pub fn hipEventDestroy(event: hipEvent_t) -> hipError_t;
    pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
    pub fn hipEventSynchronize(event: hipEvent_t) -> hipError_t;
    pub fn hipEventElapsedTime(ms: *mut f32, start: hipEvent_t, end: hipEvent_t)
        -> hipError_t;

    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        kname: *const c_char,
    ) -> hipError_t;
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        stream: hipStream_t,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;

    // ---- HIP graphs (M14) ----

    pub fn hipStreamBeginCapture(
        stream: hipStream_t,
        mode: c_uint,
    ) -> hipError_t;
    pub fn hipStreamEndCapture(
        stream: hipStream_t,
        graph: *mut hipGraph_t,
    ) -> hipError_t;
    pub fn hipStreamIsCapturing(
        stream: hipStream_t,
        status: *mut c_int,
    ) -> hipError_t;

    pub fn hipGraphCreate(graph: *mut hipGraph_t, flags: c_uint) -> hipError_t;
    pub fn hipGraphDestroy(graph: hipGraph_t) -> hipError_t;
    pub fn hipGraphInstantiate(
        graphExec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        errorNode: *mut hipGraphNode_t,
        log: *mut c_char,
        bufferSize: usize,
    ) -> hipError_t;
    pub fn hipGraphExecDestroy(graphExec: hipGraphExec_t) -> hipError_t;
    pub fn hipGraphLaunch(graphExec: hipGraphExec_t, stream: hipStream_t)
        -> hipError_t;

    pub fn hipGraphGetNodes(
        graph: hipGraph_t,
        nodes: *mut hipGraphNode_t,
        numNodes: *mut usize,
    ) -> hipError_t;

    pub fn hipGraphExecKernelNodeSetParams(
        graphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        nodeParams: *const hipKernelNodeParams,
    ) -> hipError_t;

    pub fn hipGraphKernelNodeGetParams(
        node: hipGraphNode_t,
        nodeParams: *mut hipKernelNodeParams,
    ) -> hipError_t;
}
