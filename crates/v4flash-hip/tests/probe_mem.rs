use std::ffi::c_void;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};

extern "C" {
    fn hipMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn hipFree(ptr: *mut c_void) -> i32;
    fn hipHostRegister(host: *mut c_void, size: usize, flags: u32) -> i32;
    fn hipHostUnregister(host: *mut c_void) -> i32;
    fn hipHostGetDevicePointer(dev_ptr: *mut *mut c_void, host_ptr: *mut c_void, flags: u32) -> i32;
    fn hipExtMallocWithFlags(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
}
const HIP_ATTACH_GLOBAL: u32 = 0x01;
const HIP_HOST_REGISTER_MAPPED: u32 = 0x02;

#[test]
#[ignore]
fn probe_alloc_capacity() {
    install_panic_handler().unwrap();
    let dev = Device::all()
        .unwrap()
        .into_iter()
        .find(|d| d.properties().unwrap().gcn_arch_name.starts_with("gfx1151"))
        .unwrap();
    dev.set_current().unwrap();
    let props = dev.properties().unwrap();
    eprintln!(
        "device id={}, arch={}, totalGlobalMem={} ({} GB)",
        dev.id,
        props.gcn_arch_name,
        props.total_global_mem,
        props.total_global_mem / 1_000_000_000
    );

    // Probe 1 GB at a time, keeping all allocations alive.
    let mut bufs: Vec<DeviceBuffer<u8>> = Vec::new();
    let chunk = 1_073_741_824usize; // 1 GiB
    for i in 1..=120usize {
        match DeviceBuffer::<u8>::new(dev.id, chunk) {
            Ok(b) => {
                bufs.push(b);
                eprintln!("alloc #{i}: {} GiB resident OK", i);
            }
            Err(e) => {
                eprintln!("alloc #{i} failed: {e}");
                break;
            }
        }
    }
    eprintln!("max resident (hipMalloc): {} GiB", bufs.len());

    // Free everything from hipMalloc probe.
    drop(bufs);

    // Try many small allocations.
    eprintln!("--- probing many small (256 MiB) hipMalloc ---");
    let mut small_bufs: Vec<DeviceBuffer<u8>> = Vec::new();
    let small_chunk = 256 * 1024 * 1024usize;
    for i in 1..=500usize {
        match DeviceBuffer::<u8>::new(dev.id, small_chunk) {
            Ok(b) => small_bufs.push(b),
            Err(e) => {
                eprintln!("small alloc #{i} ({} MiB total) failed: {e}", (i - 1) * 256);
                break;
            }
        }
        if i % 8 == 0 {
            eprintln!("small alloc #{i}: {} MiB total OK", i * 256);
        }
    }
    eprintln!("max resident (small): {} MiB", small_bufs.len() * 256);
    drop(small_bufs);

    // Probe hipExtMallocWithFlags variants.
    for (name, flag) in [
        ("default", 0u32),
        ("finegrained", 1u32),
        ("uncached", 3u32),
    ] {
        eprintln!("--- probing hipExtMallocWithFlags({name}) ---");
        let mut ext_ptrs: Vec<*mut c_void> = Vec::new();
        let chunk = 1_073_741_824usize;
        for i in 1..=120usize {
            let mut p: *mut c_void = std::ptr::null_mut();
            let code = unsafe { hipExtMallocWithFlags(&mut p, chunk, flag) };
            if code != 0 {
                eprintln!("ext({name}) alloc #{i} failed: code {code}");
                break;
            }
            ext_ptrs.push(p);
            eprintln!("ext({name}) alloc #{i}: {} GiB resident OK", i);
        }
        eprintln!("max resident (ext {name}): {} GiB", ext_ptrs.len());
        for p in &ext_ptrs {
            unsafe { hipFree(*p) };
        }
    }

    // Now probe hipMallocManaged.
    eprintln!("--- probing hipMallocManaged ---");
    let mut managed_ptrs: Vec<*mut c_void> = Vec::new();
    let chunk = 1_073_741_824usize;
    for i in 1..=120usize {
        let mut p: *mut c_void = std::ptr::null_mut();
        let code = unsafe { hipMallocManaged(&mut p, chunk, HIP_ATTACH_GLOBAL) };
        if code != 0 {
            eprintln!("managed alloc #{i} failed: code {code}");
            break;
        }
        managed_ptrs.push(p);
        eprintln!("managed alloc #{i}: {} GiB resident OK", i);
    }
    eprintln!("max resident (managed): {} GiB", managed_ptrs.len());
    for p in &managed_ptrs {
        unsafe { hipFree(*p) };
    }

    // Now probe hipHostRegister on host-malloc'd memory.
    eprintln!("--- probing hipHostRegister on big host allocation ---");
    // Allocate 20 GiB on host via std::alloc.
    let target_bytes = 20usize * 1024 * 1024 * 1024;
    let layout =
        std::alloc::Layout::from_size_align(target_bytes, 4096).expect("layout");
    let host_ptr = unsafe { std::alloc::alloc(layout) };
    assert!(!host_ptr.is_null(), "host alloc failed");
    eprintln!("host-allocated {} GiB at {:p}", target_bytes / (1024 * 1024 * 1024), host_ptr);
    // Touch first byte of every page to ensure pages are committed.
    let pages = target_bytes / 4096;
    for i in 0..pages {
        unsafe { *host_ptr.add(i * 4096) = 0; }
    }
    eprintln!("host pages committed");

    let rc = unsafe {
        hipHostRegister(host_ptr as *mut c_void, target_bytes, HIP_HOST_REGISTER_MAPPED)
    };
    eprintln!("hipHostRegister rc={rc}");
    if rc == 0 {
        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let rc2 = unsafe {
            hipHostGetDevicePointer(&mut dev_ptr, host_ptr as *mut c_void, 0)
        };
        eprintln!("hipHostGetDevicePointer rc={rc2}, dev_ptr={dev_ptr:p}");
        unsafe { hipHostUnregister(host_ptr as *mut c_void) };
    }
    unsafe { std::alloc::dealloc(host_ptr, layout) };
}
