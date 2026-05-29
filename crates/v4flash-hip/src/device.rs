use std::ffi::CStr;
use std::os::raw::c_char;

use color_eyre::eyre::{self, eyre};

use crate::error::check_eyre;
use crate::sys;

/// A logical HIP device identified by its integer ID. Lightweight (Copy);
/// "selecting" the device is a separate concern handled per-thread via
/// `set_current`.
#[derive(Debug, Clone, Copy)]
pub struct Device {
    pub id: i32,
}

impl Device {
    pub fn new(id: i32) -> Self {
        Device { id }
    }

    pub fn count() -> eyre::Result<i32> {
        let mut n = 0;
        check_eyre(unsafe { sys::hipGetDeviceCount(&mut n) }, "hipGetDeviceCount")?;
        Ok(n)
    }

    pub fn all() -> eyre::Result<Vec<Device>> {
        let n = Self::count()?;
        Ok((0..n).map(Device::new).collect())
    }

    /// Make this device the current device for the calling thread.
    pub fn set_current(&self) -> eyre::Result<()> {
        check_eyre(unsafe { sys::hipSetDevice(self.id) }, "hipSetDevice")
    }

    /// Block until ALL streams on this device have completed all queued
    /// work. `hipDeviceSynchronize` acts on the *current* device, so we
    /// `set_current` first. Used at teardown to drain every stream
    /// (including the pipeline lane/xfer streams and cross-device event
    /// signal packets) to a quiescent state before any stream or buffer
    /// is destroyed — otherwise a buffer's `hipFree` runs an implicit
    /// per-buffer `SyncAllStreams` that can orphan a not-yet-executed
    /// cross-device wait and busy-spin forever.
    pub fn synchronize(&self) -> eyre::Result<()> {
        self.set_current()?;
        check_eyre(unsafe { sys::hipDeviceSynchronize() }, "hipDeviceSynchronize")
    }

    /// Read full device properties.
    ///
    /// We sanity-check the struct layout immediately by comparing the
    /// `name` field to `hipDeviceGetName`. If they disagree it means our
    /// transcribed `hipDeviceProp_t` doesn't match the runtime's — we
    /// raise rather than return garbage from a misaligned read.
    pub fn properties(&self) -> eyre::Result<DeviceProperties> {
        let mut prop: sys::hipDeviceProp_t = unsafe { std::mem::zeroed() };
        check_eyre(
            unsafe { sys::hipGetDevicePropertiesR0600(&mut prop, self.id) },
            "hipGetDeviceProperties",
        )?;

        let mut name_buf = [0i8; 256];
        check_eyre(
            unsafe {
                sys::hipDeviceGetName(name_buf.as_mut_ptr() as *mut c_char, 256, self.id)
            },
            "hipDeviceGetName",
        )?;

        let struct_name = cstr_to_string(prop.name.as_ptr());
        let getter_name = cstr_to_string(name_buf.as_ptr());

        if struct_name != getter_name {
            return Err(eyre!(
                "hipDeviceProp_t layout mismatch: struct name {:?} != getter name {:?}",
                struct_name,
                getter_name
            ));
        }

        Ok(DeviceProperties {
            name: struct_name,
            gcn_arch_name: cstr_to_string(prop.gcnArchName.as_ptr()),
            total_global_mem: prop.totalGlobalMem,
            multi_processor_count: prop.multiProcessorCount,
            clock_rate_khz: prop.clockRate,
            memory_clock_rate_khz: prop.memoryClockRate,
            memory_bus_width_bits: prop.memoryBusWidth,
            l2_cache_size: prop.l2CacheSize,
            integrated: prop.integrated != 0,
            pci_bus_id: prop.pciBusID,
            pci_device_id: prop.pciDeviceID,
            pci_domain_id: prop.pciDomainID,
            major: prop.major,
            minor: prop.minor,
            warp_size: prop.warpSize,
            shared_mem_per_block: prop.sharedMemPerBlock,
            max_threads_per_block: prop.maxThreadsPerBlock,
            stream_priorities_supported: prop.streamPrioritiesSupported != 0,
        })
    }

    pub fn can_access_peer(&self, peer: Device) -> eyre::Result<bool> {
        let mut out = 0;
        check_eyre(
            unsafe { sys::hipDeviceCanAccessPeer(&mut out, self.id, peer.id) },
            "hipDeviceCanAccessPeer",
        )?;
        Ok(out != 0)
    }

    /// Caller must already have `set_current()` for this device.
    pub fn enable_peer_access(&self, peer: Device) -> eyre::Result<()> {
        check_eyre(
            unsafe { sys::hipDeviceEnablePeerAccess(peer.id, 0) },
            "hipDeviceEnablePeerAccess",
        )
    }

    pub fn stream_priority_range(&self) -> eyre::Result<(i32, i32)> {
        // Note: hipDeviceGetStreamPriorityRange uses the *current* device,
        // not an explicit ID argument. Caller is expected to set_current.
        let mut least = 0;
        let mut greatest = 0;
        check_eyre(
            unsafe { sys::hipDeviceGetStreamPriorityRange(&mut least, &mut greatest) },
            "hipDeviceGetStreamPriorityRange",
        )?;
        Ok((least, greatest))
    }
}

/// A subset of `hipDeviceProp_t` lifted into owned Rust strings. Add more
/// fields as Phase 0/1 need them.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub gcn_arch_name: String,
    pub total_global_mem: usize,
    pub multi_processor_count: i32,
    pub clock_rate_khz: i32,
    pub memory_clock_rate_khz: i32,
    pub memory_bus_width_bits: i32,
    pub l2_cache_size: i32,
    pub integrated: bool,
    pub pci_bus_id: i32,
    pub pci_device_id: i32,
    pub pci_domain_id: i32,
    pub major: i32,
    pub minor: i32,
    pub warp_size: i32,
    pub shared_mem_per_block: usize,
    pub max_threads_per_block: i32,
    pub stream_priorities_supported: bool,
}

fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: HIP runtime guarantees null-terminated strings in name/gcnArchName.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
