//! Safe wrappers around AMD HIP runtime. Phase 0 minimal surface.
//!
//! Layout:
//! - [`sys`]      — hand-rolled `extern "C"` FFI declarations
//! - [`error`]    — [`HipError`] + result conversion
//! - [`Device`]   — device enumeration + properties + peer access
//! - [`Stream`]   — streams w/ optional priority
//! - [`Event`]    — events for sync + timing
//! - [`DeviceBuffer`] / [`PinnedBuffer`] — typed RAII buffers
//! - [`Module`] / [`Function`] — kernel loading + launch
//!
//! All functions return `color_eyre::eyre::Result<T>`. Binaries should
//! call [`install_panic_handler`] from main to enable pretty backtraces.

pub mod sys;

mod buffer;
mod device;
mod error;
mod event;
mod graph;
#[macro_use]
mod launch;
mod module;
mod stream;

pub use buffer::{DeviceBuffer, PinnedBuffer};
pub use device::{Device, DeviceProperties};
pub use error::{HipError, check, check_eyre};
pub use event::Event;
pub use graph::{kernel_node_params, Graph, GraphExec};
pub use module::{Function, LaunchConfig, Module};
pub use stream::Stream;

/// Block until all work on the *current* device finishes. Stronger than
/// `Stream::synchronize` — affects every stream on the device. Useful
/// for forcing full L2 flush / memory fence at agent or system scope.
pub fn device_synchronize() -> color_eyre::eyre::Result<()> {
    error::check_eyre(unsafe { sys::hipDeviceSynchronize() }, "hipDeviceSynchronize")
}

/// Install the color-eyre panic + error report hooks. Binaries call this
/// once from main. Idempotent.
pub fn install_panic_handler() -> color_eyre::eyre::Result<()> {
    // color_eyre::install errors if called twice — collapse that into ok.
    let _ = color_eyre::install();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: device enumeration works and reports >= 1 device.
    /// Run with `cargo test -- --ignored` on a machine with a GPU.
    #[test]
    #[ignore]
    fn enumerate_devices() {
        let _ = install_panic_handler();
        let count = Device::count().expect("device count");
        assert!(count >= 1, "expected at least one HIP device, got {count}");

        for dev in Device::all().unwrap() {
            let props = dev.properties().expect("get properties");
            eprintln!(
                "device {} = {:?} ({}) — {} MiB, {} CUs, integrated={}, pci={:04x}:{:02x}:{:02x}",
                dev.id,
                props.name,
                props.gcn_arch_name,
                props.total_global_mem >> 20,
                props.multi_processor_count,
                props.integrated,
                props.pci_domain_id,
                props.pci_bus_id,
                props.pci_device_id,
            );
        }
    }

    /// Load the trivial `hello` kernel from the phase0 crate and run it.
    /// This deliberately lives in the v4flash-hip crate to keep the test
    /// independent of phase0's build artifacts; we re-compile a copy here.
    /// (Test moved to phase0 binary in commit 4 once we have it.)
    #[test]
    #[ignore]
    fn launch_kernel_on_each_device() {
        // Trivial kernel as a precompiled blob: we don't have build.rs
        // for this crate's tests, so commit 4's phase0 binary covers
        // the full launch path. This test is a placeholder asserting
        // we can at least create per-device resources.
        let _ = install_panic_handler();
        for dev in Device::all().unwrap() {
            dev.set_current().expect("set current");
            let stream = Stream::new(dev.id).expect("stream");
            let mut buf: DeviceBuffer<i32> =
                DeviceBuffer::new(dev.id, 16).expect("alloc");
            buf.fill_zero().expect("memset");
            let event = Event::new().expect("event");
            event.record(&stream).expect("record");
            stream.synchronize().expect("sync");
            eprintln!("device {}: alloc/memset/event/sync OK", dev.id);
        }
    }
}
