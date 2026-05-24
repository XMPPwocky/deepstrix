//! GGUF tensor → `DeviceBuffer<u8>` loader for kernel tests.
//!
//! Test code wants the *raw* tensor bytes on-device so the per-kernel
//! HIP code can decode the quant format directly. Production paths use
//! a smarter staging strategy (see design doc §5/§6); this loader is
//! deliberately the simplest thing that works.
//!
//! Usage:
//! ```text
//! let gguf = MappedGguf::open(model_path)?;
//! let w = weights::load_to_device(&gguf, "output.weight", device_id)?;
//! // w.buffer is a DeviceBuffer<u8> with the raw Q8_0/Q2_K/IQ2_XXS/etc bytes
//! ```
//!
//! Caller is responsible for binding the right HIP device with
//! `Device::set_current()` before invoking — the loader allocates +
//! `hipMemcpy`s into the current device's address space.

use color_eyre::eyre::{self, eyre, WrapErr};
use v4flash_core::gguf::GgufType;
use v4flash_core::MappedGguf;
use v4flash_hip::DeviceBuffer;

/// A model weight tensor materialized on a HIP device. Includes the
/// metadata so callers can sanity-check shape/dtype against expectations.
pub struct DeviceWeight {
    pub buffer: DeviceBuffer<u8>,
    pub n_elements: u64,
    pub dtype: GgufType,
    pub shape: Vec<u64>,
}

/// Allocate a fresh `DeviceBuffer<u8>` on `device_id` sized to the named
/// tensor's `byte_size`, **pread**-read the bytes from the GGUF file into
/// a transient host buffer, copy to device, drop the host buffer.
///
/// Why pread instead of mmap: bulk weight loading via mmap pages the
/// whole tensor into the OS page cache, which on UMA double-allocates
/// against the iGPU pool — V4-Flash's 80 GiB worth of routed experts
/// would cause 100+ GiB of physical RAM pressure and swap thrash.
/// pread keeps only the active tensor's bytes resident, the cache
/// effects are bounded by the largest single tensor (~500 MB).
///
/// Errors: tensor not found, zero byte_size, alloc/copy failure.
pub fn load_to_device(
    gguf: &MappedGguf,
    name: &str,
    device_id: i32,
) -> eyre::Result<DeviceWeight> {
    let tensor = gguf
        .gguf()
        .tensor(name)
        .ok_or_else(|| eyre!("tensor `{name}` not found in GGUF"))?;
    if tensor.byte_size == 0 {
        return Err(eyre!("tensor `{name}` has zero byte_size"));
    }
    let mut host: Vec<u8> = Vec::with_capacity(tensor.byte_size as usize);
    gguf.read_tensor_into(tensor, &mut host)
        .wrap_err_with(|| format!("pread `{name}`"))?;

    let mut buffer: DeviceBuffer<u8> =
        DeviceBuffer::new(device_id, host.len()).wrap_err_with(|| {
            format!(
                "alloc DeviceBuffer<u8> ({} bytes) for `{name}`",
                host.len()
            )
        })?;
    buffer
        .copy_from_host(&host)
        .wrap_err_with(|| format!("hipMemcpy `{name}` host→device"))?;
    drop(host);

    let n_elements: u64 = tensor.dims.iter().product();
    Ok(DeviceWeight {
        buffer,
        n_elements,
        dtype: tensor.dtype,
        shape: tensor.dims.clone(),
    })
}
