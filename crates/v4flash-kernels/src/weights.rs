//! GGUF tensor → `DeviceBuffer<u8>` loader for kernel tests.
//!
//! Test code wants the *raw* tensor bytes on-device so the per-kernel
//! HIP code can decode the quant format directly. Production paths use
//! a smarter staging strategy (see design doc §5/§6); this loader is
//! deliberately the simplest thing that works.
//!
//! Usage:
//! ```ignore
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
/// tensor's `byte_size`, and copy the GGUF mmap bytes into it.
///
/// Errors:
/// - tensor not found in the GGUF
/// - tensor's `byte_size == 0` (placeholder / corrupted entry)
/// - HIP malloc / memcpy failure
pub fn load_to_device(
    gguf: &MappedGguf,
    name: &str,
    device_id: i32,
) -> eyre::Result<DeviceWeight> {
    let tensor = gguf
        .gguf()
        .tensor(name)
        .ok_or_else(|| eyre!("tensor `{name}` not found in GGUF"))?;
    let bytes = gguf
        .tensor_bytes(tensor)
        .ok_or_else(|| eyre!("tensor `{name}` has zero byte_size or invalid offset"))?;

    let mut buffer: DeviceBuffer<u8> =
        DeviceBuffer::new(device_id, bytes.len()).wrap_err_with(|| {
            format!(
                "alloc DeviceBuffer<u8> ({} bytes) for `{name}`",
                bytes.len()
            )
        })?;
    buffer
        .copy_from_host(bytes)
        .wrap_err_with(|| format!("hipMemcpy `{name}` host→device"))?;

    let n_elements: u64 = tensor.dims.iter().product();
    Ok(DeviceWeight {
        buffer,
        n_elements,
        dtype: tensor.dtype,
        shape: tensor.dims.clone(),
    })
}
