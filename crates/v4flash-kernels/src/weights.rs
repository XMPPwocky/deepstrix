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
    let host = gguf
        .read_tensor(tensor)
        .wrap_err_with(|| format!("pread `{name}`"))?;

    // M18: repack Q8_0 weights from per-block [scale|q×32] interleaving
    // to per-row [scales | quants] split. Same total size; result is
    // that quants land 4-byte-aligned (versus offset+=2 mod 4 before),
    // so the matvec inner loop can issue aligned dword loads.
    let host = if tensor.dtype == GgufType::Q8_0 {
        let blocks_per_row = (tensor.dims[0] as usize) / 32;
        let row_bytes = blocks_per_row * 34;
        if host.len() % row_bytes != 0 {
            return Err(eyre!(
                "{name}: byte_size {} not a multiple of row_bytes {} (blocks_per_row {})",
                host.len(),
                row_bytes,
                blocks_per_row
            ));
        }
        let num_rows = host.len() / row_bytes;
        repack_q8_0(&host, num_rows, blocks_per_row)
    } else {
        host
    };

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

/// Repack a row-major Q8_0 weight from the GGUF on-disk layout
///   per row: [s0 q0..31 s1 q0..31 ... sN-1 q0..31]   (34 bytes/block)
/// to a split layout
///   per row: [s0 s1 ... sN-1 | q0..31 q0..31 ... q0..31]
///
/// Total bytes per row are unchanged (34·N = 2·N + 32·N). The new
/// layout puts the quants section at an aligned offset within each
/// row (2·N is 4-aligned for any even N, which all V4-Flash Q8_0
/// tensors satisfy), so the matvec inner loop can use aligned 4-byte
/// loads instead of the byte-by-byte unaligned path.
pub fn repack_q8_0(src: &[u8], num_rows: usize, blocks_per_row: usize) -> Vec<u8> {
    let row_bytes = blocks_per_row * 34;
    let mut dst = Vec::with_capacity(src.len());
    dst.resize(src.len(), 0);
    for r in 0..num_rows {
        let src_row = &src[r * row_bytes..(r + 1) * row_bytes];
        let dst_row = &mut dst[r * row_bytes..(r + 1) * row_bytes];
        let (dst_scales, dst_quants) = dst_row.split_at_mut(2 * blocks_per_row);
        for b in 0..blocks_per_row {
            let blk = &src_row[b * 34..(b + 1) * 34];
            dst_scales[b * 2..(b + 1) * 2].copy_from_slice(&blk[..2]);
            dst_quants[b * 32..(b + 1) * 32].copy_from_slice(&blk[2..]);
        }
    }
    dst
}
