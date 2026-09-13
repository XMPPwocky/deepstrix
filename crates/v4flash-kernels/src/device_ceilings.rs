//! Device ceiling probes — see kernels/device_ceilings.hip. Driven by
//! `tests/bench_device_ceilings.rs`; the results are the measured roofline
//! inputs for docs/v41/KERNEL_ROOFLINE.md.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const CEIL_GFX1201: &[u8] = include_bytes!(env!("KERNEL_DEVICE_CEILINGS_GFX1201"));
const CEIL_GFX1151: &[u8] = include_bytes!(env!("KERNEL_DEVICE_CEILINGS_GFX1151"));

pub struct DeviceCeilings {
    module: Module,
}

impl DeviceCeilings {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            CEIL_GFX1201
        } else if arch.starts_with("gfx1151") {
            CEIL_GFX1151
        } else {
            return Err(eyre!("unsupported arch for device_ceilings: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    fn cfg(n_blocks: u32, block_threads: u32) -> LaunchConfig {
        LaunchConfig { grid: (n_blocks, 1, 1), block: (block_threads, 1, 1), shared_mem_bytes: 0 }
    }

    /// Streaming read, 16 B per lane per load. `buf` is read as `uint4`s.
    pub fn read_v4(
        &self,
        stream: &Stream,
        buf: &DeviceBuffer<u8>,
        sink: &mut DeviceBuffer<u32>,
        n_blocks: u32,
        block_threads: u32,
    ) -> eyre::Result<()> {
        let n_vec = (buf.byte_len() / 16) as u64;
        let f = self.module.get_function("ceil_read_v4")?;
        launch_kernel!(f, Self::cfg(n_blocks, block_threads), stream, [buf.raw(), sink.raw(), n_vec])
    }

    /// Streaming read, 4 B per lane per load.
    pub fn read_b32(
        &self,
        stream: &Stream,
        buf: &DeviceBuffer<u8>,
        sink: &mut DeviceBuffer<u32>,
        n_blocks: u32,
        block_threads: u32,
    ) -> eyre::Result<()> {
        let n = (buf.byte_len() / 4) as u64;
        let f = self.module.get_function("ceil_read_b32")?;
        launch_kernel!(f, Self::cfg(n_blocks, block_threads), stream, [buf.raw(), sink.raw(), n])
    }

    /// Wave-per-row contiguous read (the matvec pattern). `row_bytes` must
    /// be a multiple of 16; grid = ceil(n_rows / 8) WGs of 256 threads.
    pub fn read_rows(
        &self,
        stream: &Stream,
        buf: &DeviceBuffer<u8>,
        sink: &mut DeviceBuffer<u32>,
        row_bytes: usize,
        n_rows: u32,
    ) -> eyre::Result<()> {
        if row_bytes % 16 != 0 || (n_rows as usize) * row_bytes > buf.byte_len() {
            return Err(eyre!("read_rows: bad geometry"));
        }
        let row_vecs = (row_bytes / 16) as u64;
        let f = self.module.get_function("ceil_read_rows")?;
        launch_kernel!(f, Self::cfg(n_rows.div_ceil(8), 256), stream, [buf.raw(), sink.raw(), row_vecs, n_rows])
    }

    /// Streaming write, 16 B per lane per store.
    pub fn write_v4(
        &self,
        stream: &Stream,
        buf: &mut DeviceBuffer<u8>,
        n_blocks: u32,
        block_threads: u32,
        v: u32,
    ) -> eyre::Result<()> {
        let n_vec = (buf.byte_len() / 16) as u64;
        let f = self.module.get_function("ceil_write_v4")?;
        launch_kernel!(f, Self::cfg(n_blocks, block_threads), stream, [buf.raw(), n_vec, v])
    }

    /// Read + write, 16 B per lane each way. `dst` must be at least as large as `src`.
    pub fn copy_v4(
        &self,
        stream: &Stream,
        src: &DeviceBuffer<u8>,
        dst: &mut DeviceBuffer<u8>,
        n_blocks: u32,
        block_threads: u32,
    ) -> eyre::Result<()> {
        if dst.byte_len() < src.byte_len() {
            return Err(eyre!("copy_v4: dst smaller than src"));
        }
        let n_vec = (src.byte_len() / 16) as u64;
        let f = self.module.get_function("ceil_copy_v4")?;
        launch_kernel!(f, Self::cfg(n_blocks, block_threads), stream, [src.raw(), dst.raw(), n_vec])
    }

    /// dp4a issue-rate probe. Ops emitted = n_blocks × block_threads × n_iters × 64.
    pub fn dp4a(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<i32>,
        a_in: &DeviceBuffer<i32>,
        b_in: &DeviceBuffer<i32>,
        n_iters: u32,
        n_blocks: u32,
        block_threads: u32,
    ) -> eyre::Result<()> {
        if out.len() < (n_blocks as usize) * (block_threads as usize) || a_in.len() < 8 || b_in.len() < 8 {
            return Err(eyre!("dp4a: buffers too small"));
        }
        let f = self.module.get_function("dp4a_probe")?;
        launch_kernel!(f, Self::cfg(n_blocks, block_threads), stream, [out.raw(), a_in.raw(), b_in.raw(), n_iters])
    }
}
