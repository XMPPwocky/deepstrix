//! GPU-side MXFP4 HF->ggml repack. See `kernels/mxfp4_repack.hip` for why.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const MXFP4_REPACK_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MXFP4_REPACK_GFX1201"));
const MXFP4_REPACK_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MXFP4_REPACK_GFX1151"));

pub struct Mxfp4Repack {
    module: Module,
}

impl Mxfp4Repack {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            MXFP4_REPACK_GFX1201
        } else if arch.starts_with("gfx1151") {
            MXFP4_REPACK_GFX1151
        } else {
            return Err(eyre!("unsupported arch for mxfp4_repack kernel: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// `dst` gets `out_rows * nb * 17` bytes at `dst_off`; `src` holds the raw
    /// HF bytes as the loader read them: `out_rows * nb * 16` packed nibble
    /// bytes followed by `out_rows * nb` scale bytes.
    /// As [`Self::launch`] but reading the HF bytes from an arbitrary
    /// device-visible pointer instead of a `DeviceBuffer`.
    ///
    /// The point is APU staging: on Strix Halo the "device" pool IS system RAM,
    /// so a `hipHostMalloc` buffer can be pread into by the CPU and read by the
    /// iGPU with **no copy between them**. That removes the 18.8 MB H2D a miss
    /// used to pay (1.23 ms measured on box 2) for bytes that never moved.
    /// `src_len` is only for the bounds check.
    ///
    /// # Safety contract
    /// `src` must point at `>= out_rows*nb*17` device-readable bytes that stay
    /// alive and unwritten until `stream` has synchronized.
    pub fn launch_from_ptr(
        &self,
        stream: &Stream,
        dst: &mut DeviceBuffer<u8>,
        dst_off: usize,
        src: v4flash_hip::sys::hipDeviceptr_t,
        src_len: usize,
        out_rows: u32,
        nb: u32,
    ) -> eyre::Result<()> {
        if nb % 8 != 0 {
            return Err(eyre!("mxfp4_repack needs nb % 8 == 0 (super-block is 8 blocks), got {nb}"));
        }
        let total = out_rows as usize * nb as usize;
        let packed_bytes = total * 16;
        if src_len < packed_bytes + total {
            return Err(eyre!("mxfp4_repack src too small: {src_len} < {}", packed_bytes + total));
        }
        if dst.len() < dst_off + total * 17 {
            return Err(eyre!(
                "mxfp4_repack dst too small: {} < {}",
                dst.len(),
                dst_off + total * 17
            ));
        }
        let function = self.module.get_function("mxfp4_repack_hf_to_ggml")?;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid: ((total as u32).div_ceil(block), 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let dst_v = dst.slice_view_mut(dst_off, total * 17);
        // Scales live immediately after the packed nibbles in the same buffer.
        let packed = src;
        let scale = (src as *mut u8).wrapping_add(packed_bytes) as v4flash_hip::sys::hipDeviceptr_t;
        launch_kernel!(
            function, cfg, stream,
            [dst_v.raw(), packed, scale, out_rows, nb]
        )
    }

    /// As [`Self::launch_from_ptr`] with the packed nibbles and the scale plane
    /// at INDEPENDENT pointers.
    ///
    /// Needed by the zero-copy O_DIRECT path: each region lands at its own file
    /// offset's 4096-residue, so the two are no longer contiguous. They never
    /// had to be — the kernel always took two pointers.
    ///
    /// # Safety contract
    /// Both regions must stay alive and unwritten until `stream` synchronizes.
    pub fn launch_from_ptrs(
        &self,
        stream: &Stream,
        dst: &mut DeviceBuffer<u8>,
        dst_off: usize,
        packed: v4flash_hip::sys::hipDeviceptr_t,
        scale: v4flash_hip::sys::hipDeviceptr_t,
        out_rows: u32,
        nb: u32,
    ) -> eyre::Result<()> {
        if nb % 8 != 0 {
            return Err(eyre!("mxfp4_repack needs nb % 8 == 0 (super-block is 8 blocks), got {nb}"));
        }
        let total = out_rows as usize * nb as usize;
        if dst.len() < dst_off + total * 17 {
            return Err(eyre!(
                "mxfp4_repack dst too small: {} < {}",
                dst.len(),
                dst_off + total * 17
            ));
        }
        let function = self.module.get_function("mxfp4_repack_hf_to_ggml")?;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid: ((total as u32).div_ceil(block), 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let dst_v = dst.slice_view_mut(dst_off, total * 17);
        launch_kernel!(function, cfg, stream, [dst_v.raw(), packed, scale, out_rows, nb])
    }

    pub fn launch(
        &self,
        stream: &Stream,
        dst: &mut DeviceBuffer<u8>,
        dst_off: usize,
        src: &DeviceBuffer<u8>,
        out_rows: u32,
        nb: u32,
    ) -> eyre::Result<()> {
        if nb % 8 != 0 {
            return Err(eyre!("mxfp4_repack needs nb % 8 == 0 (super-block is 8 blocks), got {nb}"));
        }
        let total = out_rows as usize * nb as usize;
        let packed_bytes = total * 16;
        if src.len() < packed_bytes + total {
            return Err(eyre!(
                "mxfp4_repack src too small: {} < {}",
                src.len(),
                packed_bytes + total
            ));
        }
        if dst.len() < dst_off + total * 17 {
            return Err(eyre!(
                "mxfp4_repack dst too small: {} < {}",
                dst.len(),
                dst_off + total * 17
            ));
        }
        let function = self.module.get_function("mxfp4_repack_hf_to_ggml")?;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid: (((total as u32) + block - 1) / block, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        // Scales live immediately after the packed nibbles in the same upload,
        // which is what makes the HF form a single contiguous pread + H2D.
        let dst_v = dst.slice_view_mut(dst_off, total * 17);
        let packed_v = src.slice_view(0, packed_bytes);
        let scale_v = src.slice_view(packed_bytes, total);
        launch_kernel!(
            function, cfg, stream,
            [dst_v.raw(), packed_v.raw(), scale_v.raw(), out_rows, nb]
        )
    }
}
