//! Q8_K activation quantization — packs an F32 vector into 256-element
//! Q8_K blocks (292 bytes each) consumed by routed-expert IQ2_XXS/Q2_K
//! matvecs. Mirrors `ds4_quantize_row_q8_K` (ds4.c:1655).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const Q8_K_QUANTIZE_GFX1201: &[u8] = include_bytes!(env!("KERNEL_Q8_K_QUANTIZE_GFX1201"));
const Q8_K_QUANTIZE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_Q8_K_QUANTIZE_GFX1151"));

pub const QK_K: u32 = 256;
pub const BLOCK_Q8_K_BYTES: usize = 292;

pub struct Q8KQuantize {
    module: Module,
}

impl Q8KQuantize {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q8_K_QUANTIZE_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q8_K_QUANTIZE_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q8_k_quantize: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        let needed_x = (n_blocks as usize) * (QK_K as usize);
        let needed_out = (n_blocks as usize) * BLOCK_Q8_K_BYTES;
        if x.len() < needed_x {
            return Err(eyre!(
                "q8_k_quantize x len: have {}, need {}",
                x.len(),
                needed_x
            ));
        }
        if out.byte_len() < needed_out {
            return Err(eyre!(
                "q8_k_quantize out bytes: have {}, need {}",
                out.byte_len(),
                needed_out
            ));
        }

        self.launch_with_offsets(stream, out, 0, x, 0, n_blocks)
    }

    /// Same as [`launch`] but with byte offset into `out` and element
    /// offset into `x`. Lets the MoE pipeline write per-slot q8k blocks
    /// into a single concatenated buffer without per-slot host copies.
    pub fn launch_with_offsets(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u8>,
        out_offset_bytes: usize,
        x: &DeviceBuffer<f32>,
        x_offset_elems: usize,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        let needed_x = (n_blocks as usize) * (QK_K as usize);
        let needed_out = (n_blocks as usize) * BLOCK_Q8_K_BYTES;
        if x.len() < x_offset_elems + needed_x {
            return Err(eyre!(
                "q8_k_quantize x: len {} < offset {} + need {}",
                x.len(),
                x_offset_elems,
                needed_x
            ));
        }
        if out.byte_len() < out_offset_bytes + needed_out {
            return Err(eyre!(
                "q8_k_quantize out: bytes {} < offset {} + need {}",
                out.byte_len(),
                out_offset_bytes,
                needed_out
            ));
        }

        let function = self.module.get_function("q8_k_quantize")?;
        // SAFETY: bounds-checked above.
        let out_ptr = unsafe { (out.raw() as *mut u8).add(out_offset_bytes) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let x_ptr = unsafe { (x.raw() as *mut f32).add(x_offset_elems) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let cfg = LaunchConfig {
            grid: (n_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out_ptr, x_ptr, n_blocks])
    }
}
