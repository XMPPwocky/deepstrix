//! Simple broadcast / tile kernel — replicate a length-n vector across
//! n_rows contiguous copies. Used by MTP draft to expand e_proj output
//! (N_EMBD) to the N_HC-row HC dim before the per-row h_proj add.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const BROADCAST_GFX1201: &[u8] = include_bytes!(env!("KERNEL_BROADCAST_TO_HC_GFX1201"));
const BROADCAST_GFX1151: &[u8] = include_bytes!(env!("KERNEL_BROADCAST_TO_HC_GFX1151"));

pub struct BroadcastToHc {
    module: Module,
}

impl BroadcastToHc {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            BROADCAST_GFX1201
        } else if arch.starts_with("gfx1151") {
            BROADCAST_GFX1151
        } else {
            return Err(eyre!("unsupported arch for broadcast_to_hc: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Replicate `src[0..n]` across `n_rows` rows in `out`.
    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        src: &DeviceBuffer<f32>,
        n: u32,
        n_rows: u32,
    ) -> eyre::Result<()> {
        if out.len() < (n_rows as usize) * (n as usize) {
            return Err(eyre!(
                "broadcast out len {} < n_rows {} * n {} = {}",
                out.len(),
                n_rows,
                n,
                (n_rows as usize) * (n as usize)
            ));
        }
        if src.len() < n as usize {
            return Err(eyre!("broadcast src len {} < n {n}", src.len()));
        }

        let function = self.module.get_function("broadcast_to_hc")?;
        let block_x = 256u32;
        let grid_x = n.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, n_rows, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), src.raw(), n, n_rows])
    }
}
