//! M28: dGPU clock keep-alive. Loads `keep_alive` kernel (single-block
//! single-thread ALU burn) and exposes a launcher used by the het
//! engine to keep the dGPU "busy" from a secondary stream so its
//! clock doesn't dip between forward_tokens.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{LaunchConfig, Module, Stream};

const KEEP_ALIVE_GFX1201: &[u8] = include_bytes!(env!("KERNEL_KEEP_ALIVE_GFX1201"));
const KEEP_ALIVE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_KEEP_ALIVE_GFX1151"));

pub struct KeepAlive {
    module: Module,
}

impl KeepAlive {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            KEEP_ALIVE_GFX1201
        } else if arch.starts_with("gfx1151") {
            KEEP_ALIVE_GFX1151
        } else {
            return Err(eyre!("unsupported arch for keep_alive: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Burn ~`iters` × few-cycles of ALU on one thread. Caller picks
    /// `iters` to set roughly how long the kernel runs. For a 9070 XT
    /// at 2.7 GHz, ~10000 iters ≈ 4µs; 100000 iters ≈ 40µs.
    pub fn launch(&self, stream: &Stream, iters: i32) -> eyre::Result<()> {
        let function = self.module.get_function("keep_alive")?;
        let mut iters_v = iters;
        let mut args: [*mut c_void; 1] = [&mut iters_v as *mut _ as *mut c_void];
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
