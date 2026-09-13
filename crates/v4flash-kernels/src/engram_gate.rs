//! Launcher for `kernels/engram_gate_add.hip` — the V4.1 Engram gate + residual
//! add on the hc copies (ARCH_SPEC §1.7). The wkv projection feeding it is the
//! ordinary Q8_0 matvec (`q8_0::Q8Matvec`), so this module only owns the tail.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const EG_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ENGRAM_GATE_ADD_GFX1201"));
const EG_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ENGRAM_GATE_ADD_GFX1151"));

pub struct EngramGateAdd {
    module: Module,
}

impl EngramGateAdd {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            EG_GFX1201
        } else if arch.starts_with("gfx1151") {
            EG_GFX1151
        } else {
            return Err(eyre!("unsupported arch for engram_gate_add: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// `h [batch, hc, dim]` += gate(h, key) · value, with `kv [batch, kv_stride]`
    /// holding key at 0 (`hc·dim` floats) and value at `value_off` (`dim`
    /// floats) — i.e. the wkv output as produced — and `qk [hc, dim]` the
    /// q_weight ⊙ k_weight product.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &Stream,
        h: &mut DeviceBuffer<f32>,
        kv: &DeviceBuffer<f32>,
        qk: &DeviceBuffer<f32>,
        hc: u32,
        dim: u32,
        kv_stride: u32,
        value_off: u32,
        eps: f32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if h.len() < (batch * hc * dim) as usize || kv.len() < (batch * kv_stride) as usize
            || qk.len() < (hc * dim) as usize || value_off + dim > kv_stride
        {
            return Err(eyre!("engram_gate_add: buffer sizes (h {}, kv {}, qk {}) vs batch {batch} hc {hc} dim {dim} stride {kv_stride}", h.len(), kv.len(), qk.len()));
        }
        let function = self.module.get_function("engram_gate_add")?;
        let cfg = LaunchConfig { grid: (hc, batch, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [h.raw(), kv.raw(), qk.raw(), hc, dim, kv_stride, value_off, eps])
    }
}
