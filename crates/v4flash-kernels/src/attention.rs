//! SWA attention compute — mirrors ds4's `layer_attention_rows_one`
//! (ds4.c:4955). Sink-aware causal softmax + weighted sum over the raw KV
//! cache. Used by V4 Flash layers L=0, L=1 (the dense / `ratio==0` layers).
//!
//! For L≥2, ds4 dispatches to `layer_attention_mixed_one` which extends
//! the softmax with compressed-KV rows + indexer masking — that's M6/M7.
//! The M5 SWA kernel is the building block both variants share.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const ATTENTION_SWA_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_SWA_GFX1201"));
const ATTENTION_SWA_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_SWA_GFX1151"));

/// Compile-time max for the kernel's shared-memory `scores`/`weights`
/// arrays. Matches the SWA window size in ds4 (`DS4_N_SWA = 128`).
pub const ATTN_SWA_MAX_KV: u32 = 128;

pub struct AttentionSwa {
    module: Module,
}

impl AttentionSwa {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            ATTENTION_SWA_GFX1201
        } else if arch.starts_with("gfx1151") {
            ATTENTION_SWA_GFX1151
        } else {
            return Err(eyre!("unsupported arch for attention_swa kernel: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Launch the SWA attention kernel.
    ///
    /// - `out`: `[n_head * head_dim]`
    /// - `q`:   `[n_head * head_dim]` (post-RoPE)
    /// - `kv`:  `[n_kv * head_dim]`   (f16-precision values in f32 cells —
    ///                                 ds4's cache stores `f16_to_f32(f32_to_f16(x))`)
    /// - `sinks`: `[n_head]`
    /// - `n_kv ≤ ATTN_SWA_MAX_KV`
    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        kv: &DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        n_head: u32,
        head_dim: u32,
        n_kv: u32,
    ) -> eyre::Result<()> {
        if n_kv > ATTN_SWA_MAX_KV {
            return Err(eyre!(
                "attention_swa: n_kv={n_kv} exceeds kernel cap {ATTN_SWA_MAX_KV}"
            ));
        }
        if n_kv == 0 {
            return Err(eyre!("attention_swa: n_kv must be > 0"));
        }
        let needed_out = (n_head as usize) * (head_dim as usize);
        if out.len() < needed_out || q.len() < needed_out {
            return Err(eyre!(
                "attention_swa: out/q have {}/{} elems, need {}",
                out.len(),
                q.len(),
                needed_out
            ));
        }
        if kv.len() < (n_kv as usize) * (head_dim as usize) {
            return Err(eyre!(
                "attention_swa: kv has {} elems, need n_kv*head_dim={}",
                kv.len(),
                (n_kv as usize) * (head_dim as usize)
            ));
        }
        if sinks.len() < n_head as usize {
            return Err(eyre!(
                "attention_swa: sinks has {} elems, need n_head={}",
                sinks.len(),
                n_head
            ));
        }

        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self.module.get_function("attention_swa")?;

        let mut out_ptr = out.raw();
        let mut q_ptr = q.raw();
        let mut kv_ptr = kv.raw();
        let mut sinks_ptr = sinks.raw();
        let mut n_head_v = n_head;
        let mut head_dim_v = head_dim;
        let mut n_kv_v = n_kv;
        let mut kq_scale_v = kq_scale;

        let mut args: [*mut c_void; 8] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut q_ptr as *mut _ as *mut c_void,
            &mut kv_ptr as *mut _ as *mut c_void,
            &mut sinks_ptr as *mut _ as *mut c_void,
            &mut n_head_v as *mut _ as *mut c_void,
            &mut head_dim_v as *mut _ as *mut c_void,
            &mut n_kv_v as *mut _ as *mut c_void,
            &mut kq_scale_v as *mut _ as *mut c_void,
        ];

        let cfg = LaunchConfig {
            grid: (n_head, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
