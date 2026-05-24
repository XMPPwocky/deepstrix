//! KV-cache append + SWA-window slide (M13.3).
//!
//! Replaces the host-side `synchronize + copy_to_host + memmove +
//! copy_from_host` round-trip in the orchestrator with a single
//! device-side kernel launch. Saves ~10 ms per token on the dGPU.
//!
//! Single-block kernel; caller must launch with `block.x = head_dim`.
//! Works for any `head_dim ≤ 1024` (HIP max block size on RDNA).

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const KV_CACHE_APPEND_GFX1201: &[u8] = include_bytes!(env!("KERNEL_KV_CACHE_APPEND_GFX1201"));
const KV_CACHE_APPEND_GFX1151: &[u8] = include_bytes!(env!("KERNEL_KV_CACHE_APPEND_GFX1151"));

pub struct KvCacheAppend {
    module: Module,
}

impl KvCacheAppend {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            KV_CACHE_APPEND_GFX1201
        } else if arch.starts_with("gfx1151") {
            KV_CACHE_APPEND_GFX1151
        } else {
            return Err(eyre!("unsupported arch for kv_cache_append: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `cache` must be `[swa_window * head_dim]` f32. `kv_new` is
    /// `[head_dim]`. `n_raw_before` is the current fill count (0..=swa_window).
    pub fn launch(
        &self,
        stream: &Stream,
        cache: &mut DeviceBuffer<f32>,
        kv_new: &DeviceBuffer<f32>,
        n_raw_before: u32,
        swa_window: u32,
        head_dim: u32,
    ) -> eyre::Result<()> {
        if head_dim == 0 || head_dim > 1024 {
            return Err(eyre!(
                "kv_cache_append: head_dim must be in [1, 1024], got {head_dim}"
            ));
        }
        if n_raw_before > swa_window {
            return Err(eyre!(
                "kv_cache_append: n_raw_before {n_raw_before} > swa_window {swa_window}"
            ));
        }
        let need = (swa_window as usize) * (head_dim as usize);
        if cache.len() < need {
            return Err(eyre!(
                "kv_cache_append: cache len {} < swa_window*head_dim {}",
                cache.len(),
                need
            ));
        }
        if kv_new.len() < head_dim as usize {
            return Err(eyre!(
                "kv_cache_append: kv_new len {} < head_dim {}",
                kv_new.len(),
                head_dim
            ));
        }

        let function = self.module.get_function("kv_cache_append")?;
        let mut cache_ptr = cache.raw();
        let mut new_ptr = kv_new.raw();
        let mut n = n_raw_before;
        let mut sw = swa_window;
        let mut hd = head_dim;
        let mut args: [*mut c_void; 5] = [
            &mut cache_ptr as *mut _ as *mut c_void,
            &mut new_ptr as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
