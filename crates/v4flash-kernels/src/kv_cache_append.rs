//! KV-cache append + SWA-window slide (M13.3).
//!
//! Replaces the host-side `synchronize + copy_to_host + memmove +
//! copy_from_host` round-trip in the orchestrator with a single
//! device-side kernel launch. Saves ~10 ms per token on the dGPU.
//!
//! Single-block kernel; caller must launch with `block.x = head_dim`.
//! Works for any `head_dim ≤ 1024` (HIP max block size on RDNA).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

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
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            cache.raw(), kv_new.raw(), n_raw_before, swa_window, head_dim
        ])
    }

    /// M50 batched: append B contiguous rows to slots
    /// [n_raw_before .. n_raw_before + B). Requires no eviction
    /// (n_raw_before + B <= swa_window). Caller must fall back to a
    /// per-token serial loop when this precondition would fail.
    pub fn launch_batched(
        &self,
        stream: &Stream,
        cache: &mut DeviceBuffer<f32>,
        kv_new: &DeviceBuffer<f32>,
        n_raw_before: u32,
        head_dim: u32,
        b: u32,
    ) -> eyre::Result<()> {
        if b == 0 {
            return Ok(());
        }
        let function = self.module.get_function("kv_cache_append_batched")?;
        let cfg = LaunchConfig {
            grid: (b, 1, 1),
            block: (head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [cache.raw(), kv_new.raw(), n_raw_before, head_dim])
    }

    /// M51 general eviction gather: when the combined sequence
    /// `[cache(r0 rows), kv_new(b rows)]` overflows the window
    /// (`r0 + b > swa_window`), the serial slide loop leaves the cache holding
    /// exactly the last `swa_window` rows of that sequence. Reproduce that
    /// final state in one launch: slot `s ← combined[r0 + b - swa_window + s]`
    /// (old cache for indices `< r0`, else `kv_new`). Handles partial eviction
    /// (`b < swa_window`, surviving prior rows) without underflow.
    ///
    /// Output goes to `out` (a separate scratch ring of `swa_window*head_dim`),
    /// so survivor reads from `cache` cannot race the writes; the caller copies
    /// `out` back into `cache`. Caller must ensure `r0 + b > swa_window`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_evict_gather(
        &self,
        stream: &Stream,
        cache: &DeviceBuffer<f32>,
        kv_new: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        r0: u32,
        b: u32,
        swa_window: u32,
        head_dim: u32,
    ) -> eyre::Result<()> {
        if swa_window == 0 {
            return Ok(());
        }
        if r0 + b <= swa_window {
            return Err(eyre!(
                "kv_cache_append_evict_gather: r0 {r0} + b {b} <= swa_window {swa_window} (no eviction; use launch_batched)"
            ));
        }
        let need = (swa_window as usize) * (head_dim as usize);
        if out.len() < need {
            return Err(eyre!(
                "kv_cache_append_evict_gather: out len {} < swa_window*head_dim {need}",
                out.len()
            ));
        }
        let function = self.module.get_function("kv_cache_append_evict_gather")?;
        let cfg = LaunchConfig {
            grid: (swa_window, 1, 1),
            block: (head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            cache.raw(), kv_new.raw(), out.raw(), r0, b, swa_window, head_dim
        ])
    }
}
