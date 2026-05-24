//! V4 Flash CSA indexer — produces the `comp_allowed` boolean mask that
//! `attention_mixed` (M6) consumes for ratio==4 layers.
//!
//! Composition (per token in a ratio==4 layer):
//!   1. F16 matvec(indexer.attn_q_b × qr_norm) → indexer_q[64, 128]
//!   2. RoPE forward on indexer_q
//!   3. F16 matvec(indexer.proj × attn_norm) → head_weights[64]
//!   4. Scale head_weights by 1/sqrt(head_dim * n_head)
//!   5. Per-comp-row score via `IndexerScore` kernel
//!   6. Top-K = DS4_N_INDEXER_TOP_K = 512 greedy selection → bool mask
//!
//! Early return: if `n_comp <= top_k`, ds4 returns all-permit without
//! computing q/weights/scores. Our pipeline mirrors that.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const INDEXER_SCORE_GFX1201: &[u8] = include_bytes!(env!("KERNEL_INDEXER_SCORE_GFX1201"));
const INDEXER_SCORE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_INDEXER_SCORE_GFX1151"));

pub const INDEXER_TOP_K: u32 = 512;
pub const INDEXER_N_HEAD: u32 = 64;
pub const INDEXER_HEAD_DIM: u32 = 128;

/// Per-comp-row scoring kernel.
pub struct IndexerScore {
    module: Module,
}

impl IndexerScore {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            INDEXER_SCORE_GFX1201
        } else if arch.starts_with("gfx1151") {
            INDEXER_SCORE_GFX1151
        } else {
            return Err(eyre!("unsupported arch for indexer_score: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `scores[c] = sum_h max(0, dot(q[h], index_comp_kv[c])) * head_weights[h]`
    /// for `c in 0..n_comp`.
    pub fn launch(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        head_weights: &DeviceBuffer<f32>,
        index_comp_kv: &DeviceBuffer<f32>,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
    ) -> eyre::Result<()> {
        if n_comp == 0 {
            return Err(eyre!("indexer_score: n_comp must be > 0"));
        }
        let function = self.module.get_function("indexer_score")?;

        let mut scores_ptr = scores.raw();
        let mut q_ptr = q.raw();
        let mut hw_ptr = head_weights.raw();
        let mut kv_ptr = index_comp_kv.raw();
        let mut n_comp_v = n_comp;
        let mut n_head_v = n_head;
        let mut head_dim_v = head_dim;
        let mut args: [*mut c_void; 7] = [
            &mut scores_ptr as *mut _ as *mut c_void,
            &mut q_ptr as *mut _ as *mut c_void,
            &mut hw_ptr as *mut _ as *mut c_void,
            &mut kv_ptr as *mut _ as *mut c_void,
            &mut n_comp_v as *mut _ as *mut c_void,
            &mut n_head_v as *mut _ as *mut c_void,
            &mut head_dim_v as *mut _ as *mut c_void,
        ];

        let cfg = LaunchConfig {
            grid: (n_comp, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
