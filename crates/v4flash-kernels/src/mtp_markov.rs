//! The DSpark exit's markov loop on the device (kernels/mtp_markov.hip, `V41_MTP_DEVICE_MARKOV`):
//! a markov-embedding row gathered by a token id that lives in device memory, and the sampled draft's
//! draw (`het::mtp::draft_dist` in f64) written back to device memory, so the exit's 5 positions
//! enqueue without a host round trip each.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const MTP_MARKOV_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MTP_MARKOV_GFX1201"));
const MTP_MARKOV_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MTP_MARKOV_GFX1151"));

/// `mtp_draft_draw`'s candidate cap (`MD_MAX_M` in the kernel).
pub const DRAW_MAX_M: usize = 64;
/// Work-groups of the multi-WG argmax (`launch_argmax`); its partial scratch is `[ARGMAX_WG]`.
pub const ARGMAX_WG: u32 = 64;

pub struct MtpMarkov {
    module: Module,
}

impl MtpMarkov {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image = if arch.starts_with("gfx1201") {
            MTP_MARKOV_GFX1201
        } else if arch.starts_with("gfx1151") {
            MTP_MARKOV_GFX1151
        } else {
            return Err(eyre!("unsupported arch for mtp_markov: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// `out[k] = f32(table[toks[idx], k])`, `k < rank`; `table` is `[n_vocab, rank]` f16 bits.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gather(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        table: &DeviceBuffer<u16>,
        toks: &DeviceBuffer<i32>,
        idx: u32,
        n_vocab: u32,
        rank: u32,
    ) -> eyre::Result<()> {
        if rank == 0 || rank > 1024 || out.len() < rank as usize || table.len() < n_vocab as usize * rank as usize || toks.len() <= idx as usize {
            return Err(eyre!("mtp_markov gather: rank {rank} out {} table {} toks {} idx {idx}", out.len(), table.len(), toks.len()));
        }
        let f = self.module.get_function("mtp_markov_gather_f16")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (rank, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [out.raw(), table.raw(), toks.raw(), idx, n_vocab, rank])
    }

    /// `sampler.launch_argmax` (argmax_one: lowest index among the maximal non-NaN logits, n - 1 if
    /// none) over `ARGMAX_WG` work-groups, into `toks[out_idx]`. Same result, two launches.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_argmax(
        &self,
        stream: &Stream,
        toks: &mut DeviceBuffer<i32>,
        out_idx: u32,
        logits: &DeviceBuffer<f32>,
        n: u32,
        part_v: &mut DeviceBuffer<f32>,
        part_i: &mut DeviceBuffer<i32>,
    ) -> eyre::Result<()> {
        if n == 0 || logits.len() < n as usize || toks.len() <= out_idx as usize || part_v.len() < ARGMAX_WG as usize || part_i.len() < ARGMAX_WG as usize {
            return Err(eyre!("mtp_markov argmax: n {n} logits {} toks {} out_idx {out_idx}", logits.len(), toks.len()));
        }
        let f = self.module.get_function("mtp_argmax_partial")?;
        let cfg = LaunchConfig { grid: (ARGMAX_WG, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [part_v.raw(), part_i.raw(), logits.raw(), n])?;
        let f = self.module.get_function("mtp_argmax_final")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [toks.raw(), part_v.raw(), part_i.raw(), ARGMAX_WG, n, out_idx])
    }

    /// The sampled draft of one position: `draft_dist(ids[..m], logits[..m], ...)`'s draw, into `toks[out_idx]`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_draw(
        &self,
        stream: &Stream,
        toks: &mut DeviceBuffer<i32>,
        ids: &DeviceBuffer<i32>,
        logits: &DeviceBuffer<f32>,
        m: u32,
        temperature: f32,
        top_p: f32,
        min_p_rel: f32,
        u: f32,
        out_idx: u32,
    ) -> eyre::Result<()> {
        if m == 0 || m as usize > DRAW_MAX_M || ids.len() < m as usize || logits.len() < m as usize || toks.len() <= out_idx as usize || !(temperature > 0.0) {
            return Err(eyre!("mtp_markov draw: m {m} ids {} logits {} toks {} out_idx {out_idx} T {temperature}", ids.len(), logits.len(), toks.len()));
        }
        let f = self.module.get_function("mtp_draft_draw")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (64, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [toks.raw(), ids.raw(), logits.raw(), m, temperature, top_p, min_p_rel, u, out_idx])
    }
}
