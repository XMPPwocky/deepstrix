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
/// `launch_topm` geometry (kernels `TM_WG` / `TM_CHUNK` / `TM_KEEP`): work-groups, max chunk, kept per WG.
pub const TOPM_WG: u32 = 64;
pub const TOPM_CHUNK: u32 = 2048;
pub const TOPM_KEEP: u32 = 64;
/// `launch_topm_radix` geometry (kernels `RX_WG` / `RX_BINS` / `RX_ST_EQOFF`): work-groups, histogram bins,
/// and the state layout (`[RX_ST_EQOFF + RX_WG]` u32, zero-initialised once; the kernels leave it reusable).
pub const RX_WG: u32 = 64;
pub const RX_BINS: u32 = 2048;
pub const RX_STATE: usize = 8 + RX_WG as usize;
/// `launch_topm_radix`'s `hist`: the global histogram (`RX_BINS`) + pass 2's per-WG rows (`RX_WG * 1024`); zero-initialise
/// once (the kernels leave the global part zero).
pub const RX_HIST_LEN: usize = RX_BINS as usize + RX_WG as usize * 1024;

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

    /// The top `m` (<= 64) ids of `logits[..n]` in the bitonic chain's order (score desc, ties to the lower id),
    /// spread over `TOPM_WG` work-groups with fixed work: `mtp_topm_chunk` (each WG its chunk's best 64) +
    /// `mtp_topm_final` (one WG over the `TOPM_WG * 64` survivors). `part` is `[TOPM_WG * 64]` u64 scratch.
    pub fn launch_topm(
        &self,
        stream: &Stream,
        selected: &mut DeviceBuffer<i32>,
        part: &mut DeviceBuffer<u64>,
        logits: &DeviceBuffer<f32>,
        n: u32,
        m: u32,
    ) -> eyre::Result<()> {
        if m == 0 || m > TOPM_KEEP || n == 0 || n.div_ceil(TOPM_WG) > TOPM_CHUNK || logits.len() < n as usize
            || selected.len() < m as usize || part.len() < (TOPM_WG * TOPM_KEEP) as usize {
            return Err(eyre!("mtp_markov topm: n {n} m {m} logits {} selected {} part {}", logits.len(), selected.len(), part.len()));
        }
        let f = self.module.get_function("mtp_topm_chunk")?;
        let cfg = LaunchConfig { grid: (TOPM_WG, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [part.raw(), logits.raw(), n])?;
        let f = self.module.get_function("mtp_topm_final")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (1024, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [selected.raw(), part.raw(), m])
    }

    /// The top `m` (<= 64) ids of `logits[..n]` in the bitonic chain's order (score desc, ties to the lower id) by
    /// radix select: three histogram passes (`mtp_rx_pass`, 11 + 11 + 10 key bits; the last WG of each fixes the
    /// threshold's next bits) and `mtp_rx_compact` (key > threshold + the lowest-idx ties, then the last WG sorts the
    /// m winners). Fixed work, 4 launches. `hist` = `[RX_HIST_LEN]` u32 and `state` = `[RX_STATE]` u32, both zeroed
    /// once (left reusable), `cand` = `[64]` u64.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_topm_radix(
        &self,
        stream: &Stream,
        selected: &mut DeviceBuffer<i32>,
        hist: &mut DeviceBuffer<u32>,
        state: &mut DeviceBuffer<u32>,
        cand: &mut DeviceBuffer<u64>,
        logits: &DeviceBuffer<f32>,
        n: u32,
        m: u32,
    ) -> eyre::Result<()> {
        if m == 0 || m > 64 || n < m || logits.len() < n as usize || selected.len() < m as usize
            || hist.len() < RX_HIST_LEN || state.len() < RX_STATE || cand.len() < 64 {
            return Err(eyre!("mtp_markov radix topm: n {n} m {m} logits {} selected {} hist {} state {} cand {}",
                logits.len(), selected.len(), hist.len(), state.len(), cand.len()));
        }
        let f = self.module.get_function("mtp_rx_pass")?;
        let cfg = LaunchConfig { grid: (RX_WG, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        for pass in 0u32..3 {
            launch_kernel!(f, cfg, stream, [hist.raw(), state.raw(), logits.raw(), n, pass, m])?;
        }
        let f = self.module.get_function("mtp_rx_compact")?;
        launch_kernel!(f, cfg, stream, [selected.raw(), cand.raw(), state.raw(), logits.raw(), n, m])
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
