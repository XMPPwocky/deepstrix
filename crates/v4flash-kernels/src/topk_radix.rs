//! Batched top-K (K <= 512) by radix select (kernels/topk_radix.hip): exactly `IndexerTopkBitonic`'s selection and order
//! (score desc, ties to the lower idx; -1 past min(K, n)), with fixed work -- 3 histogram passes + a compaction, 4
//! launches, no sampling and no give-up. `V41_IDX_TOPK_RADIX`: the decode indexer's top-512.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const TOPK_RADIX_GFX1201: &[u8] = include_bytes!(env!("KERNEL_TOPK_RADIX_GFX1201"));
const TOPK_RADIX_GFX1151: &[u8] = include_bytes!(env!("KERNEL_TOPK_RADIX_GFX1151"));

/// Work-groups per row (`TRX_WG`).
pub const TRX_WG: u32 = 64;
/// Per-row u32 scratch: the global histogram (2048) + pass 2's per-WG rows (64 x 1024) (`TRX_HIST`).
pub const TRX_HIST: usize = 2048 + 64 * 1024;
/// Per-row u32 state (`TRX_STATE`).
pub const TRX_STATE: usize = 8 + 64;
/// Max K and the per-row u64 candidate scratch (`TRX_KMAX`).
pub const TRX_KMAX: u32 = 512;

/// Per-row scratch for [`TopkRadix::launch`]: allocate once per max row count; `hist` and `state` start zeroed and
/// the kernels leave them zeroed (reusable across calls, graph-replay safe).
pub struct TopkRadixScratch {
    pub hist: DeviceBuffer<u32>,
    pub state: DeviceBuffer<u32>,
    pub cand: DeviceBuffer<u64>,
    pub rows: u32,
}

impl TopkRadixScratch {
    pub fn alloc(device_id: i32, rows: u32) -> eyre::Result<Self> {
        let r = rows.max(1) as usize;
        let mut hist = DeviceBuffer::<u32>::new(device_id, r * TRX_HIST)?;
        hist.copy_from_host(&vec![0u32; r * TRX_HIST])?;
        let mut state = DeviceBuffer::<u32>::new(device_id, r * TRX_STATE)?;
        state.copy_from_host(&vec![0u32; r * TRX_STATE])?;
        let cand = DeviceBuffer::<u64>::new(device_id, r * TRX_KMAX as usize)?;
        Ok(Self { hist, state, cand, rows: rows.max(1) })
    }
}

pub struct TopkRadix {
    module: Module,
}

impl TopkRadix {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image = if arch.starts_with("gfx1201") {
            TOPK_RADIX_GFX1201
        } else if arch.starts_with("gfx1151") {
            TOPK_RADIX_GFX1151
        } else {
            return Err(eyre!("unsupported arch for topk_radix: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// `selected[r * k .. + k]` = the best min(k, n_per[r]) ids of `scores[r * stride .. + n_per[r]]` in the chain's
    /// order, then -1. `rows <= scratch.rows`, `k <= 512`, every `n_per[r] <= stride`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &Stream,
        selected: &mut DeviceBuffer<i32>,
        scratch: &mut TopkRadixScratch,
        scores: &DeviceBuffer<f32>,
        n_per: &DeviceBuffer<u32>,
        stride: u32,
        k: u32,
        rows: u32,
    ) -> eyre::Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if k == 0 || k > TRX_KMAX || rows > scratch.rows || selected.len() < (rows * k) as usize || n_per.len() < rows as usize
            || scores.len() < (rows as usize - 1) * stride as usize {
            return Err(eyre!("topk_radix: k {k} rows {rows} (scratch {}) selected {} n_per {} scores {} stride {stride}",
                scratch.rows, selected.len(), n_per.len(), scores.len()));
        }
        let f = self.module.get_function("topk_rx_pass")?;
        let cfg = LaunchConfig { grid: (TRX_WG, rows, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        for pass in 0u32..3 {
            launch_kernel!(f, cfg, stream, [scratch.hist.raw(), scratch.state.raw(), scores.raw(), n_per.raw(), stride, pass, k])?;
        }
        let f = self.module.get_function("topk_rx_compact")?;
        launch_kernel!(f, cfg, stream, [selected.raw(), scratch.cand.raw(), scratch.state.raw(), scores.raw(), n_per.raw(), stride, k])
    }
}
