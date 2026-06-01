//! F16 weight matvec — mirrors ds4's `matvec_any` fallback for F16 weights.
//! Used by V4 Flash for every compressor + indexer projection (all F16
//! in our model). Same launch geometry as `Q8_0Matvec` (8 rows/workgroup,
//! warp-per-row) without the per-block dequant.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const F16_MATVEC_GFX1201: &[u8] = include_bytes!(env!("KERNEL_F16_MATVEC_GFX1201"));
const F16_MATVEC_GFX1151: &[u8] = include_bytes!(env!("KERNEL_F16_MATVEC_GFX1151"));
const F16_MATVEC_NARROW_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_F16_MATVEC_NARROW_GFX1201"));
const F16_MATVEC_NARROW_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_F16_MATVEC_NARROW_GFX1151"));
const F16_MATVEC_PAIR_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_F16_MATVEC_PAIR_GFX1201"));
const F16_MATVEC_PAIR_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_F16_MATVEC_PAIR_GFX1151"));

const GEMV_ROWS_PER_BLOCK: u32 = 8;
const GEMV_WARP_LANES: u32 = 32;
const NARROW_BLOCK_THREADS: u32 = 256;

/// Below this `n_rows` the original 8-rows-per-block kernel under-fills
/// the GPU (e.g. n_rows=24 → 3 blocks) and the per-row latency chain
/// dominates. The narrow variant (1 row per block, 256 threads
/// cooperating) is faster. Calibrated against the mhc_pre_* calls
/// (n_rows=24, k=16384) on gfx1201; threshold conservative enough that
/// the larger compressor matvecs (n_rows≥256) still take the wide path.
const NARROW_ROWS_THRESHOLD: u32 = 64;

pub struct F16Matvec {
    wide: Module,
    narrow: Module,
    pair: Module,
}

impl F16Matvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let (wide_img, narrow_img, pair_img): (&[u8], &[u8], &[u8]) = if arch.starts_with("gfx1201") {
            (
                F16_MATVEC_GFX1201,
                F16_MATVEC_NARROW_GFX1201,
                F16_MATVEC_PAIR_GFX1201,
            )
        } else if arch.starts_with("gfx1151") {
            (
                F16_MATVEC_GFX1151,
                F16_MATVEC_NARROW_GFX1151,
                F16_MATVEC_PAIR_GFX1151,
            )
        } else {
            return Err(eyre!("unsupported arch for f16_matvec: {arch}"));
        };
        let wide = Module::load_data(wide_img)?;
        let narrow = Module::load_data(narrow_img)?;
        let pair = Module::load_data(pair_img)?;
        Ok(Self { wide, narrow, pair })
    }

    /// Paired matvec: `kv[r] = W_kv[r] · x`, `gate[r] = W_gate[r] · x` for
    /// r in 0..n_rows. Single launch; activation reads shared in cache;
    /// half2/float2-vectorized loads with two independent accumulators
    /// per output (M14h).
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_pair(
        &self,
        stream: &Stream,
        kv: &mut DeviceBuffer<f32>,
        gate: &mut DeviceBuffer<f32>,
        kv_w: &DeviceBuffer<u8>,
        gate_w: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        let expected = (n_rows as usize) * (k as usize) * 2;
        if kv_w.byte_len() != expected || gate_w.byte_len() != expected {
            return Err(eyre!(
                "f16 matvec_pair: weight bytes mismatch (kv={}, gate={}, expected={})",
                kv_w.byte_len(),
                gate_w.byte_len(),
                expected
            ));
        }
        if kv.len() < n_rows as usize || gate.len() < n_rows as usize {
            return Err(eyre!(
                "f16 matvec_pair: out len short (kv={}, gate={}, n_rows={n_rows})",
                kv.len(),
                gate.len()
            ));
        }
        if x.len() < k as usize {
            return Err(eyre!("f16 matvec_pair: x len {} < k {k}", x.len()));
        }
        if k % 2 != 0 {
            return Err(eyre!("f16 matvec_pair: k={k} must be even for half2 loads"));
        }

        let function = self.pair.get_function("f16_matvec_pair")?;
        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            kv.raw(), gate.raw(), kv_w.raw(), gate_w.raw(), x.raw(), k, n_rows
        ])
    }

    /// M50 batched: B independent matvec_pair, sharing the two weight
    /// matrices across all B. `kv`/`gate` outputs are [B, n_rows]; `x` is
    /// [B, k]. One launch instead of B.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_pair_batched(
        &self,
        stream: &Stream,
        kv: &mut DeviceBuffer<f32>,
        gate: &mut DeviceBuffer<f32>,
        kv_w: &DeviceBuffer<u8>,
        gate_w: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        b: u32,
    ) -> eyre::Result<()> {
        if b == 0 {
            return Ok(());
        }
        if k % 2 != 0 {
            return Err(eyre!("f16 matvec_pair_batched: k={k} must be even"));
        }
        let function = self.pair.get_function("f16_matvec_pair_batched")?;
        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, b),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            kv.raw(), gate.raw(), kv_w.raw(), gate_w.raw(), x.raw(), k, n_rows
        ])
    }

    /// `out[r] = sum_i f32(weight[r, i]) * x[i]` for `r in 0..n_rows`.
    /// Weight is F16 row-major `[n_rows, k]`, passed as a `DeviceBuffer<u8>`
    /// holding raw F16 bytes (mirrors how Q8_0 weights are passed).
    /// K-split f16 matvec for narrow M (e.g., HC_MIX_DIM=24, HC_DIM=16384).
    /// Two kernels:
    ///   pass 1 partial: grid (n_k_split, 1, 1). Each WG stages its
    ///     K-slice of x in LDS once, computes ALL n_rows outputs against
    ///     that x. Eliminates the 24× redundant x reads of the legacy
    ///     narrow path (which had Grid(n_rows,1,1) and each WG re-read
    ///     all of x).
    ///   pass 2 reduce + pre_scale apply: sums partials per row,
    ///     multiplies by pre_scale[0].
    /// Requires k % n_k_split == 0, k_chunk = k/n_k_split ≤ 1024 (LDS).
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_narrow_ksplit_pre_scaled(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        pre_scale: &DeviceBuffer<f32>,
        partials: &mut DeviceBuffer<f32>,    // [n_k_split, n_rows] f32
        n_rows: u32,
        k: u32,
        n_k_split: u32,
    ) -> eyre::Result<()> {
        if k % n_k_split != 0 {
            return Err(eyre!(
                "matvec_narrow_ksplit_pre_scaled: k={k} not divisible by n_k_split={n_k_split}"
            ));
        }
        let k_chunk = k / n_k_split;
        if k_chunk > 1024 {
            return Err(eyre!(
                "matvec_narrow_ksplit_pre_scaled: k_chunk={k_chunk} exceeds LDS budget 1024"
            ));
        }
        let needed = (n_k_split as usize) * (n_rows as usize);
        if partials.len() < needed {
            return Err(eyre!(
                "matvec_narrow_ksplit_pre_scaled: partials len={} < {needed}",
                partials.len()
            ));
        }
        // F16_KSPLIT_V8=1 uses the u128 vector-load variant (8 f16 per
        // load instruction instead of 1). Requires k_chunk = k/n_k_split
        // to be a multiple of 256 (32 lanes × 8 elems).
        let use_v8 = (k_chunk % 256 == 0)
            && std::env::var("F16_KSPLIT_V8").map(|v| v != "0").unwrap_or(true);
        let f_part = if use_v8 {
            self.narrow.get_function("f16_matvec_narrow_ksplit_partial_v8")?
        } else {
            self.narrow.get_function("f16_matvec_narrow_ksplit_partial")?
        };
        let f_red = self
            .narrow
            .get_function("f16_matvec_narrow_ksplit_reduce_pre_scaled")?;
        let cfg_p = LaunchConfig {
            grid: (n_k_split, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let cfg_r = LaunchConfig {
            grid: (1, 1, 1),
            block: (32.max(n_rows.next_power_of_two()), 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(f_part, cfg_p, stream, [
            partials.raw(), weight.raw(), x.raw(), k, k_chunk, n_rows
        ])?;
        launch_kernel!(f_red, cfg_r, stream, [
            out.raw(), partials.raw(), pre_scale.raw(), n_k_split, n_rows
        ])
    }

    /// Pre-scaled matvec: same as `matvec` but each output is multiplied
    /// by the scalar in `pre_scale[0]`. Pairs with a multi-WG RMS variant
    /// that just computes inv_rms (no apply pass) — eliminates one full
    /// N=k DRAM round-trip and one kernel launch from the mhc_pre chain.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_pre_scaled(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        pre_scale: &DeviceBuffer<f32>,  // [1]
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        if n_rows < NARROW_ROWS_THRESHOLD {
            let function = self.narrow.get_function("f16_matvec_narrow_pre_scaled")?;
            let cfg = LaunchConfig {
                grid: (n_rows, 1, 1),
                block: (NARROW_BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            launch_kernel!(function, cfg, stream, [
                out.raw(), weight.raw(), x.raw(), pre_scale.raw(), k, n_rows
            ])
        } else {
            let function = self.wide.get_function("f16_matvec_pre_scaled")?;
            let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
            let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
            let cfg = LaunchConfig {
                grid: (grid_x, 1, 1),
                block: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            launch_kernel!(function, cfg, stream, [
                out.raw(), weight.raw(), x.raw(), pre_scale.raw(), k, n_rows
            ])
        }
    }

    pub fn matvec(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        let expected_weight_bytes = (n_rows as usize) * (k as usize) * 2;
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "f16 matvec weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if out.len() < n_rows as usize {
            return Err(eyre!(
                "f16 matvec out len: have {}, expected n_rows={n_rows}",
                out.len()
            ));
        }
        if x.len() < k as usize {
            return Err(eyre!(
                "f16 matvec x len: have {}, expected k={k}",
                x.len()
            ));
        }

        if n_rows < NARROW_ROWS_THRESHOLD {
            let function = self.narrow.get_function("f16_matvec_narrow")?;
            let cfg = LaunchConfig {
                grid: (n_rows, 1, 1),
                block: (NARROW_BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), x.raw(), k, n_rows])
        } else {
            let function = self.wide.get_function("f16_matvec")?;
            let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
            let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
            let cfg = LaunchConfig {
                grid: (grid_x, 1, 1),
                block: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), x.raw(), k, n_rows])
        }
    }

    /// Batched WIDE f16 matvec with grid.z = B: `out[b, r] = sum_i
    /// f32(W[r,i]) * x[b,i]`. Per-row reduction is the identical single-warp
    /// shuffle as `matvec`'s wide path, so each output is bit-identical to a
    /// per-batch loop of `matvec` — only the launch count drops to 1. Use for
    /// `n_rows >= NARROW_ROWS_THRESHOLD` (e.g. the router gate, n_rows=256).
    pub fn matvec_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let expected_weight_bytes = (n_rows as usize) * (k as usize) * 2;
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "f16 matvec_batched weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if x.len() < (batch as usize) * (k as usize) {
            return Err(eyre!("f16 matvec_batched x too small: {}", x.len()));
        }
        if out.len() < (batch as usize) * (n_rows as usize) {
            return Err(eyre!("f16 matvec_batched out too small: {}", out.len()));
        }
        let function = self.wide.get_function("f16_matvec_batched")?;
        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), x.raw(), k, n_rows])
    }

    /// M50 Phase 2: batched narrow f16 matvec with grid.z = B. For each
    /// batch element, computes `out[b, r] = sum_i f32(W[r,i]) * x[b,i]`.
    /// Uses the narrow kernel (1 block per row, 256 threads cooperating)
    /// suitable for small n_rows. Wide variant for B not implemented yet —
    /// fall back to per-batch loop if n_rows >= NARROW_ROWS_THRESHOLD.
    pub fn matvec_narrow_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let expected_w = (n_rows as usize) * (k as usize) * 2;
        if weight.byte_len() != expected_w {
            return Err(eyre!(
                "f16 matvec_narrow_batched weight bytes: {} != {}",
                weight.byte_len(),
                expected_w
            ));
        }
        if x.len() < (batch as usize) * (k as usize) {
            return Err(eyre!("f16 matvec_narrow_batched x: too small"));
        }
        if out.len() < (batch as usize) * (n_rows as usize) {
            return Err(eyre!("f16 matvec_narrow_batched out: too small"));
        }

        let function = self.narrow.get_function("f16_matvec_narrow_batched")?;
        let cfg = LaunchConfig {
            grid: (n_rows, 1, batch),
            block: (NARROW_BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), x.raw(), k, n_rows])
    }

    /// M40-P4.5: 2-wide pair variant — ONE weight, TWO input vectors → TWO
    /// outputs. Halves W bandwidth vs running `matvec` twice. NB: this is
    /// the OPPOSITE pattern from `matvec_pair` (which shares ONE input
    /// across TWO weights — used by compressor kv+gate).
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_two_inputs(
        &self,
        stream: &Stream,
        out_a: &mut DeviceBuffer<f32>,
        out_b: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x_a: &DeviceBuffer<f32>,
        x_b: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        let expected_weight_bytes = (n_rows as usize) * (k as usize) * 2;
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "f16 matvec_two_inputs weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if out_a.len() < n_rows as usize || out_b.len() < n_rows as usize {
            return Err(eyre!(
                "f16 matvec_two_inputs out lens: a={}, b={}, expected {}",
                out_a.len(),
                out_b.len(),
                n_rows
            ));
        }
        if x_a.len() < k as usize || x_b.len() < k as usize {
            return Err(eyre!(
                "f16 matvec_two_inputs x lens: a={}, b={}, expected {}",
                x_a.len(),
                x_b.len(),
                k
            ));
        }

        // M40-P7: dispatch narrow variant for tiny n_rows (e.g. HC_MIX_DIM=24)
        // — the wide variant launches only 3 WGs and runs at ~1.5% peak BW.
        // Narrow uses 1 WG per row with 256 threads cooperating per row →
        // better CU occupancy, ~3x faster on narrow shapes.
        if n_rows < NARROW_ROWS_THRESHOLD {
            let function = self.narrow.get_function("f16_matvec_narrow_two_inputs")?;
            let cfg = LaunchConfig {
                grid: (n_rows, 1, 1),
                block: (NARROW_BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            launch_kernel!(function, cfg, stream, [
                out_a.raw(), out_b.raw(), weight.raw(), x_a.raw(), x_b.raw(), k, n_rows
            ])
        } else {
            let function = self.wide.get_function("f16_matvec_two_inputs")?;
            let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
            let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
            let cfg = LaunchConfig {
                grid: (grid_x, 1, 1),
                block: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            launch_kernel!(function, cfg, stream, [
                out_a.raw(), out_b.raw(), weight.raw(), x_a.raw(), x_b.raw(), k, n_rows
            ])
        }
    }
}
