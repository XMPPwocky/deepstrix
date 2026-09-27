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
const F16_GEMM_WMMA_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_F16_GEMM_WMMA_GFX1201"));
const F16_GEMM_WMMA_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_F16_GEMM_WMMA_GFX1151"));

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

/// `V41_ROUTER_MV_H20` (default ON; `0` = `f16_matvec_batched`): the router
/// logits matvec at decode / replay ([`F16Matvec::matvec_batched_router`],
/// W [384, 5120] f16, grid (48, 1, b)) runs `f16_matvec_batched_h20`, which
/// keeps every lane's element order and the single f32 accumulator but loads
/// 20 elements per lane per chunk into registers with the next chunk's loads
/// in flight (double buffer). BIT-IDENTICAL (tests/mhc_glue_bitexact.rs; the
/// sweep review: 72 + 45 compares incl. tails and the other matvec_batched
/// shapes). 2026-09-26 sweep (F_mhc_glue/router_mv_h20, dGPU, W cold as in
/// production): b=1 21.7 -> 10.4 us, b=4 21.9 -> 13.4, b=8 41.9 -> 18.4.
/// Only when k % 640 == 0 (else the kernel would run the production loop anyway).
fn router_mv_h20_for(k: u32) -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_ROUTER_MV_H20").as_deref() != Ok("0"));
    *D && k % (GEMV_WARP_LANES * 20) == 0
}

/// `V41_MHC_GEMM_NARROW` (default ON; `0` = `f16_gemm_wmma_lds_tiled` at every
/// shape): [`F16Matvec::gemm_batched_wmma`] with `n_rows <= 32`, `k % 128 == 0`
/// and 16-byte aligned `x` / `weight` -- the prefill mHC pre-mix GEMMs (M =
/// HC_MIX_DIM = 24, K = HC_DIM = 20480; pre-attn and pre-ffn at b > 64) -- runs
/// `f16_gemm_narrow_n16_bk128_pf2`, grid (1, ceil(b/16)) x 256: one 16-column
/// n-tile per WG, all 8 waves streaming BK=128 K-chunks with b128 loads, PF=2,
/// double-buffered LDS. The same WMMA chain in the same K order with the same
/// f16 fragments: BIT-IDENTICAL (tests/mhc_glue_bitexact.rs; the sweep review
/// 108/108 at B = 1..1024). 2026-09-26 sweep (F_mhc_glue/gemm_narrow_m24, dGPU):
/// B=512 476 -> 80 us (x5.4-6.0 over three cache regimes), B=65..128 ~7x;
/// 160 calls per 1024-row chunk. The kernel writes NOTHING outside its shape
/// preconditions, so this gate is the only thing routing to it.
fn mhc_gemm_narrow_for(n_rows: u32, k: u32, weight: &DeviceBuffer<u8>, x: &DeviceBuffer<f32>) -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_MHC_GEMM_NARROW").as_deref() != Ok("0"));
    *D && n_rows <= 32
        && k % 128 == 0
        && (weight.raw() as usize) % 16 == 0
        && (x.raw() as usize) % 16 == 0
}

/// `V41_F16_MV_Z16` (default ON; `0` = `f16_matvec_batched`, grid.z = batch):
/// [`F16Matvec::matvec_batched_z16`] runs `f16_matvec_batched_z16_n<NB>` --
/// the batch split into NB-row slices on grid.z (NB = 1, 2, 4, 8 or 16, the
/// smallest >= min(batch, 16)), each weight element read once per slice instead
/// of once per batch row, 8 hoisted weight loads per lane -- with the
/// `V41_GRID_PAD` idle work-group on the wide grids. BIT-IDENTICAL
/// (tests/f16_mv_z16_bitexact.rs: every NB symbol, pad 0/1, 7 shapes, b up to 64).
/// Callers: the indexer q projection (4096 x 1280, every b <= 64), the indexer
/// head-weight projection (32 x 5120, b <= [`Z16_PROJ_MAX_B`]), the ratio-1
/// compressor (512 x 5120, [`z16_comp_ratio1_for`]) and the replay router
/// (b > [`Z16_ROUTER_MIN_B`]). 2026-09-27 round 2 (c_z16, dGPU, cold weights,
/// graph, 6 runs, ratio vs the production kernel, med / p10): idx q b = 1 0.73 /
/// 0.74, 2 0.61 / 0.58, 3 0.60 / 0.50, 4 0.49 / 0.41 (72.8 -> 38.9 us), 5 0.39,
/// 6 0.47, 8 0.31, 16 0.31, 32 0.29, 64 0.26 (639 -> 164 us, x8 index layers x2
/// lanes per replay); compressor b = 1 0.51, 2 0.86 / 0.68, 5 0.77, 8 0.84, 16
/// 0.90, 32 0.50, 64 0.30; proj b = 1 0.28, 2 0.33, 3 0.64, 4 0.80.
fn f16_mv_z16_on() -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_F16_MV_Z16").as_deref() != Ok("0"));
    *D
}

/// Batch above which the router logits matvec ([`F16Matvec::matvec_batched_router`],
/// 384 x 5120) takes the z16 kernel instead of `_h20`: z16 / h20 = 1.43 at b = 32,
/// 0.98 at 48, 0.80 at 64 (cold W, 3 runs) -- i.e. only the replay-sized passes.
pub const Z16_ROUTER_MIN_B: u32 = 48;

/// Largest batch at which the idx head-weight projection (32 x 5120: a 4-WG
/// grid) takes [`F16Matvec::matvec_batched_z16`]: 0.28 / 0.29 / 0.65 / 0.82 at
/// b = 1..4, neutral at 5-6, LOSES x1.36 at 8 and x2.4 at 16 (one slice of 4 WGs).
pub const Z16_PROJ_MAX_B: u32 = 4;

/// Whether the ratio-1 compressor matvec (512 x 5120, a 64-WG grid) takes
/// [`F16Matvec::matvec_batched_z16`] at `batch`: it wins at every measured b
/// except 3 and 4 (x1.05-1.08 / ~1.0: there the 3-4 z-slices of the production
/// grid still out-parallelise one z16 slice), so those two keep grid.z = b.
pub fn z16_comp_ratio1_for(batch: u32) -> bool {
    !(3..=4).contains(&batch)
}

pub struct F16Matvec {
    wide: Module,
    narrow: Module,
    pair: Module,
    gemm_wmma: Option<Module>,
}

impl F16Matvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let (wide_img, narrow_img, pair_img, gemm_img): (
            &[u8], &[u8], &[u8], Option<&[u8]>,
        ) = if arch.starts_with("gfx1201") {
            (
                F16_MATVEC_GFX1201,
                F16_MATVEC_NARROW_GFX1201,
                F16_MATVEC_PAIR_GFX1201,
                Some(F16_GEMM_WMMA_GFX1201),
            )
        } else if arch.starts_with("gfx1151") {
            (
                F16_MATVEC_GFX1151,
                F16_MATVEC_NARROW_GFX1151,
                F16_MATVEC_PAIR_GFX1151,
                // gfx1151 has no WMMA; the kernel compiles but body is a
                // no-op. We just never dispatch it on iGPU.
                Some(F16_GEMM_WMMA_GFX1151),
            )
        } else {
            return Err(eyre!("unsupported arch for f16_matvec: {arch}"));
        };
        let wide = Module::load_data(wide_img)?;
        let narrow = Module::load_data(narrow_img)?;
        let pair = Module::load_data(pair_img)?;
        let gemm_wmma = gemm_img.map(Module::load_data).transpose()?;
        Ok(Self { wide, narrow, pair, gemm_wmma })
    }

    /// LDS-tiled WMMA GEMM for f16 weights × f32 activations (gfx12 only).
    /// Replaces `matvec_batched` for shapes where B is large enough to
    /// benefit from cooperative tile loading (the matvec-per-batch path
    /// re-fetches weights per row and is weight-BW-bound at large B).
    /// Tile shape BM=BN=64, BK=32; out and batch dims align upward to 64.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batched_wmma(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_rows == 0 {
            return Ok(());
        }
        let module = self
            .gemm_wmma
            .as_ref()
            .ok_or_else(|| eyre!("f16 gemm_batched_wmma: no module for this arch"))?;
        let expected_weight_bytes = (n_rows as usize) * (k as usize) * 2;
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "f16 gemm_batched_wmma weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if x.len() < (batch as usize) * (k as usize) {
            return Err(eyre!("f16 gemm_batched_wmma x too small: {}", x.len()));
        }
        if out.len() < (batch as usize) * (n_rows as usize) {
            return Err(eyre!("f16 gemm_batched_wmma out too small: {}", out.len()));
        }
        if mhc_gemm_narrow_for(n_rows, k, weight, x) {
            // `V41_MHC_GEMM_NARROW`: narrow-M re-tiling, bit-identical; 2026-09-26
            // sweep: 476 -> 80 us at the B=512 mHC pre-mix. NT = 16 columns per WG.
            let function = module.get_function("f16_gemm_narrow_n16_bk128_pf2")?;
            let cfg = LaunchConfig {
                grid: (1, batch.div_ceil(16), 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            return launch_kernel!(function, cfg, stream, [
                out.raw(), weight.raw(), x.raw(), k, n_rows, batch
            ]);
        }
        // Must match BM/BN in kernels/f16_gemm_wmma.hip.
        const BM: u32 = 64;
        const BN: u32 = 64;
        const BLOCK: u32 = 128;
        let function = module.get_function("f16_gemm_wmma_lds_tiled")?;
        let cfg = LaunchConfig {
            grid: ((n_rows + BM - 1) / BM, (batch + BN - 1) / BN, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), x.raw(), k, n_rows, batch
        ])
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

    /// Batch-tiled twin of `matvec_pair_batched`: each warp handles TILE_B
    /// batch rows for one output row, so a weight pack is loaded once and
    /// reused TILE_B times instead of once per batch row.
    ///
    /// **Bit-exact** with `matvec_pair_batched` — same per-(row, b) 4-way
    /// accumulation over the same lane stride, same reduction order. Only the
    /// work-to-workgroup mapping changes. Must match TILE_B in the .hip.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_pair_batched_tiled(
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
        if k % 4 != 0 {
            return Err(eyre!("f16 matvec_pair_batched_tiled: k={k} must be %4"));
        }
        const TILE_B: u32 = 8; // must match the .hip
        let function = self.pair.get_function("f16_matvec_pair_batched_tiled")?;
        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, b.div_ceil(TILE_B)),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            kv.raw(), gate.raw(), kv_w.raw(), gate_w.raw(), x.raw(), k, n_rows, b
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

    /// Bandwidth-driven B=1 projection matvec for the Laguna decode path.
    /// Same result as `matvec` (within f16-dequant tolerance — reduction
    /// order differs) but streams each weight row with 128-bit vector loads
    /// (8× __half per load) + 4-way MLP unroll, raising achieved DRAM BW on
    /// the read-once weight stream. Falls back to the scalar `matvec` when
    /// `n_rows < NARROW_ROWS_THRESHOLD` (the narrow shapes, e.g. attn_gate)
    /// or `k % 8 != 0` (vector load precondition). Wired ONLY into the 5
    /// Laguna decode projections (wq/wk/wv/wo/wg); env-gated by the caller.
    pub fn matvec_wide_vec(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        if n_rows < NARROW_ROWS_THRESHOLD || k % 8 != 0 {
            return self.matvec(stream, out, weight, x, n_rows, k);
        }
        let expected_weight_bytes = (n_rows as usize) * (k as usize) * 2;
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "f16 matvec_wide_vec weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if out.len() < n_rows as usize {
            return Err(eyre!("f16 matvec_wide_vec out len {} < n_rows={n_rows}", out.len()));
        }
        if x.len() < k as usize {
            return Err(eyre!("f16 matvec_wide_vec x len {} < k={k}", x.len()));
        }
        let function = self.wide.get_function("f16_matvec_wide_vec")?;
        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), x.raw(), k, n_rows])
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
        self.matvec_batched_sym(stream, out, weight, x, n_rows, k, batch, "f16_matvec_batched")
    }

    /// [`Self::matvec_batched`] for the ROUTER logits (decode / replay, and the
    /// look-ahead router): `f16_matvec_batched_h20` under `V41_ROUTER_MV_H20`
    /// when `k % 640 == 0`, same args and grid, bit-identical. The other
    /// `matvec_batched` callers (compressor, indexer) keep the old kernel: the
    /// sweep measured only the router shape.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_batched_router(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        // `V41_F16_MV_Z16` above Z16_ROUTER_MIN_B rows (the replay router, ~64
        // rows per lane): 0.80 of _h20 at b = 64, equal at 48, loses below.
        if f16_mv_z16_on() && batch > Z16_ROUTER_MIN_B {
            return self.matvec_batched_z16(stream, out, weight, x, n_rows, k, batch);
        }
        let sym = if router_mv_h20_for(k) { "f16_matvec_batched_h20" } else { "f16_matvec_batched" };
        self.matvec_batched_sym(stream, out, weight, x, n_rows, k, batch, sym)
    }

    /// [`Self::matvec_batched`] with the grid.z = ceil(batch / NB) kernels under
    /// `V41_F16_MV_Z16` (see [`f16_mv_z16_on`]); identical outputs, else the
    /// production grid.z = batch launch.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_batched_z16(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if !f16_mv_z16_on() || batch == 0 {
            return self.matvec_batched(stream, out, weight, x, n_rows, k, batch);
        }
        let expected_weight_bytes = (n_rows as usize) * (k as usize) * 2;
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "f16 matvec_batched_z16 weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if x.len() < (batch as usize) * (k as usize) {
            return Err(eyre!("f16 matvec_batched_z16 x too small: {}", x.len()));
        }
        if out.len() < (batch as usize) * (n_rows as usize) {
            return Err(eyre!("f16 matvec_batched_z16 out too small: {}", out.len()));
        }
        let nb: u32 = match batch {
            1 => 1,
            2 => 2,
            3..=4 => 4,
            5..=8 => 8,
            _ => 16,
        };
        let function = self.wide.get_function(&format!("f16_matvec_batched_z16_n{nb}"))?;
        // `V41_GRID_PAD`: one idle WG column (the kernel's `row >= n_rows` guard)
        // on the wide grids only. DO NOT REMOVE: the exact 512-WG idx-q grid
        // (4096 waves per z slice) sits in gfx1201's slow-dispatch window --
        // unpadded z16 is 1.00 / 0.68 / 0.63 / 0.37 of the production kernel at
        // b = 1 / 2 / 4 / 8, padded 0.73 / 0.61 / 0.49 / 0.31. The 64-WG
        // compressor grid measured neutral, the 4- / 48-WG proj / router grids
        // were measured unpadded.
        let gx = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let pad = if gx >= 256 { crate::grid_pad() } else { 0 };
        let cfg = LaunchConfig {
            grid: (gx + pad, 1, batch.div_ceil(nb)),
            block: (GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), x.raw(), k, n_rows, batch])
    }

    /// The loaded wide matvec module (tests: explicit-symbol launches).
    pub fn wide_module(&self) -> &Module { &self.wide }

    /// The loaded WMMA GEMM module, if any (tests: explicit-symbol launches).
    pub fn gemm_module(&self) -> Option<&Module> { self.gemm_wmma.as_ref() }

    #[allow(clippy::too_many_arguments)]
    fn matvec_batched_sym(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        batch: u32,
        sym: &str,
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
        let function = self.wide.get_function(sym)?;
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
