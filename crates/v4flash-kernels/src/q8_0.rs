//! Q8_0 GEMV — mirrors ds4's `matvec_q8_0_decode_scratch` /
//! `dot_q8_0_row` (`ds4.c:2920`).
//!
//! Two-kernel pipeline:
//!   1. `q8_0_quantize_f32`: pre-quantize the f32 input vector to int8 + per-
//!      32-block f32 scale. ds4 does the equivalent in CPU before each matvec.
//!   2. `q8_0_gemv_warp8`: one warp per output row; each lane handles a
//!      block via AMD `__builtin_amdgcn_sudot4` 4-byte int8 SIMD dot product;
//!      warp-reduces the partial sums.
//!
//! Used in V4 Flash for: the output projection (`output.weight` in the GGUF —
//! Q8_0 `[n_embd=4096, n_vocab=129280]`), and (in future milestones) every
//! Q8_0-format attention projection and shared-expert MLP.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const Q8_0_MATVEC_GFX1201: &[u8] = include_bytes!(env!("KERNEL_Q8_0_MATVEC_GFX1201"));
const Q8_0_MATVEC_GFX1151: &[u8] = include_bytes!(env!("KERNEL_Q8_0_MATVEC_GFX1151"));

const Q8_0_GROUPED_MATVEC_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_Q8_0_GROUPED_MATVEC_GFX1201"));
const Q8_0_GROUPED_MATVEC_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_Q8_0_GROUPED_MATVEC_GFX1151"));

const Q8_0_MATVEC_WMMA_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_Q8_0_MATVEC_WMMA_GFX1201"));
const Q8_0_MATVEC_WMMA_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_Q8_0_MATVEC_WMMA_GFX1151"));

const SHARED_EXPERT_FUSED_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_SHARED_EXPERT_FUSED_GFX1201"));
const SHARED_EXPERT_FUSED_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_SHARED_EXPERT_FUSED_GFX1151"));

/// Q8_0 packs 32 int8 quants per 2-byte f16 scale → 34 bytes per block,
/// identical layout to ds4 / llama.cpp.
pub const Q8_0_BLOCK_ELEMS: u32 = 32;
pub const Q8_0_BLOCK_BYTES: u32 = 34;

/// One workgroup processes 8 output rows in the gemv kernel.
const GEMV_ROWS_PER_BLOCK: u32 = 8;
const GEMV_WARP_LANES: u32 = 32;
/// Max batch for `matvec_bpack` — mirrors GEMV_BPACK_MAX in q8_0_matvec.hip.
const GEMV_BPACK_MAX: u32 = 16;

/// B-packing is bit-identical to the `grid.z = batch` form, so it is used
/// automatically wherever the batch fits. `V41_GEMV_BPACK=0` rolls back.
fn bpack_ok(batch: u32) -> bool {
    if batch == 0 || batch > GEMV_BPACK_MAX {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("V41_GEMV_BPACK").as_deref() != Ok("0"))
}

/// `V41_GEMV_TB` (default ON; `0` = the runtime-batch kernels
/// `q8_0_gemv_bpack_warp8` / `q8_0_grouped_gemv_bpack` at every batch): the
/// B-packed GEMVs run the COMPILE-TIME-batch twins `q8_0_gemv_bpack_tB<b>`
/// (b = 1..10) and `q8_0_grouped_gemv_bpack_tB<b>` (b = 2..8), BIT-IDENTICAL
/// to the runtime kernel (tests/q8_0_tb_bitexact.rs; same expression, same
/// block striding, same warp-sum tree). The runtime kernel's b-loop indexes
/// acc[16] through v_movrel and waits for its loads inside the loop; the
/// unrolled twin issues all activation loads with the weight loads.
/// 2026-09-26 sweep (C1_dense_decode/tB_compile_time_batch, dGPU, cold, graph):
/// at b = 4 ratios vs runtime q_a 0.80, kv 0.74, wo_a 0.74, wo_b 0.84, shared
/// gate/up 0.81, down 0.88, Engram 0.86, head 0.88 (x1.25, reviewer x1.248);
/// b = 1 neutral (0.97-1.00), so it pays only in multi-row decode. ~69 us per
/// lane-layer at 4 rows -> ~3 ms of dGPU time per step per lane.
fn gemv_tb_on() -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_GEMV_TB").as_deref() != Ok("0"));
    *D
}

/// `q8_0_gemv_bpack_tB<b>` symbols, index b-1. Every production batch of the
/// dp4a arm (1..=8, SMALL_B_DENSE_MAX) plus 9/10 has an instantiation; above
/// that the runtime kernel runs (GEMV_BPACK_MAX = 16).
const GEMV_TB_SYMBOLS: [&str; 10] = [
    "q8_0_gemv_bpack_tB1", "q8_0_gemv_bpack_tB2", "q8_0_gemv_bpack_tB3",
    "q8_0_gemv_bpack_tB4", "q8_0_gemv_bpack_tB5", "q8_0_gemv_bpack_tB6",
    "q8_0_gemv_bpack_tB7", "q8_0_gemv_bpack_tB8", "q8_0_gemv_bpack_tB9",
    "q8_0_gemv_bpack_tB10",
];
/// `q8_0_grouped_gemv_bpack_tB<b>` symbols, index b-1 (1..=8 instantiated).
const GROUPED_TB_SYMBOLS: [&str; 8] = [
    "q8_0_grouped_gemv_bpack_tB1", "q8_0_grouped_gemv_bpack_tB2",
    "q8_0_grouped_gemv_bpack_tB3", "q8_0_grouped_gemv_bpack_tB4",
    "q8_0_grouped_gemv_bpack_tB5", "q8_0_grouped_gemv_bpack_tB6",
    "q8_0_grouped_gemv_bpack_tB7", "q8_0_grouped_gemv_bpack_tB8",
];

/// Symbol for `matvec_bpack` at `batch`: the tB twin for 1 <= b <= 10 under
/// `V41_GEMV_TB`, else the runtime kernel.
pub fn gemv_bpack_symbol(batch: u32) -> &'static str {
    if gemv_tb_on() && (1..=GEMV_TB_SYMBOLS.len() as u32).contains(&batch) {
        GEMV_TB_SYMBOLS[(batch - 1) as usize]
    } else {
        "q8_0_gemv_bpack_warp8"
    }
}

/// Symbol for `matvec_grouped_bpack` at `batch`: the tB twin for 2 <= b <= 8
/// under `V41_GEMV_TB`, else the runtime kernel. b = 1 keeps the runtime
/// kernel: the sweep review measured grouped tB1 2.5% SLOWER (76.8 -> 78.7 us,
/// 6/6 runs) on wo_a.
pub fn grouped_bpack_symbol(batch: u32) -> &'static str {
    if gemv_tb_on() && (2..=GROUPED_TB_SYMBOLS.len() as u32).contains(&batch) {
        GROUPED_TB_SYMBOLS[(batch - 1) as usize]
    } else {
        "q8_0_grouped_gemv_bpack"
    }
}

/// `V41_Q8_QUANT_WAVE` (default ON; `0` = `q8_0_quantize_f32`, one 32-thread
/// WG per block where every thread walks all 32 elements): the activation
/// quantize runs `q8_0_quantize_f32_wave`, one wave per block (lane i owns
/// element i, amax by shfl_xor fmaxf), 8 blocks per 256-thread WG. xq and
/// xscale BIT-IDENTICAL (tests/q8_0_tb_bitexact.rs; sweep review 7 shapes x 3
/// runs + 18 adversarial cases; the only deviation is an ALL-NaN block, whose
/// xscale is NaN instead of 0). 2026-09-26 sweep (C1_dense_decode/
/// quantize_wave, warm graph node): K=5120 b=4 4.46 -> 3.56 us (x1.25), K=32768
/// b=4 8.2 -> 4.5 us with the grid pad; ~5-7 us per lane-layer.
fn q8_quant_wave() -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_Q8_QUANT_WAVE").as_deref() != Ok("0"));
    *D
}

/// `V41_Q8_QUANT_GRID_PAD` (default ON; `0` = the exact grid): the quantize
/// grid gets ONE extra work-group, which the kernel's `if (b >= blocks) return`
/// guard makes a no-op (the `blocks` argument is unchanged, so the outputs are
/// byte-identical -- reviewer-verified at 22 shapes incl. 2047/2049/4095/4097).
///
/// DO NOT REMOVE THE PAD. It is real hardware dispatch behaviour, not a
/// harness artefact: q8_0_quantize_f32 at K=32768 (the wo_a input, FP:6126)
/// launches exactly 1024*b WGs, and rocprofv3 kernel-trace timestamps put the
/// DISPATCH ITSELF at 14.24 us on a 2048-WG grid vs 3.40 us at 2049, and 17.08
/// vs 5.68 us at 4096/4097 (2026-09-26 sweep, C1_dense_decode/quantize_grid_pad,
/// reviewer runs x10; not address aliasing, not graph capture, persists with a
/// producer kernel in between). 2047/4095 are equally slow and 3072 is bimodal;
/// +1 fixes every production grid 1024*b (b = 1..8) and 256*b, and the wave
/// kernel's ceil(blocks/8) grids (512+1: 4.49 us, 256+1: 3.84 us). Worth
/// ~10-11 us per lane-layer at 2 or 4 rows per lane (~0.4 ms/step per lane).
fn q8_quant_grid_pad() -> u32 {
    static D: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_Q8_QUANT_GRID_PAD").as_deref() != Ok("0")
    });
    if *D { 1 } else { 0 }
}

/// Largest batch the grid.z-chunked bpack GEMVs serve (`V41_GEMV_BPACK_Z16`).
pub const GEMV_BPACK_Z16_MAX: u32 = 64;

/// `V41_GEMV_BPACK_Z16` (default ON; `0` = the grid.z = batch kernels
/// `q8_0_gemv_batched_warp8` / `q8_0_grouped_gemv_batched`): a batched dp4a
/// GEMV of 16 < b <= 64 rows -- the CED replay regime (~64 rows per lane),
/// which exceeds the 16-row bpack cap -- runs `q8_0_gemv_bpack_z16` /
/// `q8_0_grouped_gemv_bpack_z16`: the bpack16 body with the batch split into
/// 16-row slices on grid.z, so the weight is read ceil(b/16) times instead of
/// b times. BIT-IDENTICAL per (row, b) (tests/q8_0_sweep_c2_bitexact.rs; the
/// sweep review 24 + 60 compares incl. 7/49/63/128). 2026-09-26 sweep
/// (C2_dense_prefill/replay_bpack_z16, cold, b = 64): q_b 1508 -> 576 us, wo_b
/// 1498 -> 537, wo_a 1210 -> 535, kv 52 -> 47; -2.57 ms per replay lane-layer.
/// Also off under `V41_GEMV_BPACK=0` (the b-packing rollback). The head's
/// `matvec_bpack` keeps its 16-row cap (its scratch is sized on it).
///
/// Shape gate (2026-09-27 round-2 crossover scan, b_crossover, cold, 4-5 runs):
/// the z16 grid has ceil(b/16) x fewer WGs than grid.z = b, which starves the
/// small-M projections below ~48 rows: kv (M = 512) LOSES x1.72 at b = 17 and
/// x1.11 at 32 (wins 0.83-0.88 from 48), q_a (M = 1280) LOSES x1.14 at 17 (0.80
/// at 48). M >= 2048 (q_b, wo_b, shared gate/up/down, grouped wo_a) wins at
/// every 17..64 (0.31-0.54). So: z16 when `n_rows >= Z16_MIN_ROWS_ANY_B` or
/// `batch >= Z16_MIN_BATCH_SMALL_M`.
pub fn bpack_z16_for(batch: u32, n_rows: u32) -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_GEMV_BPACK_Z16").as_deref() != Ok("0"));
    // bpack_ok(1) == "V41_GEMV_BPACK is not 0".
    *D && batch > GEMV_BPACK_MAX && batch <= GEMV_BPACK_Z16_MAX && bpack_ok(1)
        && (n_rows >= Z16_MIN_ROWS_ANY_B || batch >= Z16_MIN_BATCH_SMALL_M)
}

/// See [`bpack_z16_for`]: output rows from which z16 wins at every 16 < b <= 64.
pub const Z16_MIN_ROWS_ANY_B: u32 = 2048;
/// See [`bpack_z16_for`]: batch from which z16 also wins on kv / q_a.
pub const Z16_MIN_BATCH_SMALL_M: u32 = 48;

/// `V41_ENGRAM_I8X` (default ON; `0` = `q8_0_gemm_wmma_lds_tiled`): the prefill
/// Engram wkv GEMM (M = 25600, K = 6144, int8 + xscale activations) runs
/// `q8_0_gemm_wmma_i8x_db` ([`Q8_0MatvecWmma::gemm_i8x_db`], same inputs,
/// grid (ceil(b/128), M/128) x 256): the f16x 128x128 tile with the int8
/// activations dequantised at stage exactly as lds_tiled does, double-buffered
/// LDS, 2-deep register prefetch. BIT-IDENTICAL (tests/q8_0_sweep_c2_bitexact.rs;
/// the sweep review 17 + 48 compares incl. denormal scales). 2026-09-26 sweep
/// (C2_dense_prefill/engram_i8x_db, cold weight): 905 -> 484 us per 64-row
/// chunk, 511 us per 128-row chunk (`V41_ENGRAM_CHUNK128`). Needs M % 128 == 0
/// and K % 256 == 0 (b128 loads); other shapes keep lds_tiled.
pub fn engram_i8x_for(n_rows: u32, k: u32) -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_ENGRAM_I8X").as_deref() != Ok("0"));
    *D && n_rows % 128 == 0 && k % 256 == 0
}

/// Tile variant of the f16x WMMA GEMM ([`Q8_0MatvecWmma::gemm_f16x_tile`]).
/// Every variant is BIT-IDENTICAL to `Base` (same dequant, same per-output k
/// order, same WMMA; tests/q8_0_sweep_c2_bitexact.rs); they differ in speed per
/// shape, so the call sites pick one per projection (the selectors below).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F16xTile {
    /// `q8_0_gemm_wmma_f16x`, 128x128, grid (ceil(b/128), M/128, G).
    Base,
    /// `q8_0_gemm_wmma_f16x_db_bn64`, 128x64 double-buffered, grid (ceil(b/64), M/128, G).
    DbBn64,
    /// `q8_0_gemm_wmma_f16x_256x128`, grid (ceil(b/128), M/256, G); needs M % 256 == 0.
    T256x128,
}

/// `V41_F16X_DB_BN64` (default ON; `0` = `q8_0_gemm_wmma_f16x` at kv / q_a):
/// the occupancy-starved small-M prefill projections run the 128x64 double-
/// buffered tile (2x the WGs, 2 WGs/WGP). 2026-09-26 sweep (C2_dense_prefill/
/// f16x_db_bn64, cold weights, b = 512): kv 106 -> 59 us (x1.79), q_a 125 -> 97
/// (x1.28); at 256 rows 0.54 / 0.65. It LOSES on the big shapes (q_b / wo_a /
/// wo_b / shared, -9..25%) and on q_a at 1024 rows (+4%), hence per-site gates:
/// kv at b > 64 ([`f16x_tile_kv`]), q_a only for M <= 1280 and b <= 512
/// ([`f16x_tile_q_a`]).
fn f16x_db_bn64_on() -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_F16X_DB_BN64").as_deref() != Ok("0"));
    *D
}

/// `V41_F16X_256` (default ON; `0` = `q8_0_gemm_wmma_f16x` at q_b / wo_a): the
/// q_b (32768 x 1280) and wo_a (8 x 1024 x 4096) prefill GEMMs run the 256x128
/// tile (halves the activation-tile re-reads). 2026-09-26 sweep (C2_dense_prefill/
/// f16x_256x128, cold, b = 512): q_b 458 -> 410 us (x1.11), wo_a 367 -> 325
/// (x1.13); b = 1024 q_b x1.26, wo_a x1.09. Neutral on wo_b / shared down, LOSES
/// on kv / q_a / shared gate-up (fewer WGs), so only those two sites use it, and
/// only above 64 rows ([`f16x_tile_qb_wo_a`]): replay-sized passes were measured
/// on the base tile only.
fn f16x_256_on() -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_F16X_256").as_deref() != Ok("0"));
    *D
}

/// Tile for the prefill kv projection (M = 512): `DbBn64` above 64 rows.
pub fn f16x_tile_kv(batch: u32) -> F16xTile {
    if f16x_db_bn64_on() && batch > 64 { F16xTile::DbBn64 } else { F16xTile::Base }
}

/// Tile for a small-M dense prefill projection through `dense_gemm_prefill`
/// (q_a, M = 1280; the shared expert's M = 2304 / 5120 stay `Base`): `DbBn64`
/// only while M <= 1280 and b <= 512 (q_a loses 4% at 1024 rows).
pub fn f16x_tile_q_a(m: u32, batch: u32) -> F16xTile {
    if f16x_db_bn64_on() && m <= 1280 && batch <= 512 { F16xTile::DbBn64 } else { F16xTile::Base }
}

/// Tile for the q_b / wo_a prefill projections: `T256x128` from
/// `F16X_256_MIN_ROWS` rows when M % 256 == 0.
///
/// 2026-09-27 round-2 crossover scan (b_crossover, cold, 4 runs): below ~192
/// rows the 256x128 tile LOSES on q_b (x1.27 at b = 65, x1.18 at 128; 0.96 at
/// 192) and is neutral on wo_a (0.98 / 1.01 at 65 / 128; 0.89 at 192, 0.89 at
/// 384) -- the round-1 gate was `b > 64`, which put every 65..191-row prefill
/// tail chunk on the slower tile.
pub fn f16x_tile_qb_wo_a(m: u32, batch: u32) -> F16xTile {
    if f16x_256_on() && m % 256 == 0 && batch >= F16X_256_MIN_ROWS { F16xTile::T256x128 } else { F16xTile::Base }
}

/// See [`f16x_tile_qb_wo_a`].
pub const F16X_256_MIN_ROWS: u32 = 192;


#[allow(non_camel_case_types)]
pub struct Q8_0Matvec {
    module: Module,
}

impl Q8_0Matvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q8_0_MATVEC_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q8_0_MATVEC_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q8_0 matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Pre-quantize a length-n f32 vector into int8 + per-block f32 scale.
    /// `xq.len()` must be `n`; `xscale.len()` must be `n / 32` (n must be a
    /// multiple of 32). One workgroup per 32-element block, 32 threads each.
    pub fn quantize_input(
        &self,
        stream: &Stream,
        xq: &mut DeviceBuffer<i8>,
        xscale: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        n: u32,
    ) -> eyre::Result<()> {
        if n % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 quantize: n={n} not a multiple of 32"));
        }
        let blocks = n / Q8_0_BLOCK_ELEMS;
        if x.len() != n as usize
            || xq.len() != n as usize
            || xscale.len() != blocks as usize
        {
            return Err(eyre!(
                "q8_0 quantize len mismatch: n={n}, x={}, xq={}, xscale={}",
                x.len(),
                xq.len(),
                xscale.len()
            ));
        }
        self.launch_quantize(stream, xq, xscale, x, blocks)
    }

    /// The one quantize launch behind `quantize_input` / `quantize_input_batched`
    /// (every call site in the tree goes through those two): the wave kernel
    /// under `V41_Q8_QUANT_WAVE`, and the `+1` work-group grid pad under
    /// `V41_Q8_QUANT_GRID_PAD` -- see both knobs for the measurements. `blocks`
    /// is always the true block count; only the grid carries the pad.
    fn launch_quantize(
        &self,
        stream: &Stream,
        xq: &mut DeviceBuffer<i8>,
        xscale: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        blocks: u32,
    ) -> eyre::Result<()> {
        // One idle WG: the exact power-of-two grid dispatches 3-4x slower
        // (14.2 us vs 3.4 us at 2048 WGs). See `q8_quant_grid_pad`.
        let pad = q8_quant_grid_pad();
        let (function, cfg) = if q8_quant_wave() {
            (
                self.module.get_function("q8_0_quantize_f32_wave")?,
                LaunchConfig {
                    grid: (blocks.div_ceil(8) + pad, 1, 1),
                    block: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
            )
        } else {
            (
                self.module.get_function("q8_0_quantize_f32")?,
                LaunchConfig {
                    grid: (blocks + pad, 1, 1),
                    block: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
            )
        };
        launch_kernel!(function, cfg, stream, [xq.raw(), xscale.raw(), x.raw(), blocks])
    }

    /// M50 Phase 2: batched quantize. Equivalent to `quantize_input` over
    /// `B × n` contiguous elements. The kernel has no batch concept — it
    /// just processes `B × blocks` blocks. Buffers must be at least
    /// `B × n` (xq), `B × n/32` (xscale), `B × n` (x).
    pub fn quantize_input_batched(
        &self,
        stream: &Stream,
        xq: &mut DeviceBuffer<i8>,
        xscale: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        n: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if n % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 quantize_batched: n={n} not %32"));
        }
        let total_blocks = (n / Q8_0_BLOCK_ELEMS) * batch;
        let total_n = (n as usize) * (batch as usize);
        if x.len() < total_n || xq.len() < total_n || xscale.len() < total_blocks as usize {
            return Err(eyre!(
                "q8_0 quantize_batched: buffer too small (n*B={total_n}, blocks*B={total_blocks}, x={}, xq={}, xscale={})",
                x.len(), xq.len(), xscale.len()
            ));
        }
        self.launch_quantize(stream, xq, xscale, x, total_blocks)
    }

    /// `out[i] = sum_b f16_scale_w[i, b] * xscale[b] * dot_i8x32(qs_w[i, b], xq[b])`
    /// for i in 0..n_rows. The Q8_0 weight buffer holds `n_rows` rows of
    /// `(k/32) * 34` bytes each, row-major.
    /// The loaded module (benches: function-lookup cost).
    pub fn module(&self) -> &Module { &self.module }

    pub fn matvec(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 matvec: k={k} not a multiple of 32"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 matvec weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if out.len() != n_rows as usize {
            return Err(eyre!(
                "q8_0 matvec out len: have {}, expected n_rows={n_rows}",
                out.len()
            ));
        }
        if xq.len() != k as usize || xscale.len() != blocks as usize {
            return Err(eyre!(
                "q8_0 matvec xq/xscale len: xq={}, xscale={}, expected k={k}, blocks={blocks}",
                xq.len(),
                xscale.len()
            ));
        }

        let function = self.module.get_function("q8_0_gemv_warp8")?;

        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES; // 8 × 32 = 256
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), xq.raw(), xscale.raw(), k, n_rows, blocks])
    }

    /// M50 Phase 2: batched GEMV with `grid.z = B`. Same per-row math
    /// as `matvec`; B parallel WGs run concurrently, one per batch
    /// element. `xq[B, K]`, `xscale[B, K/32]`, `out[B, n_rows]` —
    /// row-major. Weight `[n_rows, K]` Q8_0 is shared across batch.
    ///
    /// v0 of batching: no W amortization across batch (each WG re-reads
    /// W independently). A v1 kernel will pack multiple batch elements
    /// per WG to amortize W reads.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 matvec_batched: k={k} not a multiple of 32"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 matvec_batched weight bytes: have {}, expected {}",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        let expected_out = (batch as usize) * (n_rows as usize);
        if out.len() < expected_out {
            return Err(eyre!(
                "q8_0 matvec_batched out len: have {}, expected {}",
                out.len(),
                expected_out
            ));
        }
        let expected_xq = (batch as usize) * (k as usize);
        let expected_xscale = (batch as usize) * (blocks as usize);
        if xq.len() < expected_xq || xscale.len() < expected_xscale {
            return Err(eyre!(
                "q8_0 matvec_batched xq/xscale len: xq={} (need {expected_xq}), xscale={} (need {expected_xscale})",
                xq.len(),
                xscale.len()
            ));
        }

        // `q8_0_gemv_batched_warp8` launches grid.z = batch, so EVERY batch row
        // re-reads the whole weight matrix. At the batch sizes decode and a
        // DSpark verify actually use (B<=6) that is the dominant cost and it is
        // pure waste: `matvec_bpack` reads each block once and loops the batch
        // in registers, bit-identical per (row, b). Large-B prefill keeps the
        // original kernel -- there the weight read is already amortised and the
        // 16-wide accumulator would cost registers for nothing.
        if bpack_ok(batch) {
            return self.matvec_bpack(stream, out, weight, xq, xscale, n_rows, k, batch);
        }
        if bpack_z16_for(batch, n_rows) {
            // `V41_GEMV_BPACK_Z16`: the bpack16 body over 16-row slices on grid.z
            // (replay b <= 64), bit-identical; 2026-09-26 sweep: q_b 1508 -> 576 us at b=64.
            let function = self.module.get_function("q8_0_gemv_bpack_z16")?;
            let cfg = LaunchConfig {
                grid: (n_rows.div_ceil(GEMV_ROWS_PER_BLOCK), 1, batch.div_ceil(GEMV_BPACK_MAX)),
                block: (GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES, 1, 1),
                shared_mem_bytes: 0,
            };
            return launch_kernel!(function, cfg, stream, [
                out.raw(), weight.raw(), xq.raw(), xscale.raw(), k, n_rows, blocks, batch
            ]);
        }
        let function = self.module.get_function("q8_0_gemv_batched_warp8")?;
        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES; // 8 × 32 = 256
        let cfg = LaunchConfig {
            grid: (grid_x, 1, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), xq.raw(), xscale.raw(), k, n_rows, blocks])
    }

    /// B-PACKED batched GEMV: reads the weight matrix ONCE for all `batch`
    /// activations instead of once per batch element.
    ///
    /// [`Self::matvec_batched`] puts the batch on `grid.z`, so every batch
    /// element is an independent workgroup re-reading all of W -- its own doc
    /// says so ("v0 of batching ... A v1 kernel will pack multiple batch elements
    /// per WG"). On a bandwidth-bound shape that is the whole cost: the tied
    /// vocab projection is ~700 MB and measures 1112 us/call at mean == max,
    /// i.e. already at ~630 GB/s, so B calls cost B x 1.1 ms.
    ///
    /// Numerically IDENTICAL to `matvec_batched`, by construction: the per-(row,
    /// b) accumulation keeps the same lane striding, the same
    /// `scale * xscale * dot` expression and the same warp-reduction tree. Only
    /// the weight LOAD is hoisted out of the batch loop.
    ///
    /// `batch` must be <= `GEMV_BPACK_MAX` (16) -- the accumulators are registers.
    #[allow(clippy::too_many_arguments)]
    /// Stall this stream for `ticks` of the device's realtime counter.
    ///
    /// The unit is deliberately RAW TICKS, not microseconds: the counter's rate
    /// varies by part, and the probe's whole job is to be trusted. Callers time
    /// the achieved stall with the event timers and regress the STEP against
    /// that measured delay, so no rate constant is ever assumed.
    pub fn slack_probe_spin(&self, stream: &Stream, ticks: u64) -> eyre::Result<()> {
        let function = self.module.get_function("slack_probe_spin")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (64, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [ticks])
    }

    pub fn matvec_bpack(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if batch > GEMV_BPACK_MAX {
            return Err(eyre!(
                "q8_0 matvec_bpack: batch={batch} exceeds GEMV_BPACK_MAX={GEMV_BPACK_MAX}"
            ));
        }
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 matvec_bpack: k={k} not a multiple of 32"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 matvec_bpack weight bytes: have {}, expected {}",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if out.len() < (batch as usize) * (n_rows as usize)
            || xq.len() < (batch as usize) * (k as usize)
            || xscale.len() < (batch as usize) * (blocks as usize)
        {
            return Err(eyre!("q8_0 matvec_bpack: out/xq/xscale too small for batch={batch}"));
        }
        // Compile-time-batch twin for b <= 10 (`V41_GEMV_TB`), same args and grid;
        // 2026-09-26 sweep: x1.25 at b = 4, bit-exact. The module caches the lookup.
        let function = self.module.get_function(gemv_bpack_symbol(batch))?;
        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(), k, n_rows, blocks, batch
        ])
    }

    /// `matvec_bpack` through its `_ind` twin (docs/v41/GRAPH_KEYS_DESIGN.md 2.3):
    /// operands 0..3 (out, weight, xq, xscale) marked in `ind` are read from the arena
    /// context slot at run time; the buffers passed here are still the real ones (their
    /// sizes are checked, and they are what a direct launch would use). Twins exist for
    /// the compile-time batch kernels b = 1..8 (same body, bit-identical); any other arm
    /// launches the direct kernel on those real buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_bpack_ind(
        &self,
        stream: &Stream,
        ind: crate::het::arena_ctx::Ind,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        // No twin on this arm (b > 8, or the runtime kernel under V41_GEMV_TB=0): the DIRECT
        // kernel on the real buffers (design 2.5's fallback rule; unvetted, so a capture taints).
        if !(1..=8).contains(&batch) || !gemv_tb_on() {
            return self.matvec_bpack(stream, out, weight, xq, xscale, n_rows, k, batch);
        }
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 matvec_bpack_ind: k={k} not a multiple of 32"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        if weight.byte_len() != (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize)
            || out.len() < (batch as usize) * (n_rows as usize)
            || xq.len() < (batch as usize) * (k as usize)
            || xscale.len() < (batch as usize) * (blocks as usize)
        {
            return Err(eyre!("q8_0 matvec_bpack_ind: operand sizes do not fit n_rows={n_rows} k={k} batch={batch}"));
        }
        // Production twins carry no canary code; the canary (design 2.8) has its own symbols.
        const IND_SYMBOLS: [&str; 8] = [
            "q8_0_gemv_bpack_tB1_ind", "q8_0_gemv_bpack_tB2_ind", "q8_0_gemv_bpack_tB3_ind", "q8_0_gemv_bpack_tB4_ind",
            "q8_0_gemv_bpack_tB5_ind", "q8_0_gemv_bpack_tB6_ind", "q8_0_gemv_bpack_tB7_ind", "q8_0_gemv_bpack_tB8_ind",
        ];
        const IND_CANARY_SYMBOLS: [&str; 8] = [
            "q8_0_gemv_bpack_tB1_ind_canary", "q8_0_gemv_bpack_tB2_ind_canary", "q8_0_gemv_bpack_tB3_ind_canary",
            "q8_0_gemv_bpack_tB4_ind_canary", "q8_0_gemv_bpack_tB5_ind_canary", "q8_0_gemv_bpack_tB6_ind_canary",
            "q8_0_gemv_bpack_tB7_ind_canary", "q8_0_gemv_bpack_tB8_ind_canary",
        ];
        let symbols = if ind.canary != 0 { &IND_CANARY_SYMBOLS } else { &IND_SYMBOLS };
        let function = self.module.get_function(symbols[(batch - 1) as usize])?;
        let cfg = LaunchConfig {
            grid: (n_rows.div_ceil(GEMV_ROWS_PER_BLOCK), 1, 1),
            block: (GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES, 1, 1),
            shared_mem_bytes: 0,
        };
        let (p_out, p_w, p_xq, p_xs) = (
            ind.ptr(0, out.raw() as u64),
            ind.ptr(1, weight.raw() as u64),
            ind.ptr(2, xq.raw() as u64),
            ind.ptr(3, xscale.raw() as u64),
        );
        launch_kernel!(function, cfg, stream, [ind.mask(), ind.canary, ind.tag, p_out, p_w, p_xq, p_xs, k, n_rows, blocks, batch])
    }

    /// M40-P4.5: 2-wide pair GEMV. Same as `matvec` but processes TWO input
    /// columns against ONE weight matrix per call. Reads W once and computes
    /// both outputs in the same kernel pass — halves W bandwidth vs two
    /// independent calls. Used by pair-forward where t0 and t1 share weights
    /// but have different activations.
    ///
    /// `out[i] = sum_b f16(scale_w[i,b]) * xscale[b] * dot_i8x32(qs_w[i,b], xq[b])`
    /// for both `a` and `b` columns simultaneously.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_pair(
        &self,
        stream: &Stream,
        out_a: &mut DeviceBuffer<f32>,
        out_b: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq_a: &DeviceBuffer<i8>,
        xq_b: &DeviceBuffer<i8>,
        xscale_a: &DeviceBuffer<f32>,
        xscale_b: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 matvec_pair: k={k} not a multiple of 32"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 matvec_pair weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if out_a.len() != n_rows as usize || out_b.len() != n_rows as usize {
            return Err(eyre!(
                "q8_0 matvec_pair out lens: a={}, b={}, expected n_rows={n_rows}",
                out_a.len(),
                out_b.len()
            ));
        }
        if xq_a.len() != k as usize
            || xq_b.len() != k as usize
            || xscale_a.len() != blocks as usize
            || xscale_b.len() != blocks as usize
        {
            return Err(eyre!(
                "q8_0 matvec_pair xq/xscale lens: xq_a={}, xq_b={}, xs_a={}, xs_b={}, expected k={k}, blocks={blocks}",
                xq_a.len(), xq_b.len(), xscale_a.len(), xscale_b.len()
            ));
        }

        let function = self.module.get_function("q8_0_gemv_pair_warp8")?;

        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out_a.raw(), out_b.raw(), weight.raw(), xq_a.raw(), xq_b.raw(),
            xscale_a.raw(), xscale_b.raw(), k, n_rows, blocks
        ])
    }
}

/// Q8_0 int8-WMMA GEMM (gfx12 only): `out[B,M] = W[M,K] · Xq[B,K]^T`, with
/// both Q8_0 dequant scales folded into the f16 WMMA operands at load time.
/// One wave per 16-row M-tile; batch N-tiles loop inside each K-tile so the
/// weight A-fragment is read once per K-tile and reused across all batch
/// columns. Same numeric result as `Q8_0Matvec::matvec_batched`, just via the
/// matrix cores instead of the grid.z=B dp4a path. On non-gfx12 the kernel
/// has a scalar fallback (so the gfx1151 build succeeds); it should only ever
/// be launched on the dGPU.
#[allow(non_camel_case_types)]
pub struct Q8_0MatvecWmma {
    module: Module,
}

impl Q8_0MatvecWmma {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q8_0_MATVEC_WMMA_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q8_0_MATVEC_WMMA_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q8_0 wmma matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `out[b,m] = sum_k (qW[m,k]·wscale[m,k/32]) · (qX[b,k]·xscale[b,k/32])`.
    /// `weight` is `[n_rows=M, K]` Q8_0 (row pitch `blocks*34`), `xq` is
    /// `[B, K]` int8, `xscale` is `[B, blocks]` f32, `out` is `[B, M]` f32.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        n_rows: u32, // M
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 wmma gemm: k={k} not a multiple of 32"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 wmma gemm weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        let expected_out = (batch as usize) * (n_rows as usize);
        if out.len() < expected_out {
            return Err(eyre!(
                "q8_0 wmma gemm out len: have {}, expected {}",
                out.len(),
                expected_out
            ));
        }
        let expected_xq = (batch as usize) * (k as usize);
        let expected_xscale = (batch as usize) * (blocks as usize);
        if xq.len() < expected_xq || xscale.len() < expected_xscale {
            return Err(eyre!(
                "q8_0 wmma gemm xq/xscale len: xq={} (need {expected_xq}), xscale={} (need {expected_xscale})",
                xq.len(),
                xscale.len()
            ));
        }

        let function = self.module.get_function("q8_0_gemm_wmma_i8")?;
        let grid_x = n_rows.div_ceil(16);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(), k, n_rows, batch, blocks
        ])
    }

    /// Grouped LDS-tiled GEMM (RDNA4 only). 8-group block-diagonal matmul
    /// where the per-group inputs/outputs are interleaved per-batch
    /// (xq/xscale/out all strided by n_groups within the batch dim).
    /// Used for output_proj.grouped_matvec.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_lds_tiled_grouped(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,         // [B, n_groups*rank]
        weight: &DeviceBuffer<u8>,           // [n_groups*rank, group_dim Q8_0]
        xq: &DeviceBuffer<i8>,               // [B, n_groups*group_dim]
        xscale: &DeviceBuffer<f32>,          // [B, n_groups*blocks]
        group_dim: u32,
        rank: u32,
        n_groups: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_groups == 0 {
            return Ok(());
        }
        if group_dim % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!(
                "gemm_lds_tiled_grouped: group_dim={group_dim} not %32"
            ));
        }
        if rank % 64 != 0 {
            return Err(eyre!(
                "gemm_lds_tiled_grouped: rank={rank} not %64 (BM)"
            ));
        }
        let blocks = group_dim / Q8_0_BLOCK_ELEMS;
        let function = self
            .module
            .get_function("q8_0_gemm_wmma_lds_tiled_grouped")?;
        let cfg = LaunchConfig {
            grid: (rank.div_ceil(64), batch.div_ceil(64), n_groups),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(),
            group_dim, rank, n_groups, batch, blocks
        ])
    }

    /// LDS-tiled Q8_0 GEMM (RDNA4 only). Same semantics as `gemm()` but
    /// cooperatively stages BM×BK A + BK×BN B into LDS once per K-outer
    /// iter, then runs the inner WMMA loop entirely from LDS — amortizes
    /// global weight loads across many compute ops to kill the
    /// `s_wait_loadcnt` latency stall that bottlenecks both `gemm()` and
    /// the dp4a `matvec_batched()` at long-K shapes (matvec_out, etc).
    /// Requires `M % 64 == 0` and `K % 32 == 0`.
    #[allow(clippy::too_many_arguments)]
    /// Q8_0 weights x f16 activations, 128x128x32 RDNA4 WMMA GEMM (2026-09-08).
    /// `x16` = [batch, x_pitch] f16 bits (`x_pitch` >= n_groups*k; pad it off a
    /// power of two, see the kernel), `out` = [batch, n_groups*m] f32,
    /// `weight` = [n_groups*m, blocks*34]. `n_groups` = 1 for a plain GEMM;
    /// output_a passes its 8 groups (group g reads input columns g*k.. and
    /// writes output columns g*m..). Requires m % 128 == 0, k % 32 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16x(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x16: &DeviceBuffer<u16>,
        k: u32,
        m: u32,
        n_groups: u32,
        batch: u32,
        x_pitch: u32,
    ) -> eyre::Result<()> {
        self.gemm_f16x_tile(F16xTile::Base, stream, out, weight, x16, k, m, n_groups, batch, x_pitch)
    }

    /// The loaded module (tests: explicit-symbol launches).
    pub fn module(&self) -> &Module { &self.module }

    /// [`Self::gemm_f16x`] with an explicit tile variant (2026-09-26 sweep,
    /// C2_dense_prefill): same arguments and checks, bit-identical outputs; the
    /// call sites pick the variant per projection with `f16x_tile_*`.
    /// `T256x128` additionally requires m % 256 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16x_tile(
        &self,
        tile: F16xTile,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x16: &DeviceBuffer<u16>,
        k: u32,
        m: u32,
        n_groups: u32,
        batch: u32,
        x_pitch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_groups == 0 {
            return Ok(());
        }
        if tile == F16xTile::T256x128 && m % 256 != 0 {
            return Err(eyre!("gemm_f16x 256x128: m={m} not %256"));
        }
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("gemm_f16x: k={k} not %32"));
        }
        if x_pitch < n_groups * k || x_pitch % 8 != 0 {
            return Err(eyre!("gemm_f16x: x_pitch={x_pitch} must be >= n_groups*k={} and %8", n_groups * k));
        }
        if m % 128 != 0 {
            return Err(eyre!("gemm_f16x: m={m} not %128"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let need_w = (n_groups as usize) * (m as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() < need_w {
            return Err(eyre!("gemm_f16x: weight {} B < needed {need_w}", weight.byte_len()));
        }
        if x16.len() < (batch as usize) * (x_pitch as usize) {
            return Err(eyre!("gemm_f16x: x16 too small"));
        }
        if out.len() < (batch as usize) * (n_groups as usize) * (m as usize) {
            return Err(eyre!("gemm_f16x: out too small"));
        }
        // N-blocks fastest (weight-tile L2 reuse); the variants keep that order.
        let (sym, grid) = match tile {
            F16xTile::Base => ("q8_0_gemm_wmma_f16x", (batch.div_ceil(128), m / 128, n_groups)),
            F16xTile::DbBn64 => ("q8_0_gemm_wmma_f16x_db_bn64", (batch.div_ceil(64), m / 128, n_groups)),
            F16xTile::T256x128 => ("q8_0_gemm_wmma_f16x_256x128", (batch.div_ceil(128), m / 256, n_groups)),
        };
        let function = self.module.get_function(sym)?;
        let cfg = LaunchConfig { grid, block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), x16.raw(), k, m, n_groups, batch, blocks, x_pitch
        ])
    }

    /// `q8_0_gemm_wmma_i8x_db` -- drop-in for [`Self::gemm_lds_tiled`] (same
    /// inputs: `xq[B, K]` int8, `xscale[B, K/32]`, out `[B, M]`), BIT-IDENTICAL
    /// to it; grid (ceil(b/128), M/128, 1) x 256. Requires M % 128 == 0 and
    /// K % 256 == 0 (b128 loads of the xq rows and of the weight rows).
    /// Selected for the Engram wkv under `V41_ENGRAM_I8X` ([`engram_i8x_for`]).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_i8x_db(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        n_rows: u32, // M
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if k % 256 != 0 {
            return Err(eyre!("q8_0 gemm_i8x_db: k={k} not a multiple of 256"));
        }
        if n_rows % 128 != 0 {
            return Err(eyre!("q8_0 gemm_i8x_db: n_rows={n_rows} not a multiple of 128"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 gemm_i8x_db weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(), expected_weight_bytes
            ));
        }
        let b = batch as usize;
        if xq.len() < b * (k as usize) || xscale.len() < b * (blocks as usize) || out.len() < b * (n_rows as usize) {
            return Err(eyre!(
                "q8_0 gemm_i8x_db: buffers too small for batch={batch} (xq {} xs {} out {})",
                xq.len(), xscale.len(), out.len()
            ));
        }
        let function = self.module.get_function("q8_0_gemm_wmma_i8x_db")?;
        let cfg = LaunchConfig {
            grid: (batch.div_ceil(128), n_rows / 128, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,                   // 40 KB static LDS
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(), k, n_rows, 1u32, batch, blocks
        ])
    }

    pub fn gemm_lds_tiled(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        n_rows: u32, // M
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 gemm_lds_tiled: k={k} not a multiple of 32"));
        }
        if n_rows % 64 != 0 {
            return Err(eyre!("q8_0 gemm_lds_tiled: n_rows={n_rows} not a multiple of 64 (BM)"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 gemm_lds_tiled weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(), expected_weight_bytes
            ));
        }
        let function = self.module.get_function("q8_0_gemm_wmma_lds_tiled")?;
        let grid_x = n_rows.div_ceil(64);          // BM=64
        let grid_y = batch.div_ceil(64);           // BN=64
        let cfg = LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (128, 1, 1),                    // 4 warps × wave32
            shared_mem_bytes: 0,                   // declared static in kernel
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(), k, n_rows, batch, blocks
        ])
    }
}

/// Grouped Q8_0 GEMV — each output row `idx` in `[0, n_groups*rank)` reads
/// input from group `g = idx/rank`'s 4096-element slice. Mirrors ds4's
/// `matvec_q8_0_grouped_rows_decode_scratch` (ds4.c:3618). Used by V4 Flash
/// for the attention output's first-stage projection (`attn_output_a`,
/// `[n_groups*rank=8192, group_dim=4096]` Q8_0).
///
/// Input quantisation: the flat `n_groups*group_dim` f32 input can be
/// quantised with the regular `Q8_0Matvec::quantize_input` (block boundaries
/// align with group boundaries because `group_dim % 32 == 0`), producing
/// a flat `[n_groups*group_dim]` int8 buffer and `[n_groups*blocks_per_group]`
/// scales — bit-identical to ds4's per-group quantisation loop.
#[allow(non_camel_case_types)]
pub struct Q8_0GroupedMatvec {
    module: Module,
}

impl Q8_0GroupedMatvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q8_0_GROUPED_MATVEC_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q8_0_GROUPED_MATVEC_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q8_0_grouped matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// The loaded module (tests: explicit-symbol launches).
    pub fn module(&self) -> &Module { &self.module }

    /// `out[idx] = sum_b f16(scale_w[idx,b]) * xscale[g,b] * dot_i8x32(qs_w[idx,b], xq[g,b])`
    /// for `idx` in `0..n_groups*rank`, where `g = idx/rank`.
    pub fn matvec_grouped(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        group_dim: u32,
        rank: u32,
        n_groups: u32,
    ) -> eyre::Result<()> {
        if group_dim % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!(
                "q8_0 grouped matvec: group_dim={group_dim} not a multiple of 32"
            ));
        }
        let blocks_per_group = group_dim / Q8_0_BLOCK_ELEMS;
        let out_dim = n_groups * rank;
        let expected_weight_bytes =
            (out_dim as usize) * (blocks_per_group as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 grouped matvec weight bytes: have {}, expected {} (out_dim={out_dim}, group_dim={group_dim})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        let in_total = (n_groups as usize) * (group_dim as usize);
        let scales_total = (n_groups as usize) * (blocks_per_group as usize);
        if out.len() < out_dim as usize {
            return Err(eyre!(
                "q8_0 grouped matvec out len: have {}, expected {}",
                out.len(),
                out_dim
            ));
        }
        if xq.len() < in_total || xscale.len() < scales_total {
            return Err(eyre!(
                "q8_0 grouped matvec xq/xscale len: xq={}, xscale={}, expected xq={}, xscale={}",
                xq.len(),
                xscale.len(),
                in_total,
                scales_total
            ));
        }

        let function = self.module.get_function("q8_0_grouped_gemv")?;

        let grid_x = out_dim.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(),
            group_dim, rank, blocks_per_group, n_groups
        ])
    }

    /// M50 Phase 2: batched grouped GEMV with `grid.z = B`. Per-batch
    /// xq[B, n_groups*group_dim], xscale[B, n_groups*blocks_per_group],
    /// out[B, n_groups*rank]. Weight shared across batch.
    #[allow(clippy::too_many_arguments)]
    /// B-PACKED grouped GEMV: reads the grouped weight matrix ONCE for all
    /// `batch` activations instead of once per batch element.
    ///
    /// `matvec_grouped_batched` launches `grid.z = batch`, so B=5 reads the
    /// drafter's 35.7 MB `attn_output_a` five times. Bit-identical per
    /// (row, b) -- same dp4a chain, same float accumulation order.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_grouped_bpack(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        group_dim: u32,
        rank: u32,
        n_groups: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if batch > GEMV_BPACK_MAX {
            return Err(eyre!(
                "q8_0 matvec_grouped_bpack: batch={batch} exceeds GEMV_BPACK_MAX={GEMV_BPACK_MAX}"
            ));
        }
        if group_dim % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!(
                "q8_0 matvec_grouped_bpack: group_dim={group_dim} not %32"
            ));
        }
        let blocks_per_group = group_dim / Q8_0_BLOCK_ELEMS;
        let out_dim = n_groups * rank;
        let expected_weight_bytes =
            (out_dim as usize) * (blocks_per_group as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 matvec_grouped_bpack weight bytes: {}!={expected_weight_bytes}",
                weight.byte_len()
            ));
        }
        let per_batch_in = (n_groups as usize) * (group_dim as usize);
        let per_batch_scales = (n_groups as usize) * (blocks_per_group as usize);
        if xq.len() < (batch as usize) * per_batch_in
            || xscale.len() < (batch as usize) * per_batch_scales
            || out.len() < (batch as usize) * (out_dim as usize)
        {
            return Err(eyre!(
                "q8_0 matvec_grouped_bpack: buffer too small for batch={batch} (xq {} xs {} out {})",
                xq.len(), xscale.len(), out.len()
            ));
        }

        // Compile-time-batch twin for 2 <= b <= 8 (`V41_GEMV_TB`), same args and
        // grid; 2026-09-26 sweep: wo_a 0.74 of the runtime kernel at b = 4, bit-exact.
        let function = self.module.get_function(grouped_bpack_symbol(batch))?;
        let grid_x = out_dim.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(),
            group_dim, rank, blocks_per_group, n_groups, batch
        ])
    }

    pub fn matvec_grouped_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        group_dim: u32,
        rank: u32,
        n_groups: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if group_dim % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!(
                "q8_0 grouped matvec_batched: group_dim={group_dim} not %32"
            ));
        }
        let blocks_per_group = group_dim / Q8_0_BLOCK_ELEMS;
        let out_dim = n_groups * rank;
        let expected_weight_bytes =
            (out_dim as usize) * (blocks_per_group as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 grouped matvec_batched weight bytes: {}!={expected_weight_bytes}",
                weight.byte_len()
            ));
        }
        let per_batch_in = (n_groups as usize) * (group_dim as usize);
        let per_batch_scales = (n_groups as usize) * (blocks_per_group as usize);
        if xq.len() < (batch as usize) * per_batch_in
            || xscale.len() < (batch as usize) * per_batch_scales
            || out.len() < (batch as usize) * (out_dim as usize)
        {
            return Err(eyre!(
                "q8_0 grouped matvec_batched: buffer too small for batch={batch} (xq {} xs {} out {})",
                xq.len(), xscale.len(), out.len()
            ));
        }

        // Same grid.z = batch re-read as `matvec_batched`; same bit-identical fix.
        if bpack_ok(batch) {
            return self.matvec_grouped_bpack(
                stream, out, weight, xq, xscale, group_dim, rank, n_groups, batch,
            );
        }
        if bpack_z16_for(batch, out_dim) {
            // `V41_GEMV_BPACK_Z16`: grouped bpack16 body over 16-row grid.z slices
            // (replay b <= 64), bit-identical; 2026-09-26 sweep: wo_a 1210 -> 535 us at b=64.
            let function = self.module.get_function("q8_0_grouped_gemv_bpack_z16")?;
            let cfg = LaunchConfig {
                grid: (out_dim.div_ceil(GEMV_ROWS_PER_BLOCK), 1, batch.div_ceil(GEMV_BPACK_MAX)),
                block: (GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES, 1, 1),
                shared_mem_bytes: 0,
            };
            return launch_kernel!(function, cfg, stream, [
                out.raw(), weight.raw(), xq.raw(), xscale.raw(),
                group_dim, rank, blocks_per_group, n_groups, batch
            ]);
        }
        let function = self.module.get_function("q8_0_grouped_gemv_batched")?;
        let grid_x = out_dim.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(),
            group_dim, rank, blocks_per_group, n_groups
        ])
    }

    /// M40-P4.5: 2-wide pair variant of grouped GEMV. Same weight, two
    /// input vectors → two outputs in one launch. Halves W BW vs running
    /// twice.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_grouped_pair(
        &self,
        stream: &Stream,
        out_a: &mut DeviceBuffer<f32>,
        out_b: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq_a: &DeviceBuffer<i8>,
        xq_b: &DeviceBuffer<i8>,
        xscale_a: &DeviceBuffer<f32>,
        xscale_b: &DeviceBuffer<f32>,
        group_dim: u32,
        rank: u32,
        n_groups: u32,
    ) -> eyre::Result<()> {
        if group_dim % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!(
                "q8_0 grouped matvec_pair: group_dim={group_dim} not a multiple of 32"
            ));
        }
        let blocks_per_group = group_dim / Q8_0_BLOCK_ELEMS;
        let out_dim = n_groups * rank;
        let expected_weight_bytes =
            (out_dim as usize) * (blocks_per_group as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 grouped matvec_pair weight bytes: have {}, expected {} (out_dim={out_dim}, group_dim={group_dim})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        let in_total = (n_groups as usize) * (group_dim as usize);
        let scales_total = (n_groups as usize) * (blocks_per_group as usize);
        if out_a.len() < out_dim as usize || out_b.len() < out_dim as usize {
            return Err(eyre!(
                "q8_0 grouped matvec_pair out lens: a={}, b={}, expected {}",
                out_a.len(),
                out_b.len(),
                out_dim
            ));
        }
        if xq_a.len() < in_total
            || xq_b.len() < in_total
            || xscale_a.len() < scales_total
            || xscale_b.len() < scales_total
        {
            return Err(eyre!(
                "q8_0 grouped matvec_pair xq/xscale lens: xq_a={}, xq_b={}, xs_a={}, xs_b={}, expected xq={}, xscale={}",
                xq_a.len(), xq_b.len(), xscale_a.len(), xscale_b.len(),
                in_total, scales_total
            ));
        }

        let function = self.module.get_function("q8_0_grouped_gemv_pair")?;

        let grid_x = out_dim.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out_a.raw(), out_b.raw(), weight.raw(),
            xq_a.raw(), xq_b.raw(), xscale_a.raw(), xscale_b.raw(),
            group_dim, rank, blocks_per_group, n_groups
        ])
    }
}

/// Fused shared-expert front half for dGPU decode:
/// `shared_gateup_swiglu_q8_tB<b>_r1` (kernels/shared_expert_fused.hip) =
/// gate GEMV + up GEMV + clamped swiglu + Q8_0 quantize of `mid` in ONE launch,
/// writing `(mid_xq, mid_xscale)` for the down GEMV. Replaces the 4 launches
/// `matvec_batched(gate)`, `matvec_batched(up)`, `swiglu.launch_clamped`,
/// `quantize_input_batched(mid)` of `issue_shared_expert_prefill`, BIT-IDENTICAL
/// (tests/q8_0_tb_bitexact.rs: b = 1..10, saturating and non-saturating
/// activations). One 1024-thread WG per 32 FFN rows: grid n_ff/32.
/// 2026-09-26 sweep (C1_dense_decode/shared_expert_fused_chain, dGPU, cold):
/// 5-node chain 105.9 -> 78.6 us at b = 4 with the tB down (x1.35; 4-7 us of
/// that is the fusion, the rest the tB GEMVs), b = 1 x1.12, b = 3 x1.27,
/// b = 5 x1.42. Selected by `forward_prefill::shared_fused_for(b)`.
pub struct SharedExpertFused {
    module: Module,
}

/// Largest batch with an instantiation (`_tB1_r1` .. `_tB10_r1`).
pub const SHARED_FUSED_MAX_B: u32 = 10;
const SHARED_FUSED_SYMBOLS: [&str; SHARED_FUSED_MAX_B as usize] = [
    "shared_gateup_swiglu_q8_tB1_r1", "shared_gateup_swiglu_q8_tB2_r1",
    "shared_gateup_swiglu_q8_tB3_r1", "shared_gateup_swiglu_q8_tB4_r1",
    "shared_gateup_swiglu_q8_tB5_r1", "shared_gateup_swiglu_q8_tB6_r1",
    "shared_gateup_swiglu_q8_tB7_r1", "shared_gateup_swiglu_q8_tB8_r1",
    "shared_gateup_swiglu_q8_tB9_r1", "shared_gateup_swiglu_q8_tB10_r1",
];

impl SharedExpertFused {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            SHARED_EXPERT_FUSED_GFX1201
        } else if arch.starts_with("gfx1151") {
            SHARED_EXPERT_FUSED_GFX1151
        } else {
            return Err(eyre!("unsupported arch for shared_expert_fused: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `mid_xq[b, n_ff]`, `mid_xscale[b, n_ff/32]` = Q8_0(swiglu_clamp(
    /// gate_w . x[b], up_w . x[b], clamp)) for b in 0..batch, from the Q8_0
    /// activation pair `xq[b, k]`, `xscale[b, k/32]`. `gate_w` / `up_w` are
    /// `[n_ff, k]` Q8_0 (row pitch `(k/32)*34`). Requires `k % 32 == 0`,
    /// `n_ff % 32 == 0` (a WG's 32 rows are quantized unguarded) and
    /// `1 <= batch <= SHARED_FUSED_MAX_B`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &Stream,
        mid_xq: &mut DeviceBuffer<i8>,
        mid_xscale: &mut DeviceBuffer<f32>,
        gate_w: &DeviceBuffer<u8>,
        up_w: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        k: u32,
        n_ff: u32,
        batch: u32,
        clamp: f32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if batch > SHARED_FUSED_MAX_B {
            return Err(eyre!(
                "shared_expert_fused: batch={batch} exceeds SHARED_FUSED_MAX_B={SHARED_FUSED_MAX_B}"
            ));
        }
        if k % Q8_0_BLOCK_ELEMS != 0 || n_ff % 32 != 0 || n_ff == 0 {
            return Err(eyre!("shared_expert_fused: k={k} and n_ff={n_ff} must be multiples of 32"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_ff as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if gate_w.byte_len() != expected_weight_bytes || up_w.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "shared_expert_fused weight bytes: gate {} up {}, expected {} (n_ff={n_ff}, k={k})",
                gate_w.byte_len(),
                up_w.byte_len(),
                expected_weight_bytes
            ));
        }
        let b = batch as usize;
        if xq.len() < b * (k as usize)
            || xscale.len() < b * (blocks as usize)
            || mid_xq.len() < b * (n_ff as usize)
            || mid_xscale.len() < b * (n_ff as usize / 32)
        {
            return Err(eyre!(
                "shared_expert_fused: buffers too small for batch={batch} (xq {} xs {} mid_xq {} mid_xs {})",
                xq.len(), xscale.len(), mid_xq.len(), mid_xscale.len()
            ));
        }
        let function = self.module.get_function(SHARED_FUSED_SYMBOLS[b - 1])?;
        let cfg = LaunchConfig {
            grid: (n_ff / 32, 1, 1),
            block: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid_xq.raw(), mid_xscale.raw(), gate_w.raw(), up_w.raw(), xq.raw(), xscale.raw(),
            k, n_ff, blocks, clamp
        ])
    }
}
