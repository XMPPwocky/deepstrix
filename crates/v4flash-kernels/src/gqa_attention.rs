//! Single-query GQA attention — correctness-first score -> softmax ->
//! weighted-sum for the Laguna model port.
//!
//! This is standard Grouped-Query Attention (NOT ds4's MLA). For one query
//! position it computes, per query head `h`:
//!   `out[h] = softmax_j( q[h] · k[j, h/kv_group] * scale ) · v[j, h/kv_group]`
//! over the whole causal history `j in 0..n_kv` (history is assumed to already
//! include the current position — the caller appends before calling).
//!
//! The caller has ALREADY applied qk-norm + RoPE to Q/K/V; this kernel does
//! NOT do rope, qk-norm, the softplus gate, or o_proj. Pure attention.
//!
//! CORRECTNESS-FIRST: no WMMA, no dp4a, no tiling. One workgroup per query
//! head, online (flash-style) f32 softmax so there is no `n_kv` cap. Perf,
//! WMMA and SWA-windowing are later phases. Mirrors [`crate::q4_k_dense`] /
//! [`crate::q8_0`] for the binding shape (`for_arch`, exhaustive validation).
//!
//! Data layout (caller contract — MUST match the KV-append order):
//! - `q`:       f16 `[n_head, head_dim]`            (one query position)
//! - `k_cache`: f16 `[n_kv, n_kv_head, head_dim]`   (row-major)
//! - `v_cache`: f16 `[n_kv, n_kv_head, head_dim]`   (row-major, same layout)
//! - `out`:     f32 `[n_head, head_dim]`
//!
//! Q is kept f16 to match the caller contract exactly; all accumulation is in
//! f32 inside the kernel, so f16 Q costs only the input rounding (which the
//! caller already committed to by storing an f16 cache).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const GQA_ATTENTION_GFX1201: &[u8] = include_bytes!(env!("KERNEL_GQA_ATTENTION_GFX1201"));
const GQA_ATTENTION_GFX1151: &[u8] = include_bytes!(env!("KERNEL_GQA_ATTENTION_GFX1151"));

/// Maximum `head_dim` the kernel's static LDS supports (mirrors
/// `GQA_HEAD_DIM_MAX` in `kernels/gqa_attention.hip`).
pub const GQA_HEAD_DIM_MAX: u32 = 256;

/// Workgroup size (mirrors `GQA_BLOCK` in the `.hip`).
const GQA_BLOCK: u32 = 256;

/// Flash-tiled prefill tile geometry (mirror `FBR`/`FD`/`FBLOCK` in the `.hip`).
const FLASH_BR: u32 = 32;
/// Head-grouped WMMA prefill geometry — MUST mirror `HG_G`/`HG_BR`/`HG_BLOCK`
/// in `gqa_attention.hip` (override both sides together via `-DHG_G=`/`-DHG_BR=`).
// Query heads grouped per WG. Env-overridable for A/B (MUST match the kernel's
// compile-time -DHG_G). Default 3 = max K/V reuse at 1 WG/CU; 2 lowers LDS to
// ~31 KB → 2 WG/CU (hides barriers behind a second WG).
fn flash_hg_g() -> u32 {
    std::env::var("LAGUNA_HG_G").ok().and_then(|v| v.parse().ok()).unwrap_or(3)
}
const FLASH_HG_BR: u32 = 16;
const FLASH_HG_BLOCK: u32 = 256;
/// Full-GQA-group packing factor for `..._fa2_hg_packed` (mirrors `HGP_G` in the
/// `.hip`). Fixed at 6 = full-attn kv_group; one WG owns the whole group.
const FLASH_HGP_G: u32 = 6;
/// Max `head_dim` the flash kernel's static LDS supports (mirrors `FD`).
pub const FLASH_HEAD_DIM: u32 = 128;
/// Flash kernel workgroup size (mirrors `FBLOCK`).
const FLASH_BLOCK: u32 = 256;

/// Decode-attention routing: default is the FLASH-tiled single-query kernel
/// ([`GqaAttention::single_query_flash`]); set `LAGUNA_DECODE_ATTN_NAIVE=1` to
/// fall back to the naive per-key-barrier [`GqaAttention::single_query`] kernel
/// (kept for A/B and parity fallback). Read once and cached.
pub fn decode_attn_use_naive() -> bool {
    use std::sync::OnceLock;
    static NAIVE: OnceLock<bool> = OnceLock::new();
    *NAIVE.get_or_init(|| {
        std::env::var("LAGUNA_DECODE_ATTN_NAIVE").map(|v| v == "1").unwrap_or(false)
    })
}

/// Decode-attention kernel variant. Default `splitkv` (flash decoding); set
/// `LAGUNA_DECODE_ATTN=naive|flash|splitkv` to A/B. `naive` here is the
/// per-key-barrier reference; `flash` is the batch=1 tiled kernel (the old
/// default). Read once and cached.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeAttn {
    Naive,
    Flash,
    SplitKv,
    /// Head-grouped split-KV: one WG per (kv_head, split), K/V staged once and
    /// reused across all `kv_group` query heads (kills the per-query-head
    /// redundant K/V DRAM reads). Default.
    SplitKvHg,
}
pub fn decode_attn_variant() -> DecodeAttn {
    use std::sync::OnceLock;
    static V: OnceLock<DecodeAttn> = OnceLock::new();
    *V.get_or_init(|| {
        if decode_attn_use_naive() {
            return DecodeAttn::Naive;
        }
        match std::env::var("LAGUNA_DECODE_ATTN").as_deref() {
            Ok("naive") => DecodeAttn::Naive,
            Ok("flash") => DecodeAttn::Flash,
            Ok("splitkv") => DecodeAttn::SplitKv,
            _ => DecodeAttn::SplitKvHg,
        }
    })
}

/// Crossover (in causal length `n_kv`) below which decode attention uses the
/// naive per-key kernel and at/above which it uses the split-KV family. MEASURED
/// (Laguna decode, back-to-back A/B): split-KV-HG beats naive at EVERY context
/// tested including ctx=64 (26.1 vs 23.9 tok/s), ctx=256 (26.0 vs 19.5 — the old
/// gate=512 dip), ctx=512 (25.4 vs 15.5). The naive per-key barrier storm costs
/// wall out of all proportion to the tiny attn kernel itself. So the crossover is
/// below 64; keep a minimal floor of 16 only for trivially short history where a
/// single split has no parallelism to exploit. Env `LAGUNA_DECODE_FLASH_MIN_KV`
/// overrides for A/B sweeps. Cached.
pub fn decode_flash_min_kv() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LAGUNA_DECODE_FLASH_MIN_KV").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
    })
}

/// Max `n_splits` the caller must size split-KV scratch for.
pub const DECODE_KV_SPLITS_MAX: u32 = 128;

/// Choose the number of key-splits for split-KV decode from the causal length.
/// Targets ~512 keys/split (enough work to amortise the tile setup) while
/// producing `n_head*n_splits` workgroups to fill the dGPU. Env override
/// `LAGUNA_DECODE_KV_SPLITS` (clamped to [1, DECODE_KV_SPLITS_MAX]) for sweeps.
pub fn decode_kv_splits(n_kv: u32) -> u32 {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    let ov = *OVERRIDE.get_or_init(|| {
        std::env::var("LAGUNA_DECODE_KV_SPLITS").ok().and_then(|v| v.parse::<u32>().ok())
    });
    if let Some(s) = ov {
        return s.clamp(1, DECODE_KV_SPLITS_MAX);
    }
    let s = n_kv.div_ceil(512);
    s.clamp(1, DECODE_KV_SPLITS_MAX)
}

/// Number of key-splits for the HEAD-GROUPED split-KV decode. The head-grouped
/// partial launches only `n_kv_head * n_splits` workgroups (one per KV head,
/// vs the per-head kernel's `n_head * n_splits`), so at short context it needs
/// MORE splits to keep the dGPU filled. Targets ~256 keys/split but floors the
/// split count so `n_kv_head(=8) * n_splits` stays >= ~512 WGs (≈8 deep on 64
/// CUs). Env override `LAGUNA_DECODE_KV_SPLITS` (shared with the per-head path)
/// still wins for sweeps.
pub fn decode_kv_splits_hg(n_kv: u32) -> u32 {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    let ov = *OVERRIDE.get_or_init(|| {
        std::env::var("LAGUNA_DECODE_KV_SPLITS").ok().and_then(|v| v.parse::<u32>().ok())
    });
    if let Some(s) = ov {
        return s.clamp(1, DECODE_KV_SPLITS_MAX);
    }
    // ~256 keys/split, but never fewer than 64 splits (8*64 = 512 WGs) unless
    // the history itself is shorter than that.
    let by_work = n_kv.div_ceil(256);
    let floor = 64.min(n_kv.div_ceil(32).max(1));
    by_work.max(floor).clamp(1, DECODE_KV_SPLITS_MAX)
}

pub struct GqaAttention {
    module: Module,
}

impl GqaAttention {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            GQA_ATTENTION_GFX1201
        } else if arch.starts_with("gfx1151") {
            GQA_ATTENTION_GFX1151
        } else {
            return Err(eyre!("unsupported arch for gqa attention: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Single-query GQA attention.
    ///
    /// - `out`:      f32 `[n_head * head_dim]`
    /// - `q`:        f16 `[n_head * head_dim]`
    /// - `k_cache`:  f16 `[n_kv * n_kv_head * head_dim]`
    /// - `v_cache`:  f16 `[n_kv * n_kv_head * head_dim]`
    /// - `scale`:    typically `1.0 / (head_dim as f32).sqrt()`.
    ///
    /// `n_head` must be divisible by `n_kv_head` (GQA grouping), and
    /// `head_dim <= GQA_HEAD_DIM_MAX`.
    #[allow(clippy::too_many_arguments)]
    pub fn single_query(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        n_kv: u32,
        scale: f32,
        k_base: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if n_head == 0 || n_kv_head == 0 || head_dim == 0 {
            return Err(eyre!(
                "gqa attn: n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim} must be > 0"
            ));
        }
        if head_dim > GQA_HEAD_DIM_MAX {
            return Err(eyre!(
                "gqa attn: head_dim={head_dim} exceeds GQA_HEAD_DIM_MAX={GQA_HEAD_DIM_MAX}"
            ));
        }
        if n_head % n_kv_head != 0 {
            return Err(eyre!(
                "gqa attn: n_head={n_head} not divisible by n_kv_head={n_kv_head} (GQA grouping)"
            ));
        }

        let hd = head_dim as usize;
        let nh = n_head as usize;
        let nkvh = n_kv_head as usize;
        let nkv = n_kv as usize;

        let expected_q = nh * hd;
        if q.len() != expected_q {
            return Err(eyre!(
                "gqa attn q len: have {}, expected {expected_q} (n_head={n_head}, head_dim={head_dim})",
                q.len()
            ));
        }
        let expected_out = nh * hd;
        if out.len() != expected_out {
            return Err(eyre!(
                "gqa attn out len: have {}, expected {expected_out} (n_head={n_head}, head_dim={head_dim})",
                out.len()
            ));
        }
        // The caller passes the WHOLE physical ring buffer (kv_capacity rows); the
        // kernel maps relative key j -> physical (k_base+j) % kv_capacity.
        let _ = nkv;
        let expected_kv = kv_capacity as usize * nkvh * hd;
        if k_cache.len() < expected_kv {
            return Err(eyre!(
                "gqa attn k_cache len: have {}, need >= {expected_kv} (kv_capacity={kv_capacity}, n_kv_head={n_kv_head}, head_dim={head_dim})",
                k_cache.len()
            ));
        }
        if v_cache.len() < expected_kv {
            return Err(eyre!(
                "gqa attn v_cache len: have {}, need >= {expected_kv} (kv_capacity={kv_capacity}, n_kv_head={n_kv_head}, head_dim={head_dim})",
                v_cache.len()
            ));
        }

        let function = self.module.get_function("gqa_attn_single_query")?;
        let cfg = LaunchConfig {
            grid: (n_head, 1, 1),
            block: (GQA_BLOCK, 1, 1),
            shared_mem_bytes: 0, // static LDS declared in the kernel
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), k_cache.raw(), v_cache.raw(),
            n_head, n_kv_head, head_dim, n_kv, scale, k_base, kv_capacity
        ])
    }

    /// FLASH-tiled SINGLE-query (decode) GQA attention.
    ///
    /// Same contract/output as [`single_query`], but dispatches the flash-tiled
    /// `gqa_attn_prefill_flash` kernel with `batch = 1` and `q_offset = n_kv-1`
    /// (the lone decode query sits at absolute position `n_kv-1` and attends the
    /// whole causal history `[0 ..= n_kv-1]`). This kills the naive kernel's
    /// per-key `__syncthreads` barrier storm (one barrier PER key -> one per
    /// 32-key tile) and its whole-K/V-per-key re-read, which dominate
    /// long-context decode. Quality-safe: identical online-softmax f32 math.
    ///
    /// `head_dim` must be `<= FLASH_HEAD_DIM` (128 — Laguna). Buffer shapes
    /// match [`single_query`] exactly (`q`/`out`: `[n_head*head_dim]`,
    /// `k_cache`/`v_cache`: `[n_kv*n_kv_head*head_dim]`).
    #[allow(clippy::too_many_arguments)]
    pub fn single_query_flash(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        n_kv: u32,
        scale: f32,
        swa_window: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if n_head == 0 || n_kv_head == 0 || head_dim == 0 || n_kv == 0 {
            return Err(eyre!(
                "gqa flash decode: n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim}, n_kv={n_kv} must be > 0"
            ));
        }
        if head_dim > FLASH_HEAD_DIM {
            return Err(eyre!(
                "gqa flash decode: head_dim={head_dim} exceeds FLASH_HEAD_DIM={FLASH_HEAD_DIM}"
            ));
        }
        if n_head % n_kv_head != 0 {
            return Err(eyre!(
                "gqa flash decode: n_head={n_head} not divisible by n_kv_head={n_kv_head}"
            ));
        }
        let hd = head_dim as usize;
        let expected_q = n_head as usize * hd;
        if q.len() != expected_q {
            return Err(eyre!("gqa flash decode q len: have {}, expected {expected_q}", q.len()));
        }
        if out.len() != expected_q {
            return Err(eyre!("gqa flash decode out len: have {}, expected {expected_q}", out.len()));
        }
        // Caller passes the WHOLE physical ring buffer (kv_capacity rows). n_kv is
        // the ABSOLUTE causal length; the kernel windows via swa_window and maps
        // absolute key -> physical key%kv_capacity.
        let expected_kv = kv_capacity as usize * n_kv_head as usize * hd;
        if k_cache.len() < expected_kv || v_cache.len() < expected_kv {
            return Err(eyre!(
                "gqa flash decode kv len: k={} v={}, need >= {expected_kv}",
                k_cache.len(), v_cache.len()
            ));
        }

        // batch=1, absolute query position = n_kv-1, n_kv_total = n_kv.
        let q_offset = n_kv - 1;
        let function = self.module.get_function("gqa_attn_prefill_flash")?;
        let cfg = LaunchConfig {
            grid: (n_head, 1, 1), // ceil(batch=1 / FLASH_BR) == 1
            block: (FLASH_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), k_cache.raw(), v_cache.raw(),
            n_head, n_kv_head, head_dim, q_offset, 1u32, n_kv, scale, swa_window, kv_capacity
        ])
    }

    /// SPLIT-KV DECODE ("flash decoding") single-query GQA attention.
    ///
    /// Same output contract as [`single_query`] / [`single_query_flash`], but
    /// partitions the causal history across `n_splits` workgroups per head to
    /// fill the GPU and cut the serial key-tile chain that made the batch=1
    /// flash kernel 82.9% of decode wall at 32K. Two launches: a partial kernel
    /// (grid `n_head*n_splits`) writing unnormalised partials into scratch, then
    /// a combine kernel (grid `n_head`) merging them. Numerically the identical
    /// flash recurrence, reassociated across splits (within f32 order tolerance).
    ///
    /// Scratch sizing (caller-provided, reused across calls/layers):
    ///   `out_partial >= n_head * n_splits * head_dim`, `m_partial`/`l_partial
    ///   >= n_head * n_splits`.
    #[allow(clippy::too_many_arguments)]
    pub fn single_query_splitkv(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        out_partial: &mut DeviceBuffer<f32>,
        m_partial: &mut DeviceBuffer<f32>,
        l_partial: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        n_kv: u32,
        n_splits: u32,
        scale: f32,
        k_base: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if n_head == 0 || n_kv_head == 0 || head_dim == 0 || n_kv == 0 || n_splits == 0 {
            return Err(eyre!(
                "gqa splitkv: n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim}, n_kv={n_kv}, n_splits={n_splits} must be > 0"
            ));
        }
        if head_dim > FLASH_HEAD_DIM {
            return Err(eyre!("gqa splitkv: head_dim={head_dim} exceeds {FLASH_HEAD_DIM}"));
        }
        if n_head % n_kv_head != 0 {
            return Err(eyre!("gqa splitkv: n_head={n_head} not divisible by n_kv_head={n_kv_head}"));
        }
        let hd = head_dim as usize;
        let expected_q = n_head as usize * hd;
        if q.len() != expected_q || out.len() != expected_q {
            return Err(eyre!("gqa splitkv q/out len: q={} out={}, expected {expected_q}", q.len(), out.len()));
        }
        // Whole physical ring buffer (kv_capacity rows); kernel maps relative key
        // -> physical (k_base+key)%kv_capacity. n_kv stays the windowed count.
        let expected_kv = kv_capacity as usize * n_kv_head as usize * hd;
        if k_cache.len() < expected_kv || v_cache.len() < expected_kv {
            return Err(eyre!("gqa splitkv kv len: k={} v={}, need >= {expected_kv}", k_cache.len(), v_cache.len()));
        }
        let need_part = n_head as usize * n_splits as usize * hd;
        let need_ml = n_head as usize * n_splits as usize;
        if out_partial.len() < need_part || m_partial.len() < need_ml || l_partial.len() < need_ml {
            return Err(eyre!(
                "gqa splitkv scratch too small: out_partial {} (need {need_part}), m {} l {} (need {need_ml})",
                out_partial.len(), m_partial.len(), l_partial.len()
            ));
        }

        const DEC_BLOCK: u32 = 128;
        let fp = self.module.get_function("gqa_attn_decode_partial")?;
        let cfg_p = LaunchConfig {
            grid: (n_head, n_splits, 1),
            block: (DEC_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(fp, cfg_p, stream, [
            out_partial.raw(), m_partial.raw(), l_partial.raw(),
            q.raw(), k_cache.raw(), v_cache.raw(),
            n_head, n_kv_head, head_dim, n_kv, n_splits, scale, k_base, kv_capacity
        ])?;
        let fc = self.module.get_function("gqa_attn_decode_combine")?;
        let cfg_c = LaunchConfig {
            grid: (n_head, 1, 1),
            block: (DEC_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(fc, cfg_c, stream, [
            out.raw(), out_partial.raw(), m_partial.raw(), l_partial.raw(),
            n_head, head_dim, n_splits
        ])
    }

    /// HEAD-GROUPED SPLIT-KV DECODE — GQA-aware flash decoding. Same output
    /// contract and scratch sizing as [`single_query_splitkv`], but the partial
    /// kernel launches one WG per (`kv_head`, split) = grid `n_kv_head*n_splits`
    /// instead of `n_head*n_splits`. Each WG stages its KV head's K/V tile into
    /// LDS ONCE and computes all `kv_group` (6 or 9) query heads from it, so the
    /// K/V DRAM reads that the per-head kernel duplicated `kv_group`-fold are
    /// issued once. The partials are written per DERIVED query head, so the same
    /// [`gqa_attn_decode_combine`] merges them unchanged.
    ///
    /// `n_head` must be a multiple of `n_kv_head` and `kv_group = n_head/n_kv_head
    /// <= 12` (the kernel's static per-head LDS). `head_dim <= FLASH_HEAD_DIM`.
    #[allow(clippy::too_many_arguments)]
    pub fn single_query_splitkv_hg(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        out_partial: &mut DeviceBuffer<f32>,
        m_partial: &mut DeviceBuffer<f32>,
        l_partial: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        n_kv: u32,
        n_splits: u32,
        scale: f32,
        k_base: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if n_head == 0 || n_kv_head == 0 || head_dim == 0 || n_kv == 0 || n_splits == 0 {
            return Err(eyre!(
                "gqa splitkv_hg: n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim}, n_kv={n_kv}, n_splits={n_splits} must be > 0"
            ));
        }
        if head_dim > FLASH_HEAD_DIM {
            return Err(eyre!("gqa splitkv_hg: head_dim={head_dim} exceeds {FLASH_HEAD_DIM}"));
        }
        if n_head % n_kv_head != 0 {
            return Err(eyre!("gqa splitkv_hg: n_head={n_head} not divisible by n_kv_head={n_kv_head}"));
        }
        let kv_group = n_head / n_kv_head;
        if kv_group > 12 {
            return Err(eyre!("gqa splitkv_hg: kv_group={kv_group} exceeds DEC_KVG_MAX=12"));
        }
        let hd = head_dim as usize;
        let expected_q = n_head as usize * hd;
        if q.len() != expected_q || out.len() != expected_q {
            return Err(eyre!("gqa splitkv_hg q/out len: q={} out={}, expected {expected_q}", q.len(), out.len()));
        }
        // Whole physical ring buffer (kv_capacity rows); kernel maps relative key
        // -> physical (k_base+key)%kv_capacity. n_kv stays the windowed count.
        let expected_kv = kv_capacity as usize * n_kv_head as usize * hd;
        if k_cache.len() < expected_kv || v_cache.len() < expected_kv {
            return Err(eyre!("gqa splitkv_hg kv len: k={} v={}, need >= {expected_kv}", k_cache.len(), v_cache.len()));
        }
        let need_part = n_head as usize * n_splits as usize * hd;
        let need_ml = n_head as usize * n_splits as usize;
        if out_partial.len() < need_part || m_partial.len() < need_ml || l_partial.len() < need_ml {
            return Err(eyre!(
                "gqa splitkv_hg scratch too small: out_partial {} (need {need_part}), m {} l {} (need {need_ml})",
                out_partial.len(), m_partial.len(), l_partial.len()
            ));
        }

        const DEC_BLOCK: u32 = 128;
        let fp = self.module.get_function("gqa_attn_decode_partial_hg")?;
        let cfg_p = LaunchConfig {
            grid: (n_kv_head, n_splits, 1),
            block: (DEC_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(fp, cfg_p, stream, [
            out_partial.raw(), m_partial.raw(), l_partial.raw(),
            q.raw(), k_cache.raw(), v_cache.raw(),
            n_head, n_kv_head, head_dim, n_kv, n_splits, scale, k_base, kv_capacity
        ])?;
        let fc = self.module.get_function("gqa_attn_decode_combine")?;
        let cfg_c = LaunchConfig {
            grid: (n_head, 1, 1),
            block: (DEC_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(fc, cfg_c, stream, [
            out.raw(), out_partial.raw(), m_partial.raw(), l_partial.raw(),
            n_head, head_dim, n_splits
        ])
    }

    /// Batched (prefill) GQA attention: `batch` query positions in ONE launch.
    ///
    /// Semantically identical to calling [`single_query`] once per chunk
    /// position, but dispatches all `batch * n_head` workgroups together so the
    /// dGPU is filled (sequential single-token prefill was dispatch-bound).
    ///
    /// - `out`:      f32 `[batch * n_head * head_dim]`
    /// - `q`:        f16 `[batch * n_head * head_dim]`
    /// - `k_cache`:  f16 `[n_kv_total * n_kv_head * head_dim]`
    /// - `v_cache`:  f16 `[n_kv_total * n_kv_head * head_dim]`
    /// - `q_offset`: absolute position of query row 0 (0 for a from-scratch
    ///   prefill); query row `i` attends keys `[0 ..= q_offset + i]`.
    ///
    /// The caller MUST have appended every chunk position's K/V into the cache
    /// before launching (causal masking reads only `q_offset+i+1` keys per row,
    /// so later chunk positions never leak into earlier ones).
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        batch: u32,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        q_offset: u32,
        scale: f32,
        swa_window: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0 {
            return Err(eyre!(
                "gqa prefill: batch={batch}, n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim} must be > 0"
            ));
        }
        if head_dim > GQA_HEAD_DIM_MAX {
            return Err(eyre!(
                "gqa prefill: head_dim={head_dim} exceeds GQA_HEAD_DIM_MAX={GQA_HEAD_DIM_MAX}"
            ));
        }
        if n_head % n_kv_head != 0 {
            return Err(eyre!(
                "gqa prefill: n_head={n_head} not divisible by n_kv_head={n_kv_head}"
            ));
        }
        let want_q = (batch * n_head * head_dim) as usize;
        if q.len() != want_q {
            return Err(eyre!("gqa prefill q len: have {}, expected {want_q}", q.len()));
        }
        if out.len() != want_q {
            return Err(eyre!("gqa prefill out len: have {}, expected {want_q}", out.len()));
        }
        // Caller passes the whole physical ring buffer (kv_capacity rows); for
        // global layers kv_capacity==max_kv >= q_offset+batch. The kernel maps
        // absolute key -> physical key%kv_capacity.
        let min_kv = kv_capacity as usize * (n_kv_head * head_dim) as usize;
        if k_cache.len() < min_kv || v_cache.len() < min_kv {
            return Err(eyre!(
                "gqa prefill kv cache too small: k={} v={} need >= {min_kv}",
                k_cache.len(), v_cache.len()
            ));
        }

        let function = self.module.get_function("gqa_attn_prefill")?;
        let cfg = LaunchConfig {
            grid: (n_head, batch, 1),
            block: (GQA_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), k_cache.raw(), v_cache.raw(),
            n_head, n_kv_head, head_dim, q_offset, scale, swa_window, kv_capacity
        ])
    }

    /// FLASH-tiled batched (prefill) GQA attention — same contract and output
    /// as [`prefill`], but block-tiles queries (FBR=32) and keys (FBC=32),
    /// staging one K/V tile into LDS and reusing it across the whole query
    /// block. Kills the per-query whole-K/V re-read (O(B²) -> O(B²/FBR) DRAM)
    /// and the per-key barrier storm of the naive [`prefill`] kernel.
    ///
    /// Same buffer shapes as [`prefill`]. `head_dim` must be <= `FLASH_HEAD_DIM`
    /// (128 — the flash kernel's static LDS is sized for Laguna). `n_kv_total`
    /// is the number of key/value rows present in the cache (>= q_offset+batch);
    /// derived here as `q_offset + batch`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_flash(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        batch: u32,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        q_offset: u32,
        scale: f32,
        swa_window: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0 {
            return Err(eyre!(
                "gqa flash: batch={batch}, n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim} must be > 0"
            ));
        }
        if head_dim > FLASH_HEAD_DIM {
            return Err(eyre!(
                "gqa flash: head_dim={head_dim} exceeds FLASH_HEAD_DIM={FLASH_HEAD_DIM}"
            ));
        }
        if n_head % n_kv_head != 0 {
            return Err(eyre!(
                "gqa flash: n_head={n_head} not divisible by n_kv_head={n_kv_head}"
            ));
        }
        let want_q = (batch * n_head * head_dim) as usize;
        if q.len() != want_q {
            return Err(eyre!("gqa flash q len: have {}, expected {want_q}", q.len()));
        }
        if out.len() != want_q {
            return Err(eyre!("gqa flash out len: have {}, expected {want_q}", out.len()));
        }
        let n_kv_total = q_offset + batch;
        // Caller passes the whole physical ring buffer (kv_capacity rows); global
        // layers have kv_capacity==max_kv >= n_kv_total. Kernel wraps key%kv_capacity.
        let min_kv = kv_capacity as usize * (n_kv_head * head_dim) as usize;
        if k_cache.len() < min_kv || v_cache.len() < min_kv {
            return Err(eyre!(
                "gqa flash kv cache too small: k={} v={} need >= {min_kv}",
                k_cache.len(), v_cache.len()
            ));
        }

        let function = self.module.get_function("gqa_attn_prefill_flash")?;
        let grid_y = batch.div_ceil(FLASH_BR);
        let cfg = LaunchConfig {
            grid: (n_head, grid_y, 1),
            block: (FLASH_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), k_cache.raw(), v_cache.raw(),
            n_head, n_kv_head, head_dim, q_offset, batch, n_kv_total, scale, swa_window, kv_capacity
        ])
    }

    /// WMMA FLASH-tiled batched (prefill) GQA attention — same contract and
    /// output as [`prefill_flash`], but the score (Q·Kᵀ) and AV (P·V) matmuls
    /// run on the gfx1201 (RDNA4) f16 matrix core (16×16×16 WMMA, f32 accumulate)
    /// instead of the 8-way scalar-ILP dot. dGPU-only lever (the iGPU gfx1151
    /// build gets a portable scalar fallback; never launch this there).
    ///
    /// Identical online-softmax f32 math (quality-safe); only the two matmuls
    /// are reassociated (WMMA tile order). Block is 128 threads (4 wave32 waves).
    /// Same buffer shapes and `head_dim <= FLASH_HEAD_DIM` (128) as
    /// [`prefill_flash`].
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_flash_wmma(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        batch: u32,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        q_offset: u32,
        scale: f32,
        swa_window: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0 {
            return Err(eyre!(
                "gqa flash wmma: batch={batch}, n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim} must be > 0"
            ));
        }
        if head_dim > FLASH_HEAD_DIM {
            return Err(eyre!(
                "gqa flash wmma: head_dim={head_dim} exceeds FLASH_HEAD_DIM={FLASH_HEAD_DIM}"
            ));
        }
        if n_head % n_kv_head != 0 {
            return Err(eyre!(
                "gqa flash wmma: n_head={n_head} not divisible by n_kv_head={n_kv_head}"
            ));
        }
        let want_q = (batch * n_head * head_dim) as usize;
        if q.len() != want_q {
            return Err(eyre!("gqa flash wmma q len: have {}, expected {want_q}", q.len()));
        }
        if out.len() != want_q {
            return Err(eyre!("gqa flash wmma out len: have {}, expected {want_q}", out.len()));
        }
        let n_kv_total = q_offset + batch;
        // Caller passes the whole physical ring buffer (kv_capacity rows); global
        // layers have kv_capacity==max_kv >= n_kv_total. Kernel wraps key%kv_capacity.
        let min_kv = kv_capacity as usize * (n_kv_head * head_dim) as usize;
        if k_cache.len() < min_kv || v_cache.len() < min_kv {
            return Err(eyre!(
                "gqa flash wmma kv cache too small: k={} v={} need >= {min_kv}",
                k_cache.len(), v_cache.len()
            ));
        }

        let function = self.module.get_function("gqa_attn_prefill_flash_wmma")?;
        let grid_y = batch.div_ceil(FLASH_BR);
        const WMMA_BLOCK: u32 = 128;
        let cfg = LaunchConfig {
            grid: (n_head, grid_y, 1),
            block: (WMMA_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), k_cache.raw(), v_cache.raw(),
            n_head, n_kv_head, head_dim, q_offset, batch, n_kv_total, scale, swa_window, kv_capacity
        ])
    }

    /// FA2 REGISTER-RESIDENT-O WMMA prefill — identical contract, output, and
    /// online-softmax math as [`prefill_flash_wmma`], but the running O
    /// accumulator lives in REGISTERS (a persistent WMMA C-fragment per wave)
    /// instead of the 16 KB `Os` LDS array. That drops LDS ~46 KB → ~30 KB →
    /// ≥2 WG/CU, restoring the occupancy that hides the per-key-tile barriers and
    /// flattens the O(L²) global-attention prefill falloff. Same buffer shapes
    /// and `head_dim <= FLASH_HEAD_DIM` (128) as [`prefill_flash_wmma`].
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_flash_wmma_fa2(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        batch: u32,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        q_offset: u32,
        scale: f32,
        swa_window: u32,
        kv_capacity: u32,
        kv_first: bool,
    ) -> eyre::Result<()> {
        if batch == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0 {
            return Err(eyre!(
                "gqa flash wmma fa2: batch={batch}, n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim} must be > 0"
            ));
        }
        if head_dim > FLASH_HEAD_DIM {
            return Err(eyre!(
                "gqa flash wmma fa2: head_dim={head_dim} exceeds FLASH_HEAD_DIM={FLASH_HEAD_DIM}"
            ));
        }
        if n_head % n_kv_head != 0 {
            return Err(eyre!(
                "gqa flash wmma fa2: n_head={n_head} not divisible by n_kv_head={n_kv_head}"
            ));
        }
        let want_q = (batch * n_head * head_dim) as usize;
        if q.len() != want_q {
            return Err(eyre!("gqa flash wmma fa2 q len: have {}, expected {want_q}", q.len()));
        }
        if out.len() != want_q {
            return Err(eyre!("gqa flash wmma fa2 out len: have {}, expected {want_q}", out.len()));
        }
        let n_kv_total = q_offset + batch;
        // Caller passes the whole physical ring buffer (kv_capacity rows); global
        // layers have kv_capacity==max_kv >= n_kv_total. Kernel wraps key%kv_capacity.
        let min_kv = kv_capacity as usize * (n_kv_head * head_dim) as usize;
        if k_cache.len() < min_kv || v_cache.len() < min_kv {
            return Err(eyre!(
                "gqa flash wmma fa2 kv cache too small: k={} v={} need >= {min_kv}",
                k_cache.len(), v_cache.len()
            ));
        }

        let function = self.module.get_function("gqa_attn_prefill_flash_wmma_fa2")?;
        let grid_y = batch.div_ceil(FLASH_BR);
        const WMMA_BLOCK: u32 = 128;
        // KV-first remap: enumerate all query heads + query-tiles sharing a KV head
        // by consecutive linear block ids (KV head = grid.y, the slow dim), so each
        // KV head's K/V stays Infinity-Cache-resident across its redundant reads.
        // grid.x spans kv_group query heads × grid_y query-tiles.
        let kv_group = n_head / n_kv_head;
        let grid = if kv_first {
            (kv_group * grid_y, n_kv_head, 1)
        } else {
            (n_head, grid_y, 1)
        };
        let grid_mode: u32 = if kv_first { 1 } else { 0 };
        let cfg = LaunchConfig {
            grid,
            block: (WMMA_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), k_cache.raw(), v_cache.raw(),
            n_head, n_kv_head, head_dim, q_offset, batch, n_kv_total, scale, swa_window, kv_capacity, grid_mode
        ])
    }

    /// Head-grouped WMMA FA prefill (`gqa_attn_prefill_flash_wmma_fa2_hg`). One
    /// WG owns a KV head + a sub-group of `FLASH_HG_G` query heads + a
    /// `FLASH_HG_BR`-row query tile, streaming each K/V key-tile into LDS once and
    /// reusing it across the group (fewer per-key-tile barriers + redundant K/V
    /// loads). Requires `kv_group % FLASH_HG_G == 0` (global Laguna: 6 % 3 = 0)
    /// and `head_dim <= FLASH_HEAD_DIM`. Same output contract as `fa2`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_flash_wmma_fa2_hg(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        batch: u32,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        q_offset: u32,
        scale: f32,
        swa_window: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0 {
            return Err(eyre!("gqa flash wmma fa2 hg: zero dim"));
        }
        if head_dim > FLASH_HEAD_DIM {
            return Err(eyre!("gqa flash wmma fa2 hg: head_dim={head_dim} > {FLASH_HEAD_DIM}"));
        }
        if n_head % n_kv_head != 0 {
            return Err(eyre!("gqa flash wmma fa2 hg: n_head={n_head} not div by n_kv_head={n_kv_head}"));
        }
        let kv_group = n_head / n_kv_head;
        let flash_hg_g = flash_hg_g();
        if kv_group % flash_hg_g != 0 {
            return Err(eyre!("gqa flash wmma fa2 hg: kv_group={kv_group} not div by HG_G={flash_hg_g}"));
        }
        let n_kv_total = q_offset + batch;
        let want_q = (batch * n_head * head_dim) as usize;
        if q.len() != want_q || out.len() != want_q {
            return Err(eyre!("gqa flash wmma fa2 hg: q/out len mismatch"));
        }
        let min_kv = kv_capacity as usize * (n_kv_head * head_dim) as usize;
        if k_cache.len() < min_kv || v_cache.len() < min_kv {
            return Err(eyre!("gqa flash wmma fa2 hg: kv cache too small"));
        }
        let function = self.module.get_function("gqa_attn_prefill_flash_wmma_fa2_hg")?;
        let n_subgroup = kv_group / flash_hg_g;
        let grid_x = n_kv_head * n_subgroup;
        let grid_y = batch.div_ceil(FLASH_HG_BR);
        let cfg = LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (FLASH_HG_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), k_cache.raw(), v_cache.raw(),
            n_head, n_kv_head, head_dim, q_offset, batch, n_kv_total, scale, swa_window, kv_capacity
        ])
    }

    /// FULL-GQA-group packed WMMA FA prefill (`..._fa2_hg_packed`, HGP_G=6). One
    /// WG owns ALL `kv_group` query heads of a KV head, amortizing the K/V load +
    /// softmax pass + barriers over the whole group (fa2_hg only packs 3 of 6).
    /// Requires `kv_group % FLASH_HGP_G == 0` (global Laguna: 6). Same contract as
    /// `fa2_hg`. Env-gated OFF (`LAGUNA_ATTN_HG_PACKED=1`) via the het path.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_flash_wmma_fa2_hg_packed(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        batch: u32,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        q_offset: u32,
        scale: f32,
        swa_window: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0 {
            return Err(eyre!("gqa flash wmma fa2 hg packed: zero dim"));
        }
        if head_dim > FLASH_HEAD_DIM {
            return Err(eyre!("gqa flash wmma fa2 hg packed: head_dim={head_dim} > {FLASH_HEAD_DIM}"));
        }
        if n_head % n_kv_head != 0 {
            return Err(eyre!("gqa flash wmma fa2 hg packed: n_head={n_head} not div by n_kv_head={n_kv_head}"));
        }
        let kv_group = n_head / n_kv_head;
        if kv_group % FLASH_HGP_G != 0 {
            return Err(eyre!("gqa flash wmma fa2 hg packed: kv_group={kv_group} not div by HGP_G={FLASH_HGP_G}"));
        }
        let n_kv_total = q_offset + batch;
        let want_q = (batch * n_head * head_dim) as usize;
        if q.len() != want_q || out.len() != want_q {
            return Err(eyre!("gqa flash wmma fa2 hg packed: q/out len mismatch"));
        }
        let min_kv = kv_capacity as usize * (n_kv_head * head_dim) as usize;
        if k_cache.len() < min_kv || v_cache.len() < min_kv {
            return Err(eyre!("gqa flash wmma fa2 hg packed: kv cache too small"));
        }
        let function = self.module.get_function("gqa_attn_prefill_flash_wmma_fa2_hg_packed")?;
        let n_subgroup = kv_group / FLASH_HGP_G;
        let grid_x = n_kv_head * n_subgroup;
        let grid_y = batch.div_ceil(FLASH_HG_BR);
        let cfg = LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (FLASH_HG_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), k_cache.raw(), v_cache.raw(),
            n_head, n_kv_head, head_dim, q_offset, batch, n_kv_total, scale, swa_window, kv_capacity
        ])
    }

    // ==================== FP8 (e4m3fn) KV-cache variants ====================
    // K/V cache is stored as 1-byte e4m3fn + per-(token,kv_head) f32 scale sidecar
    // (LAGUNA_FP8_KV). Same math/launch geometry as the f16 wrappers; the kernels
    // dequant during LDS/register staging via the native gfx1201 packed convert.

    /// FP8 head-grouped WMMA FA prefill (global layers). Mirrors
    /// [`prefill_flash_wmma_fa2_hg`] with e4m3fn K/V + scale sidecars.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_flash_wmma_fa2_hg_fp8(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u8>,
        v_cache: &DeviceBuffer<u8>,
        k_scale: &DeviceBuffer<f32>,
        v_scale: &DeviceBuffer<f32>,
        batch: u32,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        q_offset: u32,
        scale: f32,
        swa_window: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0 {
            return Err(eyre!("gqa flash wmma fa2 hg fp8: zero dim"));
        }
        if head_dim > FLASH_HEAD_DIM {
            return Err(eyre!("gqa flash wmma fa2 hg fp8: head_dim={head_dim} > {FLASH_HEAD_DIM}"));
        }
        if n_head % n_kv_head != 0 {
            return Err(eyre!("gqa flash wmma fa2 hg fp8: n_head={n_head} not div by n_kv_head={n_kv_head}"));
        }
        let kv_group = n_head / n_kv_head;
        let flash_hg_g = flash_hg_g();
        if kv_group % flash_hg_g != 0 {
            return Err(eyre!("gqa flash wmma fa2 hg fp8: kv_group={kv_group} not div by HG_G={flash_hg_g}"));
        }
        let n_kv_total = q_offset + batch;
        let want_q = (batch * n_head * head_dim) as usize;
        if q.len() != want_q || out.len() != want_q {
            return Err(eyre!("gqa flash wmma fa2 hg fp8: q/out len mismatch"));
        }
        let min_kv = kv_capacity as usize * (n_kv_head * head_dim) as usize;
        if k_cache.len() < min_kv || v_cache.len() < min_kv {
            return Err(eyre!("gqa flash wmma fa2 hg fp8: kv cache too small"));
        }
        let min_sc = kv_capacity as usize * n_kv_head as usize;
        if k_scale.len() < min_sc || v_scale.len() < min_sc {
            return Err(eyre!("gqa flash wmma fa2 hg fp8: scale sidecar too small"));
        }
        let function = self.module.get_function("gqa_attn_prefill_flash_wmma_fa2_hg_fp8")?;
        let n_subgroup = kv_group / flash_hg_g;
        let cfg = LaunchConfig {
            grid: (n_kv_head * n_subgroup, batch.div_ceil(FLASH_HG_BR), 1),
            block: (FLASH_HG_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), k_cache.raw(), v_cache.raw(), k_scale.raw(), v_scale.raw(),
            n_head, n_kv_head, head_dim, q_offset, batch, n_kv_total, scale, swa_window, kv_capacity
        ])
    }

    /// FP8 FA2 register-resident-O prefill (SWA layers). Mirrors
    /// [`prefill_flash_wmma_fa2`] (grid_mode 0) with e4m3fn K/V + scale sidecars.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_flash_wmma_fa2_fp8(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u8>,
        v_cache: &DeviceBuffer<u8>,
        k_scale: &DeviceBuffer<f32>,
        v_scale: &DeviceBuffer<f32>,
        batch: u32,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        q_offset: u32,
        scale: f32,
        swa_window: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0 {
            return Err(eyre!("gqa flash wmma fa2 fp8: zero dim"));
        }
        if head_dim > FLASH_HEAD_DIM || n_head % n_kv_head != 0 {
            return Err(eyre!("gqa flash wmma fa2 fp8: bad dims"));
        }
        let n_kv_total = q_offset + batch;
        let want_q = (batch * n_head * head_dim) as usize;
        if q.len() != want_q || out.len() != want_q {
            return Err(eyre!("gqa flash wmma fa2 fp8: q/out len mismatch"));
        }
        let min_kv = kv_capacity as usize * (n_kv_head * head_dim) as usize;
        if k_cache.len() < min_kv || v_cache.len() < min_kv {
            return Err(eyre!("gqa flash wmma fa2 fp8: kv cache too small"));
        }
        let min_sc = kv_capacity as usize * n_kv_head as usize;
        if k_scale.len() < min_sc || v_scale.len() < min_sc {
            return Err(eyre!("gqa flash wmma fa2 fp8: scale sidecar too small"));
        }
        let function = self.module.get_function("gqa_attn_prefill_flash_wmma_fa2_fp8")?;
        const WMMA_BLOCK: u32 = 128;
        let grid_mode: u32 = 0;
        let cfg = LaunchConfig {
            grid: (n_head, batch.div_ceil(FLASH_BR), 1),
            block: (WMMA_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), k_cache.raw(), v_cache.raw(), k_scale.raw(), v_scale.raw(),
            n_head, n_kv_head, head_dim, q_offset, batch, n_kv_total, scale, swa_window, kv_capacity, grid_mode
        ])
    }

    /// FP8 head-grouped split-KV decode. Mirrors [`single_query_splitkv_hg`]
    /// with e4m3fn K/V + scale sidecars.
    #[allow(clippy::too_many_arguments)]
    pub fn single_query_splitkv_hg_fp8(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        out_partial: &mut DeviceBuffer<f32>,
        m_partial: &mut DeviceBuffer<f32>,
        l_partial: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<u16>,
        k_cache: &DeviceBuffer<u8>,
        v_cache: &DeviceBuffer<u8>,
        k_scale: &DeviceBuffer<f32>,
        v_scale: &DeviceBuffer<f32>,
        n_head: u32,
        n_kv_head: u32,
        head_dim: u32,
        n_kv: u32,
        n_splits: u32,
        scale: f32,
        k_base: u32,
        kv_capacity: u32,
    ) -> eyre::Result<()> {
        if n_head == 0 || n_kv_head == 0 || head_dim == 0 || n_kv == 0 || n_splits == 0 {
            return Err(eyre!("gqa splitkv_hg fp8: zero dim"));
        }
        if head_dim > FLASH_HEAD_DIM || n_head % n_kv_head != 0 {
            return Err(eyre!("gqa splitkv_hg fp8: bad dims"));
        }
        let kv_group = n_head / n_kv_head;
        if kv_group > 12 {
            return Err(eyre!("gqa splitkv_hg fp8: kv_group={kv_group} exceeds DEC_KVG_MAX=12"));
        }
        let hd = head_dim as usize;
        let expected_q = n_head as usize * hd;
        if q.len() != expected_q || out.len() != expected_q {
            return Err(eyre!("gqa splitkv_hg fp8 q/out len"));
        }
        let expected_kv = kv_capacity as usize * n_kv_head as usize * hd;
        if k_cache.len() < expected_kv || v_cache.len() < expected_kv {
            return Err(eyre!("gqa splitkv_hg fp8 kv len"));
        }
        let expected_sc = kv_capacity as usize * n_kv_head as usize;
        if k_scale.len() < expected_sc || v_scale.len() < expected_sc {
            return Err(eyre!("gqa splitkv_hg fp8 scale sidecar too small"));
        }
        let need_part = n_head as usize * n_splits as usize * hd;
        let need_ml = n_head as usize * n_splits as usize;
        if out_partial.len() < need_part || m_partial.len() < need_ml || l_partial.len() < need_ml {
            return Err(eyre!("gqa splitkv_hg fp8 scratch too small"));
        }
        const DEC_BLOCK: u32 = 128;
        let fp = self.module.get_function("gqa_attn_decode_partial_hg_fp8")?;
        let cfg_p = LaunchConfig {
            grid: (n_kv_head, n_splits, 1),
            block: (DEC_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(fp, cfg_p, stream, [
            out_partial.raw(), m_partial.raw(), l_partial.raw(),
            q.raw(), k_cache.raw(), v_cache.raw(), k_scale.raw(), v_scale.raw(),
            n_head, n_kv_head, head_dim, n_kv, n_splits, scale, k_base, kv_capacity
        ])?;
        let fc = self.module.get_function("gqa_attn_decode_combine")?;
        let cfg_c = LaunchConfig {
            grid: (n_head, 1, 1),
            block: (DEC_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(fc, cfg_c, stream, [
            out.raw(), out_partial.raw(), m_partial.raw(), l_partial.raw(),
            n_head, head_dim, n_splits
        ])
    }
}
