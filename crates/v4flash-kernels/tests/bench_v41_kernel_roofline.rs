//! V4.1 per-kernel microbenchmark at the REAL production shapes, for
//! docs/v41/KERNEL_ROOFLINE.md. No model load — synthetic buffers through the
//! same `DeviceEngine` wrappers the forward pass uses, so the kernel, launch
//! geometry and argument contract are exactly production's.
//!
//! Every row records: HSA kernel name(s), shape, launches per layer, how many
//! layers per token/chunk run it, bytes moved (DRAM model: weights once,
//! activations once, outputs once), FLOPs, and event-timed p50/min. Weight
//! operands are cycled through a ring of copies totalling >= BENCH_RING_MB
//! (default 320 MB) so a 44 MB decode matvec cannot be served from the
//! 9070 XT's 64 MB Infinity Cache the way an isolated single-buffer loop
//! would (production streams ~178 MB of distinct weights per layer).
//!
//! Sections (BENCH_SECTION, comma list; default all):
//!   decode_dgpu   B=1 dGPU chain incl. dense attention at BENCH_CTX depth
//!   decode_igpu   B=1 iGPU MXFP4 MoE (6 experts, random picks per iter)
//!   prefill_dgpu  B=BENCH_B dGPU chain (encoder layer shapes; attention at
//!                 the ATTN_SCORES_STRIDE cap = the largest depth the batched
//!                 score kernel accepts without an indexer gather)
//!   prefill_igpu  B=BENCH_B iGPU MoE (group builder + kwide gate/up + down)
//!
//! Env: BENCH_CTX=8192[,32768] BENCH_B=512[,128] BENCH_ITERS=40 BENCH_WARMUP=5
//!      BENCH_N_EXPERT_BUF=384 (experts resident in the synthetic iGPU buffer)
//!      BENCH_JSON=<path> BENCH_RING_MB=320
//!
//!   HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c \
//!     cargo test --release --features v41 -p v4flash-kernels \
//!     --test bench_v41_kernel_roofline -- --ignored --nocapture
#![cfg(feature = "v41")]

use std::cell::Cell;
use std::io::Write;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::attention::{ATTN_MIXED_MAX_KEYS, ATTN_SCORES_STRIDE};
use v4flash_kernels::config::*;
use v4flash_kernels::het::batch_scratch::f16_pitch;
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;
use v4flash_kernels::sampler::{SAMPLER_N_WG, SAMPLER_TOPP_NEDGE};
use v4flash_kernels::RopeParams;

const Q8_BLOCK_BYTES: usize = 34; // 32 elems
const MXFP4_BLOCK_BYTES: usize = 17; // 32 elems
const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;
const K_SPLIT: u32 = 16;
const KV_CACHE_ROWS: usize = 1152; // SWA_WINDOW + B_MAX (state.rs)

fn pick(prefix: &str) -> Option<Device> {
    Device::all().ok()?.into_iter().find(|d| {
        d.properties().map(|p| p.gcn_arch_name.starts_with(prefix)).unwrap_or(false)
    })
}

fn env_list<T: std::str::FromStr>(name: &str, default: Vec<T>) -> Vec<T> {
    std::env::var(name)
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .filter(|v: &Vec<T>| !v.is_empty())
        .unwrap_or(default)
}

fn env_u<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn q8_bytes(rows: usize, k: usize) -> usize {
    rows * (k / 32) * Q8_BLOCK_BYTES
}
fn mxfp4_bytes(rows: usize, k: usize) -> usize {
    rows * (k / 32) * MXFP4_BLOCK_BYTES
}

/// Weight ring: `n` copies of an `elems`-element buffer, cycled per launch so
/// consecutive iterations never re-read the same lines.
struct Ring<T: Copy> {
    bufs: Vec<DeviceBuffer<T>>,
    i: Cell<usize>,
}
impl<T: Copy> Ring<T> {
    fn new(dev: i32, elems: usize, min_total_bytes: usize) -> eyre::Result<Self> {
        let bytes = elems * std::mem::size_of::<T>();
        let n = (min_total_bytes / bytes.max(1)).clamp(1, 64);
        let mut bufs = Vec::with_capacity(n);
        for _ in 0..n {
            let mut b: DeviceBuffer<T> = DeviceBuffer::new(dev, elems.max(1))?;
            b.fill_zero()?;
            bufs.push(b);
        }
        Ok(Self { bufs, i: Cell::new(0) })
    }
    fn next(&self) -> &DeviceBuffer<T> {
        let i = self.i.get();
        self.i.set((i + 1) % self.bufs.len());
        &self.bufs[i]
    }
    fn copies(&self) -> usize {
        self.bufs.len()
    }
}

struct Ctx {
    dev: Device,
    arch: String,
    label: &'static str,
    stream: Stream,
    e: DeviceEngine,
    iters: usize,
    warmup: usize,
    ring_min: usize,
    json: std::cell::RefCell<Option<std::fs::File>>,
    rope_comp: RopeParams,
    rope_dense: RopeParams,
}

impl Ctx {
    fn new(dev: Device, label: &'static str, json: Option<std::fs::File>) -> eyre::Result<Self> {
        dev.set_current()?;
        let arch = dev.properties()?.gcn_arch_name;
        let e = DeviceEngine::for_arch(dev, &arch)?;
        let stream = Stream::new(dev.id)?;
        // rope_for_layer (deepstrix-server/src/lib.rs): compressed layers use
        // theta 160000 + YaRN(16, 65536, 32/1); SWA-only layers plain 10000.
        let attn_factor = 1.0f32 / (1.0 + 0.1 * 16f32.ln());
        let rope_comp = RopeParams::from_dump_blob(&[160000.0, 1.0 / 16.0, 1.0, attn_factor, 32.0, 1.0], ROPE_ORIG_CTX)?;
        let rope_dense = RopeParams::from_dump_blob(&[10000.0, 1.0, 0.0, 1.0, 32.0, 1.0], 0)?;
        Ok(Self {
            dev,
            arch,
            label,
            stream,
            e,
            iters: env_u("BENCH_ITERS", 40usize),
            warmup: env_u("BENCH_WARMUP", 5usize),
            ring_min: env_u("BENCH_RING_MB", 320usize) * 1024 * 1024,
            json: std::cell::RefCell::new(json),
            rope_comp,
            rope_dense,
        })
    }

    fn buf<T: Copy>(&self, n: usize) -> eyre::Result<DeviceBuffer<T>> {
        let mut b: DeviceBuffer<T> = DeviceBuffer::new(self.dev.id, n.max(1))?;
        b.fill_zero()?;
        Ok(b)
    }
    fn ring(&self, bytes: usize) -> eyre::Result<Ring<u8>> {
        Ring::new(self.dev.id, bytes, self.ring_min)
    }
    fn i32s(&self, v: &[i32]) -> eyre::Result<DeviceBuffer<i32>> {
        let mut b: DeviceBuffer<i32> = DeviceBuffer::new(self.dev.id, v.len().max(1))?;
        b.copy_from_host(v)?;
        Ok(b)
    }
    fn f32s(&self, v: &[f32]) -> eyre::Result<DeviceBuffer<f32>> {
        let mut b: DeviceBuffer<f32> = DeviceBuffer::new(self.dev.id, v.len().max(1))?;
        b.copy_from_host(v)?;
        Ok(b)
    }

    /// Time `f` (event-bracketed, one sync per iter) and record a row.
    #[allow(clippy::too_many_arguments)]
    fn rec(
        &self,
        path: &str,
        name: &str,
        shape: &str,
        launches_per_layer: f64,
        layers: f64,
        bytes: f64,
        flops: f64,
        mut f: impl FnMut(&Stream) -> eyre::Result<()>,
    ) -> eyre::Result<f64> {
        for _ in 0..self.warmup {
            f(&self.stream)?;
        }
        self.stream.synchronize()?;
        let mut v = Vec::with_capacity(self.iters);
        for _ in 0..self.iters {
            let s = Event::new()?;
            let e = Event::new()?;
            s.record(&self.stream)?;
            f(&self.stream)?;
            e.record(&self.stream)?;
            self.stream.synchronize()?;
            v.push(Event::elapsed_ms(&s, &e)? as f64 * 1000.0);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = v[v.len() / 2];
        let min = v[0];
        let gbs = bytes / 1e9 / (p50 / 1e6);
        let gfs = flops / 1e9 / (p50 / 1e6);
        eprintln!(
            "{:<12} {:<5} {:<58} {:>9.1} us p50 {:>9.1} min | {:>7.1} GB/s {:>8.1} GFLOP/s | x{:<4} L{:<4} | {}",
            path, self.label, name, p50, min, gbs, gfs, launches_per_layer, layers, shape
        );
        if let Some(j) = self.json.borrow_mut().as_mut() {
            let _ = writeln!(
                j,
                "{{\"path\":\"{path}\",\"dev\":\"{}\",\"name\":\"{name}\",\"shape\":\"{}\",\"launches_per_layer\":{launches_per_layer},\"layers\":{layers},\"bytes\":{bytes},\"flops\":{flops},\"p50_us\":{p50},\"min_us\":{min}}}",
                self.label,
                shape.replace('"', "'")
            );
        }
        Ok(p50)
    }
}

fn lcg(s: &mut u64) -> u32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 33) as u32
}

/// `b` tokens × N_EXPERT_USED distinct expert ids in [0, n_exp).
fn random_selection(b: usize, n_exp: u32, seed: u64) -> Vec<i32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(b * N_EXPERT_USED);
    for _ in 0..b {
        let mut picks: Vec<i32> = Vec::with_capacity(N_EXPERT_USED);
        while picks.len() < N_EXPERT_USED {
            let e = (lcg(&mut s) % n_exp) as i32;
            if !picks.contains(&e) {
                picks.push(e);
            }
        }
        out.extend_from_slice(&picks);
    }
    out
}

// ===========================================================================
// decode, dGPU (B = 1)
// ===========================================================================
fn decode_dgpu(c: &Ctx, ctxs: &[u32]) -> eyre::Result<()> {
    let p = "decode";
    let n_embd = N_EMBD as usize;
    let hc_dim = HC_DIM as usize;
    let hcm = HC_MIX_DIM as usize;
    let f4 = 4.0f64;

    // ---- mHC pre chain (runs twice per layer: attn + ffn) ----
    {
        let residual = c.buf::<f32>(hc_dim)?;
        let mut inv = c.buf::<f32>(1)?;
        let mut partials = c.buf::<f32>(64)?;
        let mut mix = c.buf::<f32>(hcm)?;
        let mut split = c.buf::<f32>(hcm)?;
        let mut mv_partials = c.buf::<f32>(64 * hcm)?;
        let hc_fn = c.ring(hcm * hc_dim * 2)?;
        let scale = c.buf::<f32>(3)?;
        let base = c.buf::<f32>(hcm)?;
        let mut cur = c.buf::<f32>(n_embd)?;
        let mut normed = c.buf::<f32>(n_embd)?;
        let norm_w = c.buf::<f32>(n_embd)?;
        let mut carry = c.buf::<f32>(hcm)?;
        let ksplit = HC_DIM / 1024;
        c.rec(p, "rms_norm_partial_sum_sq+rms_norm_finalize_inv", "x[20480] f32, 16 WGs", 2.0, 40.0, hc_dim as f64 * f4, 2.0 * hc_dim as f64, |s| {
            c.e.rms_nw_mw.launch_inv_only(s, &mut inv, &residual, &mut partials, HC_DIM, 16, RMS_EPS)
        })?;
        c.rec(p, "f16_matvec_narrow_ksplit_partial_v8+reduce (hc_fn)", "W[24x20480] f16 + x[20480], ksplit=20", 2.0, 40.0, (hcm * hc_dim * 2 + hc_dim * 4) as f64, 2.0 * hcm as f64 * hc_dim as f64, |s| {
            c.e.f16.matvec_narrow_ksplit_pre_scaled(s, &mut mix, hc_fn.next(), &residual, &inv, &mut mv_partials, HC_MIX_DIM, HC_DIM, ksplit)
        })?;
        c.rec(p, "hc_sinkhorn_par", "mix[24] -> split[24], 20 iters", 2.0, 40.0, 200.0, 2000.0, |s| {
            c.e.hc_sinkhorn.launch(s, &mut split, &mix, &scale, &base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)
        })?;
        c.rec(p, "hc_weighted_sum", "x[4x5120] w[4] -> [5120]", 2.0, 40.0, (hc_dim + n_embd) as f64 * f4, 2.0 * hc_dim as f64, |s| {
            c.e.hc_weighted.launch(s, &mut cur, &residual, &carry, N_EMBD, N_HC)
        })?;
        c.rec(p, "rms_norm_weighted (5120)", "x[5120] w[5120]", 2.0, 40.0, 3.0 * n_embd as f64 * f4, 3.0 * n_embd as f64, |s| {
            c.e.rms_w.launch_weighted(s, &mut normed, &cur, &norm_w, N_EMBD, RMS_EPS)
        })?;
        c.rec(p, "memcpy D2D hc_pre_carry (4 f32)", "4 f32", 2.0, 40.0, 32.0, 0.0, |s| {
            let cur_pre = split.slice_view(0, N_HC as usize);
            carry.slice_view_mut(0, N_HC as usize).copy_from_buffer_async(&cur_pre, s)
        })?;
    }

    // ---- q chain ----
    {
        let x = c.buf::<f32>(n_embd)?;
        let mut xq = c.buf::<i8>(n_embd)?;
        let mut xs = c.buf::<f32>(n_embd / 32)?;
        let mut qr = c.buf::<f32>(N_LORA_Q as usize)?;
        let mut qr_normed = c.buf::<f32>(N_LORA_Q as usize)?;
        let mut qr_xq = c.buf::<i8>(N_LORA_Q as usize)?;
        let mut qr_xs = c.buf::<f32>(N_LORA_Q as usize / 32)?;
        let q_norm_w = c.buf::<f32>(N_LORA_Q as usize)?;
        let mut q = c.buf::<f32>(Q_FLAT as usize)?;
        let mut q_normed = c.buf::<f32>(Q_FLAT as usize)?;
        let wqa = c.ring(q8_bytes(N_LORA_Q as usize, n_embd))?;
        let wqb = c.ring(q8_bytes(Q_FLAT as usize, N_LORA_Q as usize))?;
        c.rec(p, "q8_0_quantize_f32 (5120)", "x[5120] -> i8 + scale", 2.0, 40.0, n_embd as f64 * 5.125, n_embd as f64, |s| {
            c.e.q8.quantize_input(s, &mut xq, &mut xs, &x, N_EMBD)
        })?;
        let wb = q8_bytes(N_LORA_Q as usize, n_embd) as f64;
        c.rec(p, "q8_0_gemv_warp8 wq_a", &format!("W[1280x5120] Q8_0 {:.1} MB, ring x{}", wb / 1e6, wqa.copies()), 1.0, 40.0, wb + n_embd as f64 * 1.125 + N_LORA_Q as f64 * 4.0, 2.0 * N_LORA_Q as f64 * n_embd as f64, |s| {
            c.e.q8.matvec(s, &mut qr, wqa.next(), &xq, &xs, N_LORA_Q, N_EMBD)
        })?;
        c.rec(p, "rms_norm_weighted_quantize_q8 (1280)", "qr[1280]", 1.0, 40.0, N_LORA_Q as f64 * 9.125, 3.0 * N_LORA_Q as f64, |s| {
            c.e.rms_w.launch_weighted_quantize_q8(s, &mut qr_normed, &mut qr_xq, &mut qr_xs, &qr, &q_norm_w, N_LORA_Q, RMS_EPS)
        })?;
        let wb = q8_bytes(Q_FLAT as usize, N_LORA_Q as usize) as f64;
        c.rec(p, "q8_0_gemv_warp8 wq_b", &format!("W[32768x1280] Q8_0 {:.1} MB, ring x{}", wb / 1e6, wqb.copies()), 1.0, 40.0, wb + N_LORA_Q as f64 * 1.125 + Q_FLAT as f64 * 4.0, 2.0 * Q_FLAT as f64 * N_LORA_Q as f64, |s| {
            c.e.q8.matvec(s, &mut q, wqb.next(), &qr_xq, &qr_xs, Q_FLAT, N_LORA_Q)
        })?;
        c.rec(p, "memcpy D2D q->q_normed (no q-norm in V4.1)", "32768 f32", 1.0, 40.0, 2.0 * Q_FLAT as f64 * 4.0, 0.0, |s| {
            q_normed.copy_from_buffer_async(&q, s)
        })?;
        let rope = c.rope_comp;
        c.rec(p, "rope_tail fwd q", "[64 heads x 512], rot 64", 1.0, 40.0, 2.0 * 64.0 * 64.0 * 4.0, 64.0 * 64.0 * 6.0, |s| {
            c.e.rope.launch_forward(s, &mut q_normed, N_HEAD, N_HEAD_DIM, N_ROT, 4000, &rope)
        })?;
    }

    // ---- kv chain ----
    {
        let xq = c.buf::<i8>(n_embd)?;
        let xs = c.buf::<f32>(n_embd / 32)?;
        let mut kv_raw = c.buf::<f32>(N_HEAD_DIM as usize)?;
        let mut kv_normed = c.buf::<f32>(N_HEAD_DIM as usize)?;
        let kv_norm_w = c.buf::<f32>(N_HEAD_DIM as usize)?;
        let wkv = c.ring(q8_bytes(N_HEAD_DIM as usize, n_embd))?;
        let mut cache = c.buf::<u16>(KV_CACHE_ROWS * N_HEAD_DIM as usize)?;
        let wb = q8_bytes(N_HEAD_DIM as usize, n_embd) as f64;
        c.rec(p, "q8_0_gemv_warp8 wkv", &format!("W[512x5120] Q8_0 {:.1} MB, ring x{}", wb / 1e6, wkv.copies()), 1.0, 40.0, wb + n_embd as f64 * 1.125 + 2048.0, 2.0 * 512.0 * n_embd as f64, |s| {
            c.e.q8.matvec(s, &mut kv_raw, wkv.next(), &xq, &xs, N_HEAD_DIM, N_EMBD)
        })?;
        c.rec(p, "rms_norm_weighted (512)", "kv[512]", 1.0, 40.0, 3.0 * 512.0 * 4.0, 1536.0, |s| {
            c.e.rms_w.launch_weighted(s, &mut kv_normed, &kv_raw, &kv_norm_w, N_HEAD_DIM, RMS_EPS)
        })?;
        let rope = c.rope_comp;
        c.rec(p, "rope_tail fwd kv", "[1 x 512], rot 64", 1.0, 40.0, 512.0, 384.0, |s| {
            c.e.rope.launch_forward(s, &mut kv_normed, 1, N_HEAD_DIM, N_ROT, 4000, &rope)
        })?;
        c.rec(p, "fp8_act_quant_inplace (window KV)", "[1 x 512] block 32", 1.0, 40.0, 4096.0, 2048.0, |s| {
            c.e.fp4kv.launch_fp8_window(s, &mut kv_normed, 1, N_HEAD_DIM)
        })?;
        c.rec(p, "f16_roundtrip (512)", "[512]", 1.0, 40.0, 4096.0, 512.0, |s| c.e.f16rt.launch(s, &mut kv_normed, N_HEAD_DIM))?;
        c.rec(p, "kv_cache_append", "1 row x 512 -> f16 cache", 1.0, 40.0, 2048.0 + 1024.0, 512.0, |s| {
            c.e.kv_append.launch(s, &mut cache, &kv_normed, 200, KV_CACHE_ROWS as u32, N_HEAD_DIM)
        })?;
    }

    // ---- compressor: ratio 2 (layers 2, 8, 14) every token; boundary every 2nd ----
    {
        let cw = 512usize; // comp_width at ratio 2
        let x = c.buf::<f32>(n_embd)?;
        let mut kv_cur = c.buf::<f32>(cw)?;
        let mut sc_cur = c.buf::<f32>(cw)?;
        let wkv = c.ring(cw * n_embd * 2)?;
        let wgate = c.ring(cw * n_embd * 2)?;
        let mut state_kv = c.buf::<f32>(2 * cw)?;
        let mut state_sc = c.buf::<f32>(2 * cw)?;
        let ape = c.buf::<u8>(cw * 2 * 2)?;
        let mut pooled = c.buf::<f32>(N_HEAD_DIM as usize)?;
        let mut row = c.buf::<f32>(N_HEAD_DIM as usize)?;
        let norm_w = c.buf::<f32>(N_HEAD_DIM as usize)?;
        let mut comp_kv = c.buf::<u16>(4096 * N_HEAD_DIM as usize)?;
        let wb = (2 * cw * n_embd * 2) as f64;
        c.rec(p, "f16_matvec_pair (compressor wkv+wgate, ratio 2)", &format!("2x W[512x5120] f16 {:.1} MB, ring x{}", wb / 1e6, wkv.copies()), 1.0, 3.0, wb + n_embd as f64 * 4.0, 2.0 * 2.0 * 512.0 * n_embd as f64, |s| {
            c.e.f16.matvec_pair(s, &mut kv_cur, &mut sc_cur, wkv.next(), wgate.next(), &x, cw as u32, N_EMBD)
        })?;
        c.rec(p, "compressor_state_write (ratio 2)", "row [512]+[512] -> state", 1.0, 3.0, 4.0 * 512.0 * 4.0, 1024.0, |s| {
            c.e.compressor_state_write.launch(s, &mut state_kv, &mut state_sc, &kv_cur, &sc_cur, &ape, cw as u32, 1, 1)
        })?;
        c.rec(p, "compressor_pool (ratio 2, boundary)", "state[2x512] -> pooled[512]", 0.5, 3.0, 5.0 * 512.0 * 4.0, 4.0 * 512.0 * 4.0, |s| {
            c.e.compressor_pool.launch(s, &mut pooled, &state_kv, &state_sc, N_HEAD_DIM, 2)
        })?;
        c.rec(p, "rms_norm_weighted (comp row 512)", "[512]", 0.5, 3.0, 6144.0, 1536.0, |s| {
            c.e.rms_w.launch_weighted(s, &mut row, &pooled, &norm_w, N_HEAD_DIM, RMS_EPS)
        })?;
        let rope = c.rope_comp;
        c.rec(p, "rope_tail fwd comp row", "[1 x 512]", 0.5, 3.0, 512.0, 384.0, |s| {
            c.e.rope.launch_forward(s, &mut row, 1, N_HEAD_DIM, N_ROT, 2000, &rope)
        })?;
        c.rec(p, "fp4_kv_quant_inplace (comp row)", "[1 x 512] E2M1/E4M3 per 16", 0.5, 3.0, 4096.0, 4096.0, |s| {
            c.e.fp4kv.launch(s, &mut row, 1, N_HEAD_DIM)
        })?;
        c.rec(p, "comp_kv_append", "[512] f32 -> f16 store", 0.5, 3.0, 3072.0, 512.0, |s| {
            c.e.comp_kv_append.launch(s, &mut comp_kv, &row, 1000, N_HEAD_DIM)
        })?;
        // ratio 1 (layer 20): one f16 matvec, norm, rope, fp4, append every token.
        let wkv1 = c.ring(cw * n_embd * 2)?;
        let wb1 = (cw * n_embd * 2) as f64;
        c.rec(p, "f16_matvec_wide_vec (compressor wkv, ratio 1, L20)", &format!("W[512x5120] f16 {:.1} MB, ring x{}", wb1 / 1e6, wkv1.copies()), 1.0, 1.0, wb1 + n_embd as f64 * 4.0, 2.0 * 512.0 * n_embd as f64, |s| {
            c.e.f16.matvec(s, &mut pooled, wkv1.next(), &x, cw as u32, N_EMBD)
        })?;
    }

    // ---- attention: SWA layers 0-1, dense compressed attention layers 2-39 ----
    {
        let n_head = N_HEAD as usize;
        let hd = N_HEAD_DIM as usize;
        let q = c.buf::<f32>(n_head * hd)?;
        let raw_kv = c.buf::<u16>(128 * hd)?;
        let sinks = c.buf::<f32>(n_head)?;
        let mut heads = c.buf::<f32>(n_head * hd)?;
        c.rec(p, "attention_swa (L0-1)", "64 heads x 128 keys x 512", 1.0, 2.0, (n_head * hd * 4 + 128 * hd * 2 + n_head * hd * 4) as f64, 64.0 * 128.0 * 512.0 * 4.0, |s| {
            c.e.attn_swa.launch(s, &mut heads, &q, &raw_kv, &sinks, N_HEAD, N_HEAD_DIM, 128)
        })?;
        let mut scores = c.buf::<f32>(n_head * ATTN_MIXED_MAX_KEYS as usize)?;
        let mut inv = c.buf::<f32>(n_head)?;
        let mut partials = c.buf::<f32>(K_SPLIT as usize * n_head * hd)?;
        for &ctx in ctxs {
            // ratio-2 stores (layers 2-19: 18 layers) hold ctx/2 rows; the
            // ratio-1 store (layers 20-39: 20 layers) holds ctx rows. Reuse
            // layers read the SAME store back to back, so production sees it
            // cache-warm; a cold ring bounds the other end.
            for &(n_comp, layers, tag) in &[(ctx / 2, 18.0f64, "ratio2"), (ctx, 20.0, "ratio1"), (512u32, 38.0, "top512-equiv")] {
                if n_comp + 128 > ATTN_MIXED_MAX_KEYS {
                    continue;
                }
                if tag == "top512-equiv" && ctx != ctxs[0] {
                    continue;
                }
                let n_total = 128 + n_comp;
                let comp_bytes = n_comp as usize * hd * 2;
                let comp_elems = n_comp as usize * hd;
                for &(warm, wtag) in &[(true, "warm"), (false, "cold")] {
                    if tag == "top512-equiv" && !warm {
                        continue;
                    }
                    let ring: Ring<u16> = if warm { Ring::new(c.dev.id, comp_elems, 1)? } else { Ring::new(c.dev.id, comp_elems, c.ring_min)? };
                    let shape = format!("64 heads, n_raw=128 n_comp={n_comp} ({tag} @ctx {ctx}), comp store {wtag} x{}", ring.copies());
                    let sb = n_head as f64 * n_total as f64 * 4.0;
                    let fl = 64.0 * n_total as f64 * 512.0 * 2.0;
                    c.rec(p, &format!("attention_mixed_score_b1_htiled_wmma [{tag},{wtag}]"), &shape, 1.0, layers, (n_head * hd * 4 + 128 * hd * 2) as f64 + comp_bytes as f64 + sb, fl, |s| {
                        c.e.attn_mixed.launch_score_b1_htiled_wmma(s, &mut scores, &q, &raw_kv, Some(ring.next()), 128, 0, n_comp, N_HEAD, N_HEAD_DIM, n_total)
                    })?;
                    c.rec(p, &format!("attention_mixed_softmax_only [{tag},{wtag}]"), &shape, 1.0, layers, 2.0 * sb, 5.0 * n_head as f64 * n_total as f64, |s| {
                        c.e.attn_mixed.launch_softmax_only(s, &mut scores, &sinks, &mut inv, N_HEAD, 128, n_comp)
                    })?;
                    c.rec(p, &format!("attention_mixed_wsum_b1_htiled_ksplit_ldsv [{tag},{wtag}]"), &shape, 1.0, layers, sb + comp_bytes as f64 + (128 * hd * 2) as f64 + (K_SPLIT as usize * n_head * hd * 4) as f64, fl, |s| {
                        c.e.attn_mixed.launch_wsum_b1_htiled_ksplit_ldsv(s, &mut partials, &scores, &raw_kv, Some(ring.next()), N_HEAD, N_HEAD_DIM, 128, n_comp, K_SPLIT)
                    })?;
                    c.rec(p, &format!("attention_mixed_reduce_partials_apply_inv [{tag},{wtag}]"), &shape, 1.0, layers, (K_SPLIT as usize * n_head * hd * 4 + n_head * hd * 4) as f64, (K_SPLIT as usize * n_head * hd) as f64, |s| {
                        c.e.attn_mixed.launch_reduce_partials_apply_inv(s, &mut heads, &partials, &inv, N_HEAD, N_HEAD_DIM, K_SPLIT)
                    })?;
                }
            }
        }
    }

    // ---- output projection ----
    {
        let mut heads = c.buf::<f32>(Q_FLAT as usize)?;
        let mut hxq = c.buf::<i8>(Q_FLAT as usize)?;
        let mut hxs = c.buf::<f32>(Q_FLAT as usize / 32)?;
        let mut low = c.buf::<f32>(OUT_LOW as usize)?;
        let mut lxq = c.buf::<i8>(OUT_LOW as usize)?;
        let mut lxs = c.buf::<f32>(OUT_LOW as usize / 32)?;
        let mut out = c.buf::<f32>(n_embd)?;
        let woa = c.ring(q8_bytes((N_GROUPS * RANK) as usize, GROUP_DIM as usize))?;
        let wob = c.ring(q8_bytes(n_embd, OUT_LOW as usize))?;
        let rope = c.rope_comp;
        c.rec(p, "rope_tail inverse heads", "[64 x 512], rot 64", 1.0, 40.0, 2.0 * 64.0 * 64.0 * 4.0, 64.0 * 64.0 * 6.0, |s| {
            c.e.rope.launch_inverse(s, &mut heads, N_HEAD, N_HEAD_DIM, N_ROT, 4000, &rope)
        })?;
        c.rec(p, "q8_0_quantize_f32 (32768)", "heads[32768]", 1.0, 40.0, Q_FLAT as f64 * 5.125, Q_FLAT as f64, |s| {
            c.e.q8.quantize_input(s, &mut hxq, &mut hxs, &heads, Q_FLAT)
        })?;
        let wb = q8_bytes((N_GROUPS * RANK) as usize, GROUP_DIM as usize) as f64;
        c.rec(p, "q8_0_grouped_gemv wo_a", &format!("8 x W[1024x4096] Q8_0 {:.1} MB, ring x{}", wb / 1e6, woa.copies()), 1.0, 40.0, wb + Q_FLAT as f64 * 1.125 + OUT_LOW as f64 * 4.0, 2.0 * 8.0 * 1024.0 * 4096.0, |s| {
            c.e.q8_grouped.matvec_grouped(s, &mut low, woa.next(), &hxq, &hxs, GROUP_DIM, RANK, N_GROUPS)
        })?;
        c.rec(p, "q8_0_quantize_f32 (8192)", "low[8192]", 1.0, 40.0, OUT_LOW as f64 * 5.125, OUT_LOW as f64, |s| {
            c.e.q8.quantize_input(s, &mut lxq, &mut lxs, &low, OUT_LOW)
        })?;
        let wb = q8_bytes(n_embd, OUT_LOW as usize) as f64;
        c.rec(p, "q8_0_gemv_warp8 wo_b", &format!("W[5120x8192] Q8_0 {:.1} MB, ring x{}", wb / 1e6, wob.copies()), 1.0, 40.0, wb + OUT_LOW as f64 * 1.125 + n_embd as f64 * 4.0, 2.0 * n_embd as f64 * OUT_LOW as f64, |s| {
            c.e.q8.matvec(s, &mut out, wob.next(), &lxq, &lxs, N_EMBD, OUT_LOW)
        })?;
    }

    // ---- hc_post (attn) + ffn_combine (vec_add + hc_post) ----
    {
        let mut out_hc = c.buf::<f32>(hc_dim)?;
        let block_out = c.buf::<f32>(n_embd)?;
        let residual = c.buf::<f32>(hc_dim)?;
        let split = c.buf::<f32>(hcm)?;
        let mut moe = c.buf::<f32>(n_embd)?;
        let shared = c.buf::<f32>(n_embd)?;
        c.rec(p, "hc_post (from split)", "residual[4x5120] + block[5120] -> [4x5120]", 2.0, 40.0, (2 * hc_dim + n_embd) as f64 * f4, 2.0 * hc_dim as f64 * 5.0, |s| {
            c.e.hc_post.launch_from_split(s, &mut out_hc, &block_out, &residual, &split, N_HC, N_EMBD, N_HC)
        })?;
        c.rec(p, "vec_add_inplace (5120)", "[5120] += [5120]", 1.0, 40.0, 3.0 * n_embd as f64 * f4, n_embd as f64, |s| {
            c.e.vec_add.launch(s, &mut moe, &shared, N_EMBD)
        })?;
    }

    // ---- router + shared expert ----
    {
        let x = c.buf::<f32>(n_embd)?;
        let mut logits = c.buf::<f32>(N_EXPERT as usize)?;
        let gate_w = c.ring(N_EXPERT as usize * n_embd * 2)?;
        let bias = c.buf::<f32>(N_EXPERT as usize)?;
        let mut sel = c.buf::<i32>(N_EXPERT_USED)?;
        let mut ew = c.buf::<f32>(N_EXPERT_USED)?;
        let wb = (N_EXPERT as usize * n_embd * 2) as f64;
        c.rec(p, "f16_matvec_wide_vec router gate", &format!("W[384x5120] f16 {:.1} MB, ring x{}", wb / 1e6, gate_w.copies()), 1.0, 40.0, wb + n_embd as f64 * 4.0, 2.0 * 384.0 * n_embd as f64, |s| {
            c.e.f16.matvec(s, &mut logits, gate_w.next(), &x, N_EXPERT, N_EMBD)
        })?;
        c.rec(p, "router_topk", "384 logits -> top-6", 1.0, 40.0, 384.0 * 8.0, 384.0 * 20.0, |s| {
            c.e.router_topk.launch(s, &mut sel, &mut ew, &logits, Some(&bias), N_EXPERT, N_EXPERT_USED as u32, EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS)
        })?;
        let xq = c.buf::<i8>(n_embd)?;
        let xs = c.buf::<f32>(n_embd / 32)?;
        let mut g = c.buf::<f32>(N_FF_SHARED as usize)?;
        let mut u = c.buf::<f32>(N_FF_SHARED as usize)?;
        let mut mid = c.buf::<f32>(N_FF_SHARED as usize)?;
        let mut mxq = c.buf::<i8>(N_FF_SHARED as usize)?;
        let mut mxs = c.buf::<f32>(N_FF_SHARED as usize / 32)?;
        let mut out = c.buf::<f32>(n_embd)?;
        let wg = c.ring(q8_bytes(N_FF_SHARED as usize, n_embd))?;
        let wu = c.ring(q8_bytes(N_FF_SHARED as usize, n_embd))?;
        let wd = c.ring(q8_bytes(n_embd, N_FF_SHARED as usize))?;
        let wb = q8_bytes(N_FF_SHARED as usize, n_embd) as f64;
        c.rec(p, "q8_0_gemv_warp8 shared gate", &format!("W[2304x5120] Q8_0 {:.1} MB, ring x{}", wb / 1e6, wg.copies()), 1.0, 40.0, wb + n_embd as f64 * 1.125 + 2304.0 * 4.0, 2.0 * 2304.0 * n_embd as f64, |s| {
            c.e.q8.matvec(s, &mut g, wg.next(), &xq, &xs, N_FF_SHARED, N_EMBD)
        })?;
        c.rec(p, "q8_0_gemv_warp8 shared up", &format!("W[2304x5120] Q8_0 {:.1} MB, ring x{}", wb / 1e6, wu.copies()), 1.0, 40.0, wb + n_embd as f64 * 1.125 + 2304.0 * 4.0, 2.0 * 2304.0 * n_embd as f64, |s| {
            c.e.q8.matvec(s, &mut u, wu.next(), &xq, &xs, N_FF_SHARED, N_EMBD)
        })?;
        c.rec(p, "swiglu (2304)", "[2304]", 1.0, 40.0, 3.0 * 2304.0 * 4.0, 2304.0 * 8.0, |s| {
            c.e.swiglu.launch_clamped(s, &mut mid, &g, &u, N_FF_SHARED, SWIGLU_CLAMP_EXP)
        })?;
        c.rec(p, "q8_0_quantize_f32 (2304)", "[2304]", 1.0, 40.0, 2304.0 * 5.125, 2304.0, |s| {
            c.e.q8.quantize_input(s, &mut mxq, &mut mxs, &mid, N_FF_SHARED)
        })?;
        c.rec(p, "q8_0_gemv_warp8 shared down", &format!("W[5120x2304] Q8_0 {:.1} MB, ring x{}", wb / 1e6, wd.copies()), 1.0, 40.0, wb + 2304.0 * 1.125 + n_embd as f64 * 4.0, 2.0 * 2304.0 * n_embd as f64, |s| {
            c.e.q8.matvec(s, &mut out, wd.next(), &mxq, &mxs, N_EMBD, N_FF_SHARED)
        })?;
    }

    // ---- Engram (layers 1, 14) ----
    {
        let ein = ENGRAM_IN as usize;
        let eout = ENGRAM_OUT as usize;
        let rows = c.buf::<f32>(ein)?;
        let mut xq = c.buf::<i8>(ein)?;
        let mut xs = c.buf::<f32>(ein / 32)?;
        let mut kv = c.buf::<f32>(eout)?;
        let mut h = c.buf::<f32>(hc_dim)?;
        let qk = c.buf::<f32>(hc_dim)?;
        let wkv = c.ring(q8_bytes(eout, ein))?;
        c.rec(p, "q8_0_quantize_f32 (engram 6144)", "[6144]", 1.0, 2.0, ein as f64 * 5.125, ein as f64, |s| {
            c.e.q8.quantize_input(s, &mut xq, &mut xs, &rows, ENGRAM_IN)
        })?;
        let wb = q8_bytes(eout, ein) as f64;
        c.rec(p, "q8_0_gemv_warp8 engram wkv", &format!("W[25600x6144] Q8_0 {:.1} MB, ring x{}", wb / 1e6, wkv.copies()), 1.0, 2.0, wb + ein as f64 * 1.125 + eout as f64 * 4.0, 2.0 * eout as f64 * ein as f64, |s| {
            c.e.q8.matvec(s, &mut kv, wkv.next(), &xq, &xs, ENGRAM_OUT, ENGRAM_IN)
        })?;
        c.rec(p, "engram_gate_add", "h[4x5120] kv[25600] qk[4x5120]", 1.0, 2.0, (2 * hc_dim + eout + hc_dim) as f64 * f4, 4.0 * n_embd as f64 * 8.0, |s| {
            c.e.engram_gate.launch(s, &mut h, &kv, &qk, N_HC, N_EMBD, ENGRAM_OUT, N_HC * N_EMBD, RMS_EPS, 1)
        })?;
    }

    // ---- head + sampler (once per token) ----
    {
        let residual = c.buf::<f32>(hc_dim)?;
        let carry = c.buf::<f32>(hcm)?;
        let mut embd = c.buf::<f32>(n_embd)?;
        let mut normed = c.buf::<f32>(n_embd)?;
        let norm_w = c.buf::<f32>(n_embd)?;
        let mut xq = c.buf::<i8>(n_embd)?;
        let mut xs = c.buf::<f32>(n_embd / 32)?;
        let nv = N_VOCAB as usize;
        let mut logits = c.buf::<f32>(nv)?;
        let head = c.ring(q8_bytes(nv, n_embd))?;
        c.rec(p, "hc_weighted_sum (head collapse)", "[4x5120] -> [5120]", 1.0, 1.0, (hc_dim + n_embd) as f64 * f4, 2.0 * hc_dim as f64, |s| {
            c.e.hc_weighted.launch(s, &mut embd, &residual, &carry, N_EMBD, N_HC)
        })?;
        c.rec(p, "rms_norm_weighted (head 5120)", "[5120]", 1.0, 1.0, 3.0 * n_embd as f64 * f4, 3.0 * n_embd as f64, |s| {
            c.e.rms_w.launch_weighted(s, &mut normed, &embd, &norm_w, N_EMBD, RMS_EPS)
        })?;
        c.rec(p, "q8_0_quantize_f32 (head 5120)", "[5120]", 1.0, 1.0, n_embd as f64 * 5.125, n_embd as f64, |s| {
            c.e.q8.quantize_input(s, &mut xq, &mut xs, &normed, N_EMBD)
        })?;
        let wb = q8_bytes(nv, n_embd) as f64;
        c.rec(p, "q8_0_gemv_warp8 head", &format!("W[129280x5120] Q8_0 {:.1} MB, ring x{}", wb / 1e6, head.copies()), 1.0, 1.0, wb + n_embd as f64 * 1.125 + nv as f64 * 4.0, 2.0 * nv as f64 * n_embd as f64, |s| {
            c.e.q8.matvec(s, &mut logits, head.next(), &xq, &xs, N_VOCAB, N_EMBD)
        })?;
        let mut next = c.buf::<i32>(1)?;
        let mut pmax = c.buf::<f32>(SAMPLER_N_WG as usize)?;
        let mut pz = c.buf::<f32>(SAMPLER_N_WG as usize)?;
        let mut mass = c.buf::<f32>((SAMPLER_N_WG * SAMPLER_TOPP_NEDGE) as usize)?;
        let mut bracket = c.buf::<f32>(2)?;
        let mut thr = c.buf::<f32>(1)?;
        let u01 = c.f32s(&[0.37])?;
        c.rec(p, "sampler multinomial top_p=0.95 (logits_max/expsum/topp_mass/bracket/sample)", "129280 logits", 1.0, 1.0, 4.0 * nv as f64 * 4.0, 6.0 * nv as f64, |s| {
            c.e.sampler.launch_multinomial_topp(s, &mut next, &logits, &mut pmax, &mut pz, &mut mass, &mut bracket, &mut thr, &u01, N_VOCAB, 1.0, 0.0, 0.95)
        })?;
        c.rec(p, "sampler argmax", "129280 logits", 1.0, 1.0, nv as f64 * 4.0, nv as f64, |s| {
            c.e.sampler.launch_argmax(s, &mut next, &logits, N_VOCAB)
        })?;
    }
    Ok(())
}

// ===========================================================================
// decode, iGPU (B = 1): MXFP4 MoE over 6 random experts per iteration
// ===========================================================================
fn decode_igpu(c: &Ctx, n_exp_buf: u32) -> eyre::Result<()> {
    let p = "decode";
    let n_embd = N_EMBD as usize;
    let gbpe = mxfp4_bytes(N_FF_EXP as usize, n_embd);
    let dbpe = mxfp4_bytes(n_embd, N_FF_EXP as usize);
    eprintln!("igpu decode MoE: {} experts resident in the synthetic buffer ({:.2} GB), random 6 per iter", n_exp_buf, 3.0 * n_exp_buf as f64 * gbpe as f64 / 1e9);
    let gate = c.buf::<u8>(n_exp_buf as usize * gbpe)?;
    let up = c.buf::<u8>(n_exp_buf as usize * gbpe)?;
    let down = c.buf::<u8>(n_exp_buf as usize * dbpe)?;
    let x = c.buf::<f32>(n_embd)?;
    let mut xq = c.buf::<u8>(BLOCKS_Q8K_GATE_IN as usize * BLOCK_Q8_K_BYTES)?;
    let mut mid = c.buf::<f32>(N_EXPERT_USED * N_FF_EXP as usize)?;
    let mut midq = c.buf::<u8>(N_EXPERT_USED * BLOCKS_Q8K_DOWN_IN as usize * BLOCK_Q8_K_BYTES)?;
    let mut out = c.buf::<f32>(n_embd)?;
    let ew = c.f32s(&[0.25; N_EXPERT_USED])?;
    // remap[e] = -e-1: every expert is an iGPU slot (== id); nothing on the dGPU.
    let remap_h: Vec<i32> = (0..N_EXPERT as i32).map(|e| -e - 1).collect();
    let remap = c.i32s(&remap_h)?;
    // A pool of random selections, one per iteration.
    let n_sel = 256usize;
    let sel_all = random_selection(n_sel, n_exp_buf, 0x1234);
    let sel_dev = c.i32s(&sel_all)?;
    let sel_i = Cell::new(0usize);
    let next_sel = || {
        let i = sel_i.get();
        sel_i.set((i + 1) % n_sel);
        sel_dev.slice_view(i * N_EXPERT_USED, N_EXPERT_USED)
    };
    c.rec(p, "q8_k_quantize (20 blk)", "x[5120] -> Q8_K", 1.0, 40.0, n_embd as f64 * 5.14, n_embd as f64, |s| {
        c.e.q8k.launch(s, &mut xq, &x, BLOCKS_Q8K_GATE_IN)
    })?;
    let wb = (N_EXPERT_USED * 2 * gbpe) as f64;
    c.rec(p, "mxfp4_pair_matvec_fused_swiglu_batch_hetsplit", &format!("6 x (gate+up)[2304x5120] MXFP4 = {:.1} MB, random experts", wb / 1e6), 1.0, 40.0, wb + BLOCKS_Q8K_GATE_IN as f64 * 292.0 + (N_EXPERT_USED * N_FF_EXP as usize * 4) as f64, 2.0 * 2.0 * N_EXPERT_USED as f64 * N_FF_EXP as f64 * n_embd as f64, |s| {
        let sel = next_sel();
        c.e.mxfp4pair.launch_fused_swiglu_batch_hetsplit(s, &mut mid, &gate, &up, &xq, &ew, &sel, &remap, 0, 0, gbpe as u32, gbpe as u32, N_EXPERT_USED as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN)
    })?;
    let nb_mid = BLOCKS_Q8K_DOWN_IN * N_EXPERT_USED as u32;
    c.rec(p, "q8_k_quantize (mid 54 blk)", "mid[6x2304] -> Q8_K", 1.0, 40.0, (N_EXPERT_USED * N_FF_EXP as usize) as f64 * 5.14, (N_EXPERT_USED * N_FF_EXP as usize) as f64, |s| {
        c.e.q8k.launch(s, &mut midq, &mid, nb_mid)
    })?;
    let wb = (N_EXPERT_USED * dbpe) as f64;
    c.rec(p, "mxfp4_matvec_par_batched_hetsplit (down)", &format!("6 x down[5120x2304] MXFP4 = {:.1} MB, random experts", wb / 1e6), 1.0, 40.0, wb + nb_mid as f64 * 292.0 + n_embd as f64 * 4.0, 2.0 * N_EXPERT_USED as f64 * N_FF_EXP as f64 * n_embd as f64, |s| {
        let sel = next_sel();
        c.e.mxfp4.launch_batched_hetsplit(s, &mut out, &down, &midq, &sel, &remap, 0, 0, dbpe as u32, (BLOCKS_Q8K_DOWN_IN as usize * BLOCK_Q8_K_BYTES) as u32, N_EXPERT_USED as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN)
    })?;
    // The whole chain back to back (what the captured graph replays).
    c.rec(p, "igpu MoE chain (q8k+gateup+q8k+down)", "6 random experts", 1.0, 40.0, (N_EXPERT_USED * (2 * gbpe + dbpe)) as f64, 3.0 * 2.0 * N_EXPERT_USED as f64 * N_FF_EXP as f64 * n_embd as f64, |s| {
        let sel = next_sel();
        c.e.q8k.launch(s, &mut xq, &x, BLOCKS_Q8K_GATE_IN)?;
        c.e.mxfp4pair.launch_fused_swiglu_batch_hetsplit(s, &mut mid, &gate, &up, &xq, &ew, &sel, &remap, 0, 0, gbpe as u32, gbpe as u32, N_EXPERT_USED as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN)?;
        c.e.q8k.launch(s, &mut midq, &mid, nb_mid)?;
        c.e.mxfp4.launch_batched_hetsplit(s, &mut out, &down, &midq, &sel, &remap, 0, 0, dbpe as u32, (BLOCKS_Q8K_DOWN_IN as usize * BLOCK_Q8_K_BYTES) as u32, N_EXPERT_USED as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN)
    })?;
    Ok(())
}

// ===========================================================================
// prefill, dGPU (B rows of one lane)
// ===========================================================================
fn prefill_dgpu(c: &Ctx, b: u32, n_comps: &[u32]) -> eyre::Result<()> {
    let p = &format!("prefill_b{b}");
    let bu = b as usize;
    let n_embd = N_EMBD as usize;
    let hc_dim = HC_DIM as usize;
    let hcm = HC_MIX_DIM as usize;
    let f4 = 4.0f64;
    let bf = b as f64;
    let pos_h: Vec<i32> = (0..b as i32).map(|i| 4000 + i).collect();
    let pos_per_b = c.i32s(&pos_h)?;

    // ---- mHC pre (x2 per layer) ----
    {
        let residual = c.buf::<f32>(bu * hc_dim)?;
        let mut flat = c.buf::<f32>(bu * hc_dim)?;
        let mut mix = c.buf::<f32>(bu * hcm)?;
        let mut split = c.buf::<f32>(bu * hcm)?;
        let hc_fn = c.ring(hcm * hc_dim * 2)?;
        let scale = c.buf::<f32>(3)?;
        let base = c.buf::<f32>(hcm)?;
        let mut cur = c.buf::<f32>(bu * n_embd)?;
        let mut normed = c.buf::<f32>(bu * n_embd)?;
        let norm_w = c.buf::<f32>(n_embd)?;
        let carry = c.buf::<f32>(bu * hcm)?;
        c.rec(p, "rms_norm_no_weight_batched (20480)", &format!("[{b} x 20480]"), 2.0, 20.0, 2.0 * bf * hc_dim as f64 * f4, 2.0 * bf * hc_dim as f64, |s| {
            c.e.rms_nw.launch_batched(s, &mut flat, &residual, 1, HC_DIM, RMS_EPS, b)
        })?;
        c.rec(p, "f16_matvec_narrow_batched (hc_fn)", &format!("W[24x20480] f16 x [{b} x 20480]"), 2.0, 20.0, (hcm * hc_dim * 2) as f64 + bf * hc_dim as f64 * f4 + bf * hcm as f64 * f4, 2.0 * bf * hcm as f64 * hc_dim as f64, |s| {
            c.e.f16.matvec_narrow_batched(s, &mut mix, hc_fn.next(), &flat, HC_MIX_DIM, HC_DIM, b)
        })?;
        {
            let w64 = c.ring(64 * hc_dim * 2)?;
            let mut mix64 = c.buf::<f32>(bu * 64)?;
            c.rec(p, "PROPOSAL f16_gemm_wmma_lds_tiled hc_fn (M padded 24->64)", &format!("W[64x20480] f16 x [{b} x 20480] f32"), 2.0, 20.0, (64 * hc_dim * 2) as f64 + bf * hc_dim as f64 * f4 + bf * 64.0 * f4, 2.0 * bf * 64.0 * hc_dim as f64, |s| {
                c.e.f16.gemm_batched_wmma(s, &mut mix64, w64.next(), &flat, 64, HC_DIM, b)
            })?;
        }
        c.rec(p, "hc_sinkhorn_par_batched", &format!("[{b} x 24]"), 2.0, 20.0, 2.0 * bf * hcm as f64 * f4, bf * 2000.0, |s| {
            c.e.hc_sinkhorn.launch_batched(s, &mut split, &mix, &scale, &base, N_HC, SINKHORN_ITERS, SINKHORN_EPS, b)
        })?;
        c.rec(p, "hc_weighted_sum_batched", &format!("[{b} x 4 x 5120] -> [{b} x 5120]"), 2.0, 20.0, bf * (hc_dim + n_embd) as f64 * f4, 2.0 * bf * hc_dim as f64, |s| {
            c.e.hc_weighted.launch_batched(s, &mut cur, &residual, &carry, N_EMBD, N_HC, HC_MIX_DIM, b)
        })?;
        c.rec(p, "rms_norm_weighted_batched (5120)", &format!("[{b} x 5120]"), 2.0, 20.0, 2.0 * bf * n_embd as f64 * f4, 3.0 * bf * n_embd as f64, |s| {
            c.e.rms_w.launch_weighted_batched(s, &mut normed, &cur, &norm_w, N_EMBD, RMS_EPS, b)
        })?;
    }

    // ---- q chain ----
    {
        let x = c.buf::<f32>(bu * n_embd)?;
        let pitch_e = f16_pitch(N_EMBD) as usize;
        let mut x16 = c.buf::<u16>(bu * pitch_e)?;
        let mut qr = c.buf::<f32>(bu * N_LORA_Q as usize)?;
        let mut qr_normed = c.buf::<f32>(bu * N_LORA_Q as usize)?;
        let q_norm_w = c.buf::<f32>(N_LORA_Q as usize)?;
        let pitch_q = f16_pitch(N_LORA_Q) as usize;
        let mut qr16 = c.buf::<u16>(bu * pitch_q)?;
        let mut q = c.buf::<f32>(bu * Q_FLAT as usize)?;
        let mut q_normed = c.buf::<f32>(bu * Q_FLAT as usize)?;
        let wqa = c.ring(q8_bytes(N_LORA_Q as usize, n_embd))?;
        let wqb = c.ring(q8_bytes(Q_FLAT as usize, N_LORA_Q as usize))?;
        c.rec(p, "f32_to_f16_cast_2d (5120)", &format!("[{b} x 5120] f32 -> f16"), 2.0, 20.0, bf * n_embd as f64 * 6.0, bf * n_embd as f64, |s| {
            c.e.q8k.launch_cast_f16_2d(s, &mut x16, &x, b, N_EMBD, f16_pitch(N_EMBD))
        })?;
        let wb = q8_bytes(N_LORA_Q as usize, n_embd) as f64;
        c.rec(p, "q8_0_gemm_wmma_f16x wq_a", &format!("W[1280x5120] Q8_0 x [{b} x 5120] f16"), 1.0, 20.0, wb + bf * n_embd as f64 * 2.0 + bf * N_LORA_Q as f64 * 4.0, 2.0 * bf * N_LORA_Q as f64 * n_embd as f64, |s| {
            c.e.q8_wmma.gemm_f16x(s, &mut qr, wqa.next(), &x16, N_EMBD, N_LORA_Q, 1, b, f16_pitch(N_EMBD))
        })?;
        c.rec(p, "rms_norm_weighted_batched (1280)", &format!("[{b} x 1280]"), 1.0, 20.0, 2.0 * bf * N_LORA_Q as f64 * f4, 3.0 * bf * N_LORA_Q as f64, |s| {
            c.e.rms_w.launch_weighted_batched(s, &mut qr_normed, &qr, &q_norm_w, N_LORA_Q, RMS_EPS, b)
        })?;
        c.rec(p, "f32_to_f16_cast_2d (1280)", &format!("[{b} x 1280]"), 1.0, 20.0, bf * N_LORA_Q as f64 * 6.0, bf * N_LORA_Q as f64, |s| {
            c.e.q8k.launch_cast_f16_2d(s, &mut qr16, &qr_normed, b, N_LORA_Q, f16_pitch(N_LORA_Q))
        })?;
        let wb = q8_bytes(Q_FLAT as usize, N_LORA_Q as usize) as f64;
        c.rec(p, "q8_0_gemm_wmma_f16x wq_b", &format!("W[32768x1280] Q8_0 x [{b} x 1280] f16 -> [{b} x 32768] f32"), 1.0, 20.0, wb + bf * N_LORA_Q as f64 * 2.0 + bf * Q_FLAT as f64 * 4.0, 2.0 * bf * Q_FLAT as f64 * N_LORA_Q as f64, |s| {
            c.e.q8_wmma.gemm_f16x(s, &mut q, wqb.next(), &qr16, N_LORA_Q, Q_FLAT, 1, b, f16_pitch(N_LORA_Q))
        })?;
        c.rec(p, "memcpy D2D q->q_normed", &format!("[{b} x 32768] f32"), 1.0, 20.0, 2.0 * bf * Q_FLAT as f64 * 4.0, 0.0, |s| {
            let n = bu * Q_FLAT as usize;
            let src = q.slice_view(0, n);
            q_normed.slice_view_mut(0, n).copy_from_buffer_async(&src, s)
        })?;
        let rope = c.rope_comp;
        c.rec(p, "rope_tail_batched fwd q", &format!("[{b} x 64 x 512], rot 64"), 1.0, 20.0, 2.0 * bf * 64.0 * 64.0 * 4.0, bf * 64.0 * 64.0 * 6.0, |s| {
            c.e.rope.launch_forward_batched(s, &mut q_normed, &pos_per_b, N_HEAD, N_HEAD_DIM, N_ROT, b, &rope)
        })?;
    }

    // ---- kv chain + window append ----
    {
        let pitch_e = f16_pitch(N_EMBD) as usize;
        let x16 = c.buf::<u16>(bu * pitch_e)?;
        let mut kv_raw = c.buf::<f32>(bu * N_HEAD_DIM as usize)?;
        let mut kv_normed = c.buf::<f32>(bu * N_HEAD_DIM as usize)?;
        let kv_norm_w = c.buf::<f32>(N_HEAD_DIM as usize)?;
        let wkv = c.ring(q8_bytes(N_HEAD_DIM as usize, n_embd))?;
        let mut cache = c.buf::<u16>(KV_CACHE_ROWS * N_HEAD_DIM as usize)?;
        let wb = q8_bytes(N_HEAD_DIM as usize, n_embd) as f64;
        c.rec(p, "q8_0_gemm_wmma_f16x wkv", &format!("W[512x5120] Q8_0 x [{b} x 5120] f16"), 1.0, 20.0, wb + bf * n_embd as f64 * 2.0 + bf * 512.0 * 4.0, 2.0 * bf * 512.0 * n_embd as f64, |s| {
            c.e.q8_wmma.gemm_f16x(s, &mut kv_raw, wkv.next(), &x16, N_EMBD, N_HEAD_DIM, 1, b, f16_pitch(N_EMBD))
        })?;
        c.rec(p, "rms_norm_weighted_batched (512)", &format!("[{b} x 512]"), 1.0, 20.0, 2.0 * bf * 512.0 * f4, 3.0 * bf * 512.0, |s| {
            c.e.rms_w.launch_weighted_batched(s, &mut kv_normed, &kv_raw, &kv_norm_w, N_HEAD_DIM, RMS_EPS, b)
        })?;
        let rope = c.rope_comp;
        c.rec(p, "rope_tail_batched fwd kv", &format!("[{b} x 1 x 512]"), 1.0, 20.0, 2.0 * bf * 64.0 * 4.0, bf * 384.0, |s| {
            c.e.rope.launch_forward_batched(s, &mut kv_normed, &pos_per_b, 1, N_HEAD_DIM, N_ROT, b, &rope)
        })?;
        c.rec(p, "fp8_act_quant_inplace (window KV, batched)", &format!("[{b} x 512]"), 1.0, 20.0, 2.0 * bf * 512.0 * f4, bf * 2048.0, |s| {
            c.e.fp4kv.launch_fp8_window(s, &mut kv_normed, b, N_HEAD_DIM)
        })?;
        c.rec(p, "f16_roundtrip (b x 512)", &format!("[{b} x 512]"), 1.0, 20.0, 2.0 * bf * 512.0 * f4, bf * 512.0, |s| {
            c.e.f16rt.launch(s, &mut kv_normed, b * N_HEAD_DIM)
        })?;
        c.rec(p, "kv_cache_append_batched", &format!("[{b} x 512] -> f16 cache"), 1.0, 20.0, bf * 512.0 * 6.0, bf * 512.0, |s| {
            c.e.kv_append.launch_batched(s, &mut cache, &kv_normed, 128, N_HEAD_DIM, b)
        })?;
    }

    // ---- compressor (ratio 2: layers 2, 8, 14; ratio 1: layer 20) ----
    {
        let cw = 512usize;
        let x = c.buf::<f32>(bu * n_embd)?;
        let mut kv_cur = c.buf::<f32>(bu * cw)?;
        let mut sc_cur = c.buf::<f32>(bu * cw)?;
        let wkv = c.ring(cw * n_embd * 2)?;
        let wgate = c.ring(cw * n_embd * 2)?;
        let wb = (2 * cw * n_embd * 2) as f64;
        c.rec(p, "f16_matvec_pair_batched_tiled (compressor, ratio 2)", &format!("2x W[512x5120] f16 x [{b} x 5120], TILE_B=8"), 1.0, 3.0, wb * (bf / 8.0) + bf * n_embd as f64 * 4.0 + 2.0 * bf * 512.0 * 4.0, 2.0 * 2.0 * bf * 512.0 * n_embd as f64, |s| {
            c.e.f16.matvec_pair_batched_tiled(s, &mut kv_cur, &mut sc_cur, wkv.next(), wgate.next(), &x, cw as u32, N_EMBD, b)
        })?;
        c.rec(p, "f16_matvec_batched (compressor wkv, ratio 1, L20)", &format!("W[512x5120] f16 x [{b} x 5120] (grid.z = b)"), 1.0, 1.0, (cw * n_embd * 2) as f64 * bf + bf * n_embd as f64 * 4.0, 2.0 * bf * 512.0 * n_embd as f64, |s| {
            c.e.f16.matvec_batched(s, &mut kv_cur, wkv.next(), &x, cw as u32, N_EMBD, b)
        })?;
        c.rec(p, "PROPOSAL f16_gemm_wmma_lds_tiled compressor wkv (ratio 1, L20)", &format!("W[512x5120] f16 x [{b} x 5120] f32"), 1.0, 1.0, (cw * n_embd * 2) as f64 + bf * n_embd as f64 * 4.0 + bf * 512.0 * 4.0, 2.0 * bf * 512.0 * n_embd as f64, |s| {
            c.e.f16.gemm_batched_wmma(s, &mut kv_cur, wkv.next(), &x, cw as u32, N_EMBD, b)
        })?;
        let n_bnd = b / 2;
        let rows_state = 2u32;
        let mut snap_kv = c.buf::<f32>(n_bnd as usize * rows_state as usize * cw)?;
        let mut snap_sc = c.buf::<f32>(n_bnd as usize * rows_state as usize * cw)?;
        let state_kv = c.buf::<f32>(rows_state as usize * cw)?;
        let state_sc = c.buf::<f32>(rows_state as usize * cw)?;
        let mut state_kv_w = c.buf::<f32>(rows_state as usize * cw)?;
        let mut state_sc_w = c.buf::<f32>(rows_state as usize * cw)?;
        let ape = c.buf::<u8>(cw * 2 * 2)?;
        let gs_h: Vec<i32> = (0..n_bnd as i32).map(|k| 4000 + 2 * k).collect();
        let group_start = c.i32s(&gs_h)?;
        let pm = c.i32s(&[0, 1])?;
        c.rec(p, "compressor_snapshot_gather (ratio 2)", &format!("{n_bnd} boundaries x 2 rows x 512"), 1.0, 3.0, 2.0 * (bf * cw as f64 * 4.0) * 2.0, bf * cw as f64 * 2.0, |s| {
            c.e.compressor_state_snapshot.launch_gather(s, &mut snap_kv, &mut snap_sc, &kv_cur, &sc_cur, &state_kv, &state_sc, &ape, &group_start, cw as u32, 2, rows_state, 4000, b, n_bnd)
        })?;
        c.rec(p, "compressor_state_write_batched (end of chunk, 2 rows)", "2 rows x 512", 1.0, 3.0, 4.0 * 2.0 * cw as f64 * 4.0, 2.0 * cw as f64, |s| {
            let kv_seg = kv_cur.slice_view((bu - 2) * cw, 2 * cw);
            let sc_seg = sc_cur.slice_view((bu - 2) * cw, 2 * cw);
            c.e.compressor_state_write.launch_batched(s, &mut state_kv_w, &mut state_sc_w, &kv_seg, &sc_seg, &ape, &pm, &pm, cw as u32, 2)
        })?;
        let mut pooled = c.buf::<f32>(n_bnd as usize * N_HEAD_DIM as usize)?;
        let mut rows = c.buf::<f32>(n_bnd as usize * N_HEAD_DIM as usize)?;
        let norm_w = c.buf::<f32>(N_HEAD_DIM as usize)?;
        let mut comp_kv = c.buf::<u16>(8192 * N_HEAD_DIM as usize)?;
        let nb = n_bnd as f64;
        c.rec(p, "compressor_pool_batched (ratio 2)", &format!("{n_bnd} x state[2x512] -> [512]"), 1.0, 3.0, nb * 5.0 * 512.0 * 4.0, nb * 4.0 * 512.0 * 4.0, |s| {
            c.e.compressor_pool.launch_batched(s, &mut pooled, &snap_kv, &snap_sc, N_HEAD_DIM, 2, n_bnd)
        })?;
        c.rec(p, "rms_norm_weighted_batched (comp rows 512)", &format!("[{n_bnd} x 512]"), 1.0, 3.0, 2.0 * nb * 512.0 * f4, 3.0 * nb * 512.0, |s| {
            c.e.rms_w.launch_weighted_batched(s, &mut rows, &pooled, &norm_w, N_HEAD_DIM, RMS_EPS, n_bnd)
        })?;
        let rope = c.rope_comp;
        c.rec(p, "rope_tail_batched fwd comp rows", &format!("[{n_bnd} x 1 x 512]"), 1.0, 3.0, 2.0 * nb * 64.0 * 4.0, nb * 384.0, |s| {
            c.e.rope.launch_forward_batched(s, &mut rows, &group_start, 1, N_HEAD_DIM, N_ROT, n_bnd, &rope)
        })?;
        c.rec(p, "fp4_kv_quant_inplace (comp rows)", &format!("[{n_bnd} x 512] E2M1/E4M3 per 16"), 1.0, 3.0, 2.0 * nb * 512.0 * f4, nb * 4096.0, |s| {
            c.e.fp4kv.launch(s, &mut rows, n_bnd, N_HEAD_DIM)
        })?;
        c.rec(p, "comp_kv_append_batched", &format!("[{n_bnd} x 512] -> f16 store"), 1.0, 3.0, nb * 512.0 * 6.0, nb * 512.0, |s| {
            c.e.comp_kv_append.launch_batched(s, &mut comp_kv, &rows, 1000, N_HEAD_DIM, n_bnd)
        })?;
    }

    // ---- attention ----
    {
        let n_head = N_HEAD as usize;
        let hd = N_HEAD_DIM as usize;
        let q = c.buf::<f32>(bu * n_head * hd)?;
        let cache = c.buf::<u16>(KV_CACHE_ROWS * hd)?;
        let sinks = c.buf::<f32>(n_head)?;
        let mut heads = c.buf::<f32>(bu * n_head * hd)?;
        let nrp = c.i32s(&vec![128i32; bu])?;
        let nrop_h: Vec<i32> = (0..b as i32).collect();
        let nrop = c.i32s(&nrop_h)?;
        c.rec(p, "attention_swa_batched (L0-1)", &format!("[{b}] x 64 heads x 128 keys x 512"), 1.0, 2.0, bf * (n_head * hd * 4 * 2) as f64 + (640 * hd * 2) as f64, bf * 64.0 * 128.0 * 512.0 * 4.0, |s| {
            c.e.attn_swa.launch_batched(s, &mut heads, &q, &cache, &sinks, &nrp, &nrop, N_HEAD, N_HEAD_DIM, b, SWA_WINDOW)
        })?;
        let max_total = n_comps.iter().map(|&n| 128 + n).max().unwrap_or(128).max(ATTN_SCORES_STRIDE);
        // f16 scores at stride `max_total` per (b, head); the f32-typed buffer is 2x oversized by construction.
        let mut scores = c.buf::<f32>(bu * n_head * max_total as usize)?;
        for &n_comp in n_comps {
            let n_total = 128 + n_comp;
            let stride = max_total;
            let comp = c.buf::<u16>(n_comp as usize * hd)?;
            let comp_opt = if n_comp > 0 { Some(&comp) } else { None };
            let ncp = c.i32s(&vec![n_comp as i32; bu])?;
            let layers_here = if n_comp == 0 { 2.0 } else { 18.0 };
            let shape = format!("[{b}] x 64 heads x (128 + {n_comp}) keys x 512, f16 scores{}", if n_comp == 0 { " (SWA-layer replacement proposal)" } else { "" });
            let sb = bf * n_head as f64 * n_total as f64 * 2.0;
            let fl = bf * 64.0 * n_total as f64 * 512.0 * 2.0;
            let qb = bf * (n_head * hd * 4) as f64;
            let kb = (n_comp as usize * hd * 2 + 640 * hd * 2) as f64;
            c.rec(p, "attention_mixed_score_batched_htiled_wmma_f16s", &shape, 1.0, layers_here, qb + kb + sb, fl, |s| {
                c.e.attn_mixed.launch_score_batched_htiled_wmma_f16s(s, &mut scores, &q, &cache, comp_opt, &nrp, &nrop, &ncp, None, N_HEAD, N_HEAD_DIM, n_total, b, 0, stride)
            })?;
            c.rec(p, "attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s", &shape, 1.0, layers_here, sb + kb + qb, fl, |s| {
                c.e.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s(s, &mut heads, &mut scores, &sinks, &cache, comp_opt, &nrp, &nrop, &ncp, N_HEAD, N_HEAD_DIM, b, 0, stride)
            })?;
        }
    }

    // ---- output projection ----
    {
        let mut heads = c.buf::<f32>(bu * Q_FLAT as usize)?;
        let pitch_h = f16_pitch(Q_FLAT) as usize;
        let mut heads16 = c.buf::<u16>(bu * pitch_h)?;
        let mut low = c.buf::<f32>(bu * OUT_LOW as usize)?;
        let pitch_l = f16_pitch(OUT_LOW) as usize;
        let mut low16 = c.buf::<u16>(bu * pitch_l)?;
        let mut out = c.buf::<f32>(bu * n_embd)?;
        let woa = c.ring(q8_bytes((N_GROUPS * RANK) as usize, GROUP_DIM as usize))?;
        let wob = c.ring(q8_bytes(n_embd, OUT_LOW as usize))?;
        let rope = c.rope_comp;
        c.rec(p, "rope_tail_batched inverse heads", &format!("[{b} x 64 x 512]"), 1.0, 20.0, 2.0 * bf * 64.0 * 64.0 * 4.0, bf * 64.0 * 64.0 * 6.0, |s| {
            c.e.rope.launch_inverse_batched(s, &mut heads, &pos_per_b, N_HEAD, N_HEAD_DIM, N_ROT, b, &rope)
        })?;
        c.rec(p, "f32_to_f16_cast_2d (32768)", &format!("[{b} x 32768] f32 -> f16"), 1.0, 20.0, bf * Q_FLAT as f64 * 6.0, bf * Q_FLAT as f64, |s| {
            c.e.q8k.launch_cast_f16_2d(s, &mut heads16, &heads, b, Q_FLAT, f16_pitch(Q_FLAT))
        })?;
        let wb = q8_bytes((N_GROUPS * RANK) as usize, GROUP_DIM as usize) as f64;
        c.rec(p, "q8_0_gemm_wmma_f16x wo_a (8 groups)", &format!("8 x W[1024x4096] Q8_0 x [{b} x 4096] f16"), 1.0, 20.0, wb + bf * Q_FLAT as f64 * 2.0 + bf * OUT_LOW as f64 * 4.0, 2.0 * bf * 8.0 * 1024.0 * 4096.0, |s| {
            c.e.q8_wmma.gemm_f16x(s, &mut low, woa.next(), &heads16, GROUP_DIM, RANK, N_GROUPS, b, f16_pitch(Q_FLAT))
        })?;
        c.rec(p, "f32_to_f16_cast_2d (8192)", &format!("[{b} x 8192]"), 1.0, 20.0, bf * OUT_LOW as f64 * 6.0, bf * OUT_LOW as f64, |s| {
            c.e.q8k.launch_cast_f16_2d(s, &mut low16, &low, b, OUT_LOW, f16_pitch(OUT_LOW))
        })?;
        let wb = q8_bytes(n_embd, OUT_LOW as usize) as f64;
        c.rec(p, "q8_0_gemm_wmma_f16x wo_b", &format!("W[5120x8192] Q8_0 x [{b} x 8192] f16"), 1.0, 20.0, wb + bf * OUT_LOW as f64 * 2.0 + bf * n_embd as f64 * 4.0, 2.0 * bf * n_embd as f64 * OUT_LOW as f64, |s| {
            c.e.q8_wmma.gemm_f16x(s, &mut out, wob.next(), &low16, OUT_LOW, N_EMBD, 1, b, f16_pitch(OUT_LOW))
        })?;
    }

    // ---- hc_post, router, shared expert, combine ----
    {
        let mut out_hc = c.buf::<f32>(bu * hc_dim)?;
        let block_out = c.buf::<f32>(bu * n_embd)?;
        let residual = c.buf::<f32>(bu * hc_dim)?;
        let split = c.buf::<f32>(bu * hcm)?;
        c.rec(p, "hc_post_from_split_batched", &format!("[{b}] x (residual[4x5120] + block[5120])"), 2.0, 20.0, bf * (2 * hc_dim + n_embd) as f64 * f4, 2.0 * bf * hc_dim as f64 * 5.0, |s| {
            c.e.hc_post.launch_from_split_batched(s, &mut out_hc, &block_out, &residual, &split, N_HC, N_EMBD, N_HC, b)
        })?;
        let x = c.buf::<f32>(bu * n_embd)?;
        let mut logits = c.buf::<f32>(bu * N_EXPERT as usize)?;
        let gate_w = c.ring(N_EXPERT as usize * n_embd * 2)?;
        let bias = c.buf::<f32>(N_EXPERT as usize)?;
        let mut sel = c.buf::<i32>(bu * N_EXPERT_USED)?;
        let mut ew = c.buf::<f32>(bu * N_EXPERT_USED)?;
        let wb = (N_EXPERT as usize * n_embd * 2) as f64;
        c.rec(p, "f16_gemm_wmma_lds_tiled router gate", &format!("W[384x5120] f16 x [{b} x 5120] f32"), 1.0, 20.0, wb + bf * n_embd as f64 * 4.0 + bf * 384.0 * 4.0, 2.0 * bf * 384.0 * n_embd as f64, |s| {
            c.e.f16.gemm_batched_wmma(s, &mut logits, gate_w.next(), &x, N_EXPERT, N_EMBD, b)
        })?;
        c.rec(p, "router_topk_batched", &format!("[{b} x 384] -> top-6"), 1.0, 20.0, bf * 384.0 * 4.0 + bf * 48.0, bf * 384.0 * 20.0, |s| {
            c.e.router_topk.launch_batched(s, &mut sel, &mut ew, &logits, Some(&bias), N_EXPERT, N_EXPERT_USED as u32, EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS, b)
        })?;
        let pitch_e = f16_pitch(N_EMBD) as usize;
        let mut x16 = c.buf::<u16>(bu * pitch_e)?;
        let nfs = N_FF_SHARED as usize;
        let mut g = c.buf::<f32>(bu * nfs)?;
        let mut u = c.buf::<f32>(bu * nfs)?;
        let mut mid = c.buf::<f32>(bu * nfs)?;
        let pitch_m = f16_pitch(N_FF_SHARED) as usize;
        let mut mid16 = c.buf::<u16>(bu * pitch_m)?;
        let mut out = c.buf::<f32>(bu * n_embd)?;
        let wg = c.ring(q8_bytes(nfs, n_embd))?;
        let wu = c.ring(q8_bytes(nfs, n_embd))?;
        let wd = c.ring(q8_bytes(n_embd, nfs))?;
        c.rec(p, "f32_to_f16_cast_2d (shared in 5120)", &format!("[{b} x 5120]"), 1.0, 20.0, bf * n_embd as f64 * 6.0, bf * n_embd as f64, |s| {
            c.e.q8k.launch_cast_f16_2d(s, &mut x16, &x, b, N_EMBD, f16_pitch(N_EMBD))
        })?;
        let wb = q8_bytes(nfs, n_embd) as f64;
        c.rec(p, "q8_0_gemm_wmma_f16x shared gate", &format!("W[2304x5120] Q8_0 x [{b} x 5120] f16"), 1.0, 20.0, wb + bf * n_embd as f64 * 2.0 + bf * nfs as f64 * 4.0, 2.0 * bf * nfs as f64 * n_embd as f64, |s| {
            c.e.q8_wmma.gemm_f16x(s, &mut g, wg.next(), &x16, N_EMBD, N_FF_SHARED, 1, b, f16_pitch(N_EMBD))
        })?;
        c.rec(p, "q8_0_gemm_wmma_f16x shared up", &format!("W[2304x5120] Q8_0 x [{b} x 5120] f16"), 1.0, 20.0, wb + bf * n_embd as f64 * 2.0 + bf * nfs as f64 * 4.0, 2.0 * bf * nfs as f64 * n_embd as f64, |s| {
            c.e.q8_wmma.gemm_f16x(s, &mut u, wu.next(), &x16, N_EMBD, N_FF_SHARED, 1, b, f16_pitch(N_EMBD))
        })?;
        c.rec(p, "swiglu (b x 2304)", &format!("[{b} x 2304]"), 1.0, 20.0, 3.0 * bf * nfs as f64 * f4, bf * nfs as f64 * 8.0, |s| {
            c.e.swiglu.launch_clamped(s, &mut mid, &g, &u, b * N_FF_SHARED, SWIGLU_CLAMP_EXP)
        })?;
        c.rec(p, "f32_to_f16_cast_2d (2304)", &format!("[{b} x 2304]"), 1.0, 20.0, bf * nfs as f64 * 6.0, bf * nfs as f64, |s| {
            c.e.q8k.launch_cast_f16_2d(s, &mut mid16, &mid, b, N_FF_SHARED, f16_pitch(N_FF_SHARED))
        })?;
        c.rec(p, "q8_0_gemm_wmma_f16x shared down", &format!("W[5120x2304] Q8_0 x [{b} x 2304] f16"), 1.0, 20.0, wb + bf * nfs as f64 * 2.0 + bf * n_embd as f64 * 4.0, 2.0 * bf * nfs as f64 * n_embd as f64, |s| {
            c.e.q8_wmma.gemm_f16x(s, &mut out, wd.next(), &mid16, N_FF_SHARED, N_EMBD, 1, b, f16_pitch(N_FF_SHARED))
        })?;
        let mut moe = c.buf::<f32>(bu * n_embd)?;
        c.rec(p, "vec_add_inplace (b x 5120)", &format!("[{b} x 5120]"), 1.0, 20.0, 3.0 * bf * n_embd as f64 * f4, bf * n_embd as f64, |s| {
            c.e.vec_add.launch(s, &mut moe, &out, b * N_EMBD)
        })?;
    }

    // ---- Engram (layers 1, 14): ENGRAM_CHUNK-row passes ----
    {
        let ein = ENGRAM_IN as usize;
        let eout = ENGRAM_OUT as usize;
        let n = ENGRAM_CHUNK.min(b) as usize;
        let nf = n as f64;
        let rows = c.buf::<f32>(n * ein)?;
        let mut xq = c.buf::<i8>(n * ein)?;
        let mut xs = c.buf::<f32>(n * ein / 32)?;
        let mut kv = c.buf::<f32>(n * eout)?;
        let mut h = c.buf::<f32>(n * hc_dim)?;
        let qk = c.buf::<f32>(hc_dim)?;
        let wkv = c.ring(q8_bytes(eout, ein))?;
        let passes = bf / nf;
        c.rec(p, "q8_0_quantize_f32_batched (engram)", &format!("[{n} x 6144]"), passes, 2.0, nf * ein as f64 * 5.125, nf * ein as f64, |s| {
            c.e.q8.quantize_input_batched(s, &mut xq, &mut xs, &rows, ENGRAM_IN, n as u32)
        })?;
        let wb = q8_bytes(eout, ein) as f64;
        c.rec(p, "q8_0_gemv_batched_warp8 engram wkv", &format!("W[25600x6144] Q8_0 {:.0} MB x [{n} x 6144] (per {n}-row pass)", wb / 1e6), passes, 2.0, wb + nf * ein as f64 * 1.125 + nf * eout as f64 * 4.0, 2.0 * nf * eout as f64 * ein as f64, |s| {
            c.e.q8.matvec_batched(s, &mut kv, wkv.next(), &xq, &xs, ENGRAM_OUT, ENGRAM_IN, n as u32)
        })?;
        c.rec(p, "engram_gate_add (batched)", &format!("[{n}] x h[4x5120] kv[25600]"), passes, 2.0, nf * (2 * hc_dim + eout) as f64 * f4 + hc_dim as f64 * f4, nf * 4.0 * n_embd as f64 * 8.0, |s| {
            c.e.engram_gate.launch(s, &mut h, &kv, &qk, N_HC, N_EMBD, ENGRAM_OUT, N_HC * N_EMBD, RMS_EPS, n as u32)
        })?;
        // Proposed replacements (KERNEL_PERF_REVIEW #2): the int8 LDS-tiled WMMA GEMM and
        // the f16-activation WMMA GEMM at the same shape, per 64-row pass and per whole lane.
        let pitch_ei = f16_pitch(ENGRAM_IN) as usize;
        for &(nn, tagp) in &[(n, "per 64-row pass"), (bu, "whole lane")] {
            let nnf = nn as f64;
            let passes_here = bf / nnf;
            let xq_b = c.buf::<i8>(nn * ein)?;
            let xs_b = c.buf::<f32>(nn * ein / 32)?;
            let x16_b = c.buf::<u16>(nn * pitch_ei)?;
            let mut kv_b = c.buf::<f32>(nn * eout)?;
            c.rec(p, "PROPOSAL q8_0_gemm_wmma_lds_tiled engram wkv", &format!("W[25600x6144] Q8_0 x [{nn} x 6144] i8 ({tagp})"), passes_here, 2.0, wb + nnf * ein as f64 * 1.125 + nnf * eout as f64 * 4.0, 2.0 * nnf * eout as f64 * ein as f64, |s| {
                c.e.q8_wmma.gemm_lds_tiled(s, &mut kv_b, wkv.next(), &xq_b, &xs_b, ENGRAM_OUT, ENGRAM_IN, nn as u32)
            })?;
            c.rec(p, "PROPOSAL q8_0_gemm_wmma_f16x engram wkv", &format!("W[25600x6144] Q8_0 x [{nn} x 6144] f16 ({tagp})"), passes_here, 2.0, wb + nnf * ein as f64 * 2.0 + nnf * eout as f64 * 4.0, 2.0 * nnf * eout as f64 * ein as f64, |s| {
                c.e.q8_wmma.gemm_f16x(s, &mut kv_b, wkv.next(), &x16_b, ENGRAM_IN, ENGRAM_OUT, 1, nn as u32, f16_pitch(ENGRAM_IN))
            })?;
        }
    }
    Ok(())
}

// ===========================================================================
// prefill, iGPU (B rows of one lane): MXFP4 MoE, by-expert kwide kernels
// ===========================================================================
fn prefill_igpu(c: &Ctx, b: u32, n_exp_buf: u32) -> eyre::Result<()> {
    let p = &format!("prefill_b{b}");
    let bu = b as usize;
    let bf = b as f64;
    let n_embd = N_EMBD as usize;
    let gbpe = mxfp4_bytes(N_FF_EXP as usize, n_embd);
    let dbpe = mxfp4_bytes(n_embd, N_FF_EXP as usize);
    let n_exp = n_exp_buf as usize;
    eprintln!("igpu prefill MoE: {n_exp} experts resident ({:.2} GB), B={b}, random top-6 per row", 3.0 * n_exp as f64 * gbpe as f64 / 1e9);
    let gate = c.buf::<u8>(n_exp * gbpe)?;
    let up = c.buf::<u8>(n_exp * gbpe)?;
    let down = c.buf::<u8>(n_exp * dbpe)?;
    let x = c.buf::<f32>(bu * n_embd)?;
    let mut xq = c.buf::<u8>(bu * BLOCKS_Q8K_GATE_IN as usize * BLOCK_Q8_K_BYTES)?;
    let mut mid = c.buf::<f32>(bu * N_EXPERT_USED * N_FF_EXP as usize)?;
    let mut midq = c.buf::<u8>(bu * N_EXPERT_USED * BLOCKS_Q8K_DOWN_IN as usize * BLOCK_Q8_K_BYTES)?;
    let mut partials = c.buf::<f32>(bu * N_EXPERT_USED * n_embd)?;
    let mut out = c.buf::<f32>(bu * n_embd)?;
    let ew = c.f32s(&vec![0.25f32; bu * N_EXPERT_USED])?;
    let sel_h = random_selection(bu, n_exp_buf, 0xbeef);
    let d_selected = c.i32s(&sel_h)?;
    let max_per_expert = b; // BatchIgpuShared::max_per_expert() == rows
    let mut group_count = c.buf::<i32>(N_EXPERT as usize)?;
    let mut expert_members = c.buf::<i32>(N_EXPERT as usize * bu)?;
    let wi_len = N_EXPERT as usize + bu * N_EXPERT_USED;
    let mut work_items = c.buf::<i32>(wi_len)?;
    let mut n_wi_dev = c.buf::<i32>(1)?;
    const CHUNK: u32 = 32;

    c.rec(p, "q8_k_quantize (b x 20 blk)", &format!("[{b} x 5120] -> Q8_K"), 1.0, 20.0, bf * n_embd as f64 * 5.14, bf * n_embd as f64, |s| {
        c.e.q8k.launch(s, &mut xq, &x, BLOCKS_Q8K_GATE_IN * b)
    })?;
    c.rec(p, "moe_group_builder (+memset)", &format!("[{b} x 6] picks -> per-expert lists"), 1.0, 20.0, bf * 6.0 * 4.0 * 3.0, bf * 6.0, |s| {
        group_count.fill_zero_async(s)?;
        c.e.moe_group_builder.launch(s, &mut group_count, &mut expert_members, &d_selected, b, N_EXPERT_USED as u32, N_EXPERT, max_per_expert)
    })?;
    // work items are built once here (production also syncs + reads back n_work_items).
    n_wi_dev.fill_zero()?;
    c.rec(p, "moe_work_items_builder (+memset, +host readback in production)", "384 experts, chunk 32", 1.0, 20.0, 384.0 * 12.0, 384.0, |s| {
        n_wi_dev.fill_zero_async(s)?;
        c.e.moe_group_builder.launch_work_items(s, &mut work_items, &mut n_wi_dev, &group_count, N_EXPERT, CHUNK, wi_len as u32)
    })?;
    let mut n_wi_h = [0i32; 1];
    n_wi_dev.copy_to_host(&mut n_wi_h)?;
    let n_wi = n_wi_h[0] as u32;
    eprintln!("n_work_items = {n_wi} (b={b}, chunk={CHUNK})");
    // Expert bytes actually touched: every expert with >= 1 member.
    let mut gc_h = vec![0i32; N_EXPERT as usize];
    group_count.copy_to_host(&mut gc_h)?;
    let touched = gc_h.iter().filter(|&&g| g > 0).count();
    let wb_gu = (touched * 2 * gbpe) as f64;
    let wb_d = (touched * dbpe) as f64;
    let picks = bf * N_EXPERT_USED as f64;
    c.rec(p, "mxfp4_pair_matvec_fused_swiglu_kwide (gate+up)", &format!("{touched} experts touched x (gate+up) MXFP4 = {:.2} GB, {n_wi} work items x {} row-WGs", wb_gu / 1e9, N_FF_EXP / 8), 1.0, 20.0, wb_gu + bf * 20.0 * 292.0 + picks * N_FF_EXP as f64 * 4.0, 2.0 * 2.0 * picks * N_FF_EXP as f64 * n_embd as f64, |s| {
        c.e.mxfp4pair.launch_fused_swiglu_kwide(s, &mut mid, &gate, &up, &xq, &ew, &group_count, &expert_members, &work_items, n_wi, gbpe as u32, gbpe as u32, N_EXPERT_USED as u32, max_per_expert, CHUNK, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN)
    })?;
    let nb_mid = BLOCKS_Q8K_DOWN_IN * N_EXPERT_USED as u32 * b;
    c.rec(p, "q8_k_quantize (mid, b x 54 blk)", &format!("[{b} x 6 x 2304] -> Q8_K"), 1.0, 20.0, picks * N_FF_EXP as f64 * 5.14, picks * N_FF_EXP as f64, |s| {
        c.e.q8k.launch(s, &mut midq, &mid, nb_mid)
    })?;
    c.rec(p, "mxfp4_matvec_par_by_expert_kwide2 (down)", &format!("{touched} experts touched x down MXFP4 = {:.2} GB, partials [{b} x 6 x 5120]", wb_d / 1e9), 1.0, 20.0, wb_d + picks * 9.0 * 292.0 + picks * n_embd as f64 * 4.0, 2.0 * picks * N_FF_EXP as f64 * n_embd as f64, |s| {
        c.e.mxfp4.launch_by_expert_kwide2(s, &mut partials, &down, &midq, &group_count, &expert_members, &work_items, n_wi, dbpe as u32, (BLOCKS_Q8K_DOWN_IN as usize * BLOCK_Q8_K_BYTES) as u32, N_EXPERT_USED as u32, max_per_expert, CHUNK, N_EMBD, BLOCKS_Q8K_DOWN_IN)
    })?;
    c.rec(p, "q2_k_reduce_partials", &format!("[{b} x 6 x 5120] -> [{b} x 5120]"), 1.0, 20.0, picks * n_embd as f64 * 4.0 + bf * n_embd as f64 * 4.0, picks * n_embd as f64, |s| {
        c.e.q2k.launch_reduce_partials(s, &mut out, &partials, N_EXPERT_USED as u32, N_EMBD, b)
    })?;
    Ok(())
}

#[test]
#[ignore]
fn bench_v41_kernel_roofline() -> eyre::Result<()> {
    install_panic_handler()?;
    let sections = std::env::var("BENCH_SECTION").unwrap_or_else(|_| "decode_dgpu,decode_igpu,prefill_dgpu,prefill_igpu".into());
    let ctxs: Vec<u32> = env_list("BENCH_CTX", vec![8192]);
    let bs: Vec<u32> = env_list("BENCH_B", vec![512]);
    let n_comps: Vec<u32> = env_list("BENCH_PREFILL_NCOMP", vec![0, 2048, 8192, 16384]);
    let n_exp_buf: u32 = env_u("BENCH_N_EXPERT_BUF", N_EXPERT);
    let json = std::env::var("BENCH_JSON").ok().map(|p| std::fs::File::create(p)).transpose()?;
    let json2 = json.as_ref().map(|f| f.try_clone()).transpose()?;
    eprintln!("V4.1 kernel roofline bench: sections={sections} ctx={ctxs:?} B={bs:?} prefill n_comp={n_comps:?} expert_buf={n_exp_buf}");
    eprintln!("{:<12} {:<5} {:<58} {:>13} {:>13} | {:>12} {:>15} | launches | shape", "path", "dev", "kernel", "p50", "min", "achieved", "achieved");

    if sections.contains("dgpu") {
        let dev = pick("gfx1201").ok_or_else(|| eyre!("no gfx1201 dGPU visible"))?;
        let c = Ctx::new(dev, "dgpu", json)?;
        if sections.contains("decode_dgpu") {
            decode_dgpu(&c, &ctxs)?;
        }
        if sections.contains("prefill_dgpu") {
            for &b in &bs {
                prefill_dgpu(&c, b, &n_comps)?;
            }
        }
    }
    if sections.contains("igpu") {
        let dev = pick("gfx1151").ok_or_else(|| eyre!("no gfx1151 iGPU visible"))?;
        let c = Ctx::new(dev, "igpu", json2)?;
        if sections.contains("decode_igpu") {
            decode_igpu(&c, n_exp_buf)?;
        }
        if sections.contains("prefill_igpu") {
            for &b in &bs {
                prefill_igpu(&c, b, n_exp_buf)?;
            }
        }
    }
    Ok(())
}
