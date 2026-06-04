//! Standalone PREFILL attention probe — no GGUF, no model load, no mmap.
//! Runs production WMMA attention kernels on synthetic buffers, over a
//! batch of B tokens. Mirrors `bench_attention_isolated.rs` (decode probe).
//!
//! Buffers are zero-initialized. The math is data-independent for timing
//! (FMAs and BW don't depend on values), so timing is representative;
//! outputs are zero. Per-token causal prefixes are set uniformly to
//! (n_raw, n_comp) — close enough for timing.
//!
//! Env vars:
//!   BENCH_N_RAW    n_raw (SWA cap = 128). Default 128.
//!   BENCH_N_COMP   n_comp. ratio=4 @ 64K → 16384; ratio=128 @ 64K → 512.
//!                  Default 16384.
//!   BENCH_BATCH    tokens per chunk (≤ B_MAX=256). Default 256.
//!   BENCH_ITERS    timed iterations. Default 50.
//!   BENCH_WARMUP   warmup iterations (discarded). Default 3.
//!   BENCH_PHASE    "score_wmma" | "smwsum_wmma" | "smwsum_ldsv" |
//!                  "smwsum_ldsv_db" | "smwsum_regv_db" — which kernel to
//!                  time. Default "smwsum_wmma".
//!
//! Run normally:
//!   HIP_VISIBLE_DEVICES=0,1 BENCH_N_COMP=16384 BENCH_BATCH=256 \
//!     nix develop -c cargo test --release -p v4flash-kernels \
//!     --test bench_prefill_attention_isolated bench_prefill_attention_isolated \
//!     -- --ignored --nocapture
//!
//! Run under rocprofv3 (per-instruction ATT): same pattern as the decode
//! probe — see bench_attention_isolated.rs header.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::attention::{AttentionMixed, ATTN_MIXED_MAX_KEYS};
use v4flash_kernels::config::{N_HEAD, N_HEAD_DIM};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201"))
}

/// f32 → f16 bits (IEEE 754 round-to-nearest-even). Matches the GPU's
/// `(_Float16)` cast — both use RTNE. Used to upload f32 reference data
/// as the f16 V buffers the production kernels consume.
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32;
    let mant = (bits & 0x7fffff) as u32;
    if exp == 0xff {
        let m = if mant != 0 { 0x200 } else { 0 };
        return sign | 0x7c00 | m as u16;
    }
    exp = exp - 127 + 15;
    if exp >= 0x1f { return sign | 0x7c00; }
    if exp <= 0 {
        if exp < -10 { return sign; }
        let m = (mant | 0x800000) >> (1 - exp);
        let rounded = (m + 0x1000 + ((m >> 13) & 1)) >> 13;
        return sign | rounded as u16;
    }
    let rounded_mant = (mant + 0x1000 + ((mant >> 13) & 1)) >> 13;
    if rounded_mant & 0x400 != 0 {
        return sign | ((exp as u16 + 1) << 10);
    }
    sign | ((exp as u16) << 10) | rounded_mant as u16
}

fn pack_f16(xs: &[f32]) -> Vec<u16> {
    xs.iter().map(|&x| f32_to_f16_bits(x)).collect()
}

fn percentile(xs_sorted: &[f32], p: f32) -> f32 {
    if xs_sorted.is_empty() {
        return 0.0;
    }
    let k = ((xs_sorted.len() - 1) as f32 * p / 100.0).round() as usize;
    xs_sorted[k.min(xs_sorted.len() - 1)]
}

#[test]
#[ignore]
fn bench_prefill_attention_isolated() -> eyre::Result<()> {
    install_panic_handler()?;

    let n_raw: u32 = std::env::var("BENCH_N_RAW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let n_comp: u32 = std::env::var("BENCH_N_COMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16384);
    let batch: u32 = std::env::var("BENCH_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let phase: String =
        std::env::var("BENCH_PHASE").unwrap_or_else(|_| "smwsum_wmma".to_string());
    let do_score_wmma = phase == "score_wmma";
    let do_smwsum_wmma = phase == "smwsum_wmma";
    let do_smwsum_ldsv = phase == "smwsum_ldsv";
    let do_smwsum_ldsv_db = phase == "smwsum_ldsv_db";
    let do_smwsum_regv_db = phase == "smwsum_regv_db";
    let do_smwsum_ldsv_f16s = phase == "smwsum_ldsv_f16s";

    let n_total = n_raw + n_comp;
    if n_total > ATTN_MIXED_MAX_KEYS {
        return Err(eyre!(
            "n_raw+n_comp={n_total} exceeds cap {ATTN_MIXED_MAX_KEYS}"
        ));
    }
    let ratio_hint = if n_comp > 1000 {
        "ratio=4"
    } else if n_comp > 0 {
        "ratio=128"
    } else {
        "ratio=0 (no comp)"
    };
    eprintln!(
        "isolated PREFILL attention probe: B={batch} n_raw={n_raw} n_comp={n_comp} \
         n_total={n_total} {ratio_hint}, iters={iters} warmup={warmup} phase={phase}"
    );

    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let attn = AttentionMixed::for_arch(&arch)?;

    let head_dim = N_HEAD_DIM;
    let n_head = N_HEAD;
    let b = batch as usize;

    // q: [B, n_head, head_dim] post-RoPE
    let mut q: DeviceBuffer<f32> =
        DeviceBuffer::new(dgpu.id, b * (n_head as usize) * (head_dim as usize))?;
    q.fill_zero()?;

    // raw_kv: [SWA_WINDOW=128, head_dim] f16 (shared across batch)
    let n_raw_capacity: usize = 128;
    let mut raw_kv: DeviceBuffer<u16> =
        DeviceBuffer::new(dgpu.id, n_raw_capacity * (head_dim as usize))?;
    raw_kv.fill_zero()?;

    // comp_kv: [n_comp, head_dim] f16 (shared across batch)
    let comp_kv: Option<DeviceBuffer<u16>> = if n_comp > 0 {
        let mut c: DeviceBuffer<u16> =
            DeviceBuffer::new(dgpu.id, (n_comp as usize) * (head_dim as usize))?;
        c.fill_zero()?;
        Some(c)
    } else {
        None
    };

    // sinks: [n_head]
    let mut sinks: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_head as usize)?;
    sinks.fill_zero()?;

    // Per-token causal prefixes [B] — uniform for the bench.
    let mut n_raw_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, b)?;
    let mut n_raw_offset_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, b)?;
    let mut n_comp_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, b)?;
    n_raw_per.copy_from_host_async(&vec![n_raw as i32; b], &stream)?;
    n_raw_offset_per.copy_from_host_async(&vec![0i32; b], &stream)?;
    n_comp_per.copy_from_host_async(&vec![n_comp as i32; b], &stream)?;
    stream.synchronize()?;

    // scores scratch: [B, n_head, ATTN_MIXED_MAX_KEYS]
    let mut scores: DeviceBuffer<f32> =
        DeviceBuffer::new(dgpu.id, b * (n_head as usize) * (ATTN_MIXED_MAX_KEYS as usize))?;
    scores.fill_zero()?;

    // out: [B, n_head, head_dim]
    let mut out: DeviceBuffer<f32> =
        DeviceBuffer::new(dgpu.id, b * (n_head as usize) * (head_dim as usize))?;
    out.fill_zero()?;

    let launch_iter = |stream: &Stream,
                           scores: &mut DeviceBuffer<f32>,
                           out: &mut DeviceBuffer<f32>|
     -> eyre::Result<()> {
        if do_score_wmma {
            attn.launch_score_batched_htiled_wmma(
                stream,
                scores,
                &q,
                &raw_kv,
                comp_kv.as_ref(),
                &n_raw_per,
                &n_raw_offset_per,
                &n_comp_per,
                n_head,
                head_dim,
                n_total,
                batch,
            )?;
        }
        if do_smwsum_wmma {
            attn.launch_softmax_wsum_batched_htiled_wmma(
                stream,
                out,
                scores,
                &sinks,
                &raw_kv,
                comp_kv.as_ref(),
                &n_raw_per,
                &n_raw_offset_per,
                &n_comp_per,
                n_head,
                head_dim,
                batch,
            )?;
        }
        if do_smwsum_ldsv {
            attn.launch_softmax_wsum_batched_htiled_wmma_ldsv(
                stream,
                out,
                scores,
                &sinks,
                &raw_kv,
                comp_kv.as_ref(),
                &n_raw_per,
                &n_raw_offset_per,
                &n_comp_per,
                n_head,
                head_dim,
                batch,
            )?;
        }
        if do_smwsum_ldsv_f16s {
            attn.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s(
                stream,
                out,
                scores,
                &sinks,
                &raw_kv,
                comp_kv.as_ref(),
                &n_raw_per,
                &n_raw_offset_per,
                &n_comp_per,
                n_head,
                head_dim,
                batch,
                0,
            )?;
        }
        if do_smwsum_ldsv_db {
            attn.launch_softmax_wsum_batched_htiled_wmma_ldsv_db(
                stream,
                out,
                scores,
                &sinks,
                &raw_kv,
                comp_kv.as_ref(),
                &n_raw_per,
                &n_raw_offset_per,
                &n_comp_per,
                n_head,
                head_dim,
                batch,
            )?;
        }
        if do_smwsum_regv_db {
            attn.launch_softmax_wsum_batched_htiled_wmma_regv_db(
                stream,
                out,
                scores,
                &sinks,
                &raw_kv,
                comp_kv.as_ref(),
                &n_raw_per,
                &n_raw_offset_per,
                &n_comp_per,
                n_head,
                head_dim,
                batch,
            )?;
        }
        Ok(())
    };

    for _ in 0..warmup {
        launch_iter(&stream, &mut scores, &mut out)?;
    }
    stream.synchronize()?;

    eprintln!("running {iters} timed iters...");
    let mut walls_ms: Vec<f32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&stream)?;
        launch_iter(&stream, &mut scores, &mut out)?;
        end.record(&stream)?;
        stream.synchronize()?;
        walls_ms.push(Event::elapsed_ms(&start, &end)?);
    }
    walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = walls_ms[0];
    let p50 = percentile(&walls_ms, 50.0);
    let p90 = percentile(&walls_ms, 90.0);
    let p99 = percentile(&walls_ms, 99.0);
    let max = walls_ms[walls_ms.len() - 1];
    let mean: f32 = walls_ms.iter().sum::<f32>() / walls_ms.len() as f32;
    eprintln!(
        "prefill attention {phase} (B={batch}): min={:.3} mean={:.3} p50={:.3} p90={:.3} \
         p99={:.3} max={:.3} (ms)",
        min, mean, p50, p90, p99, max,
    );
    eprintln!(
        "= {:.3} ms/token p50 ({:.1} µs); one ratio=4 layer-chunk",
        p50 / batch as f32,
        p50 * 1000.0,
    );
    Ok(())
}

/// M52 cross-chunk SWA regression test: the production WMMA chain
/// (score_batched_htiled_wmma + softmax_wsum_batched_htiled_wmma_ldsv) must
/// correctly honor `n_raw_offset_per[b] != 0` (per-token starting slot into
/// the oversized raw KV cache). Validated against a CPU reference that
/// re-implements the per-token attention math with the same per-token offset
/// semantics. This is the test the T=540 regression would have caught at
/// `cargo test` time instead of via manual wine-tasting essays.
#[test]
#[ignore]
fn prefill_attention_htiled_offset_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let attn = AttentionMixed::for_arch(&arch)?;

    let head_dim = N_HEAD_DIM;
    let n_head = N_HEAD;
    let batch: u32 = 4;
    let b = batch as usize;

    // Mimics the post-eviction cross-chunk geometry: n_raw_before = W = 128,
    // token i lives at chunk slot W + i, causal window starts at slot i + 1.
    // So n_raw_per[i] = W (saturated) and n_raw_offset_per[i] = i + 1.
    let n_raw_per_h: Vec<i32> = vec![128, 128, 128, 128];
    let n_raw_offset_per_h: Vec<i32> = vec![1, 2, 3, 4];
    // Mix in some comp coverage so both raw + comp paths are exercised.
    let n_comp_per_h: Vec<i32> = vec![500, 510, 520, 530];

    let max_raw_slot: usize = (0..b)
        .map(|i| (n_raw_offset_per_h[i] + n_raw_per_h[i]) as usize)
        .max()
        .unwrap();
    let max_comp_slot = *n_comp_per_h.iter().max().unwrap() as usize;

    // Deterministic pseudo-random fill (no rand dep), matching the
    // companion test's RNG so failures are reproducible.
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let qh: Vec<f32> = (0..b * n_head as usize * head_dim as usize)
        .map(|_| next())
        .collect();
    let rawh: Vec<f32> = (0..max_raw_slot * head_dim as usize)
        .map(|_| next())
        .collect();
    let comph: Vec<f32> = (0..max_comp_slot * head_dim as usize)
        .map(|_| next())
        .collect();
    let sinkh: Vec<f32> = (0..n_head as usize).map(|_| next()).collect();

    // CPU oracle: per (b, h) attention with per-token offset.
    let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
    let hd = head_dim as usize;
    let nh = n_head as usize;
    let mut out_cpu = vec![0f32; b * nh * hd];
    for bi in 0..b {
        let n_r = n_raw_per_h[bi] as usize;
        let off = n_raw_offset_per_h[bi] as usize;
        let n_c = n_comp_per_h[bi] as usize;
        for h in 0..nh {
            let q_base = (bi * nh + h) * hd;
            let mut scores = Vec::with_capacity(n_r + n_c);
            for r in 0..n_r {
                let kv_base = (r + off) * hd;
                let mut s = 0f32;
                for d in 0..hd {
                    s += qh[q_base + d] * rawh[kv_base + d];
                }
                scores.push(s * kq_scale);
            }
            for c in 0..n_c {
                let kv_base = c * hd;
                let mut s = 0f32;
                for d in 0..hd {
                    s += qh[q_base + d] * comph[kv_base + d];
                }
                scores.push(s * kq_scale);
            }
            let sink_h = sinkh[h];
            let max_s = scores.iter().copied().fold(sink_h, f32::max);
            let mut denom = (sink_h - max_s).exp();
            let mut weights = Vec::with_capacity(scores.len());
            for &s in &scores {
                let w = (s - max_s).exp();
                weights.push(w);
                denom += w;
            }
            let inv = 1.0f32 / denom;
            let o_base = (bi * nh + h) * hd;
            for d in 0..hd {
                let mut sum = 0f32;
                for r in 0..n_r {
                    sum += weights[r] * rawh[(r + off) * hd + d];
                }
                for c in 0..n_c {
                    sum += weights[n_r + c] * comph[c * hd + d];
                }
                out_cpu[o_base + d] = sum * inv;
            }
        }
    }

    // GPU run through the htiled chain.
    let mut q = DeviceBuffer::new(dgpu.id, qh.len())?;
    q.copy_from_host(&qh)?;
    let raw_kv_h = pack_f16(&rawh);
    let mut raw_kv = DeviceBuffer::<u16>::new(dgpu.id, raw_kv_h.len())?;
    raw_kv.copy_from_host(&raw_kv_h)?;
    let comp_kv_h = pack_f16(&comph);
    let mut comp_kv = DeviceBuffer::<u16>::new(dgpu.id, comp_kv_h.len())?;
    comp_kv.copy_from_host(&comp_kv_h)?;
    let mut sinks = DeviceBuffer::new(dgpu.id, sinkh.len())?;
    sinks.copy_from_host(&sinkh)?;
    let mut n_raw_per = DeviceBuffer::new(dgpu.id, b)?;
    n_raw_per.copy_from_host(&n_raw_per_h)?;
    let mut n_raw_offset_per = DeviceBuffer::new(dgpu.id, b)?;
    n_raw_offset_per.copy_from_host(&n_raw_offset_per_h)?;
    let mut n_comp_per = DeviceBuffer::new(dgpu.id, b)?;
    n_comp_per.copy_from_host(&n_comp_per_h)?;

    let out_len = b * nh * hd;
    let mut out_gpu = DeviceBuffer::new(dgpu.id, out_len)?;
    out_gpu.fill_zero()?;
    let mut scores_g =
        DeviceBuffer::new(dgpu.id, b * nh * ATTN_MIXED_MAX_KEYS as usize)?;
    scores_g.fill_zero()?;
    let n_total_max = n_raw_per_h
        .iter()
        .zip(n_comp_per_h.iter())
        .map(|(&r, &c)| (r + c) as u32)
        .max()
        .unwrap();

    let use_f16s = std::env::var_os("CHAIN_F16S").is_some();
    if use_f16s {
        attn.launch_score_batched_htiled_wmma_f16s(
            &stream,
            &mut scores_g,
            &q,
            &raw_kv,
            Some(&comp_kv),
            &n_raw_per,
            &n_raw_offset_per,
            &n_comp_per,
            None, // no CSA mask in this isolated bench — dense path
            n_head,
            head_dim,
            n_total_max,
            batch,
            0,
        )?;
        attn.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s(
            &stream,
            &mut out_gpu,
            &mut scores_g,
            &sinks,
            &raw_kv,
            Some(&comp_kv),
            &n_raw_per,
            &n_raw_offset_per,
            &n_comp_per,
            n_head,
            head_dim,
            batch,
            0,
        )?;
    } else {
        attn.launch_score_batched_htiled_wmma(
            &stream,
            &mut scores_g,
            &q,
            &raw_kv,
            Some(&comp_kv),
            &n_raw_per,
            &n_raw_offset_per,
            &n_comp_per,
            n_head,
            head_dim,
            n_total_max,
            batch,
        )?;
        attn.launch_softmax_wsum_batched_htiled_wmma_ldsv(
            &stream,
            &mut out_gpu,
            &mut scores_g,
            &sinks,
            &raw_kv,
            Some(&comp_kv),
            &n_raw_per,
            &n_raw_offset_per,
            &n_comp_per,
            n_head,
            head_dim,
            batch,
        )?;
    }
    stream.synchronize()?;

    let mut got = vec![0f32; out_len];
    out_gpu.copy_to_host(&mut got)?;

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (a, c) in out_cpu.iter().zip(got.iter()) {
        let abs = (a - c).abs();
        max_abs = max_abs.max(abs);
        let rel = abs / a.abs().max(1e-4);
        max_rel = max_rel.max(rel);
    }
    eprintln!(
        "htiled-with-offset vs CPU: max_abs={max_abs:.2e} max_rel={max_rel:.2e} \
         (B={batch}, n_raw_offset_per={n_raw_offset_per_h:?})"
    );
    // f32 reductions on GPU (32-lane shuffle tree) vs CPU (sequential) differ
    // by ULPs through softmax; 1e-3 is the same tolerance the companion test
    // uses against the monolithic GPU reference.
    if max_abs > 1e-3 {
        return Err(eyre!(
            "htiled with n_raw_offset_per != 0 diverges from CPU oracle: \
             max_abs={max_abs:.2e} (the cross-chunk SWA path is broken)"
        ));
    }
    eprintln!("PASS: htiled honors n_raw_offset_per (M52 cross-chunk SWA path is correct)");
    Ok(())
}

/// Correctness oracle for the fused FlashAttention-style WMMA kernel.
/// Compares `attention_mixed_fused_wmma` (online softmax, no scores buffer,
/// LDS-staged Q + V) against the production WMMA score → WMMA LDS-V smwsum
/// chain on the same synthetic inputs. Covers both `n_raw_offset_per=0`
/// (vanilla prefill) and `n_raw_offset_per!=0` (M52 cross-chunk SWA).
///
/// Tolerance: max_abs < 1e-3, rel_L2 < 1e-2. Looser than the strict
/// split-vs-mono test because online softmax has a different reduction
/// order than full-tensor softmax — f16 precision through the rescale
/// path bounds the difference.
#[test]
#[ignore]
fn prefill_attention_fused_matches_split() -> eyre::Result<()> {
    install_panic_handler()?;

    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let attn = AttentionMixed::for_arch(&arch)?;

    let head_dim = N_HEAD_DIM;
    let n_head = N_HEAD;
    let batch: u32 = 8;
    let b = batch as usize;

    // Two cases, run sequentially with the same buffers.
    let cases: &[(&str, Vec<i32>, Vec<i32>, Vec<i32>)] = &[
        (
            "offset=0",
            vec![128, 128, 128, 128, 96, 96, 96, 96],     // n_raw
            vec![0, 0, 0, 0, 0, 0, 0, 0],                  // offset
            vec![1500, 1510, 1520, 1530, 1400, 1410, 1420, 1430], // n_comp
        ),
        (
            "offset!=0 (M52)",
            vec![128, 128, 128, 128, 128, 128, 128, 128], // n_raw
            vec![1, 5, 9, 13, 17, 21, 25, 29],            // offset
            vec![1500, 1510, 1520, 1530, 1540, 1550, 1560, 1570], // n_comp
        ),
    ];

    let max_raw_slot: usize = cases
        .iter()
        .flat_map(|(_, nr, off, _)| nr.iter().zip(off.iter()))
        .map(|(&r, &o)| (r + o) as usize)
        .max()
        .unwrap();
    let max_comp_slot: usize = cases
        .iter()
        .flat_map(|(_, _, _, nc)| nc.iter())
        .map(|&c| c as usize)
        .max()
        .unwrap();

    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let qh: Vec<f32> = (0..b * n_head as usize * head_dim as usize)
        .map(|_| next())
        .collect();
    let rawh: Vec<f32> = (0..max_raw_slot * head_dim as usize)
        .map(|_| next())
        .collect();
    let comph: Vec<f32> = (0..max_comp_slot * head_dim as usize)
        .map(|_| next())
        .collect();
    let sinkh: Vec<f32> = (0..n_head as usize).map(|_| next()).collect();

    let mut q = DeviceBuffer::new(dgpu.id, qh.len())?;
    q.copy_from_host(&qh)?;
    let raw_kv_h = pack_f16(&rawh);
    let mut raw_kv = DeviceBuffer::<u16>::new(dgpu.id, raw_kv_h.len())?;
    raw_kv.copy_from_host(&raw_kv_h)?;
    let comp_kv_h = pack_f16(&comph);
    let mut comp_kv = DeviceBuffer::<u16>::new(dgpu.id, comp_kv_h.len())?;
    comp_kv.copy_from_host(&comp_kv_h)?;
    let mut sinks = DeviceBuffer::new(dgpu.id, sinkh.len())?;
    sinks.copy_from_host(&sinkh)?;

    let mut n_raw_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, b)?;
    let mut n_raw_offset_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, b)?;
    let mut n_comp_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, b)?;

    let out_len = b * n_head as usize * head_dim as usize;
    let mut out_ref = DeviceBuffer::new(dgpu.id, out_len)?;
    let mut out_fused = DeviceBuffer::new(dgpu.id, out_len)?;
    let mut scores_g: DeviceBuffer<f32> =
        DeviceBuffer::new(dgpu.id, b * n_head as usize * ATTN_MIXED_MAX_KEYS as usize)?;

    for (label, nr, off, nc) in cases.iter() {
        n_raw_per.copy_from_host(nr)?;
        n_raw_offset_per.copy_from_host(off)?;
        n_comp_per.copy_from_host(nc)?;
        out_ref.fill_zero()?;
        out_fused.fill_zero()?;
        scores_g.fill_zero()?;
        let n_total_max = nr
            .iter()
            .zip(nc.iter())
            .map(|(&r, &c)| (r + c) as u32)
            .max()
            .unwrap();

        // Reference: production WMMA score → WMMA LDS-V smwsum.
        attn.launch_score_batched_htiled_wmma(
            &stream,
            &mut scores_g,
            &q,
            &raw_kv,
            Some(&comp_kv),
            &n_raw_per,
            &n_raw_offset_per,
            &n_comp_per,
            n_head,
            head_dim,
            n_total_max,
            batch,
        )?;
        attn.launch_softmax_wsum_batched_htiled_wmma_ldsv(
            &stream,
            &mut out_ref,
            &mut scores_g,
            &sinks,
            &raw_kv,
            Some(&comp_kv),
            &n_raw_per,
            &n_raw_offset_per,
            &n_comp_per,
            n_head,
            head_dim,
            batch,
        )?;

        // Candidate: fused kernel (Steps 2-6).
        attn.launch_fused_wmma(
            &stream,
            &mut out_fused,
            &q,
            &sinks,
            &raw_kv,
            Some(&comp_kv),
            &n_raw_per,
            &n_raw_offset_per,
            &n_comp_per,
            n_head,
            head_dim,
            n_total_max,
            batch,
        )?;
        stream.synchronize()?;

        let mut r = vec![0f32; out_len];
        let mut f = vec![0f32; out_len];
        out_ref.copy_to_host(&mut r)?;
        out_fused.copy_to_host(&mut f)?;

        let mut max_abs = 0f32;
        let mut err_sq = 0f64;
        let mut ref_sq = 0f64;
        for (a, c) in r.iter().zip(f.iter()) {
            let d = (a - c).abs();
            max_abs = max_abs.max(d);
            err_sq += (d as f64) * (d as f64);
            ref_sq += (*a as f64) * (*a as f64);
        }
        let rel_l2 = (err_sq / ref_sq.max(1e-30)).sqrt();
        eprintln!(
            "[{label}] fused vs split-chain: max_abs={max_abs:.2e} rel_L2={rel_l2:.3e} \
             (B={batch}, n_total_max={n_total_max})"
        );
        if max_abs > 1e-3 || rel_l2 > 1e-2 {
            return Err(eyre!(
                "[{label}] fused diverges: max_abs={max_abs:.2e} rel_L2={rel_l2:.3e}"
            ));
        }
    }

    eprintln!("PASS: fused kernel matches split chain in both offset cases");
    Ok(())
}

/// Correctness: double-buffered smwsum_ldsv_db must match single-buffered
/// _ldsv exactly. Same WMMA ops in same order — the only difference is
/// pipelining of stage_v, which doesn't change the math. Expect max_abs = 0.
#[test]
#[ignore]
fn prefill_attention_ldsv_db_matches_ldsv() -> eyre::Result<()> {
    install_panic_handler()?;

    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let attn = AttentionMixed::for_arch(&arch)?;

    let head_dim = N_HEAD_DIM;
    let n_head = N_HEAD;
    let batch: u32 = 8;
    let b = batch as usize;

    let cases: &[(&str, Vec<i32>, Vec<i32>, Vec<i32>)] = &[
        (
            "offset=0 large-comp",
            vec![128, 128, 128, 128, 96, 96, 96, 96],
            vec![0; 8],
            vec![1500, 1510, 1520, 1530, 1400, 1410, 1420, 1430],
        ),
        (
            "offset!=0 (M52)",
            vec![128; 8],
            vec![1, 5, 9, 13, 17, 21, 25, 29],
            vec![1500, 1510, 1520, 1530, 1540, 1550, 1560, 1570],
        ),
        (
            "partial-tile boundary",
            vec![17, 31, 33, 47, 1, 16, 15, 0],   // exercises n_raw % 16 != 0
            vec![0; 8],
            vec![17, 31, 33, 47, 1, 16, 15, 0],   // and n_comp % 16 != 0
        ),
    ];

    let max_raw_slot: usize = cases
        .iter()
        .flat_map(|(_, nr, off, _)| nr.iter().zip(off.iter()))
        .map(|(&r, &o)| (r + o) as usize)
        .max()
        .unwrap();
    let max_comp_slot: usize = cases
        .iter()
        .flat_map(|(_, _, _, nc)| nc.iter())
        .map(|&c| c as usize)
        .max()
        .unwrap();

    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let rawh: Vec<f32> = (0..max_raw_slot.max(1) * head_dim as usize)
        .map(|_| next())
        .collect();
    let comph: Vec<f32> = (0..max_comp_slot.max(1) * head_dim as usize)
        .map(|_| next())
        .collect();
    let sinkh: Vec<f32> = (0..n_head as usize).map(|_| next()).collect();
    // Pre-generate scores so we can reset to the same input for each kernel.
    let scores_seed: Vec<f32> = (0..b * n_head as usize * ATTN_MIXED_MAX_KEYS as usize)
        .map(|_| next() * 4.0)
        .collect();

    let raw_kv_h = pack_f16(&rawh);
    let mut raw_kv = DeviceBuffer::<u16>::new(dgpu.id, raw_kv_h.len())?;
    raw_kv.copy_from_host(&raw_kv_h)?;
    let comp_kv_h = pack_f16(&comph);
    let mut comp_kv = DeviceBuffer::<u16>::new(dgpu.id, comp_kv_h.len())?;
    comp_kv.copy_from_host(&comp_kv_h)?;
    let mut sinks = DeviceBuffer::new(dgpu.id, sinkh.len())?;
    sinks.copy_from_host(&sinkh)?;

    let mut n_raw_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, b)?;
    let mut n_raw_offset_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, b)?;
    let mut n_comp_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, b)?;

    let out_len = b * n_head as usize * head_dim as usize;
    let mut out_ref = DeviceBuffer::new(dgpu.id, out_len)?;
    let mut out_db = DeviceBuffer::new(dgpu.id, out_len)?;
    let mut out_regv = DeviceBuffer::new(dgpu.id, out_len)?;
    let mut scores_g: DeviceBuffer<f32> =
        DeviceBuffer::new(dgpu.id, scores_seed.len())?;

    for (label, nr, off, nc) in cases.iter() {
        n_raw_per.copy_from_host(nr)?;
        n_raw_offset_per.copy_from_host(off)?;
        n_comp_per.copy_from_host(nc)?;

        // Reference: single-buffered _ldsv.
        scores_g.copy_from_host(&scores_seed)?;
        out_ref.fill_zero()?;
        attn.launch_softmax_wsum_batched_htiled_wmma_ldsv(
            &stream,
            &mut out_ref,
            &mut scores_g,
            &sinks,
            &raw_kv,
            Some(&comp_kv),
            &n_raw_per,
            &n_raw_offset_per,
            &n_comp_per,
            n_head,
            head_dim,
            batch,
        )?;

        // Candidate 1: _ldsv_db (LDS double-buffered).
        scores_g.copy_from_host(&scores_seed)?;
        out_db.fill_zero()?;
        attn.launch_softmax_wsum_batched_htiled_wmma_ldsv_db(
            &stream,
            &mut out_db,
            &mut scores_g,
            &sinks,
            &raw_kv,
            Some(&comp_kv),
            &n_raw_per,
            &n_raw_offset_per,
            &n_comp_per,
            n_head,
            head_dim,
            batch,
        )?;

        // Candidate 2: _regv_db (register-V double-buffered).
        scores_g.copy_from_host(&scores_seed)?;
        out_regv.fill_zero()?;
        attn.launch_softmax_wsum_batched_htiled_wmma_regv_db(
            &stream,
            &mut out_regv,
            &mut scores_g,
            &sinks,
            &raw_kv,
            Some(&comp_kv),
            &n_raw_per,
            &n_raw_offset_per,
            &n_comp_per,
            n_head,
            head_dim,
            batch,
        )?;
        stream.synchronize()?;

        let mut r = vec![0f32; out_len];
        let mut d = vec![0f32; out_len];
        let mut v = vec![0f32; out_len];
        out_ref.copy_to_host(&mut r)?;
        out_db.copy_to_host(&mut d)?;
        out_regv.copy_to_host(&mut v)?;

        let mut max_abs_db = 0f32;
        let mut max_abs_regv = 0f32;
        for ((a, c), w) in r.iter().zip(d.iter()).zip(v.iter()) {
            max_abs_db = max_abs_db.max((a - c).abs());
            max_abs_regv = max_abs_regv.max((a - w).abs());
        }
        eprintln!(
            "[{label}] _ldsv_db vs _ldsv: max_abs={max_abs_db:.2e}   \
             _regv_db vs _ldsv: max_abs={max_abs_regv:.2e}"
        );
        if max_abs_db > 0.0 {
            return Err(eyre!(
                "[{label}] _ldsv_db diverges: max_abs={max_abs_db:.2e}"
            ));
        }
        if max_abs_regv > 0.0 {
            return Err(eyre!(
                "[{label}] _regv_db diverges: max_abs={max_abs_regv:.2e}"
            ));
        }
    }
    eprintln!("PASS: both DB variants bit-identical to _ldsv across all cases");
    Ok(())
}

