//! Standalone PREFILL attention probe — no GGUF, no model load, no mmap.
//! Runs the batched split attention chain
//! (`attention_mixed_score_batched_htiled` + `attention_mixed_softmax_wsum_batched_htiled`)
//! on a single layer with synthetic buffers, over a batch of B tokens.
//! Mirrors `bench_attention_isolated.rs` (the decode probe) for prefill.
//!
//! Buffers are zero-initialized. The math is data-independent for timing
//! (FMAs and BW don't depend on values), so timing is representative;
//! outputs are zero. Per-token causal prefixes are set uniformly to
//! (n_raw, n_comp) — close enough for timing (real chunks vary by ±B/ratio,
//! negligible at depth).
//!
//! Env vars:
//!   BENCH_N_RAW    n_raw (SWA cap = 128). Default 128.
//!   BENCH_N_COMP   n_comp. ratio=4 @ 64K → 16384; ratio=128 @ 64K → 512.
//!                  Default 16384.
//!   BENCH_BATCH    tokens per chunk (≤ B_MAX=256). Default 256.
//!   BENCH_ITERS    timed iterations. Default 50.
//!   BENCH_WARMUP   warmup iterations (discarded). Default 3.
//!   BENCH_PHASE    "both" | "score_ht" | "smwsum_ht" | "score_wmma" |
//!                  "smwsum_wmma" | "mono" — which to time. "both" runs the
//!                  production htiled score + htiled smwsum chain.
//!                  "mono" runs the old monolithic attention_mixed_batched
//!                  (requires n_raw+n_comp ≤ ATTN_PREFILL_LDS_MAX=2304).
//!                  Default "both".
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
use v4flash_kernels::attention::{AttentionMixed, ATTN_MIXED_MAX_KEYS, ATTN_PREFILL_LDS_MAX};
use v4flash_kernels::config::{N_HEAD, N_HEAD_DIM};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201"))
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
    let phase: String = std::env::var("BENCH_PHASE").unwrap_or_else(|_| "both".to_string());
    let do_score_ht = phase == "both" || phase == "score_ht";
    let do_score_wmma = phase == "score_wmma";
    let do_smwsum_ht = phase == "both" || phase == "smwsum_ht";
    let do_smwsum_wmma = phase == "smwsum_wmma";
    let do_mono = phase == "mono";

    let n_total = n_raw + n_comp;
    if n_total > ATTN_MIXED_MAX_KEYS {
        return Err(eyre!(
            "n_raw+n_comp={n_total} exceeds cap {ATTN_MIXED_MAX_KEYS}"
        ));
    }
    if do_mono && n_total > ATTN_PREFILL_LDS_MAX {
        return Err(eyre!(
            "phase=mono needs n_raw+n_comp={n_total} ≤ ATTN_PREFILL_LDS_MAX={ATTN_PREFILL_LDS_MAX} \
             (the monolithic kernel's LDS scores[] array overflows above that)"
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

    // raw_kv: [SWA_WINDOW=128, head_dim] (shared across batch)
    let n_raw_capacity: usize = 128;
    let mut raw_kv: DeviceBuffer<f32> =
        DeviceBuffer::new(dgpu.id, n_raw_capacity * (head_dim as usize))?;
    raw_kv.fill_zero()?;

    // comp_kv: [n_comp, head_dim] (shared across batch)
    let comp_kv: Option<DeviceBuffer<f32>> = if n_comp > 0 {
        let mut c: DeviceBuffer<f32> =
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
        if do_mono {
            attn.launch_batched(
                stream,
                out,
                &q,
                &raw_kv,
                comp_kv.as_ref(),
                &sinks,
                &n_raw_per,
                &n_comp_per,
                n_head,
                head_dim,
                batch,
            )?;
            return Ok(());
        }
        if do_score_ht {
            attn.launch_score_batched_htiled(
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
        if do_score_wmma {
            attn.launch_score_batched_htiled_wmma(
                stream,
                scores,
                &q,
                &raw_kv,
                comp_kv.as_ref(),
                &n_raw_per,
                &n_comp_per,
                n_head,
                head_dim,
                n_total,
                batch,
            )?;
        }
        if do_smwsum_ht {
            attn.launch_softmax_wsum_batched_htiled(
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
        if do_smwsum_wmma {
            attn.launch_softmax_wsum_batched_htiled_wmma(
                stream,
                out,
                scores,
                &sinks,
                &raw_kv,
                comp_kv.as_ref(),
                &n_raw_per,
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

/// Correctness: the batched split must match the proven monolithic
/// `attention_mixed_batched` on random data, at a shape both handle
/// (n_total ≤ ATTN_PREFILL_LDS_MAX). Per-token prefixes are varied so the
/// causal-masking paths (different n_raw/n_comp per token) are exercised.
#[test]
#[ignore]
fn prefill_attention_split_matches_mono() -> eyre::Result<()> {
    install_panic_handler()?;

    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let attn = AttentionMixed::for_arch(&arch)?;

    let head_dim = N_HEAD_DIM;
    let n_head = N_HEAD;
    let batch: u32 = 17; // odd, exercises non-aligned batch
    let b = batch as usize;
    let n_raw_max: u32 = 128;
    let n_comp_max: u32 = 2000; // n_total ≤ 128+2000 = 2128 ≤ 2304

    // Deterministic pseudo-random fill (no rand dep).
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
    let rawh: Vec<f32> = (0..n_raw_max as usize * head_dim as usize)
        .map(|_| next())
        .collect();
    let comph: Vec<f32> = (0..n_comp_max as usize * head_dim as usize)
        .map(|_| next())
        .collect();
    let sinkh: Vec<f32> = (0..n_head as usize).map(|_| next()).collect();
    // Per-token prefixes: vary causally so masking differs per token.
    let nrawh: Vec<i32> = (0..b).map(|i| (96 + (i as i32 % 33)).min(128)).collect();
    let ncomph: Vec<i32> = (0..b).map(|i| 1500 + (i as i32 * 17) % 500).collect();

    let mut q = DeviceBuffer::new(dgpu.id, qh.len())?;
    q.copy_from_host(&qh)?;
    let mut raw_kv = DeviceBuffer::new(dgpu.id, rawh.len())?;
    raw_kv.copy_from_host(&rawh)?;
    let mut comp_kv = DeviceBuffer::new(dgpu.id, comph.len())?;
    comp_kv.copy_from_host(&comph)?;
    let mut sinks = DeviceBuffer::new(dgpu.id, sinkh.len())?;
    sinks.copy_from_host(&sinkh)?;
    let mut n_raw_per = DeviceBuffer::new(dgpu.id, b)?;
    n_raw_per.copy_from_host(&nrawh)?;
    let mut n_raw_offset_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, b)?;
    n_raw_offset_per.copy_from_host(&vec![0i32; b])?;
    let mut n_comp_per = DeviceBuffer::new(dgpu.id, b)?;
    n_comp_per.copy_from_host(&ncomph)?;

    let out_len = b * n_head as usize * head_dim as usize;
    let mut out_mono = DeviceBuffer::new(dgpu.id, out_len)?;
    out_mono.fill_zero()?;
    let mut out_htfull = DeviceBuffer::new(dgpu.id, out_len)?;
    out_htfull.fill_zero()?;
    let mut scores_htfull =
        DeviceBuffer::new(dgpu.id, b * n_head as usize * ATTN_MIXED_MAX_KEYS as usize)?;
    scores_htfull.fill_zero()?;
    let mut scores_wmma =
        DeviceBuffer::new(dgpu.id, b * n_head as usize * ATTN_MIXED_MAX_KEYS as usize)?;
    scores_wmma.fill_zero()?;
    // Fresh f32 score reference — NOT consumed by softmax (the wsum kernels
    // overwrite scores_g in place with weights), so it stays as raw scores.
    let mut scores_f32ref =
        DeviceBuffer::new(dgpu.id, b * n_head as usize * ATTN_MIXED_MAX_KEYS as usize)?;
    scores_f32ref.fill_zero()?;

    let n_total_max = nrawh
        .iter()
        .zip(ncomph.iter())
        .map(|(&r, &c)| (r + c) as u32)
        .max()
        .unwrap();

    // Monolithic reference.
    attn.launch_batched(
        &stream,
        &mut out_mono,
        &q,
        &raw_kv,
        Some(&comp_kv),
        &sinks,
        &n_raw_per,
        &n_comp_per,
        n_head,
        head_dim,
        batch,
    )?;
    // Full head-tiled chain (htiled score + htiled smwsum) — the production
    // batched attention path.
    attn.launch_score_batched_htiled(
        &stream,
        &mut scores_htfull,
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
    attn.launch_softmax_wsum_batched_htiled(
        &stream,
        &mut out_htfull,
        &mut scores_htfull,
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
    // Fresh raw f32 score reference (not softmaxed).
    attn.launch_score_batched_htiled(
        &stream,
        &mut scores_f32ref,
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
    // WMMA score (f16 GEMM) — validated against the f32 score directly.
    attn.launch_score_batched_htiled_wmma(
        &stream,
        &mut scores_wmma,
        &q,
        &raw_kv,
        Some(&comp_kv),
        &n_raw_per,
        &n_comp_per,
        n_head,
        head_dim,
        n_total_max,
        batch,
    )?;
    stream.synchronize()?;

    {
        let nelem = b * n_head as usize * ATTN_MIXED_MAX_KEYS as usize;
        let mut sf = vec![0f32; nelem];
        let mut sw = vec![0f32; nelem];
        scores_f32ref.copy_to_host(&mut sf)?;
        scores_wmma.copy_to_host(&mut sw)?;
        let mut err_sq = 0f64;
        let mut ref_sq = 0f64;
        let mut max_abs = 0f32;
        for (a, c) in sf.iter().zip(sw.iter()) {
            let d = (a - c).abs();
            max_abs = max_abs.max(d);
            err_sq += (d as f64) * (d as f64);
            ref_sq += (*a as f64) * (*a as f64);
        }
        let rel_l2 = (err_sq / ref_sq).sqrt();
        eprintln!(
            "WMMA-score vs f32-score: rel_L2={rel_l2:.3e} max_abs={max_abs:.2e} \
             (B={batch}, n_total_max={n_total_max})"
        );
        if rel_l2 > 1e-2 {
            return Err(eyre!(
                "WMMA score diverges from f32 score: rel_L2={rel_l2:.3e} (layout likely wrong)"
            ));
        }
    }

    let mut m = vec![0f32; out_len];
    let mut shtf = vec![0f32; out_len];
    out_mono.copy_to_host(&mut m)?;
    out_htfull.copy_to_host(&mut shtf)?;

    let cmp = |label: &str, ref_v: &[f32], got: &[f32]| -> eyre::Result<()> {
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for (a, c) in ref_v.iter().zip(got.iter()) {
            let abs = (a - c).abs();
            if abs > max_abs {
                max_abs = abs;
            }
            let rel = abs / (a.abs().max(1e-4));
            if rel > max_rel {
                max_rel = rel;
            }
        }
        eprintln!(
            "{label} vs mono: max_abs={max_abs:.2e} max_rel={max_rel:.2e} (B={batch}, \
             n_total_max={n_total_max})"
        );
        // Different reduction order (32-lane shuffle vs 256-thread tree; wave-
        // parallel vs single-thread softmax) → fp differences only.
        if max_abs > 1e-3 {
            return Err(eyre!("{label} diverges from mono: max_abs={max_abs:.2e}"));
        }
        Ok(())
    };
    cmp("htiled-full", &m, &shtf)?;
    eprintln!("PASS: head-tiled split chain matches monolithic within 1e-3");
    Ok(())
}

/// M52 cross-chunk SWA regression test: the htiled chain must correctly honor
/// `n_raw_offset_per[b] != 0` (per-token starting slot into the oversized raw
/// KV cache). Validated against a CPU reference that re-implements the
/// per-token attention math with the same per-token offset semantics.
///
/// The monolithic kernel (used as the oracle in `prefill_attention_split_matches_mono`)
/// reads `raw_kv[r * head_dim]` and cannot express per-token offsets, so it's
/// useless for this case — hence the CPU oracle. This is the test the T=540
/// regression would have caught at `cargo test` time instead of via manual
/// wine-tasting essays.
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
    let mut raw_kv = DeviceBuffer::new(dgpu.id, rawh.len())?;
    raw_kv.copy_from_host(&rawh)?;
    let mut comp_kv = DeviceBuffer::new(dgpu.id, comph.len())?;
    comp_kv.copy_from_host(&comph)?;
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

    attn.launch_score_batched_htiled(
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
    attn.launch_softmax_wsum_batched_htiled(
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
