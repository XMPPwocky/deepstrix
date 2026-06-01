//! Standalone attention probe — no GGUF, no model load, no mmap. Runs
//! the split decode attention chain (`attention_mixed_score` +
//! `attention_mixed_softmax_wsum`) on a single layer with synthetic
//! buffers. Designed for clean rocprofv3 PMC + perf iteration.
//!
//! Buffers are zero-initialized. The math is data-independent for
//! timing (FMAs and BW don't depend on values), so timing is
//! representative; outputs are zero.
//!
//! Env vars:
//!   BENCH_N_RAW    n_raw (SWA cap = 128). Default 128.
//!   BENCH_N_COMP   n_comp. For ratio=4 at pos=64K, set 16000.
//!                  For ratio=128 at pos=64K, set 500. Default 500.
//!   BENCH_ITERS    timed iterations. Default 100.
//!   BENCH_WARMUP   warmup iterations (discarded). Default 5.
//!   BENCH_PHASE    "both" | "score" | "smwsum" — which kernels to time.
//!                  Default "both".
//!
//! Run normally:
//!   HIP_VISIBLE_DEVICES=0,1 BENCH_N_COMP=16000 \
//!     nix develop -c cargo test --release -p v4flash-kernels \
//!     --test bench_attention_isolated bench_attention_isolated \
//!     -- --ignored --nocapture
//!
//! Run under rocprofv3:
//!   nix develop -c bash -c '
//!     export PATH=/nix/store/c9874ja4w6hkfbrv2fsx0r6zplrplwni-rocprofiler-sdk-7.2.3/bin:$PATH
//!     cargo build --release -p v4flash-kernels --test bench_attention_isolated
//!     TEST_BIN=$(find target/release/deps -name "bench_attention_isolated-*" -executable -not -name "*.d" | head -1)
//!     rocprofv3 -i /tmp/attn_counters.txt -d /tmp/attn_prof -o run -- \
//!       "$TEST_BIN" bench_attention_isolated --ignored --nocapture
//!   '

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

fn percentile(xs_sorted: &[f32], p: f32) -> f32 {
    if xs_sorted.is_empty() { return 0.0; }
    let k = ((xs_sorted.len() - 1) as f32 * p / 100.0).round() as usize;
    xs_sorted[k.min(xs_sorted.len() - 1)]
}

#[test]
#[ignore]
fn bench_attention_isolated() -> eyre::Result<()> {
    install_panic_handler()?;

    let n_raw: u32 = std::env::var("BENCH_N_RAW")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let n_comp: u32 = std::env::var("BENCH_N_COMP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(500);
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let phase: String = std::env::var("BENCH_PHASE")
        .unwrap_or_else(|_| "both".to_string());
    // "score" / "smwsum" / "both" — single-token decode kernels.
    // "batched_score" / "batched_smwsum" / "batched_both" — batched WMMA
    //   kernels run at B=1 (decode shape, but with the LDS-V+f16-scores
    //   structure we shipped for prefill). The data layout is compatible.
    let do_score        = phase == "both" || phase == "score";
    let do_smwsum       = phase == "both" || phase == "smwsum";
    let do_smwsum_ldsv  = phase == "smwsum_ldsv";
    let do_smwsum_ksplit = phase == "smwsum_ksplit";
    let do_b_score      = phase == "batched_both" || phase == "batched_score";
    let do_b_smwsum     = phase == "batched_both" || phase == "batched_smwsum";
    let do_b1_score     = phase == "b1_score";

    if n_raw + n_comp > ATTN_MIXED_MAX_KEYS {
        return Err(eyre!(
            "n_raw+n_comp={} exceeds cap {ATTN_MIXED_MAX_KEYS}",
            n_raw + n_comp
        ));
    }
    let n_total = n_raw + n_comp;
    let ratio_hint = if n_comp > 1000 { "ratio=4" }
                     else if n_comp > 0 { "ratio=128" }
                     else { "ratio=0 (no comp)" };
    eprintln!(
        "isolated attention probe: n_raw={n_raw} n_comp={n_comp} \
         n_total={n_total} {ratio_hint}, iters={iters} warmup={warmup} \
         phase={phase}"
    );

    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let attn = AttentionMixed::for_arch(&arch)?;

    // === Synthetic buffers ===
    let head_dim = N_HEAD_DIM;
    let n_head = N_HEAD;

    // q: [n_head, head_dim] post-RoPE
    let mut q: DeviceBuffer<f32> = DeviceBuffer::new(
        dgpu.id, (n_head as usize) * (head_dim as usize))?;
    // ATT_SKIP_FILL=1: skip the data fill_zero kernels so the warmup
    // smwsum is the FIRST dispatched kernel. Lets `rocprofv3 --att
    // --att-consecutive-kernels 1` capture smwsum without a regex
    // filter (which hangs ATT per [[rocprofv3-rdna4]]).
    let skip_fill = std::env::var_os("ATT_SKIP_FILL").is_some();
    if !skip_fill { q.fill_zero()?; } else { drop(()); }

    // raw_kv: [n_raw_max=SWA_WINDOW=128, head_dim] f16
    let n_raw_capacity: usize = 128;
    let mut raw_kv: DeviceBuffer<u16> = DeviceBuffer::new(
        dgpu.id, n_raw_capacity * (head_dim as usize))?;
    if !skip_fill { raw_kv.fill_zero()?; }

    // comp_kv: [n_comp, head_dim] f16
    let mut comp_kv: Option<DeviceBuffer<u16>> = if n_comp > 0 {
        let mut b: DeviceBuffer<u16> = DeviceBuffer::new(
            dgpu.id, (n_comp as usize) * (head_dim as usize))?;
        if !skip_fill { b.fill_zero()?; }
        Some(b)
    } else { None };

    // sinks: [n_head]
    let mut sinks: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_head as usize)?;
    if !skip_fill { sinks.fill_zero()?; }

    // attn_scores scratch: [n_head, ATTN_MIXED_MAX_KEYS]
    let mut scores: DeviceBuffer<f32> = DeviceBuffer::new(
        dgpu.id, (n_head as usize) * (ATTN_MIXED_MAX_KEYS as usize))?;
    if !skip_fill { scores.fill_zero()?; }

    // out: [n_head, head_dim]
    let mut out: DeviceBuffer<f32> = DeviceBuffer::new(
        dgpu.id, (n_head as usize) * (head_dim as usize))?;
    if !skip_fill { out.fill_zero()?; }

    // Buffers for the K-split decode smwsum pipeline.
    let k_split: u32 = std::env::var("BENCH_KSPLIT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let mut partials: DeviceBuffer<f32> = DeviceBuffer::new(
        dgpu.id, (k_split as usize) * (n_head as usize) * (head_dim as usize))?;
    if !skip_fill { partials.fill_zero()?; }
    let mut inv_per_head: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_head as usize)?;
    if !skip_fill { inv_per_head.fill_zero()?; }

    // Per-batch counters for the batched kernels (B=1).
    let mut n_raw_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, 1)?;
    n_raw_per.copy_from_host(&[n_raw as i32])?;
    let mut n_raw_offset_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, 1)?;
    n_raw_offset_per.copy_from_host(&[0i32])?;
    let mut n_comp_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, 1)?;
    n_comp_per.copy_from_host(&[n_comp as i32])?;

    // === Run loop ===
    let mut launch_iter = |stream: &Stream,
                           scores: &mut DeviceBuffer<f32>,
                           out: &mut DeviceBuffer<f32>|
     -> eyre::Result<()> {
        if do_score {
            attn.launch_score(
                stream, scores, &q, &raw_kv, comp_kv.as_ref(),
                n_head, head_dim, n_raw, n_comp,
            )?;
        }
        if do_smwsum {
            attn.launch_softmax_wsum(
                stream, out, scores, &sinks, &raw_kv, comp_kv.as_ref(),
                n_head, head_dim, n_raw, n_comp,
            )?;
        }
        if do_smwsum_ldsv {
            attn.launch_softmax_wsum_ldsv(
                stream, out, scores, &sinks, &raw_kv, comp_kv.as_ref(),
                n_head, head_dim, n_raw, n_comp,
            )?;
        }
        if do_smwsum_ksplit {
            // 3-pass: softmax_only → wsum_b1_htiled_ksplit_ldsv → reduce.
            attn.launch_softmax_only(
                stream, scores, &sinks, &mut inv_per_head,
                n_head, n_raw, n_comp,
            )?;
            attn.launch_wsum_b1_htiled_ksplit_ldsv(
                stream, &mut partials, scores, &raw_kv, comp_kv.as_ref(),
                n_head, head_dim, n_raw, n_comp, k_split,
            )?;
            attn.launch_reduce_partials_apply_inv(
                stream, out, &partials, &inv_per_head,
                n_head, head_dim, k_split,
            )?;
        }
        if do_b_score {
            attn.launch_score_batched_htiled_wmma_f16s(
                stream, scores, &q, &raw_kv, comp_kv.as_ref(),
                &n_raw_per, &n_raw_offset_per, &n_comp_per,
                n_head, head_dim, n_total, /*batch=*/1,
            )?;
        }
        if do_b1_score {
            attn.launch_score_b1_htiled_wmma_f16s(
                stream, scores, &q, &raw_kv, comp_kv.as_ref(),
                n_raw, /*raw_off=*/0, n_comp, n_head, head_dim, n_total,
            )?;
        }
        if do_b_smwsum {
            attn.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s(
                stream, out, scores, &sinks, &raw_kv, comp_kv.as_ref(),
                &n_raw_per, &n_raw_offset_per, &n_comp_per,
                n_head, head_dim, /*batch=*/1,
            )?;
        }
        Ok(())
    };

    // Warmup
    for _ in 0..warmup {
        launch_iter(&stream, &mut scores, &mut out)?;
    }
    stream.synchronize()?;

    // Correctness: when comparing score vs batched_score (or smwsum vs
    // batched_smwsum), run one round of each on the same inputs and diff
    // the output buffers. Non-zero synth input would give a stronger
    // signal, but even all-zero we'd catch a structural bug (both should
    // produce identical zero output).
    if std::env::var_os("BENCH_DIFF_BATCHED").is_some() {
        let mut s_ref  = vec![0.0f32; (n_head as usize) * (ATTN_MIXED_MAX_KEYS as usize)];
        let mut s_test = vec![0.0f32; (n_head as usize) * (ATTN_MIXED_MAX_KEYS as usize)];
        let mut o_ref  = vec![0.0f32; (n_head as usize) * (head_dim as usize)];
        let mut o_test = vec![0.0f32; (n_head as usize) * (head_dim as usize)];
        // single-token reference: score then smwsum
        attn.launch_score(&stream, &mut scores, &q, &raw_kv, comp_kv.as_ref(),
            n_head, head_dim, n_raw, n_comp)?;
        stream.synchronize()?;
        scores.copy_to_host(&mut s_ref)?;
        attn.launch_softmax_wsum(&stream, &mut out, &mut scores, &sinks,
            &raw_kv, comp_kv.as_ref(), n_head, head_dim, n_raw, n_comp)?;
        stream.synchronize()?;
        out.copy_to_host(&mut o_ref)?;

        // batched WMMA at B=1
        attn.launch_score_batched_htiled_wmma_f16s(&stream, &mut scores, &q,
            &raw_kv, comp_kv.as_ref(),
            &n_raw_per, &n_raw_offset_per, &n_comp_per,
            n_head, head_dim, n_total, 1)?;
        stream.synchronize()?;
        scores.copy_to_host(&mut s_test)?;
        attn.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s(&stream, &mut out,
            &mut scores, &sinks, &raw_kv, comp_kv.as_ref(),
            &n_raw_per, &n_raw_offset_per, &n_comp_per,
            n_head, head_dim, 1)?;
        stream.synchronize()?;
        out.copy_to_host(&mut o_test)?;

        let mut s_max = 0.0f32;
        for (a, b) in s_ref.iter().zip(s_test.iter()) {
            let d = (a - b).abs(); if d > s_max { s_max = d; }
        }
        let mut o_max = 0.0f32;
        for (a, b) in o_ref.iter().zip(o_test.iter()) {
            let d = (a - b).abs(); if d > o_max { o_max = d; }
        }
        eprintln!("DIFF batched-vs-single: scores max_abs={s_max:.3e}, out max_abs={o_max:.3e}");
    }

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
        "attention {phase}: min={:.3} mean={:.3} p50={:.3} p90={:.3} p99={:.3} max={:.3} (ms)",
        min, mean, p50, p90, p99, max,
    );
    // Per-token estimate (one layer × per-token cost; the model has
    // ~21 ratio=4 + ~20 ratio=128 compressed layers + 2 dense).
    eprintln!(
        "= {:.1} µs per layer; rough per-token contribution at 41 \
         compressed layers: {:.1} ms",
        p50 * 1000.0, p50 * 41.0,
    );
    Ok(())
}
