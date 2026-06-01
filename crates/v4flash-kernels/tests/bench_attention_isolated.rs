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
    let do_score = phase == "both" || phase == "score";
    let do_smwsum = phase == "both" || phase == "smwsum";

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
    q.fill_zero()?;

    // raw_kv: [n_raw_max=SWA_WINDOW=128, head_dim] f16
    let n_raw_capacity: usize = 128;
    let mut raw_kv: DeviceBuffer<u16> = DeviceBuffer::new(
        dgpu.id, n_raw_capacity * (head_dim as usize))?;
    raw_kv.fill_zero()?;

    // comp_kv: [n_comp, head_dim] f16
    let mut comp_kv: Option<DeviceBuffer<u16>> = if n_comp > 0 {
        let mut b: DeviceBuffer<u16> = DeviceBuffer::new(
            dgpu.id, (n_comp as usize) * (head_dim as usize))?;
        b.fill_zero()?;
        Some(b)
    } else { None };

    // sinks: [n_head]
    let mut sinks: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_head as usize)?;
    sinks.fill_zero()?;

    // attn_scores scratch: [n_head, ATTN_MIXED_MAX_KEYS]
    let mut scores: DeviceBuffer<f32> = DeviceBuffer::new(
        dgpu.id, (n_head as usize) * (ATTN_MIXED_MAX_KEYS as usize))?;
    scores.fill_zero()?;

    // out: [n_head, head_dim]
    let mut out: DeviceBuffer<f32> = DeviceBuffer::new(
        dgpu.id, (n_head as usize) * (head_dim as usize))?;
    out.fill_zero()?;

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
        Ok(())
    };

    // Warmup
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
