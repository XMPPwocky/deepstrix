//! Isolated probe for the 5 kernels in the mhc_pre_attn / mhc_pre_ffn
//! chain at production decode shape. The chain currently runs as a
//! captured 5-kernel graph; this bench measures each kernel separately
//! to see where time actually goes.
//!
//! Shapes (V4-Flash decode B=1):
//!   1. rms_nw        : in/out [HC_DIM=16384]
//!   2. f16.matvec    : weight [HC_MIX_DIM=24, HC_DIM=16384] f16, in [HC_DIM], out [HC_MIX_DIM]
//!   3. hc_sinkhorn   : mix [HC_MIX_DIM=24], scale [3], base [HC_MIX_DIM]
//!   4. hc_weighted   : x [N_HC=4, N_EMBD=4096], weights [N_HC], out [N_EMBD]
//!   5. rms_w         : in/out [N_EMBD=4096], weight [N_EMBD]
//!
//! ATT_SKIP_FILL=1 skips the fill_zero kernels so the WARMUP kernel under
//! BENCH_PHASE is the first dispatched kernel — lets rocprofv3 --att
//! capture it with --att-consecutive-kernels 1 and no regex (which hangs).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, N_EMBD, N_HC, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS};
use v4flash_kernels::{F16Matvec, HcSinkhorn, HcWeightedSum, RmsNorm, RmsNormNoWeight, RmsNormNoWeightMultiWG};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        let arch = d.properties()?.gcn_arch_name;
        if arch.starts_with("gfx1201") || arch.starts_with("gfx1200") {
            return Ok(d);
        }
    }
    Err(eyre!("no dGPU (gfx120x) found"))
}

fn percentile(xs: &[f32], p: f32) -> f32 {
    let idx = ((p / 100.0) * (xs.len() as f32 - 1.0)).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn stats(walls: &mut [f32], label: &str) {
    walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = walls[0];
    let p50 = percentile(walls, 50.0);
    let p90 = percentile(walls, 90.0);
    let p99 = percentile(walls, 99.0);
    let max = walls[walls.len() - 1];
    eprintln!(
        "{label}: min={:.4} p50={:.4} p90={:.4} p99={:.4} max={:.4} (ms = {:.1} µs p50)",
        min, p50, p90, p99, max, p50 * 1000.0
    );
}

#[test]
#[ignore]
fn bench_mhc_pre_isolated() -> eyre::Result<()> {
    install_panic_handler()?;

    let iters: usize = std::env::var("BENCH_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let phase: String = std::env::var("BENCH_PHASE")
        .unwrap_or_else(|_| "all".to_string());
    let skip_fill = std::env::var_os("ATT_SKIP_FILL").is_some();

    let dgpu = pick_dgpu()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    eprintln!("device: {arch}, iters={iters} warmup={warmup} phase={phase}");
    dgpu.set_current()?;
    let stream = Stream::new(dgpu.id)?;

    // Kernels
    let rms_nw = RmsNormNoWeight::for_arch(&arch)?;
    let rms_nw_mw = RmsNormNoWeightMultiWG::for_arch(&arch)?;
    let mut rms_partial: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, 64)?;
    let rms_w = RmsNorm::for_arch(&arch)?;
    let f16 = F16Matvec::for_arch(&arch)?;
    let hc_sink = HcSinkhorn::for_arch(&arch)?;
    let hc_wsum = HcWeightedSum::for_arch(&arch)?;

    // Buffers (production shape).
    let mut residual: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, HC_DIM as usize)?;
    let mut flat:     DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, HC_DIM as usize)?;
    let mut mix:      DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, HC_MIX_DIM as usize)?;
    let mut split:    DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, HC_MIX_DIM as usize)?;
    let mut attn_cur: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, N_EMBD as usize)?;
    let mut out:      DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, N_EMBD as usize)?;

    // f16 weight: [HC_MIX_DIM, HC_DIM] f16 = 24 × 16384 × 2 = 768 KB.
    let hc_fn_w_bytes = (HC_MIX_DIM as usize) * (HC_DIM as usize) * 2;
    let mut hc_fn_w: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, hc_fn_w_bytes)?;

    let mut sk_scale: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, 3)?;
    let mut sk_base:  DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, HC_MIX_DIM as usize)?;
    let mut attn_norm: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, N_EMBD as usize)?;

    if !skip_fill {
        residual.fill_zero()?;
        flat.fill_zero()?;
        mix.fill_zero()?;
        split.fill_zero()?;
        attn_cur.fill_zero()?;
        out.fill_zero()?;
        hc_fn_w.fill_zero()?;
        sk_scale.fill_zero()?;
        sk_base.fill_zero()?;
        attn_norm.fill_zero()?;
    }

    // Time the FULL chain too (as production would), plus each component.
    let do_chain  = phase == "all" || phase == "chain";
    let do_rms_nw = phase == "all" || phase == "rms_nw";
    let do_f16mv  = phase == "all" || phase == "f16_matvec";
    let do_sink   = phase == "all" || phase == "sinkhorn";
    let do_wsum   = phase == "all" || phase == "hc_weighted";
    let do_rms_w  = phase == "all" || phase == "rms_w";
    let do_rms_nw_mw = phase == "all" || phase == "rms_nw_mw";
    let rms_n_wgs: u32 = std::env::var("BENCH_RMS_NWGS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(16);

    // Warmup all kernels we'll bench.
    for _ in 0..warmup {
        if do_rms_nw { rms_nw.launch(&stream, &mut flat, &residual, 1, HC_DIM, RMS_EPS)?; }
        if do_f16mv  { f16.matvec(&stream, &mut mix, &hc_fn_w, &flat, HC_MIX_DIM, HC_DIM)?; }
        if do_sink   { hc_sink.launch(&stream, &mut split, &mix, &sk_scale, &sk_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?; }
        if do_wsum   { hc_wsum.launch(&stream, &mut attn_cur, &residual, &split, N_EMBD, N_HC)?; }
        if do_rms_w  { rms_w.launch_weighted(&stream, &mut out, &attn_cur, &attn_norm, N_EMBD, RMS_EPS)?; }
    }
    stream.synchronize()?;

    // === Per-kernel timing — inlined to avoid borrow-checker issues with
    // mutable buffer captures across multiple closures.
    if do_rms_nw {
        let mut walls: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let s = Event::new()?; let e = Event::new()?;
            s.record(&stream)?;
            rms_nw.launch(&stream, &mut flat, &residual, 1, HC_DIM, RMS_EPS)?;
            e.record(&stream)?; stream.synchronize()?;
            walls.push(Event::elapsed_ms(&s, &e)?);
        }
        stats(&mut walls, "1. rms_nw   (HC_DIM=16384)        ");
    }
    if do_rms_nw_mw {
        let mut walls: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let s = Event::new()?; let e = Event::new()?;
            s.record(&stream)?;
            rms_nw_mw.launch(&stream, &mut flat, &residual, &mut rms_partial, HC_DIM, rms_n_wgs, RMS_EPS)?;
            e.record(&stream)?; stream.synchronize()?;
            walls.push(Event::elapsed_ms(&s, &e)?);
        }
        stats(&mut walls, format!("1b. rms_nw multi-WG (n_wgs={rms_n_wgs})    ").as_str());
    }
    if do_f16mv {
        let mut walls: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let s = Event::new()?; let e = Event::new()?;
            s.record(&stream)?;
            f16.matvec(&stream, &mut mix, &hc_fn_w, &flat, HC_MIX_DIM, HC_DIM)?;
            e.record(&stream)?; stream.synchronize()?;
            walls.push(Event::elapsed_ms(&s, &e)?);
        }
        stats(&mut walls, "2. f16.matvec (24×16384)           ");
    }
    if do_sink {
        let mut walls: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let s = Event::new()?; let e = Event::new()?;
            s.record(&stream)?;
            hc_sink.launch(&stream, &mut split, &mix, &sk_scale, &sk_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?;
            e.record(&stream)?; stream.synchronize()?;
            walls.push(Event::elapsed_ms(&s, &e)?);
        }
        stats(&mut walls, "3. hc_sinkhorn (n_hc=4, iters=20)   ");
    }
    if do_wsum {
        let mut walls: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let s = Event::new()?; let e = Event::new()?;
            s.record(&stream)?;
            hc_wsum.launch(&stream, &mut attn_cur, &residual, &split, N_EMBD, N_HC)?;
            e.record(&stream)?; stream.synchronize()?;
            walls.push(Event::elapsed_ms(&s, &e)?);
        }
        stats(&mut walls, "4. hc_weighted_sum (N_HC=4×N_EMBD)  ");
    }
    if do_rms_w {
        let mut walls: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let s = Event::new()?; let e = Event::new()?;
            s.record(&stream)?;
            rms_w.launch_weighted(&stream, &mut out, &attn_cur, &attn_norm, N_EMBD, RMS_EPS)?;
            e.record(&stream)?; stream.synchronize()?;
            walls.push(Event::elapsed_ms(&s, &e)?);
        }
        stats(&mut walls, "5. rms_w    (N_EMBD=4096)           ");
    }
    if do_chain {
        // The full chain back-to-back, ONE elapsed-time measurement for
        // the whole chain — closest analog to the captured graph at decode.
        let mut walls: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let s = Event::new()?; let e = Event::new()?;
            s.record(&stream)?;
            rms_nw.launch(&stream, &mut flat, &residual, 1, HC_DIM, RMS_EPS)?;
            f16.matvec(&stream, &mut mix, &hc_fn_w, &flat, HC_MIX_DIM, HC_DIM)?;
            hc_sink.launch(&stream, &mut split, &mix, &sk_scale, &sk_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?;
            hc_wsum.launch(&stream, &mut attn_cur, &residual, &split, N_EMBD, N_HC)?;
            rms_w.launch_weighted(&stream, &mut out, &attn_cur, &attn_norm, N_EMBD, RMS_EPS)?;
            e.record(&stream)?; stream.synchronize()?;
            walls.push(Event::elapsed_ms(&s, &e)?);
        }
        stats(&mut walls, "FULL CHAIN (1+2+3+4+5, single-WG rms)");

        // Chained with multi-WG rms_nw substituted for kernel 1.
        let mut walls: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let s = Event::new()?; let e = Event::new()?;
            s.record(&stream)?;
            rms_nw_mw.launch(&stream, &mut flat, &residual, &mut rms_partial, HC_DIM, rms_n_wgs, RMS_EPS)?;
            f16.matvec(&stream, &mut mix, &hc_fn_w, &flat, HC_MIX_DIM, HC_DIM)?;
            hc_sink.launch(&stream, &mut split, &mix, &sk_scale, &sk_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?;
            hc_wsum.launch(&stream, &mut attn_cur, &residual, &split, N_EMBD, N_HC)?;
            rms_w.launch_weighted(&stream, &mut out, &attn_cur, &attn_norm, N_EMBD, RMS_EPS)?;
            e.record(&stream)?; stream.synchronize()?;
            walls.push(Event::elapsed_ms(&s, &e)?);
        }
        stats(&mut walls, "FULL CHAIN (1+2+3+4+5, multi-WG rms) ");
    }
    Ok(())
}
