//! Where the ARENA (multistream decode) mHC pre-mix goes, at V4.1 shape and a
//! lane's row count (`BENCH_B`, default 1..=4). Times each kernel of
//! `forward_prefill.rs` stage 1 / stage 8 and whole chains replayed as HIP
//! graphs (production replays each stage as one graph):
//!   A  current pre_attn: rms_nw(batched, unused on this path) + per row
//!      {rms inv-only, matvec_pre_scaled} + sinkhorn + hc_weighted + rms_w
//!   B  A without the unused rms_nw
//!   C  B with the K-split pre-scaled matvec per row (decode's default)
//!   D  current pre_ffn: rms_nw + matvec_narrow_batched + sinkhorn + hc_weighted + rms_w
//! Min and p50 over BENCH_ITERS (default 300); min resists contention from a
//! live server on the same GPU.
//!
//!   cargo test -p v4flash-kernels --release --features v41 --test bench_mhc_arena_v41 -- --ignored --nocapture
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

fn time<F: FnMut(&Stream) -> eyre::Result<()>>(s: &Stream, iters: usize, mut f: F) -> eyre::Result<(f32, f32)> {
    for _ in 0..10 {
        f(s)?;
    }
    s.synchronize()?;
    let mut v = Vec::with_capacity(iters);
    for _ in 0..iters {
        let a = Event::new()?;
        let b = Event::new()?;
        a.record(s)?;
        f(s)?;
        b.record(s)?;
        s.synchronize()?;
        v.push(Event::elapsed_ms(&a, &b)? * 1000.0);
    }
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    Ok((v[0], v[v.len() / 2]))
}

#[test]
#[ignore]
fn bench_mhc_arena_v41() -> eyre::Result<()> {
    install_panic_handler()?;
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    let bs: Vec<u32> = match std::env::var("BENCH_B").ok().and_then(|s| s.parse().ok()) {
        Some(b) => vec![b],
        None => vec![1, 2, 3, 4],
    };
    let dgpu = pick_dgpu()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    dgpu.set_current()?;
    let id = dgpu.id;
    let s = Stream::new(id)?;
    eprintln!("device {arch}  HC_DIM={HC_DIM} HC_MIX_DIM={HC_MIX_DIM} N_EMBD={N_EMBD}  iters={iters}  (us: min / p50)");

    let rms_nw = RmsNormNoWeight::for_arch(&arch)?;
    let rms_nw_mw = RmsNormNoWeightMultiWG::for_arch(&arch)?;
    let rms_w = RmsNorm::for_arch(&arch)?;
    let f16 = F16Matvec::for_arch(&arch)?;
    let sink = HcSinkhorn::for_arch(&arch)?;
    let wsum = HcWeightedSum::for_arch(&arch)?;

    let (hcd, hmd, ne) = (HC_DIM as usize, HC_MIX_DIM as usize, N_EMBD as usize);
    let bmax = *bs.iter().max().unwrap() as usize;
    let z = |n: usize| -> eyre::Result<DeviceBuffer<f32>> {
        let mut d = DeviceBuffer::new(id, n)?;
        d.fill_zero()?;
        Ok(d)
    };
    let residual = {
        let h: Vec<f32> = (0..bmax * hcd).map(|i| ((i * 2654435761) % 1000) as f32 * 1e-3 - 0.5).collect();
        let mut d = DeviceBuffer::new(id, h.len())?;
        d.copy_from_host(&h)?;
        d
    };
    let mut flat = z(bmax * hcd)?;
    let mut mix = z(bmax * hmd)?;
    let mut split = z(bmax * hmd)?;
    let carry = z(bmax * hmd)?;
    let mut attn_cur = z(bmax * ne)?;
    let mut attn_norm_out = z(bmax * ne)?;
    let norm_w = z(ne)?;
    let mut inv = z(1)?;
    let mut rms_part = z(64)?;
    let mut mv_part = z(64 * hmd)?;
    let scale = z(3)?;
    let base = z(hmd)?;
    let w: DeviceBuffer<u8> = {
        let mut d = DeviceBuffer::new(id, hmd * hcd * 2)?;
        d.fill_zero()?;
        d
    };
    let ksplit = HC_DIM / 1024;

    for &b in &bs {
        let bu = b as usize;
        eprintln!("--- b={b}");
        let r = time(&s, iters, |s| rms_nw.launch_batched(s, &mut flat, &residual, 1, HC_DIM, RMS_EPS, b))?;
        eprintln!("  rms_nw (batched)            {:7.1} / {:7.1}", r.0, r.1);
        let r = time(&s, iters, |s| {
            let row = residual.slice_view(0, hcd);
            rms_nw_mw.launch_inv_only(s, &mut inv, &row, &mut rms_part, HC_DIM, 16, RMS_EPS)
        })?;
        eprintln!("  rms inv-only (1 row)        {:7.1} / {:7.1}", r.0, r.1);
        let r = time(&s, iters, |s| {
            let row = residual.slice_view(0, hcd);
            let mut m = mix.slice_view_mut(0, hmd);
            f16.matvec_pre_scaled(s, &mut m, &w, &row, &inv, HC_MIX_DIM, HC_DIM)
        })?;
        eprintln!("  matvec_pre_scaled (1 row)   {:7.1} / {:7.1}", r.0, r.1);
        let r = time(&s, iters, |s| {
            let row = residual.slice_view(0, hcd);
            let mut m = mix.slice_view_mut(0, hmd);
            f16.matvec_narrow_ksplit_pre_scaled(s, &mut m, &w, &row, &inv, &mut mv_part, HC_MIX_DIM, HC_DIM, ksplit)
        })?;
        eprintln!("  ksplit_pre_scaled (1 row)   {:7.1} / {:7.1}", r.0, r.1);
        let r = time(&s, iters, |s| f16.matvec_narrow_batched(s, &mut mix, &w, &flat, HC_MIX_DIM, HC_DIM, b))?;
        eprintln!("  matvec_narrow_batched       {:7.1} / {:7.1}", r.0, r.1);
        let r = time(&s, iters, |s| sink.launch_batched(s, &mut split, &mix, &scale, &base, N_HC, SINKHORN_ITERS, SINKHORN_EPS, b))?;
        eprintln!("  sinkhorn (batched)          {:7.1} / {:7.1}", r.0, r.1);
        let r = time(&s, iters, |s| wsum.launch_batched(s, &mut attn_cur, &residual, &carry, N_EMBD, N_HC, HC_MIX_DIM, b))?;
        eprintln!("  hc_weighted (batched)       {:7.1} / {:7.1}", r.0, r.1);
        let r = time(&s, iters, |s| rms_w.launch_weighted_batched(s, &mut attn_norm_out, &attn_cur, &norm_w, N_EMBD, RMS_EPS, b))?;
        eprintln!("  rms_w (batched)             {:7.1} / {:7.1}", r.0, r.1);

        // Whole chains, captured once and replayed (as the arena does).
        for (name, variant) in [("A current pre_attn     ", 0), ("B  - unused rms_nw     ", 1), ("C  + ksplit per row    ", 2), ("D current pre_ffn      ", 3)] {
            s.begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let enq = (|| -> eyre::Result<()> {
                if variant == 0 || variant == 3 {
                    rms_nw.launch_batched(&s, &mut flat, &residual, 1, HC_DIM, RMS_EPS, b)?;
                }
                if variant == 3 {
                    f16.matvec_narrow_batched(&s, &mut mix, &w, &flat, HC_MIX_DIM, HC_DIM, b)?;
                } else {
                    for r in 0..bu {
                        let row = residual.slice_view(r * hcd, hcd);
                        rms_nw_mw.launch_inv_only(&s, &mut inv, &row, &mut rms_part, HC_DIM, 16, RMS_EPS)?;
                        let mut m = mix.slice_view_mut(r * hmd, hmd);
                        if variant == 2 {
                            f16.matvec_narrow_ksplit_pre_scaled(&s, &mut m, &w, &row, &inv, &mut mv_part, HC_MIX_DIM, HC_DIM, ksplit)?;
                        } else {
                            f16.matvec_pre_scaled(&s, &mut m, &w, &row, &inv, HC_MIX_DIM, HC_DIM)?;
                        }
                    }
                }
                sink.launch_batched(&s, &mut split, &mix, &scale, &base, N_HC, SINKHORN_ITERS, SINKHORN_EPS, b)?;
                wsum.launch_batched(&s, &mut attn_cur, &residual, &carry, N_EMBD, N_HC, HC_MIX_DIM, b)?;
                rms_w.launch_weighted_batched(&s, &mut attn_norm_out, &attn_cur, &norm_w, N_EMBD, RMS_EPS, b)?;
                Ok(())
            })();
            let graph = s.end_capture()?;
            enq?;
            let exec = graph.instantiate()?;
            let r = time(&s, iters, |s| exec.launch(s))?;
            eprintln!("  chain {name} {:7.1} / {:7.1}", r.0, r.1);
        }
        // The mixes alone (what a side stream would carry) vs the collapse alone.
        s.begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let enq = (|| -> eyre::Result<()> {
            for r in 0..bu {
                let row = residual.slice_view(r * hcd, hcd);
                rms_nw_mw.launch_inv_only(&s, &mut inv, &row, &mut rms_part, HC_DIM, 16, RMS_EPS)?;
                let mut m = mix.slice_view_mut(r * hmd, hmd);
                f16.matvec_pre_scaled(&s, &mut m, &w, &row, &inv, HC_MIX_DIM, HC_DIM)?;
            }
            sink.launch_batched(&s, &mut split, &mix, &scale, &base, N_HC, SINKHORN_ITERS, SINKHORN_EPS, b)
        })();
        let graph = s.end_capture()?;
        enq?;
        let exec = graph.instantiate()?;
        let r = time(&s, iters, |s| exec.launch(s))?;
        eprintln!("  mixes only (B minus collapse) {:7.1} / {:7.1}", r.0, r.1);
    }
    Ok(())
}
