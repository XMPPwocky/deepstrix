//! W0 (docs/v41/EMBED_PHASE_DESIGN.md §19.11): the GPU window's first ~10 minutes,
//! before G1. Pins the hardware facts `qe_attn_g4`'s expected rate rests on. dGPU
//! (gfx1201) only; `#[ignore]` -- exclusive GPU window only:
//!
//! ```text
//! CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release -p v4flash-kernels \
//!     --features v41 --test qe_attn_g4_w0 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! - (a) the existing `wmma_f16_throughput_probe_parallel`: clk per WMMA, sustained rate;
//! - (b) WMMA + V VALU + E exp mixes: serial vs co-issue (the slope in V);
//! - (c) the v_exp_f32 issue rate vs plain VALU;
//! - (d) lane-linear ds_load_b128 B/clk/CU at 8 / 16 waves, CU vs WGP module;
//! - (e) the NW16 split-barrier re-arrive pattern: must neither hang (timeout) nor read
//!   stale data -- run FIRST of the barrier-dependent gates; if it fails, G1-G3 run the
//!   NW8 cell (`V41_EMBED_ATTN=g4n8`, the joined pattern) only;
//! - (f) `v_fma_mixlo_f16` vs `v_cvt_f16_f32` bit identity over all 2^32 f32 patterns
//!   (only gates the post-window P-pack polish).

use std::time::{Duration, Instant};

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::qe_attn_w0::{QeAttnW0, MIXES};
use v4flash_kernels::wmma_probe::WmmaProbe;

fn dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 device"))
}

/// Median device ms of `f` over 5 runs after 1 warmup.
fn time_ms(stream: &Stream, mut f: impl FnMut(&Stream) -> eyre::Result<()>) -> eyre::Result<f64> {
    f(stream)?;
    stream.synchronize()?;
    let mut v = Vec::new();
    for _ in 0..5 {
        let (a, b) = (Event::new()?, Event::new()?);
        a.record(stream)?;
        f(stream)?;
        b.record(stream)?;
        b.synchronize()?;
        v.push(f64::from(Event::elapsed_ms(&a, &b)?));
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(v[2])
}

#[test]
#[ignore]
fn qe_attn_g4_w0() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = dgpu()?;
    dev.set_current()?;
    let p = dev.properties()?;
    let cus = p.multi_processor_count as u32 * 2; // HIP reports WGPs on RDNA
    let ghz = p.clock_rate_khz as f64 / 1e6;
    eprintln!("W0 on {} ({}): {cus} CUs, reported sclk {ghz:.2} GHz", p.name, p.gcn_arch_name);
    let stream = Stream::new(dev.id)?;
    let w0 = QeAttnW0::for_arch(&p.gcn_arch_name)?;

    // (e) first: the barrier pattern the primary cell depends on. A hang here is the
    // failure mode, so poll with a deadline instead of blocking.
    let mut err = DeviceBuffer::<u32>::new(dev.id, 1)?;
    let mut done = DeviceBuffer::<u32>::new(dev.id, 1)?;
    let n_wg = cus * 4;
    for (sleep_wave, mask) in [(0u32, 0u32), (0, 0x5555_5555), (7, 0x0000_ffff), (15, 0xffff_ffff), (3, 0x9249_2492)] {
        err.copy_from_host(&[0])?;
        done.copy_from_host(&[0])?;
        let ev = Event::new_no_timing()?;
        w0.barrier_rearrive(&stream, &mut err, &mut done, 64, sleep_wave, mask, n_wg)?;
        ev.record(&stream)?;
        let t0 = Instant::now();
        while !ev.query()? {
            assert!(t0.elapsed() < Duration::from_secs(20), "(e) barrier re-arrive: HANG (sleep wave {sleep_wave} mask {mask:#x}) -- the NW16 cell is unsafe; run g4n8 only");
            std::thread::sleep(Duration::from_millis(5));
        }
        let (mut e, mut d) = ([0u32], [0u32]);
        err.copy_to_host(&mut e)?;
        done.copy_to_host(&mut d)?;
        eprintln!("(e) barrier re-arrive: sleep wave {sleep_wave} mask {mask:#010x}: {} stale reads, {}/{n_wg} WGs done in {:.1} ms", e[0], d[0], t0.elapsed().as_secs_f64() * 1e3);
        assert_eq!(e[0], 0, "(e) stale LDS data under the early re-arrive -- the NW16 cell is unsafe; run g4n8 only");
        assert_eq!(d[0], n_wg, "(e) not every WG finished");
    }

    // (a) the existing parallel WMMA probe.
    let probe = WmmaProbe::for_arch(&p.gcn_arch_name)?;
    let a_bits: Vec<u16> = (0..32).map(|i| 0x3c00 + (i % 4) as u16).collect();
    let mut d_a = DeviceBuffer::<u16>::new(dev.id, a_bits.len())?;
    d_a.copy_from_host(&a_bits)?;
    let iters = 20_000u32;
    for &(blocks, threads) in &[(cus, 256u32), (cus * 2, 256), (cus, 512)] {
        let mut out = DeviceBuffer::<f32>::new(dev.id, (blocks * threads / 32 * 8) as usize)?;
        let ms = time_ms(&stream, |s| probe.launch_parallel(s, &mut out, &d_a, &d_a, iters, blocks, threads))?;
        let wmmas = blocks as f64 * (threads / 32) as f64 * iters as f64 * 8.0;
        let tf = wmmas * 8192.0 / (ms * 1e-3) / 1e12;
        // clk per WMMA per SIMD = time * clk / (WMMAs per SIMD)
        let per_simd = wmmas / (cus as f64 * 2.0);
        eprintln!("(a) wmma parallel {blocks} x {threads}: {ms:.2} ms, {tf:.1} TFLOPS, {:.2} clk/WMMA/SIMD at the reported clock", ms * 1e-3 * ghz * 1e9 / per_simd);
    }

    // (b) / (c) the mixes, 8 waves / SIMD (one 512-thread WG per CU).
    let mut out = DeviceBuffer::<f32>::new(dev.id, (cus * 512) as usize)?;
    let mix_iters = 4000u32;
    let mut base = None;
    for &mix in &MIXES {
        let ms = time_ms(&stream, |s| w0.mix(s, mix, &mut out, &d_a, mix_iters, cus))?;
        // Per wave per iteration, in clocks of ONE SIMD issuing for its 8 waves.
        let clk = ms * 1e-3 * ghz * 1e9 / (mix_iters as f64 * 8.0);
        let note = match (mix.0, base) {
            (32, None) => {
                base = Some(clk);
                format!("{:.2} clk/WMMA", clk / 32.0)
            }
            (32, Some(b)) => format!("+{:.1} clk over WMMA-only for {} VALU + {} exp ({:.2} clk/op)", clk - b, mix.1, mix.2, (clk - b) / f64::from((mix.1 + mix.2).max(1))),
            _ => format!("{:.2} clk/op", clk / f64::from((mix.1 + mix.2).max(1))),
        };
        eprintln!("(b/c) mix W{:<2} V{:<3} E{:<2}: {ms:7.2} ms, {clk:7.1} clk/iter/wave -- {note}", mix.0, mix.1, mix.2);
    }

    // (d) ds_load_b128 bandwidth.
    let lds_iters = 20_000u32;
    for wgp in [false, true] {
        for threads in [256u32, 512] {
            // 48 KB per WG: 1 WG per CU in CU mode, 2 per WGP in WGP mode.
            let n_wg = cus;
            let mut o = DeviceBuffer::<u32>::new(dev.id, (n_wg * threads) as usize)?;
            let ms = time_ms(&stream, |s| w0.lds_b128(s, wgp, &mut o, lds_iters, n_wg, threads))?;
            let bytes = n_wg as f64 * (threads / 32) as f64 * lds_iters as f64 * 16.0 * 512.0;
            eprintln!(
                "(d) ds_load_b128 {} {:2} waves/WG: {ms:.2} ms, {:.1} B/clk/CU at the reported clock",
                if wgp { "WGP" } else { "CU " }, threads / 32, bytes / (ms * 1e-3 * ghz * 1e9) / cus as f64
            );
        }
    }

    // (f) fma_mix vs cvt.
    let mut mism = DeviceBuffer::<u32>::new(dev.id, 1)?;
    let mut first = DeviceBuffer::<u32>::new(dev.id, 1)?;
    mism.copy_from_host(&[0])?;
    first.copy_from_host(&[u32::MAX])?;
    w0.fma_mix_vs_cvt(&stream, &mut mism, &mut first)?;
    stream.synchronize()?;
    let (mut m, mut f) = ([0u32], [0u32]);
    mism.copy_to_host(&mut m)?;
    first.copy_to_host(&mut f)?;
    eprintln!("(f) v_fma_mixlo_f16 vs v_cvt_f16_f32 over 2^32 f32 patterns: {} mismatches (first input {:#010x})", m[0], f[0]);
    Ok(())
}
