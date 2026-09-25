//! Paired A/B of the arena mHC pre-mix: OLD kernel chain vs `MhcArena` mix +
//! the same collapse (hc_weighted + rms_w), both
//! captured as HIP graphs (production replays each stage as one graph), per
//! sub-block (pre-attn / pre-ffn) and lane rows b. Each round runs A and B
//! back to back in RANDOM order and records both; the statistic is the paired
//! difference d = A - B (positive = new faster), reported as the median with
//! a 95% bootstrap CI (2000 resamples) plus P(B faster). Pairing cancels the
//! drift of a live server sharing the GPU.
//!
//!   BENCH_ROUNDS=400 cargo test -p v4flash-kernels --release --features v41 \
//!     --test bench_mhc_arena_ab -- --ignored --nocapture
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, N_EMBD, N_HC, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS};
use v4flash_kernels::mhc_arena::{MIX_NORMED, MIX_PRE_SCALED};
use v4flash_kernels::{F16Matvec, HcSinkhorn, HcWeightedSum, MhcArena, RmsNorm, RmsNormNoWeight, RmsNormNoWeightMultiWG};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn time_once(s: &Stream, exec: &v4flash_hip::GraphExec) -> eyre::Result<f64> {
    let a = Event::new()?;
    let b = Event::new()?;
    a.record(s)?;
    exec.launch(s)?;
    b.record(s)?;
    s.synchronize()?;
    Ok(Event::elapsed_ms(&a, &b)? as f64 * 1000.0)
}

fn capture<F: FnOnce(&Stream) -> eyre::Result<()>>(s: &Stream, f: F) -> eyre::Result<v4flash_hip::GraphExec> {
    s.begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let r = f(s);
    let g = s.end_capture()?;
    r?;
    Ok(g.instantiate()?)
}

#[test]
#[ignore]
fn bench_mhc_arena_ab() -> eyre::Result<()> {
    install_panic_handler()?;
    let rounds: usize = std::env::var("BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201"))?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let rms_nw = RmsNormNoWeight::for_arch(&arch)?;
    let rms_nw_mw = RmsNormNoWeightMultiWG::for_arch(&arch)?;
    let rms_w = RmsNorm::for_arch(&arch)?;
    let f16 = F16Matvec::for_arch(&arch)?;
    let sink = HcSinkhorn::for_arch(&arch)?;
    let wsum = HcWeightedSum::for_arch(&arch)?;
    let arena = MhcArena::for_arch(&arch)?;
    let (hcd, m, ne, bmax) = (HC_DIM as usize, HC_MIX_DIM as usize, N_EMBD as usize, 4usize);
    let z = |n: usize| -> eyre::Result<DeviceBuffer<f32>> {
        let mut d = DeviceBuffer::new(id, n)?;
        d.fill_zero()?;
        Ok(d)
    };
    let x = {
        let h: Vec<f32> = (0..bmax * hcd).map(|i| ((i * 2654435761) % 1000) as f32 * 1e-3 - 0.5).collect();
        let mut d = DeviceBuffer::new(id, h.len())?;
        d.copy_from_host(&h)?;
        d
    };
    let w: DeviceBuffer<u8> = {
        let mut d = DeviceBuffer::new(id, m * hcd * 2)?;
        d.fill_zero()?;
        d
    };
    let (scale, base, carry, norm_w) = (z(3)?, z(m)?, z(bmax * m)?, z(ne)?);
    let (mut mix, mut split, mut cur, mut norm, mut flat) = (z(bmax * m)?, z(bmax * m)?, z(bmax * ne)?, z(bmax * ne)?, z(bmax * hcd)?);
    let (mut inv, mut part) = (z(1)?, z(16)?);
    let mut counters: DeviceBuffer<u32> = DeviceBuffer::new(id, bmax)?;
    counters.fill_zero()?;
    let mut rng = Lcg(0xab_ab_ab);

    if std::env::var("BENCH_PARTS").is_ok() {
        eprintln!("components (graph replay), us: min / p10");
        for b in [1u32, 2] {
            let bu = b as usize;
            let mut show = |name: &str, exec: v4flash_hip::GraphExec| -> eyre::Result<()> {
                for _ in 0..20 { time_once(&s, &exec)?; }
                let mut v: Vec<f64> = (0..rounds).map(|_| time_once(&s, &exec)).collect::<eyre::Result<_>>()?;
                v.sort_by(|x, y| x.partial_cmp(y).unwrap());
                eprintln!("  b={b} {name:<34} {:>7.1} / {:>7.1}", v[0], v[(v.len() - 1) / 10]);
                Ok(())
            };
            show("old attn mixes (inv+matvec per row)", capture(&s, |s| { for r in 0..bu { let row = x.slice_view(r * hcd, hcd); rms_nw_mw.launch_inv_only(s, &mut inv, &row, &mut part, HC_DIM, 16, RMS_EPS)?; let mut mr = mix.slice_view_mut(r * m, m); f16.matvec_pre_scaled(s, &mut mr, &w, &row, &inv, HC_MIX_DIM, HC_DIM)?; } Ok(()) })?)?;
            show("old sinkhorn", capture(&s, |s| sink.launch_batched(s, &mut split, &mix, &scale, &base, N_HC, SINKHORN_ITERS, SINKHORN_EPS, b))?)?;
            show("old collapse (wsum + rms_w)", capture(&s, |s| { wsum.launch_batched(s, &mut cur, &x, &carry, N_EMBD, N_HC, HC_MIX_DIM, b)?; rms_w.launch_weighted_batched(s, &mut norm, &cur, &norm_w, N_EMBD, RMS_EPS, b) })?)?;
            show("new mix PRE_SCALED (incl sinkhorn)", capture(&s, |s| arena.launch_mix(s, &mut split, &mut mix, &mut counters, &w, &x, &scale, &base, HC_DIM, MIX_PRE_SCALED, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b))?)?;
            show("new mix NORMED (incl sinkhorn)", capture(&s, |s| arena.launch_mix(s, &mut split, &mut mix, &mut counters, &w, &x, &scale, &base, HC_DIM, MIX_NORMED, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b))?)?;
        }
        return Ok(());
    }
    eprintln!("gfx1201 paired A/B, {rounds} rounds each, us (A = old chain, B = MhcArena; d = A - B)");
    eprintln!("{:<9} {:>2} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>18} {:>7}", "stage", "b", "A min", "B min", "A p10", "B p10", "A med", "B med", "d med", "d 95% CI", "P(B<A)");
    for b in 1..=bmax as u32 {
        let bu = b as usize;
        for stage in ["pre_attn", "pre_ffn"] {
            let old = capture(&s, |s| {
                if stage == "pre_attn" {
                    for r in 0..bu {
                        let row = x.slice_view(r * hcd, hcd);
                        rms_nw_mw.launch_inv_only(s, &mut inv, &row, &mut part, HC_DIM, 16, RMS_EPS)?;
                        let mut mr = mix.slice_view_mut(r * m, m);
                        f16.matvec_pre_scaled(s, &mut mr, &w, &row, &inv, HC_MIX_DIM, HC_DIM)?;
                    }
                } else {
                    rms_nw.launch_batched(s, &mut flat, &x, 1, HC_DIM, RMS_EPS, b)?;
                    f16.matvec_narrow_batched(s, &mut mix, &w, &flat, HC_MIX_DIM, HC_DIM, b)?;
                }
                sink.launch_batched(s, &mut split, &mix, &scale, &base, N_HC, SINKHORN_ITERS, SINKHORN_EPS, b)?;
                wsum.launch_batched(s, &mut cur, &x, &carry, N_EMBD, N_HC, HC_MIX_DIM, b)?;
                rms_w.launch_weighted_batched(s, &mut norm, &cur, &norm_w, N_EMBD, RMS_EPS, b)
            })?;
            let mode = if stage == "pre_attn" { MIX_PRE_SCALED } else { MIX_NORMED };
            let new = capture(&s, |s| {
                arena.launch_mix(s, &mut split, &mut mix, &mut counters, &w, &x, &scale, &base, HC_DIM, mode, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b)?;
                wsum.launch_batched(s, &mut cur, &x, &carry, N_EMBD, N_HC, HC_MIX_DIM, b)?;
                rms_w.launch_weighted_batched(s, &mut norm, &cur, &norm_w, N_EMBD, RMS_EPS, b)
            })?;
            for _ in 0..20 {
                time_once(&s, &old)?;
                time_once(&s, &new)?;
            }
            let (mut a, mut bb, mut d) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..rounds {
                let (ta, tb) = if rng.next() & 1 == 0 {
                    let ta = time_once(&s, &old)?;
                    (ta, time_once(&s, &new)?)
                } else {
                    let tb = time_once(&s, &new)?;
                    (time_once(&s, &old)?, tb)
                };
                a.push(ta);
                bb.push(tb);
                d.push(ta - tb);
            }
            let wins = d.iter().filter(|&&v| v > 0.0).count() as f64 / rounds as f64;
            let mut boots = Vec::with_capacity(2000);
            for _ in 0..2000 {
                let mut sample: Vec<f64> = (0..rounds).map(|_| d[(rng.next() as usize) % rounds]).collect();
                boots.push(median(&mut sample));
            }
            boots.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let (lo, hi) = (boots[50], boots[1949]);
            let dm = median(&mut d.clone());
            let pct = |v: &mut Vec<f64>, p: f64| -> f64 {
                v.sort_by(|x, y| x.partial_cmp(y).unwrap());
                v[((v.len() - 1) as f64 * p) as usize]
            };
            let (amin, bmin, a10, b10) = (pct(&mut a, 0.0), pct(&mut bb, 0.0), pct(&mut a, 0.1), pct(&mut bb, 0.1));
            eprintln!(
                "{:<9} {:>2} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}   [{:>6.1}, {:>6.1}] {:>7.2}",
                stage, b, amin, bmin, a10, b10, median(&mut a), median(&mut bb), dm, lo, hi, wins
            );
        }
    }
    Ok(())
}
