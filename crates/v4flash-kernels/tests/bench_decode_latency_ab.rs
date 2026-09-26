//! Paired, interleaved A/B of the 2026-09-26 decode-latency changes, at the
//! arena decode's production call sequences (B = 1, 2, 3 rows per lane):
//!
//!   mhc  : V41_MHC_FAST -- the three arena mHC stage graphs per lane-layer
//!          A = g.mhc_pre_attn [mix PRE_SCALED, hc_weighted, carry memcpy, rms_w]
//!              g.mhc_pre_ffn  [hc_weighted, rms_w]
//!              g.mhc_mix_ffn_late [mix NORMED, carry memcpy]
//!          B = the same graphs as one `MhcArena::launch_fast` each
//!          (+ "lane-layer" = all three back to back, what one lane pays per layer)
//!   topk : V41_TOPK_SELECT_ILP -- the indexer's batched top-k launcher on a
//!          decode row (select + the early-out chain launches), old select
//!          kernel vs the ILP one, at n = 65K / 131K / 235K.
//!
//! Every round runs A and B back to back in RANDOM order, in two regimes
//! (cold = idle stream, as `de.events.stage` sees it; queued = behind a ~50 us
//! spin kernel, i.e. device time only). The statistic is the paired
//! difference d = A - B (positive = B faster): median with a 95% bootstrap CI
//! (2000 resamples), P(B < A), and A/B min / p10 / median. Pairing cancels the
//! drift of a live server sharing the GPU; absolute numbers do not transfer.
//!
//!   BENCH_SECTION=mhc|topk|all BENCH_ROUNDS=1000 CARGO_TARGET_DIR=target-v41 \
//!   nix develop --command cargo test --release --features v41 -p v4flash-kernels \
//!     --test bench_decode_latency_ab -- --ignored --nocapture
//!
//! VRAM: < 20 MB.
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use std::cell::RefCell;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, GraphExec, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, INDEXER_TOP_K, N_EMBD, N_HC, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mhc_arena::{FastCollapse, FastMix, MIX_NORMED, MIX_PRE_SCALED};
use v4flash_kernels::wmma_probe::WmmaProbe;
use v4flash_kernels::ATTN_MIXED_MAX_KEYS;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xffffff) as f32 / 16777216.0
    }
}

fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    if mant & 0x1000 != 0 {
        h += 1;
    }
    sign | (h as u16)
}

fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * p) as usize]
}

fn capture<F: FnOnce(&Stream) -> eyre::Result<()>>(s: &Stream, f: F) -> eyre::Result<GraphExec> {
    s.begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let r = f(s);
    let g = s.end_capture()?;
    r?;
    Ok(g.instantiate()?)
}

struct Harness {
    s: Stream,
    ea: Event,
    eb: Event,
    probe: WmmaProbe,
    spin_out: DeviceBuffer<f32>,
    spin_a: DeviceBuffer<f32>,
    spin_b: DeviceBuffer<f32>,
    spin_iters: u32,
}

impl Harness {
    fn new(id: i32, arch: &str) -> eyre::Result<Self> {
        let mut spin_a = DeviceBuffer::new(id, 1)?;
        spin_a.copy_from_host(&[1.0f32])?;
        let mut spin_b = DeviceBuffer::new(id, 1)?;
        spin_b.copy_from_host(&[0.999f32])?;
        let mut h = Self {
            s: Stream::new(id)?,
            ea: Event::new()?,
            eb: Event::new()?,
            probe: WmmaProbe::for_arch(arch)?,
            spin_out: DeviceBuffer::new(id, 32)?,
            spin_a,
            spin_b,
            spin_iters: 700,
        };
        let t = pct(&(0..20).map(|_| h.time(false, |s, hh| hh.spin(s))).collect::<eyre::Result<Vec<_>>>()?, 0.5);
        h.spin_iters = ((h.spin_iters as f64) * 50.0 / t.max(1.0)).clamp(100.0, 200000.0) as u32;
        Ok(h)
    }
    fn spin(&mut self, s: &Stream) -> eyre::Result<()> {
        self.probe.launch_fma_f32(s, &mut self.spin_out, &self.spin_a, &self.spin_b, self.spin_iters, 1, 32)
    }
    fn time<F: FnMut(&Stream, &mut Self) -> eyre::Result<()>>(&mut self, queued: bool, mut f: F) -> eyre::Result<f64> {
        // SAFETY: the stream outlives the call and is only used on this thread.
        let st = unsafe { &*(&self.s as *const Stream) };
        if queued {
            self.spin(st)?;
        }
        self.ea.record(st)?;
        f(st, self)?;
        self.eb.record(st)?;
        st.synchronize()?;
        Ok(Event::elapsed_ms(&self.ea, &self.eb)? as f64 * 1000.0)
    }
}

type Run<'a> = Box<dyn FnMut(&Stream) -> eyre::Result<()> + 'a>;

/// Paired A/B: `rounds` rounds, each A and B in random order, both regimes.
fn paired(h: &mut Harness, rng: &mut Lcg, rounds: usize, label: &str, a: &mut Run<'_>, b: &mut Run<'_>) -> eyre::Result<()> {
    for _ in 0..10 {
        h.time(false, |s, _| a(s))?;
        h.time(false, |s, _| b(s))?;
    }
    for queued in [false, true] {
        let (mut va, mut vb, mut d) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds), Vec::with_capacity(rounds));
        for _ in 0..rounds {
            let (ta, tb) = if rng.next() & 1 == 0 {
                let ta = h.time(queued, |s, _| a(s))?;
                (ta, h.time(queued, |s, _| b(s))?)
            } else {
                let tb = h.time(queued, |s, _| b(s))?;
                (h.time(queued, |s, _| a(s))?, tb)
            };
            va.push(ta);
            vb.push(tb);
            d.push(ta - tb);
        }
        let wins = d.iter().filter(|&&v| v > 0.0).count() as f64 / rounds as f64;
        let mut boots = Vec::with_capacity(2000);
        for _ in 0..2000 {
            let sample: Vec<f64> = (0..rounds).map(|_| d[(rng.next() as usize) % rounds]).collect();
            boots.push(pct(&sample, 0.5));
        }
        boots.sort_by(|x, y| x.partial_cmp(y).unwrap());
        eprintln!(
            "  {:<34} {:<6} A {:>6.1}/{:>6.1}/{:>6.1}  B {:>6.1}/{:>6.1}/{:>6.1}  d {:>6.1} [{:>6.1}, {:>6.1}]  P(B<A) {:.2}",
            label,
            if queued { "queued" } else { "cold" },
            pct(&va, 0.0),
            pct(&va, 0.1),
            pct(&va, 0.5),
            pct(&vb, 0.0),
            pct(&vb, 0.1),
            pct(&vb, 0.5),
            pct(&d, 0.5),
            boots[50],
            boots[1949],
            wins
        );
    }
    Ok(())
}

fn up<T: Copy>(id: i32, h: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}

fn section_mhc(e: &DeviceEngine, h: &mut Harness, rounds: usize, rng: &mut Lcg) -> eyre::Result<()> {
    let id = e.device.id;
    let (hcd, m, ne, bmax) = (HC_DIM as usize, HC_MIX_DIM as usize, N_EMBD as usize, 3usize);
    let rnd = |rng: &mut Lcg, n: usize, sc: f32| -> Vec<f32> { (0..n).map(|_| (rng.unit() - 0.5) * 2.0 * sc).collect() };
    let residual = up(id, &rnd(rng, bmax * hcd, 1.0))?;
    let after_attn = up(id, &rnd(rng, bmax * hcd, 1.0))?;
    let w16 = |rng: &mut Lcg| -> Vec<u8> { (0..m * hcd).flat_map(|_| f16_bits((rng.unit() - 0.5) * 0.02).to_le_bytes()).collect() };
    let w_attn = up(id, &w16(rng))?;
    let w_ffn = up(id, &w16(rng))?;
    let scale = up(id, &[0.5f32, 0.5, 0.5])?;
    let base = up(id, &rnd(rng, m, 0.5))?;
    let attn_norm = up(id, &rnd(rng, ne, 1.0))?;
    let ffn_norm = up(id, &rnd(rng, ne, 1.0))?;
    let carry0: Vec<f32> = (0..bmax * m).map(|i| if i % m < 4 { 0.25 } else { 0.1 }).collect();
    let carry = RefCell::new(up(id, &carry0)?);
    let split = RefCell::new(up(id, &carry0)?);
    let mix = RefCell::new(DeviceBuffer::<f32>::new(id, bmax * m)?);
    let inv_rows = RefCell::new(DeviceBuffer::<f32>::new(id, 16)?);
    let counters = RefCell::new({
        let mut c = DeviceBuffer::<u32>::new(id, 16)?;
        c.fill_zero()?;
        c
    });
    let attn_cur = RefCell::new(DeviceBuffer::<f32>::new(id, bmax * ne)?);
    let attn_in = RefCell::new(DeviceBuffer::<f32>::new(id, bmax * ne)?);
    let ffn_cur = RefCell::new(DeviceBuffer::<f32>::new(id, bmax * ne)?);
    let ffn_in = RefCell::new(DeviceBuffer::<f32>::new(id, bmax * ne)?);
    eprintln!("\n== mHC: old arena graphs (A) vs MhcArena::launch_fast (B), us: min/p10/med, d = A - B median [95% CI] ==");
    for b in 1..=bmax as u32 {
        let bu = b as usize;
        let s = &h.s;
        // A: the production graphs (V41_MHC_ARENA_FUSED=1, V41_MHC_FFN_LATE=1).
        let a_pre_attn = capture(s, |s| {
            e.mhc_arena.launch_mix(s, &mut split.borrow_mut(), &mut mix.borrow_mut(), &mut counters.borrow_mut(), &w_attn, &residual,
                &scale, &base, HC_DIM, MIX_PRE_SCALED, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b)?;
            e.hc_weighted.launch_batched(s, &mut attn_cur.borrow_mut(), &residual, &carry.borrow(), N_EMBD, N_HC, HC_MIX_DIM, b)?;
            let src = split.borrow().slice_view(0, bu * m);
            carry.borrow_mut().slice_view_mut(0, bu * m).copy_from_buffer_async(&src, s)?;
            e.rms_w.launch_weighted_batched(s, &mut attn_in.borrow_mut(), &attn_cur.borrow(), &attn_norm, N_EMBD, RMS_EPS, b)
        })?;
        let a_pre_ffn = capture(s, |s| {
            e.hc_weighted.launch_batched(s, &mut ffn_cur.borrow_mut(), &after_attn, &carry.borrow(), N_EMBD, N_HC, HC_MIX_DIM, b)?;
            e.rms_w.launch_weighted_batched(s, &mut ffn_in.borrow_mut(), &ffn_cur.borrow(), &ffn_norm, N_EMBD, RMS_EPS, b)
        })?;
        let a_mix_late = capture(s, |s| {
            e.mhc_arena.launch_mix(s, &mut split.borrow_mut(), &mut mix.borrow_mut(), &mut counters.borrow_mut(), &w_ffn, &after_attn,
                &scale, &base, HC_DIM, MIX_NORMED, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b)?;
            let src = split.borrow().slice_view(0, bu * m);
            carry.borrow_mut().slice_view_mut(0, bu * m).copy_from_buffer_async(&src, s)
        })?;
        // B: one launch per stage.
        let b_pre_attn = capture(s, |s| {
            let (mut sp, mut mx, mut cn) = (split.borrow_mut(), mix.borrow_mut(), counters.borrow_mut());
            let (mut cu, mut no) = (attn_cur.borrow_mut(), attn_in.borrow_mut());
            e.mhc_arena.launch_fast(
                s,
                Some(FastMix { weight: &w_attn, x: &residual, scale: &scale, base: &base, mode: MIX_PRE_SCALED, split_out: &mut sp, mix_out: &mut mx, counters: &mut cn, inv_rows: &mut inv_rows.borrow_mut() }),
                Some(FastCollapse { x: &residual, cur_out: &mut cu, norm_out: &mut no, norm_w: &attn_norm }),
                &mut carry.borrow_mut(), true, HC_DIM, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b,
            )
        })?;
        let b_pre_ffn = capture(s, |s| {
            let (mut cu, mut no) = (ffn_cur.borrow_mut(), ffn_in.borrow_mut());
            e.mhc_arena.launch_fast(
                s, None, Some(FastCollapse { x: &after_attn, cur_out: &mut cu, norm_out: &mut no, norm_w: &ffn_norm }),
                &mut carry.borrow_mut(), false, HC_DIM, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b,
            )
        })?;
        let b_mix_late = capture(s, |s| {
            let (mut sp, mut mx, mut cn) = (split.borrow_mut(), mix.borrow_mut(), counters.borrow_mut());
            e.mhc_arena.launch_fast(
                s,
                Some(FastMix { weight: &w_ffn, x: &after_attn, scale: &scale, base: &base, mode: MIX_NORMED, split_out: &mut sp, mix_out: &mut mx, counters: &mut cn, inv_rows: &mut inv_rows.borrow_mut() }),
                None, &mut carry.borrow_mut(), true, HC_DIM, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b,
            )
        })?;
        eprintln!(" b = {b}");
        let pairs: [(&str, &GraphExec, &GraphExec); 3] = [
            ("mhc_pre_attn", &a_pre_attn, &b_pre_attn),
            ("mhc_pre_ffn", &a_pre_ffn, &b_pre_ffn),
            ("mhc_mix_ffn_late", &a_mix_late, &b_mix_late),
        ];
        for (name, ga, gb) in pairs {
            let mut ra: Run = Box::new(|s: &Stream| ga.launch(s));
            let mut rb: Run = Box::new(|s: &Stream| gb.launch(s));
            paired(h, rng, rounds, name, &mut ra, &mut rb)?;
        }
        let mut ra: Run = Box::new(|s: &Stream| {
            a_pre_attn.launch(s)?;
            a_pre_ffn.launch(s)?;
            a_mix_late.launch(s)
        });
        let mut rb: Run = Box::new(|s: &Stream| {
            b_pre_attn.launch(s)?;
            b_pre_ffn.launch(s)?;
            b_mix_late.launch(s)
        });
        paired(h, rng, rounds, "lane-layer (all 3 graphs)", &mut ra, &mut rb)?;
    }
    Ok(())
}

fn section_topk(e: &DeviceEngine, h: &mut Harness, rounds: usize, rng: &mut Lcg) -> eyre::Result<()> {
    let id = e.device.id;
    let bmax = 3usize;
    let stride = ATTN_MIXED_MAX_KEYS;
    let n_words = (ATTN_MIXED_MAX_KEYS + 31) / 32;
    eprintln!("\n== indexer top-k on decode rows: old select kernel (A) vs ILP select (B), incl. the early-out chain launches ==");
    for &n in &[65_536u32, 131_072, 235_000] {
        // Scores: the indexer's weighted-ReLU sums are >= 0 with a long tail;
        // any continuous distribution exercises the same threshold path.
        let hs: Vec<f32> = (0..bmax * stride as usize)
            .map(|i| if (i % stride as usize) < n as usize { (rng.unit() * rng.unit()) * 8.0 } else { -3.4028235e38 })
            .collect();
        let scores = up(id, &hs)?;
        let levels = v4flash_kernels::indexer::topk_merge_levels(n, INDEXER_TOP_K);
        let per_row: usize = levels.iter().map(|&v| v as usize).sum::<usize>().max(1);
        let scratch = RefCell::new(DeviceBuffer::<u32>::new(id, bmax * per_row)?);
        let sel = RefCell::new(DeviceBuffer::<i32>::new(id, bmax * INDEXER_TOP_K as usize)?);
        let done = RefCell::new(DeviceBuffer::<u32>::new(id, bmax)?);
        for b in 1..=bmax as u32 {
            let n_idx = up(id, &vec![n; b as usize])?;
            let run = |s: &Stream, ilp: bool| -> eyre::Result<()> {
                e.indexer_topk_bitonic.launch_batched_sel(
                    s, &mut sel.borrow_mut(), None, &mut scratch.borrow_mut(), &scores, &n_idx, n, stride, n_words, INDEXER_TOP_K, b,
                    Some(&mut done.borrow_mut()), ilp,
                )
            };
            let mut ra: Run = Box::new(|s: &Stream| run(s, false));
            let mut rb: Run = Box::new(|s: &Stream| run(s, true));
            paired(h, rng, rounds, &format!("topk n={n} b={b}"), &mut ra, &mut rb)?;
            h.s.synchronize()?;
            let mut dn = vec![0u32; b as usize];
            done.borrow().slice_view(0, b as usize).copy_to_host(&mut dn)?;
            if dn.iter().any(|&d| d != 1) {
                return Err(eyre!("select fell back to the chain at n={n} b={b}: {dn:?} (not the production path)"));
            }
        }
    }
    Ok(())
}

#[test]
#[ignore]
fn bench_decode_latency_ab() -> eyre::Result<()> {
    install_panic_handler()?;
    let rounds: usize = std::env::var("BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let section = std::env::var("BENCH_SECTION").unwrap_or_else(|_| "all".into());
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201"))?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let mut h = Harness::new(e.device.id, &arch)?;
    let mut rng = Lcg(0xab_2026_0926);
    eprintln!("gfx1201 paired A/B, {rounds} rounds per regime, us");
    if section == "all" || section == "mhc" {
        section_mhc(&e, &mut h, rounds, &mut rng)?;
    }
    if section == "all" || section == "topk" {
        section_topk(&e, &mut h, rounds, &mut rng)?;
    }
    if section == "all" || section == "attnmeta" {
        section_attnmeta(&e, &mut h, rounds, &mut rng)?;
    }
    if section == "all" || section == "attnscore" {
        section_attnscore(&e, &mut h, rounds, &mut rng)?;
    }
    Ok(())
}

/// V41_ATTN_DEC_SCORE: the batched htiled f16s score kernel (A) vs its
/// decode-shape twin (B), alone and followed by the smwsum that consumes the
/// scores, at the decode shape (128 window + 512 gathered keys per row).
fn section_attnscore(e: &DeviceEngine, h: &mut Harness, rounds: usize, rng: &mut Lcg) -> eyre::Result<()> {
    use v4flash_kernels::config::{N_HEAD, N_HEAD_DIM};
    let id = e.device.id;
    let bmax = 3usize;
    let (nh, hd) = (N_HEAD as usize, N_HEAD_DIM as usize);
    let (n_raw, n_comp, raw_slots, stride) = (128u32, INDEXER_TOP_K, 256usize, 3072u32);
    let f16v = |rng: &mut Lcg, n: usize| -> Vec<u16> { (0..n).map(|_| f16_bits((rng.unit() - 0.5) * 2.0)).collect() };
    let q = up(id, &(0..bmax * nh * hd).map(|_| (rng.unit() - 0.5) * 0.1).collect::<Vec<f32>>())?;
    let raw_kv = up(id, &f16v(rng, bmax * raw_slots * hd))?;
    let active = up(id, &f16v(rng, bmax * n_comp as usize * hd))?;
    let sinks = up(id, &(0..nh).map(|_| rng.unit()).collect::<Vec<f32>>())?;
    let scores = RefCell::new(DeviceBuffer::<f32>::new(id, bmax * nh * stride as usize)?);
    let heads = RefCell::new(DeviceBuffer::<f32>::new(id, bmax * nh * hd)?);
    let nrp = up(id, &vec![n_raw as i32; bmax])?;
    let nrop = up(id, &(0..bmax).map(|r| (r * raw_slots) as i32).collect::<Vec<i32>>())?;
    let ncp = up(id, &vec![n_comp as i32; bmax])?;
    eprintln!("\n== attention score: batched htiled f16s (A) vs decode twin (B); n_raw {n_raw} + {n_comp} gathered ==");
    for b in 1..=bmax as u32 {
        let bu = b as usize;
        let (nr, no, nc) = (nrp.slice_view(0, bu), nrop.slice_view(0, bu), ncp.slice_view(0, bu));
        let score = |s: &Stream, dec: bool| -> eyre::Result<()> {
            if dec {
                e.attn_dec.launch_score_f16s_rows(
                    s, &mut scores.borrow_mut(), &q, &raw_kv, Some(&active), &nr, &no, &nc, None, N_HEAD, N_HEAD_DIM, n_raw + n_comp, b, n_comp,
                    stride, None,
                )
            } else {
                e.attn_mixed.launch_score_batched_htiled_wmma_f16s_rows(
                    s, &mut scores.borrow_mut(), &q, &raw_kv, Some(&active), &nr, &no, &nc, None, N_HEAD, N_HEAD_DIM, n_raw + n_comp, b, n_comp,
                    stride, None,
                )
            }
        };
        let smwsum = |s: &Stream| -> eyre::Result<()> {
            e.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows(
                s, &mut heads.borrow_mut(), &mut scores.borrow_mut(), &sinks, &raw_kv, Some(&active), &nr, &no, &nc, N_HEAD, N_HEAD_DIM, b,
                n_comp, stride, None,
            )
        };
        eprintln!(" b = {b}");
        let mut ra: Run = Box::new(|s: &Stream| score(s, false));
        let mut rb: Run = Box::new(|s: &Stream| score(s, true));
        paired(h, rng, rounds, "score", &mut ra, &mut rb)?;
        let mut ra: Run = Box::new(|s: &Stream| { score(s, false)?; smwsum(s) });
        let mut rb: Run = Box::new(|s: &Stream| { score(s, true)?; smwsum(s) });
        paired(h, rng, rounds, "score + smwsum", &mut ra, &mut rb)?;
    }
    Ok(())
}

/// V41_ATTN_META_FILL: the attention stage's per-row metadata uploads, followed
/// by the kernels that read them (gather + score + smwsum), as production
/// queues them on a decode lane. A = the pageable copies (reuse layer: n_raw,
/// n_raw_off, dense n_comp, sparse n_comp; index source: + n_index_comp),
/// B = one `AttnMetaFill` launch.
fn section_attnmeta(e: &DeviceEngine, h: &mut Harness, rounds: usize, rng: &mut Lcg) -> eyre::Result<()> {
    use v4flash_kernels::attn_meta::MetaDst;
    use v4flash_kernels::config::{N_HEAD, N_HEAD_DIM};
    let id = e.device.id;
    let bmax = 3usize;
    let (nh, hd) = (N_HEAD as usize, N_HEAD_DIM as usize);
    let (n_raw, n_comp, raw_slots, stride) = (128u32, INDEXER_TOP_K, 256usize, 3072u32);
    let f16v = |rng: &mut Lcg, n: usize| -> Vec<u16> { (0..n).map(|_| f16_bits((rng.unit() - 0.5) * 2.0)).collect() };
    let q = up(id, &(0..bmax * nh * hd).map(|_| (rng.unit() - 0.5) * 0.1).collect::<Vec<f32>>())?;
    let raw_kv = up(id, &f16v(rng, bmax * raw_slots * hd))?;
    let store_rows = 16384usize;
    let store = up(id, &f16v(rng, store_rows * hd))?;
    let active = RefCell::new(up(id, &f16v(rng, bmax * n_comp as usize * hd))?);
    let sinks = up(id, &(0..nh).map(|_| rng.unit()).collect::<Vec<f32>>())?;
    let scores = RefCell::new(DeviceBuffer::<f32>::new(id, bmax * nh * stride as usize)?);
    let heads = RefCell::new(DeviceBuffer::<f32>::new(id, bmax * nh * hd)?);
    let sel = up(id, &(0..bmax * n_comp as usize).map(|_| (rng.next() % store_rows as u64) as i32).collect::<Vec<i32>>())?;
    let nrp = RefCell::new(DeviceBuffer::<i32>::new(id, 16)?);
    let nrop = RefCell::new(DeviceBuffer::<i32>::new(id, 16)?);
    let ncp = RefCell::new(DeviceBuffer::<i32>::new(id, 16)?);
    let nidx = RefCell::new(DeviceBuffer::<u32>::new(id, 16)?);
    eprintln!("\n== attention metadata: pageable copies (A) vs one kernel-argument launch (B), + gather/score/smwsum ==");
    for b in 1..=bmax as u32 {
        let bu = b as usize;
        let h_nrp: Vec<i32> = vec![n_raw as i32; bu];
        let h_nrop: Vec<i32> = (0..bu).map(|r| (r * raw_slots) as i32).collect();
        let n_full: Vec<u32> = (0..bu).map(|r| 180_000 + r as u32).collect();
        let h_ncp_dense: Vec<i32> = n_full.iter().map(|&v| v as i32).collect();
        let h_ncp: Vec<i32> = n_full.iter().map(|&v| v.min(n_comp) as i32).collect();
        let h_nidx_i: Vec<i32> = n_full.iter().map(|&v| v as i32).collect();
        let kernels = |s: &Stream| -> eyre::Result<()> {
            e.indexer_gather.launch_batched_rows(s, &mut active.borrow_mut(), &store, &sel, INDEXER_TOP_K, N_HEAD_DIM, b, None)?;
            let (nr, no, nc) = (nrp.borrow().slice_view(0, bu), nrop.borrow().slice_view(0, bu), ncp.borrow().slice_view(0, bu));
            e.attn_mixed.launch_score_batched_htiled_wmma_f16s_rows(
                s, &mut scores.borrow_mut(), &q, &raw_kv, Some(&active.borrow()), &nr, &no, &nc, None, N_HEAD, N_HEAD_DIM,
                n_raw + n_comp, b, n_comp, stride, None,
            )?;
            e.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows(
                s, &mut heads.borrow_mut(), &mut scores.borrow_mut(), &sinks, &raw_kv, Some(&active.borrow()), &nr, &no, &nc,
                N_HEAD, N_HEAD_DIM, b, n_comp, stride, None,
            )
        };
        let copies = |s: &Stream, source: bool| -> eyre::Result<()> {
            nrp.borrow_mut().slice_view_mut(0, bu).copy_from_host_async(&h_nrp, s)?;
            nrop.borrow_mut().slice_view_mut(0, bu).copy_from_host_async(&h_nrop, s)?;
            ncp.borrow_mut().slice_view_mut(0, bu).copy_from_host_async(&h_ncp_dense, s)?;
            if source {
                nidx.borrow_mut().slice_view_mut(0, bu).copy_from_host_async(&n_full, s)?;
            }
            ncp.borrow_mut().slice_view_mut(0, bu).copy_from_host_async(&h_ncp, s)
        };
        let fill = |s: &Stream, source: bool| -> eyre::Result<()> {
            let (mut a, mut bb, mut c, mut d) = (nrp.borrow_mut(), nrop.borrow_mut(), ncp.borrow_mut(), nidx.borrow_mut());
            e.attn_meta.launch(
                s,
                [
                    Some(MetaDst::new(&mut a, &h_nrp)?),
                    Some(MetaDst::new(&mut bb, &h_nrop)?),
                    Some(MetaDst::new(&mut c, &h_ncp)?),
                    if source { Some(MetaDst::new(&mut d, &h_nidx_i)?) } else { None },
                ],
            )
        };
        eprintln!(" b = {b}");
        for source in [false, true] {
            let tag = if source { "index source" } else { "reuse layer" };
            let mut ra: Run = Box::new(|s: &Stream| copies(s, source));
            let mut rb: Run = Box::new(|s: &Stream| fill(s, source));
            paired(h, rng, rounds, &format!("uploads only, {tag}"), &mut ra, &mut rb)?;
            let mut ra: Run = Box::new(|s: &Stream| { copies(s, source)?; kernels(s) });
            let mut rb: Run = Box::new(|s: &Stream| { fill(s, source)?; kernels(s) });
            paired(h, rng, rounds, &format!("uploads + attn, {tag}"), &mut ra, &mut rb)?;
        }
    }
    Ok(())
}
