//! Kernel-level breakdown of the arena decode's LATENCY-BOUND dGPU stages
//! (mHC, attention, prefill indexer on decode rows, and the small copies
//! inside them), at decode shapes: B = 1, 2, 3 rows per lane.
//!
//! Every item reproduces a production call sequence from
//! `het/forward_prefill.rs` (same kernels, same launch geometry, same
//! graph-vs-direct launch). Two regimes per item:
//!   cold   = event, launch, event on an idle stream (what `de.events.stage`
//!            measures when the host is the bottleneck: includes the host
//!            submit latency of the first node);
//!   queued = the same behind a ~50 us single-wave spin kernel, so the host
//!            has already enqueued everything when the GPU reaches the first
//!            event: pure device time incl. inter-node gaps.
//! All items of a section are measured once per round in a RANDOM order, so
//! the drift of a live server sharing the GPU hits every item alike. Report:
//! min / p10 / median (us). Absolute numbers are confounded by live traffic;
//! compare within a table.
//!
//!   BENCH_SECTION=mhc|attn|indexer|all BENCH_ROUNDS=200 \
//!   CARGO_TARGET_DIR=target-v41 nix develop --command cargo test --release \
//!     --features v41 -p v4flash-kernels --test bench_decode_latency_breakdown \
//!     -- --ignored --nocapture
//!
//! VRAM: < 80 MB at the largest indexer shape (3 rows x 235K keys).
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, GraphExec, PinnedBuffer, Stream};
use v4flash_kernels::config::{
    HC_DIM, HC_MIX_DIM, INDEXER_TOP_K, N_EMBD, N_HC, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD, N_INDEXER_HEAD_DIM,
    N_LORA_Q, N_ROT, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS,
};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mhc_arena::{FastCollapse, FastMix, MIX_NORMED, MIX_PRE_SCALED};
use v4flash_kernels::rope::RopeParams;
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
    half_from_f32(v)
}
fn half_from_f32(v: f32) -> u16 {
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

/// Timing harness: one stream, one event pair, a single-wave spin kernel.
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
            spin_iters: 2000,
        };
        // Calibrate the spinner to ~50 us.
        for _ in 0..3 {
            h.spin_only()?;
        }
        let t = pct(&(0..20).map(|_| h.spin_only()).collect::<eyre::Result<Vec<_>>>()?, 0.5);
        h.spin_iters = ((h.spin_iters as f64) * 50.0 / t.max(1.0)).clamp(100.0, 200000.0) as u32;
        let t2 = pct(&(0..20).map(|_| h.spin_only()).collect::<eyre::Result<Vec<_>>>()?, 0.5);
        eprintln!("spinner: {} iters = {:.1} us", h.spin_iters, t2);
        Ok(h)
    }
    fn spin(&mut self, s: &Stream) -> eyre::Result<()> {
        self.probe.launch_fma_f32(s, &mut self.spin_out, &self.spin_a, &self.spin_b, self.spin_iters, 1, 32)
    }
    fn spin_only(&mut self) -> eyre::Result<f64> {
        self.time(false, |s, hh| hh.spin(s))
    }
    /// Time `f` on the harness stream; `queued` puts the spinner ahead of the
    /// first event.
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

type Item<'a> = (String, Box<dyn FnMut(&Stream) -> eyre::Result<()> + 'a>);

/// Run every item `rounds` times in both regimes, random order per round.
fn run_items(h: &mut Harness, rounds: usize, items: &mut [Item<'_>], rng: &mut Lcg) -> eyre::Result<()> {
    let n = items.len();
    let mut cold: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); n];
    let mut qd: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); n];
    for (_, f) in items.iter_mut() {
        for _ in 0..5 {
            h.time(false, |s, _| f(s))?;
            h.time(true, |s, _| f(s))?;
        }
    }
    let mut order: Vec<usize> = (0..2 * n).collect();
    for _ in 0..rounds {
        for i in (1..order.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            order.swap(i, j);
        }
        for &o in &order {
            let (i, queued) = (o % n, o >= n);
            let f = &mut items[i].1;
            let t = h.time(queued, |s, _| f(s))?;
            if queued { qd[i].push(t) } else { cold[i].push(t) }
        }
    }
    eprintln!("  {:<58} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7}", "item", "cold min", "p10", "med", "q min", "p10", "med");
    for i in 0..n {
        eprintln!(
            "  {:<58} {:>7.1} {:>7.1} {:>7.1} | {:>7.1} {:>7.1} {:>7.1}",
            items[i].0,
            pct(&cold[i], 0.0),
            pct(&cold[i], 0.1),
            pct(&cold[i], 0.5),
            pct(&qd[i], 0.0),
            pct(&qd[i], 0.1),
            pct(&qd[i], 0.5)
        );
    }
    Ok(())
}

fn dev_f32(id: i32, h: &[f32]) -> eyre::Result<DeviceBuffer<f32>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}
fn dev_rand_f32(id: i32, n: usize, rng: &mut Lcg, scale: f32) -> eyre::Result<DeviceBuffer<f32>> {
    let h: Vec<f32> = (0..n).map(|_| (rng.unit() - 0.5) * 2.0 * scale).collect();
    dev_f32(id, &h)
}
fn dev_rand_f16(id: i32, n: usize, rng: &mut Lcg, scale: f32) -> eyre::Result<DeviceBuffer<u16>> {
    let h: Vec<u16> = (0..n).map(|_| f16_bits((rng.unit() - 0.5) * 2.0 * scale)).collect();
    let mut d = DeviceBuffer::new(id, n)?;
    d.copy_from_host(&h)?;
    Ok(d)
}
fn dev_rand_f16_u8(id: i32, n_halves: usize, rng: &mut Lcg, scale: f32) -> eyre::Result<DeviceBuffer<u8>> {
    let h: Vec<u8> = (0..n_halves)
        .flat_map(|_| f16_bits((rng.unit() - 0.5) * 2.0 * scale).to_le_bytes())
        .collect();
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(&h)?;
    Ok(d)
}
fn dev_i32(id: i32, h: &[i32]) -> eyre::Result<DeviceBuffer<i32>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}
fn dev_u32(id: i32, h: &[u32]) -> eyre::Result<DeviceBuffer<u32>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}

fn dgpu() -> eyre::Result<(Device, String)> {
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201"))?;
    let arch = dev.properties()?.gcn_arch_name;
    Ok((dev, arch))
}

// ---------------------------------------------------------------------------
// mHC: the four arena stages and their nodes.
// ---------------------------------------------------------------------------
fn section_mhc(e: &DeviceEngine, h: &mut Harness, rounds: usize, rng: &mut Lcg) -> eyre::Result<()> {
    let id = e.device.id;
    let (hcd, m, ne, bmax) = (HC_DIM as usize, HC_MIX_DIM as usize, N_EMBD as usize, 3usize);
    let residual = dev_rand_f32(id, bmax * hcd, rng, 1.0)?;
    let after_attn = dev_rand_f32(id, bmax * hcd, rng, 1.0)?;
    let attn_out = dev_rand_f32(id, bmax * ne, rng, 1.0)?;
    let w_attn = dev_rand_f16_u8(id, m * hcd, rng, 0.01)?;
    let w_ffn = dev_rand_f16_u8(id, m * hcd, rng, 0.01)?;
    let scale = dev_f32(id, &[0.5, 0.5, 0.5])?;
    let base = dev_rand_f32(id, m, rng, 0.5)?;
    let norm_w = dev_rand_f32(id, ne, rng, 1.0)?;
    let carry0: Vec<f32> = (0..bmax * m).map(|i| if i % m < 4 { 0.25 } else { 0.1 }).collect();
    let carry = std::cell::RefCell::new(dev_f32(id, &carry0)?);
    let split = std::cell::RefCell::new(dev_f32(id, &carry0)?);
    let mix = std::cell::RefCell::new(DeviceBuffer::<f32>::new(id, bmax * m)?);
    let inv_rows = std::cell::RefCell::new(DeviceBuffer::<f32>::new(id, 16)?);
    let counters = std::cell::RefCell::new({
        let mut c = DeviceBuffer::<u32>::new(id, 16)?;
        c.fill_zero()?;
        c
    });
    let cur = std::cell::RefCell::new(DeviceBuffer::<f32>::new(id, bmax * ne)?);
    let norm = std::cell::RefCell::new(DeviceBuffer::<f32>::new(id, bmax * ne)?);
    let post_out = std::cell::RefCell::new(DeviceBuffer::<f32>::new(id, bmax * hcd)?);
    let mut dummy = DeviceBuffer::<f32>::new(id, 64)?;
    dummy.fill_zero()?;
    let dummy = std::cell::RefCell::new(dummy);

    for b in 1..=bmax as u32 {
        let bu = b as usize;
        eprintln!("\n== mHC, b = {b} (arena decode, V41_MHC_ARENA_FUSED=1, FFN_LATE=1) ==");
        let mix_k = |s: &Stream, mode: u32, iters: u32, x: &DeviceBuffer<f32>, w: &DeviceBuffer<u8>| -> eyre::Result<()> {
            e.mhc_arena.launch_mix(
                s, &mut split.borrow_mut(), &mut mix.borrow_mut(), &mut counters.borrow_mut(), w, x, &scale, &base,
                HC_DIM, mode, RMS_EPS, iters, SINKHORN_EPS, b,
            )
        };
        let wsum = |s: &Stream, x: &DeviceBuffer<f32>| -> eyre::Result<()> {
            e.hc_weighted.launch_batched(s, &mut cur.borrow_mut(), x, &carry.borrow(), N_EMBD, N_HC, HC_MIX_DIM, b)
        };
        let memcpy = |s: &Stream| -> eyre::Result<()> {
            let src = split.borrow().slice_view(0, bu * m);
            carry.borrow_mut().slice_view_mut(0, bu * m).copy_from_buffer_async(&src, s)
        };
        let rmsw = |s: &Stream| -> eyre::Result<()> {
            e.rms_w.launch_weighted_batched(s, &mut norm.borrow_mut(), &cur.borrow(), &norm_w, N_EMBD, RMS_EPS, b)
        };
        let post = |s: &Stream| -> eyre::Result<()> {
            e.hc_post.launch_from_split_batched(s, &mut post_out.borrow_mut(), &attn_out, &residual, &split.borrow(), N_HC, N_EMBD, N_HC, b)
        };
        let empty = |s: &Stream| -> eyre::Result<()> { e.vec_scale.launch(s, &mut dummy.borrow_mut(), 1.0, 1) };
        let rep = |n: usize, f: &dyn Fn(&Stream) -> eyre::Result<()>| -> eyre::Result<GraphExec> {
            capture(&h.s, |s| {
                for _ in 0..n {
                    f(s)?;
                }
                Ok(())
            })
        };
        // Production stage graphs.
        let g_pre_attn = capture(&h.s, |s| {
            mix_k(s, MIX_PRE_SCALED, SINKHORN_ITERS, &residual, &w_attn)?;
            wsum(s, &residual)?;
            memcpy(s)?;
            rmsw(s)
        })?;
        let g_pre_ffn = capture(&h.s, |s| {
            wsum(s, &after_attn)?;
            rmsw(s)
        })?;
        let g_mix_late = capture(&h.s, |s| {
            mix_k(s, MIX_NORMED, SINKHORN_ITERS, &after_attn, &w_ffn)?;
            memcpy(s)
        })?;
        // Single nodes, and x10 in one graph (per-node steady state).
        let g_mix_pre = rep(1, &|s| mix_k(s, MIX_PRE_SCALED, SINKHORN_ITERS, &residual, &w_attn))?;
        let g_mix_pre_it1 = rep(1, &|s| mix_k(s, MIX_PRE_SCALED, 1, &residual, &w_attn))?;
        let g_mix_norm = rep(1, &|s| mix_k(s, MIX_NORMED, SINKHORN_ITERS, &after_attn, &w_ffn))?;
        let g_mix_norm_it1 = rep(1, &|s| mix_k(s, MIX_NORMED, 1, &after_attn, &w_ffn))?;
        let g_wsum = rep(1, &|s| wsum(s, &residual))?;
        let g_rmsw = rep(1, &|s| rmsw(s))?;
        let g_memcpy = rep(1, &|s| memcpy(s))?;
        let g_post = rep(1, &|s| post(s))?;
        let g_empty = rep(1, &|s| empty(s))?;
        let g10_mix_pre = rep(10, &|s| mix_k(s, MIX_PRE_SCALED, SINKHORN_ITERS, &residual, &w_attn))?;
        let g10_mix_norm = rep(10, &|s| mix_k(s, MIX_NORMED, SINKHORN_ITERS, &after_attn, &w_ffn))?;
        let g10_wsum = rep(10, &|s| wsum(s, &residual))?;
        let g10_rmsw = rep(10, &|s| rmsw(s))?;
        let g10_memcpy = rep(10, &|s| memcpy(s))?;
        let g10_post = rep(10, &|s| post(s))?;
        let g10_empty = rep(10, &|s| empty(s))?;
        // V41_MHC_FAST: MhcArena::launch_fast (one launch per stage) and its parts.
        let fast = |s: &Stream, mode: Option<u32>, iters: u32, collapse: bool, write_carry: bool| -> eyre::Result<()> {
            let (mut sp, mut mx, mut cn, mut cu, mut no, mut iv) =
                (split.borrow_mut(), mix.borrow_mut(), counters.borrow_mut(), cur.borrow_mut(), norm.borrow_mut(), inv_rows.borrow_mut());
            let (x, w) = if mode == Some(MIX_PRE_SCALED) { (&residual, &w_attn) } else { (&after_attn, &w_ffn) };
            let m = mode.map(|md| FastMix { weight: w, x, scale: &scale, base: &base, mode: md, split_out: &mut sp, mix_out: &mut mx, counters: &mut cn, inv_rows: &mut iv });
            let c = if collapse { Some(FastCollapse { x: &residual, cur_out: &mut cu, norm_out: &mut no, norm_w: &norm_w }) } else { None };
            e.mhc_arena.launch_fast(s, m, c, &mut carry.borrow_mut(), write_carry, HC_DIM, RMS_EPS, iters, SINKHORN_EPS, b)
        };
        let f_pre_attn = rep(1, &|s| fast(s, Some(MIX_PRE_SCALED), SINKHORN_ITERS, true, true))?;
        let f_pre_ffn = rep(1, &|s| fast(s, None, SINKHORN_ITERS, true, false))?;
        let f_mix_late = rep(1, &|s| fast(s, Some(MIX_NORMED), SINKHORN_ITERS, false, true))?;
        let f_mix_pre = rep(1, &|s| fast(s, Some(MIX_PRE_SCALED), SINKHORN_ITERS, false, false))?;
        let f_mix_pre_it1 = rep(1, &|s| fast(s, Some(MIX_PRE_SCALED), 1, false, false))?;
        let f_mix_norm_it1 = rep(1, &|s| fast(s, Some(MIX_NORMED), 1, false, false))?;
        let f10_pre_attn = rep(10, &|s| fast(s, Some(MIX_PRE_SCALED), SINKHORN_ITERS, true, true))?;
        let f10_pre_ffn = rep(10, &|s| fast(s, None, SINKHORN_ITERS, true, false))?;
        let f10_mix_late = rep(10, &|s| fast(s, Some(MIX_NORMED), SINKHORN_ITERS, false, true))?;
        let mut items: Vec<Item> = vec![
            ("FAST mhc_pre_attn [mix_pre+collapse+carry]".into(), Box::new(|s: &Stream| f_pre_attn.launch(s))),
            ("FAST mhc_pre_ffn [collapse]".into(), Box::new(|s: &Stream| f_pre_ffn.launch(s))),
            ("FAST mhc_mix_ffn_late [mix_normed+carry]".into(), Box::new(|s: &Stream| f_mix_late.launch(s))),
            ("fast mix PRE_SCALED only (sinkhorn 20)".into(), Box::new(|s: &Stream| f_mix_pre.launch(s))),
            ("fast mix PRE_SCALED only (sinkhorn 1)".into(), Box::new(|s: &Stream| f_mix_pre_it1.launch(s))),
            ("fast mix NORMED only (sinkhorn 1)".into(), Box::new(|s: &Stream| f_mix_norm_it1.launch(s))),
            ("x10 FAST mhc_pre_attn".into(), Box::new(|s: &Stream| f10_pre_attn.launch(s))),
            ("x10 FAST mhc_pre_ffn".into(), Box::new(|s: &Stream| f10_pre_ffn.launch(s))),
            ("x10 FAST mhc_mix_ffn_late".into(), Box::new(|s: &Stream| f10_mix_late.launch(s))),
            ("STAGE mhc_pre_attn graph [mix_pre+wsum+memcpy+rms_w]".into(), Box::new(|s: &Stream| g_pre_attn.launch(s))),
            ("STAGE mhc_pre_ffn graph [wsum+rms_w]".into(), Box::new(|s: &Stream| g_pre_ffn.launch(s))),
            ("STAGE mhc_mix_ffn_late graph [mix_normed+memcpy]".into(), Box::new(|s: &Stream| g_mix_late.launch(s))),
            ("STAGE mhc_post_attn direct [hc_post]".into(), Box::new(|s: &Stream| post(s))),
            ("node mix PRE_SCALED (sinkhorn 20)".into(), Box::new(|s: &Stream| g_mix_pre.launch(s))),
            ("node mix PRE_SCALED (sinkhorn 1)".into(), Box::new(|s: &Stream| g_mix_pre_it1.launch(s))),
            ("node mix NORMED (sinkhorn 20)".into(), Box::new(|s: &Stream| g_mix_norm.launch(s))),
            ("node mix NORMED (sinkhorn 1)".into(), Box::new(|s: &Stream| g_mix_norm_it1.launch(s))),
            ("node hc_weighted_sum".into(), Box::new(|s: &Stream| g_wsum.launch(s))),
            ("node rms_norm_weighted".into(), Box::new(|s: &Stream| g_rmsw.launch(s))),
            ("node memcpy D2D carry".into(), Box::new(|s: &Stream| g_memcpy.launch(s))),
            ("node hc_post (graph)".into(), Box::new(|s: &Stream| g_post.launch(s))),
            ("node EMPTY (vec_scale n=1)".into(), Box::new(|s: &Stream| g_empty.launch(s))),
            ("x10 mix PRE_SCALED".into(), Box::new(|s: &Stream| g10_mix_pre.launch(s))),
            ("x10 mix NORMED".into(), Box::new(|s: &Stream| g10_mix_norm.launch(s))),
            ("x10 hc_weighted_sum".into(), Box::new(|s: &Stream| g10_wsum.launch(s))),
            ("x10 rms_norm_weighted".into(), Box::new(|s: &Stream| g10_rmsw.launch(s))),
            ("x10 memcpy D2D carry".into(), Box::new(|s: &Stream| g10_memcpy.launch(s))),
            ("x10 hc_post".into(), Box::new(|s: &Stream| g10_post.launch(s))),
            ("x10 EMPTY".into(), Box::new(|s: &Stream| g10_empty.launch(s))),
        ];
        run_items(h, rounds, &mut items, rng)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Launch overhead: event pair, direct launches, graph launches (the arena
// decode replays each stage as its own graph).
// ---------------------------------------------------------------------------
fn section_overhead(e: &DeviceEngine, h: &mut Harness, rounds: usize, rng: &mut Lcg) -> eyre::Result<()> {
    let id = e.device.id;
    let ne = N_EMBD as usize;
    let mut dummy = DeviceBuffer::<f32>::new(id, 64)?;
    dummy.fill_zero()?;
    let dummy = std::cell::RefCell::new(dummy);
    let x = dev_rand_f32(id, ne, rng, 1.0)?;
    let w = dev_rand_f32(id, ne, rng, 1.0)?;
    let out = std::cell::RefCell::new(DeviceBuffer::<f32>::new(id, ne)?);
    let empty = |s: &Stream| -> eyre::Result<()> { e.vec_scale.launch(s, &mut dummy.borrow_mut(), 1.0, 1) };
    let rmsw = |s: &Stream| -> eyre::Result<()> { e.rms_w.launch_weighted_batched(s, &mut out.borrow_mut(), &x, &w, N_EMBD, RMS_EPS, 1) };
    let g1 = capture(&h.s, |s| empty(s))?;
    let g10 = capture(&h.s, |s| { for _ in 0..10 { empty(s)?; } Ok(()) })?;
    let gr = capture(&h.s, |s| rmsw(s))?;
    eprintln!("\n== launch overhead (b = 1) ==");
    let mut items: Vec<Item> = vec![
        ("event pair only".into(), Box::new(|_s: &Stream| Ok(()))),
        ("direct EMPTY x1".into(), Box::new(|s: &Stream| empty(s))),
        ("direct EMPTY x10".into(), Box::new(|s: &Stream| { for _ in 0..10 { empty(s)?; } Ok(()) })),
        ("graph[EMPTY] x1".into(), Box::new(|s: &Stream| g1.launch(s))),
        ("graph[EMPTY x10] x1".into(), Box::new(|s: &Stream| g10.launch(s))),
        ("graph[EMPTY] x2 back-to-back".into(), Box::new(|s: &Stream| { g1.launch(s)?; g1.launch(s) })),
        ("graph[EMPTY] x5 back-to-back".into(), Box::new(|s: &Stream| { for _ in 0..5 { g1.launch(s)?; } Ok(()) })),
        ("direct rms_norm_weighted n=5120".into(), Box::new(|s: &Stream| rmsw(s))),
        ("graph[rms_norm_weighted] x1".into(), Box::new(|s: &Stream| gr.launch(s))),
        ("graph[rms_w] then direct EMPTY".into(), Box::new(|s: &Stream| { gr.launch(s)?; empty(s) })),
        ("direct EMPTY then graph[rms_w]".into(), Box::new(|s: &Stream| { empty(s)?; gr.launch(s) })),
    ];
    run_items(h, rounds, &mut items, rng)
}

// ---------------------------------------------------------------------------
// Attention: the decode-row attention stage after the indexer (sparse top-512
// gathered comp rows + a 128-row window), plus the stage's small H2D uploads.
// ---------------------------------------------------------------------------
fn section_attn(e: &DeviceEngine, h: &mut Harness, rounds: usize, rng: &mut Lcg) -> eyre::Result<()> {
    let id = e.device.id;
    let bmax = 3usize;
    let (nh, hd) = (N_HEAD as usize, N_HEAD_DIM as usize);
    let n_raw = 128u32;
    let n_comp = INDEXER_TOP_K;
    let raw_slots = 256usize; // per-row window region
    let q = dev_rand_f32(id, bmax * nh * hd, rng, 0.05)?;
    let raw_kv = dev_rand_f16(id, bmax * raw_slots * hd, rng, 1.0)?;
    let comp_store_rows = 16384usize; // stands in for the (much larger) f16 main store
    let comp_store = dev_rand_f16(id, comp_store_rows * hd, rng, 1.0)?;
    let active = std::cell::RefCell::new(dev_rand_f16(id, bmax * n_comp as usize * hd, rng, 1.0)?);
    let sinks = dev_rand_f32(id, nh, rng, 1.0)?;
    let stride = 3072u32;
    let scores = std::cell::RefCell::new(DeviceBuffer::<f32>::new(id, bmax * nh * stride as usize)?);
    let heads = std::cell::RefCell::new(DeviceBuffer::<f32>::new(id, bmax * nh * hd)?);
    let n_raw_per = std::cell::RefCell::new(dev_i32(id, &vec![n_raw as i32; bmax])?);
    let n_raw_off = std::cell::RefCell::new(dev_i32(id, &(0..bmax).map(|r| (r * raw_slots) as i32).collect::<Vec<_>>())?);
    let n_comp_per = std::cell::RefCell::new(dev_i32(id, &vec![n_comp as i32; bmax])?);
    let n_idx_per = std::cell::RefCell::new(dev_u32(id, &vec![100_000u32; bmax])?);
    let sel: Vec<i32> = (0..bmax * n_comp as usize).map(|_| (rng.next() % comp_store_rows as u64) as i32).collect();
    let sel = dev_i32(id, &sel)?;
    let host_i32: Vec<i32> = vec![n_raw as i32; bmax];
    let host_u32: Vec<u32> = vec![100_000u32; bmax];
    let mut pinned = PinnedBuffer::<i32>::new(bmax)?;
    pinned.as_mut_slice().copy_from_slice(&host_i32);

    for b in 1..=bmax as u32 {
        let bu = b as usize;
        eprintln!("\n== attention, b = {b} (n_raw {n_raw}, n_comp {n_comp} gathered, f16 scores, stride {stride}) ==");
        let score = |s: &Stream| -> eyre::Result<()> {
            e.attn_mixed.launch_score_batched_htiled_wmma_f16s_rows(
                s, &mut scores.borrow_mut(), &q, &raw_kv, Some(&active.borrow()), &n_raw_per.borrow().slice_view(0, bu),
                &n_raw_off.borrow().slice_view(0, bu), &n_comp_per.borrow().slice_view(0, bu), None, N_HEAD, N_HEAD_DIM,
                n_raw + n_comp, b, n_comp, stride, None,
            )
        };
        let smwsum = |s: &Stream| -> eyre::Result<()> {
            e.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows(
                s, &mut heads.borrow_mut(), &mut scores.borrow_mut(), &sinks, &raw_kv, Some(&active.borrow()),
                &n_raw_per.borrow().slice_view(0, bu), &n_raw_off.borrow().slice_view(0, bu),
                &n_comp_per.borrow().slice_view(0, bu), N_HEAD, N_HEAD_DIM, b, n_comp, stride, None,
            )
        };
        let gather = |s: &Stream| -> eyre::Result<()> {
            e.indexer_gather.launch_batched_rows(s, &mut active.borrow_mut(), &comp_store, &sel, INDEXER_TOP_K, N_HEAD_DIM, b, None)
        };
        let h2d_i32 = |s: &Stream, buf: &std::cell::RefCell<DeviceBuffer<i32>>| -> eyre::Result<()> {
            buf.borrow_mut().slice_view_mut(0, bu).copy_from_host_async(&host_i32[..bu], s)
        };
        let h2d_u32 = |s: &Stream| -> eyre::Result<()> {
            n_idx_per.borrow_mut().slice_view_mut(0, bu).copy_from_host_async(&host_u32[..bu], s)
        };
        let h2d_pinned = |s: &Stream| -> eyre::Result<()> {
            n_raw_per.borrow_mut().slice_view_mut(0, bu).copy_from_host_async(&pinned.as_slice()[..bu], s)
        };
        let mut items: Vec<Item> = vec![
            (
                "STAGE attn (reuse layer) 3xH2D+[H2D+gather]+score+smwsum".into(),
                Box::new(|s: &Stream| {
                    h2d_i32(s, &n_raw_per)?;
                    h2d_i32(s, &n_raw_off)?;
                    h2d_i32(s, &n_comp_per)?;
                    h2d_i32(s, &n_comp_per)?;
                    gather(s)?;
                    score(s)?;
                    smwsum(s)
                }),
            ),
            ("kernels only: score+smwsum".into(), Box::new(|s: &Stream| { score(s)?; smwsum(s) })),
            ("score_batched_htiled_wmma_f16s".into(), Box::new(|s: &Stream| score(s))),
            ("indexer_gather (512 rows x 1 KB per row)".into(), Box::new(|s: &Stream| gather(s))),
            ("H2D pageable i32[b] x1".into(), Box::new(|s: &Stream| h2d_i32(s, &n_raw_per))),
            ("H2D pageable i32[b] x4".into(), Box::new(|s: &Stream| { for _ in 0..4 { h2d_i32(s, &n_raw_per)?; } Ok(()) })),
            ("H2D pageable u32[b] x1 (indexer n_idx)".into(), Box::new(|s: &Stream| h2d_u32(s))),
            ("H2D pinned i32[b] x1".into(), Box::new(|s: &Stream| h2d_pinned(s))),
            ("H2D pinned i32[b] x4".into(), Box::new(|s: &Stream| { for _ in 0..4 { h2d_pinned(s)?; } Ok(()) })),
        ];
        run_items(h, rounds, &mut items, rng)?;
        eprintln!("  (smwsum alone = 'score+smwsum' row minus 'score' row: smwsum rewrites the scores in place, so it always runs after a score)");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Prefill indexer on decode rows (index-source layers: 2, 8, 14 ratio 2;
// 20, 24, 28, 32, 36 ratio 1). V4.1 E2M1 keys, per-row key bases (arena).
// ---------------------------------------------------------------------------
fn section_indexer(e: &DeviceEngine, h: &mut Harness, rounds: usize, rng: &mut Lcg) -> eyre::Result<()> {
    let id = e.device.id;
    let wmma = e.indexer_score_wmma.as_ref().ok_or_else(|| eyre!("no IndexerScoreWmma"))?;
    let (nih, nihd) = (N_INDEXER_HEAD as usize, N_INDEXER_HEAD_DIM as usize);
    let bmax = 3usize;
    let n_list: Vec<u32> = std::env::var("BENCH_N_IDX")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![65_536, 131_072, 235_000]);
    let n_max = *n_list.iter().max().unwrap() as usize;
    let stride = ATTN_MIXED_MAX_KEYS;
    // Weights + activations.
    let wq = dev_rand_f16_u8(id, nih * nihd * N_LORA_Q as usize, rng, 0.02)?;
    let wproj = dev_rand_f16_u8(id, nih * N_EMBD as usize, rng, 0.02)?;
    let qr = dev_rand_f32(id, bmax * N_LORA_Q as usize, rng, 1.0)?;
    let xn = dev_rand_f32(id, bmax * N_EMBD as usize, rng, 1.0)?;
    let iq = std::cell::RefCell::new(DeviceBuffer::<f32>::new(id, bmax * nih * nihd)?);
    let hw = std::cell::RefCell::new(DeviceBuffer::<f32>::new(id, bmax * nih)?);
    let pos = dev_i32(id, &(0..bmax).map(|r| 100_000 + r as i32 * 7).collect::<Vec<_>>())?;
    let rope = RopeParams { freq_base: 10000.0, freq_scale: 1.0, ext_factor: 0.0, attn_factor: 1.0, beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65536 };
    // Keys: per-row stores of n_max rows (valid nibbles, small block exponents).
    let row_bytes = v4flash_kernels::E2M1_KEY_ROW_BYTES;
    let keys = {
        let mut hk = vec![0u8; bmax * n_max * row_bytes];
        for r in 0..bmax * n_max {
            let row = &mut hk[r * row_bytes..(r + 1) * row_bytes];
            for x in row[..64].iter_mut() {
                *x = (rng.next() & 0xff) as u8;
            }
            for x in row[64..68].iter_mut() {
                *x = ((rng.next() % 5) as i8 - 3) as u8;
            }
        }
        let mut d = DeviceBuffer::<u8>::new(id, hk.len())?;
        d.copy_from_host(&hk)?;
        d
    };
    let keys_base = dev_u32(id, &(0..bmax).map(|r| (r * n_max) as u32).collect::<Vec<_>>())?;
    let scores = std::cell::RefCell::new(DeviceBuffer::<f32>::new(id, bmax * stride as usize)?);
    let sel = std::cell::RefCell::new(DeviceBuffer::<i32>::new(id, bmax * INDEXER_TOP_K as usize)?);
    // The gather reads its OWN copy of the selection, folded into the 16K-row
    // stand-in store (the real store is up to 235K rows x 1 KB).
    let sel_g = std::cell::RefCell::new(DeviceBuffer::<i32>::new(id, bmax * INDEXER_TOP_K as usize)?);
    let levels = v4flash_kernels::indexer::topk_merge_levels(n_max as u32, INDEXER_TOP_K);
    let per_row: usize = levels.iter().map(|&n| n as usize).sum::<usize>().max(1);
    let topk_scratch = std::cell::RefCell::new(DeviceBuffer::<u32>::new(id, bmax * per_row)?);
    let done = std::cell::RefCell::new(DeviceBuffer::<u32>::new(id, bmax)?);
    let comp_store_rows = 16384usize;
    let comp_store = dev_rand_f16(id, comp_store_rows * N_HEAD_DIM as usize, rng, 1.0)?;
    let active = std::cell::RefCell::new(DeviceBuffer::<u16>::new(id, bmax * INDEXER_TOP_K as usize * N_HEAD_DIM as usize)?);
    let n_idx_dev = std::cell::RefCell::new(DeviceBuffer::<u32>::new(id, bmax)?);
    let n_words = ((ATTN_MIXED_MAX_KEYS + 31) / 32) as u32;
    let scale = 1.0f32 / ((N_INDEXER_HEAD_DIM as f32) * (N_INDEXER_HEAD as f32)).sqrt();
    eprintln!("indexer: keys {:.1} MB, scores {:.1} MB, topk ladder {:?}", keys.len() as f64 / 1e6, (bmax * stride as usize * 4) as f64 / 1e6, levels);

    for &n in &n_list {
        for b in 1..=bmax as u32 {
            let bu = b as usize;
            let n_host: Vec<u32> = vec![n; bu];
            // The gather must see an in-range selection: run the real topk once.
            eprintln!("\n== prefill indexer on decode rows, n_idx = {n}, b = {b} ==");
            let h2d = |s: &Stream| -> eyre::Result<()> { n_idx_dev.borrow_mut().slice_view_mut(0, bu).copy_from_host_async(&n_host, s) };
            let matvec_q = |s: &Stream| -> eyre::Result<()> {
                e.f16.matvec_batched(s, &mut iq.borrow_mut(), &wq, &qr, N_INDEXER_HEAD * N_INDEXER_HEAD_DIM, N_LORA_Q, b)
            };
            let ropek = |s: &Stream| -> eyre::Result<()> {
                e.rope.launch_forward_batched(s, &mut iq.borrow_mut(), &pos.slice_view(0, bu), N_INDEXER_HEAD, N_INDEXER_HEAD_DIM, N_ROT, b, &rope)
            };
            let qat = |s: &Stream| -> eyre::Result<()> { e.indexer_qat.launch_fp4(s, &mut iq.borrow_mut(), b * N_INDEXER_HEAD) };
            let proj = |s: &Stream| -> eyre::Result<()> {
                e.f16.matvec_batched(s, &mut hw.borrow_mut(), &wproj, &xn, N_INDEXER_HEAD, N_EMBD, b)
            };
            let vscale = |s: &Stream| -> eyre::Result<()> { e.vec_scale.launch(s, &mut hw.borrow_mut(), scale, b * N_INDEXER_HEAD) };
            let score = |s: &Stream| -> eyre::Result<()> {
                wmma.launch_batched_mw_e2m1_rows(s, &mut scores.borrow_mut(), &iq.borrow(), &hw.borrow(), &keys, &n_idx_dev.borrow(), n, stride, b, Some(&keys_base))
            };
            let topk = |s: &Stream| -> eyre::Result<()> {
                e.indexer_topk_bitonic.launch_batched(
                    s, &mut sel.borrow_mut(), None, &mut topk_scratch.borrow_mut(), &scores.borrow(), &n_idx_dev.borrow(), n, stride, n_words,
                    INDEXER_TOP_K, b, Some(&mut done.borrow_mut()),
                )
            };
            let gather = |s: &Stream| -> eyre::Result<()> {
                e.indexer_gather.launch_batched_rows(s, &mut active.borrow_mut(), &comp_store, &sel_g.borrow(), INDEXER_TOP_K, N_HEAD_DIM, b, None)
            };
            // Prime: a real selection (indices < n) before the gather is timed on its own.
            {
                h2d(&h.s)?;
                matvec_q(&h.s)?;
                ropek(&h.s)?;
                qat(&h.s)?;
                proj(&h.s)?;
                vscale(&h.s)?;
                score(&h.s)?;
                topk(&h.s)?;
                h.s.synchronize()?;
                let mut hs = vec![0i32; bu * INDEXER_TOP_K as usize];
                sel.borrow().slice_view(0, hs.len()).copy_to_host(&mut hs)?;
                let mut dn = vec![0u32; bu];
                done.borrow().slice_view(0, bu).copy_to_host(&mut dn)?;
                let bad = hs.iter().filter(|&&v| v < 0 || v as u32 >= n).count();
                eprintln!("  (primed: select done flags {dn:?}, {bad} out-of-range picks; gather indexes a {comp_store_rows}-row store mod its size)");
                // Keep the gather in-bounds for its 16K-row stand-in store.
                let hs2: Vec<i32> = hs.iter().map(|&v| v.rem_euclid(comp_store_rows as i32)).collect();
                sel_g.borrow_mut().slice_view_mut(0, hs2.len()).copy_from_host(&hs2)?;
            }
            let full = |s: &Stream| -> eyre::Result<()> {
                h2d(s)?;
                matvec_q(s)?;
                ropek(s)?;
                qat(s)?;
                proj(s)?;
                vscale(s)?;
                score(s)?;
                topk(s)?;
                gather(s)?;
                h2d(s)
            };
            let mut items: Vec<Item> = vec![
                ("STAGE prefill_indexer (full sequence, 10 launches+H2Ds)".into(), Box::new(|s: &Stream| full(s))),
                ("H2D pageable n_idx".into(), Box::new(|s: &Stream| h2d(s))),
                ("f16 matvec_q [4096x1280] (10.5 MB)".into(), Box::new(|s: &Stream| matvec_q(s))),
                ("rope".into(), Box::new(|s: &Stream| ropek(s))),
                ("qat fp4".into(), Box::new(|s: &Stream| qat(s))),
                ("f16 matvec_proj [32x5120]".into(), Box::new(|s: &Stream| proj(s))),
                ("vec_scale".into(), Box::new(|s: &Stream| vscale(s))),
                ("score mw_e2m1_rows".into(), Box::new(|s: &Stream| score(s))),
                ("topk (select + chain launches)".into(), Box::new(|s: &Stream| topk(s))),
                ("gather".into(), Box::new(|s: &Stream| gather(s))),
            ];
            run_items(h, rounds, &mut items, rng)?;
        }
    }
    Ok(())
}

#[test]
#[ignore]
fn bench_decode_latency_breakdown() -> eyre::Result<()> {
    install_panic_handler()?;
    let rounds: usize = std::env::var("BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let section = std::env::var("BENCH_SECTION").unwrap_or_else(|_| "all".into());
    let (dev, arch) = dgpu()?;
    dev.set_current()?;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let mut h = Harness::new(e.device.id, &arch)?;
    let mut rng = Lcg(0x5eed_1234);
    eprintln!("gfx1201, {rounds} rounds per item, us");
    if section == "all" || section == "overhead" {
        section_overhead(&e, &mut h, rounds, &mut rng)?;
    }
    if section == "all" || section == "mhc" {
        section_mhc(&e, &mut h, rounds, &mut rng)?;
    }
    if section == "all" || section == "attn" {
        section_attn(&e, &mut h, rounds, &mut rng)?;
    }
    if section == "all" || section == "indexer" {
        section_indexer(&e, &mut h, rounds, &mut rng)?;
    }
    Ok(())
}
