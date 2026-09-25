//! Paired A/B of the ROUTER READBACK on the dGPU, with a shared-expert-shaped
//! chain queued behind the router the way production queues it
//! (router -> [readback] -> selected_ready -> shared expert):
//!
//!   rbstream  (live since 07:08) copies on a NON-blocking stream behind the event
//!   null      (before 07:08)     blocking null-stream hipMemcpy per segment
//!   oncompute copies on the compute stream BEFORE the event
//!   pack      one `ReadbackPack` kernel on the compute stream BEFORE the event
//!   none      no readback (the floor)
//!
//! Every round runs all variants back to back in RANDOM order; the statistic is
//! the paired difference vs `rbstream` (positive = faster), median + 95%
//! bootstrap CI, because a live server shares the GPU. Metrics per variant:
//!   wait    host time from "everything enqueued" to "picks in host memory"
//!   d2h     the part after the router event (what `lh.sel_d2h` measures)
//!   shared  event time of the shared-expert chain (`dgpu.shared_expert`)
//!
//!   BENCH_ROUNDS=300 BENCH_B=1 cargo test -p v4flash-kernels --release --features v41 \
//!     --test bench_router_readback_ab -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use std::time::Instant;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, PinnedBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT_USED, N_FF_SHARED};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::readback_pack::PackSeg;
use v4flash_kernels::BLOCK_Q8_K_BYTES;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn pct(v: &[f64], p: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn boot_ci(d: &[f64], rng: &mut Lcg) -> (f64, f64) {
    let mut meds = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let s: Vec<f64> = (0..d.len()).map(|_| d[(rng.next() as usize) % d.len()]).collect();
        meds.push(median(&s));
    }
    (pct(&meds, 0.025), pct(&meds, 0.975))
}

/// The copy variants' host unpack, as `pre_moe_route` does it (`to_vec` per output).
fn unpack_typed(p_i32: &PinnedBuffer<i32>, p_f32: &PinnedBuffer<f32>, p_u8: &PinnedBuffer<u8>, n_sel: usize, b: usize) -> i64 {
    let (ri, rf) = (p_i32.as_slice(), p_f32.as_slice());
    let (sel, look, orig) = (ri[..n_sel].to_vec(), ri[n_sel..2 * n_sel].to_vec(), ri[2 * n_sel..3 * n_sel].to_vec());
    let (range, ew) = (rf[..b].to_vec(), rf[b..b + n_sel].to_vec());
    let xq = p_u8.as_slice().to_vec();
    sel[0] as i64 + look[0] as i64 + orig[0] as i64 + range[0] as i64 + ew[0] as i64 + xq[0] as i64
}

const VARIANTS: [&str; 5] = ["rbstream", "null", "oncompute", "pack", "none"];

#[test]
#[ignore]
fn bench_router_readback_ab() -> eyre::Result<()> {
    install_panic_handler()?;
    let rounds: usize = std::env::var("BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let b: usize = std::env::var("BENCH_B").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201"))?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let cs = Stream::new(id)?;
    let rb = Stream::new_non_blocking(id)?;

    // Weights: a 3-deep ring per matrix so the 64 MB infinity cache cannot
    // serve them (production's shared expert is a different matrix per layer).
    let (ne, nf) = (N_EMBD as usize, N_FF_SHARED as usize);
    let ring = 3usize;
    let q8 = |m: usize, k: usize, c: usize| -> eyre::Result<DeviceBuffer<u8>> {
        let bytes = m * (k / 32) * 34;
        let host: Vec<u8> = (0..bytes).map(|i| ((i * 31 + c * 7) % 251) as u8).collect();
        let mut w = DeviceBuffer::<u8>::new(id, bytes)?;
        w.copy_from_host(&host)?;
        Ok(w)
    };
    let mut w_gate = Vec::new();
    let mut w_up = Vec::new();
    let mut w_down = Vec::new();
    let mut w_pre = Vec::new();
    for c in 0..ring {
        w_gate.push(q8(nf, ne, c)?);
        w_up.push(q8(nf, ne, c + 11)?);
        w_down.push(q8(ne, nf, c + 23)?);
        w_pre.push(q8(ne, 8192, c + 37)?);
    }
    let x: Vec<f32> = (0..b * 8192).map(|i| ((i % 97) as f32 - 48.0) / 50.0).collect();
    let mut xd = DeviceBuffer::<f32>::new(id, x.len())?;
    xd.copy_from_host(&x)?;
    let mut xq_i8 = DeviceBuffer::<i8>::new(id, b * 8192)?;
    let mut xs = DeviceBuffer::<f32>::new(id, b * 8192 / 32)?;
    let mut h_pre = DeviceBuffer::<f32>::new(id, b * ne)?;
    let mut h_ff = DeviceBuffer::<f32>::new(id, b * nf)?;
    let mut h_out = DeviceBuffer::<f32>::new(id, b * ne)?;

    // Router outputs: picks, look-ahead picks, prior's picks, range, weights, box-2 xq.
    let n_sel = b * N_EXPERT_USED;
    let xq_bytes = b * BLOCKS_Q8K_GATE_IN as usize * BLOCK_Q8_K_BYTES;
    let mk_i32 = |n: usize, k: i32| -> eyre::Result<DeviceBuffer<i32>> {
        let h: Vec<i32> = (0..n as i32).map(|i| i * 7 + k).collect();
        let mut d = DeviceBuffer::new(id, n)?;
        d.copy_from_host(&h)?;
        Ok(d)
    };
    let d_sel = mk_i32(n_sel, 1)?;
    let d_look = mk_i32(n_sel, 2)?;
    let d_orig = mk_i32(n_sel, 3)?;
    let mut d_range = DeviceBuffer::<f32>::new(id, b)?;
    d_range.copy_from_host(&vec![0.5f32; b])?;
    let mut d_ew = DeviceBuffer::<f32>::new(id, n_sel)?;
    d_ew.copy_from_host(&vec![0.25f32; n_sel])?;
    let mut d_xq = DeviceBuffer::<u8>::new(id, xq_bytes)?;
    // Staging, as production: typed pinned buffers for the copies, one u32
    // buffer for the pack.
    let mut p_i32 = PinnedBuffer::<i32>::new(3 * n_sel)?;
    let mut p_f32 = PinnedBuffer::<f32>::new(b + n_sel)?;
    let mut p_u8 = PinnedBuffer::<u8>::new(xq_bytes)?;
    let mut p_pack = PinnedBuffer::<u32>::new(4 * n_sel + b + xq_bytes / 4)?;

    let ev_router = Event::new_no_timing()?;
    let (sx0, sx1) = (Event::new()?, Event::new()?);
    let mut ri = 0usize;
    let mut rng = Lcg(0x5eed_1234);
    // [variant][round] -> (wait, d2h, shared) us
    let mut res: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); VARIANTS.len()];

    // What the staging must hold: the device data, as words
    // [sel | look | orig | range | ew | xq].
    let device_words = |d_sel: &DeviceBuffer<i32>, d_look: &DeviceBuffer<i32>, d_orig: &DeviceBuffer<i32>,
                        d_range: &DeviceBuffer<f32>, d_ew: &DeviceBuffer<f32>, d_xq: &DeviceBuffer<u8>|
     -> eyre::Result<(Vec<u32>, Vec<u8>)> {
        let mut want: Vec<u32> = Vec::new();
        let mut hi = vec![0i32; n_sel];
        for d in [d_sel, d_look, d_orig] {
            d.copy_to_host(&mut hi)?;
            want.extend(hi.iter().map(|&v| v as u32));
        }
        let mut hr = vec![0f32; b];
        d_range.copy_to_host(&mut hr)?;
        want.extend(hr.iter().map(|v| v.to_bits()));
        let mut he = vec![0f32; n_sel];
        d_ew.copy_to_host(&mut he)?;
        want.extend(he.iter().map(|v| v.to_bits()));
        let mut hx = vec![0u8; xq_bytes];
        d_xq.copy_to_host(&mut hx)?;
        want.extend(hx.chunks_exact(4).map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]])));
        Ok((want, hx))
    };
    let mut checked = 0usize;

    for round in 0..rounds + 5 {
        let mut order: Vec<usize> = (0..VARIANTS.len()).collect();
        for i in (1..order.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            order.swap(i, j);
        }
        for &v in &order {
            ri = (ri + 1) % ring;
            cs.synchronize()?;
            // "attention chain" ahead of the router, so the host enqueues
            // everything while the GPU is still busy (as in production).
            for k in 0..3 {
                let wi = (ri + k) % ring;
                e.q8.quantize_input_batched(&cs, &mut xq_i8, &mut xs, &xd, 8192, b as u32)?;
                e.q8.matvec_bpack(&cs, &mut h_pre, &w_pre[wi], &xq_i8, &xs, ne as u32, 8192, b as u32)?;
            }
            // Router tail: box 2's xq quantize (production's last kernel before the event).
            e.q8k.launch(&cs, &mut d_xq, &h_pre, BLOCKS_Q8K_GATE_IN * b as u32)?;
            match VARIANTS[v] {
                "oncompute" => {
                    d_sel.copy_to_pinned_async(&mut p_i32, 0, &cs)?;
                    d_look.copy_to_pinned_async(&mut p_i32, n_sel, &cs)?;
                    d_orig.copy_to_pinned_async(&mut p_i32, 2 * n_sel, &cs)?;
                    d_range.copy_to_pinned_async(&mut p_f32, 0, &cs)?;
                    d_ew.copy_to_pinned_async(&mut p_f32, b, &cs)?;
                    d_xq.copy_to_pinned_async(&mut p_u8, 0, &cs)?;
                }
                "pack" => {
                    let segs = [
                        PackSeg::words(&d_sel, n_sel)?,
                        PackSeg::words(&d_look, n_sel)?,
                        PackSeg::words(&d_orig, n_sel)?,
                        PackSeg::words(&d_range, b)?,
                        PackSeg::words(&d_ew, n_sel)?,
                        PackSeg::bytes(&d_xq, xq_bytes)?,
                    ];
                    e.rb_pack.launch(&cs, &mut p_pack, &segs)?;
                }
                _ => {}
            }
            ev_router.record(&cs)?;
            // Shared expert: quantize, gate, up, down.
            sx0.record(&cs)?;
            e.q8.quantize_input_batched(&cs, &mut xq_i8, &mut xs, &h_pre, ne as u32, b as u32)?;
            e.q8.matvec_bpack(&cs, &mut h_ff, &w_gate[ri], &xq_i8, &xs, nf as u32, ne as u32, b as u32)?;
            e.q8.matvec_bpack(&cs, &mut h_ff, &w_up[ri], &xq_i8, &xs, nf as u32, ne as u32, b as u32)?;
            e.q8.quantize_input_batched(&cs, &mut xq_i8, &mut xs, &h_ff, nf as u32, b as u32)?;
            e.q8.matvec_bpack(&cs, &mut h_out, &w_down[ri], &xq_i8, &xs, ne as u32, nf as u32, b as u32)?;
            sx1.record(&cs)?;

            let t0 = Instant::now();
            ev_router.synchronize()?;
            let t1 = Instant::now();
            let mut sink = 0i64;
            match VARIANTS[v] {
                "rbstream" => {
                    rb.wait_event(&ev_router)?;
                    d_sel.copy_to_pinned_async(&mut p_i32, 0, &rb)?;
                    d_look.copy_to_pinned_async(&mut p_i32, n_sel, &rb)?;
                    d_orig.copy_to_pinned_async(&mut p_i32, 2 * n_sel, &rb)?;
                    d_range.copy_to_pinned_async(&mut p_f32, 0, &rb)?;
                    d_ew.copy_to_pinned_async(&mut p_f32, b, &rb)?;
                    d_xq.copy_to_pinned_async(&mut p_u8, 0, &rb)?;
                    rb.synchronize()?;
                    sink += unpack_typed(&p_i32, &p_f32, &p_u8, n_sel, b);
                }
                "null" => {
                    let mut hs = vec![0i32; n_sel];
                    d_sel.copy_to_host(&mut hs)?;
                    d_look.copy_to_host(&mut hs)?;
                    d_orig.copy_to_host(&mut hs)?;
                    let mut hr = vec![0f32; b];
                    d_range.copy_to_host(&mut hr)?;
                    let mut he = vec![0f32; n_sel];
                    d_ew.copy_to_host(&mut he)?;
                    let mut hx = vec![0u8; xq_bytes];
                    d_xq.copy_to_host(&mut hx)?;
                    sink += hs[0] as i64 + hx[0] as i64;
                }
                "oncompute" => sink += unpack_typed(&p_i32, &p_f32, &p_u8, n_sel, b),
                "pack" => {
                    // As `pre_moe_route` does: i32/f32 per element, xq in one memcpy.
                    let w = p_pack.as_slice();
                    let i32s = |v: &[u32]| -> Vec<i32> { v.iter().map(|&x| x as i32).collect() };
                    let f32s = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
                    let (sel, look, orig) = (i32s(&w[..n_sel]), i32s(&w[n_sel..2 * n_sel]), i32s(&w[2 * n_sel..3 * n_sel]));
                    let (range, ew) = (f32s(&w[3 * n_sel..3 * n_sel + b]), f32s(&w[3 * n_sel + b..4 * n_sel + b]));
                    let xw = &w[4 * n_sel + b..4 * n_sel + b + xq_bytes / 4];
                    // SAFETY: a live, initialized &[u32] viewed as bytes.
                    let xq = unsafe { std::slice::from_raw_parts(xw.as_ptr().cast::<u8>(), xw.len() * 4) }.to_vec();
                    sink += sel[0] as i64 + look[0] as i64 + orig[0] as i64 + range[0] as i64 + ew[0] as i64 + xq[0] as i64;
                }
                _ => {}
            }
            let t2 = Instant::now();
            std::hint::black_box(sink);
            cs.synchronize()?;
            // Staging == device data, right after the variant that filled it
            // (outside the timed window; the next variant changes the data).
            if round % 25 == 0 && (VARIANTS[v] == "pack" || VARIANTS[v] == "oncompute") {
                let (want, hx) = device_words(&d_sel, &d_look, &d_orig, &d_range, &d_ew, &d_xq)?;
                let ok = if VARIANTS[v] == "pack" {
                    p_pack.as_slice()[..want.len()] == want[..]
                } else {
                    p_u8.as_slice() == &hx[..]
                        && p_i32.as_slice().iter().map(|&x| x as u32).eq(want[..3 * n_sel].iter().copied())
                        && p_f32.as_slice().iter().map(|x| x.to_bits()).eq(want[3 * n_sel..3 * n_sel + b + n_sel].iter().copied())
                };
                if !ok {
                    return Err(eyre!("{} staging != device data (round {round})", VARIANTS[v]));
                }
                checked += 1;
            }
            if round >= 5 {
                let us = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e6;
                let shared = Event::elapsed_ms(&sx0, &sx1)? as f64 * 1000.0;
                res[v].push((us(t0, t2), us(t1, t2), shared));
            }
        }
    }

    println!("staging verified {checked} times");
    println!("B={b} rounds={rounds} (us; median / p10; paired d = rbstream - X, + = X faster)");
    println!("{:<10} {:>16} {:>16} {:>16} | {:>26} {:>26}", "variant", "wait", "d2h", "shared", "d(wait) [95% CI]", "d(shared) [95% CI]");
    let base = 0usize;
    for (v, name) in VARIANTS.iter().enumerate() {
        let col = |f: fn(&(f64, f64, f64)) -> f64| -> Vec<f64> { res[v].iter().map(f).collect() };
        let (w, d, s) = (col(|r| r.0), col(|r| r.1), col(|r| r.2));
        let dw: Vec<f64> = res[base].iter().zip(&res[v]).map(|(a, b)| a.0 - b.0).collect();
        let ds: Vec<f64> = res[base].iter().zip(&res[v]).map(|(a, b)| a.2 - b.2).collect();
        let (wl, wh) = boot_ci(&dw, &mut rng);
        let (sl, sh) = boot_ci(&ds, &mut rng);
        println!(
            "{:<10} {:>7.1} / {:>6.1} {:>7.1} / {:>6.1} {:>7.1} / {:>6.1} | {:>7.1} [{:>6.1},{:>6.1}] {:>7.1} [{:>6.1},{:>6.1}]",
            name, median(&w), pct(&w, 0.1), median(&d), pct(&d, 0.1), median(&s), pct(&s, 0.1),
            median(&dw), wl, wh, median(&ds), sl, sh
        );
    }
    Ok(())
}
