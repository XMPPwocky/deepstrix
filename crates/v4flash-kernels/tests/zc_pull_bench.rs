//! dGPU bundle slice 2 microbench + correctness (`V41_DGPU_ZC_PUSH`, docs/v41/DGPU_BUNDLE_DESIGN.md 3):
//! the iGPU pulling a lane-layer's picks / weights / Q8_K rows from the dGPU's pinned readback pack
//! (`hipHostMalloc(0)` allocated with the dGPU current, written by the `ReadbackPack` kernel before an
//! event) with async copies on an iGPU stream, vs today's three SDMA peer pushes on a dGPU stream.
//!
//! Per round: new values on the dGPU -> pack kernel -> event; the iGPU stream waits the event and
//! copies; the iGPU result is checked against the values (a stale / torn read fails the test).
//! Reported: host time of the three async copy CALLS (must be small: a synchronous staged copy
//! would block the host thread -- the risk the window must rule out), iGPU device time of the pull,
//! dGPU device time of the peer push. Small (< 40 MB per GPU), runs beside a live hub (the numbers
//! then include its load).
//!
//!   ZC_ROUNDS=400 ZC_B=4 cargo test -p v4flash-kernels --release --features v41 \
//!     --test zc_pull_bench -- --ignored --nocapture --test-threads=1
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use std::time::Instant;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, PinnedBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_GATE_IN, N_EXPERT_USED};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::sync::{peer_push_f32, peer_push_i32, peer_push_u8};
use v4flash_kernels::readback_pack::{PackPlan, PackSeg};
use v4flash_kernels::BLOCK_Q8_K_BYTES;

fn pick(prefix: &str) -> eyre::Result<Device> {
    Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with(prefix)).unwrap_or(false))
        .ok_or_else(|| eyre!("no {prefix}"))
}

fn pctl(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

#[test]
#[ignore]
fn zc_pull_matches_and_is_async() -> eyre::Result<()> {
    install_panic_handler()?;
    let rounds: usize = std::env::var("ZC_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    let b: usize = std::env::var("ZC_B").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let dg = pick("gfx1201")?;
    let ig = pick("gfx1151")?;
    let n_sel = b * N_EXPERT_USED;
    let xq_bytes = b * BLOCKS_Q8K_GATE_IN as usize * BLOCK_Q8_K_BYTES;

    // dGPU side, as production: router outputs + the pack kernel + a pinned pack allocated with the
    // dGPU current (BatchDgpuScratch).
    dg.set_current()?;
    let arch = dg.properties()?.gcn_arch_name;
    let e = DeviceEngine::for_arch(dg, &arch)?;
    let cs = Stream::new(dg.id)?;
    let xfer = Stream::new(dg.id)?;
    let mut d_sel = DeviceBuffer::<i32>::new(dg.id, n_sel)?;
    let mut d_ew = DeviceBuffer::<f32>::new(dg.id, n_sel)?;
    let mut d_xq = DeviceBuffer::<u8>::new(dg.id, xq_bytes)?;
    let mut pack = PinnedBuffer::<u32>::new(2 * n_sel + xq_bytes / 4)?;
    let ready = Event::new()?;
    let (p0, p1) = (Event::new()?, Event::new()?);

    // iGPU side: the landing buffers.
    ig.set_current()?;
    let is = Stream::new(ig.id)?;
    let mut i_sel = DeviceBuffer::<i32>::new(ig.id, n_sel)?;
    let mut i_ew = DeviceBuffer::<f32>::new(ig.id, n_sel)?;
    let mut i_xq = DeviceBuffer::<u8>::new(ig.id, xq_bytes)?;
    let (z0, z1) = (Event::new()?, Event::new()?);

    let (mut host_us, mut pull_us, mut push_us) = (Vec::new(), Vec::new(), Vec::new());
    let mut mismatches = 0usize;
    for r in 0..rounds {
        // New values each round (a stale read of the previous round's pack must FAIL).
        let hs: Vec<i32> = (0..n_sel).map(|i| (i * 37 + r * 11) as i32 % 384).collect();
        let he: Vec<f32> = (0..n_sel).map(|i| (i + r) as f32 * 0.001).collect();
        let hx: Vec<u8> = (0..xq_bytes).map(|i| ((i * 13 + r * 7) % 251) as u8).collect();
        dg.set_current()?;
        d_sel.copy_from_host(&hs)?;
        d_ew.copy_from_host(&he)?;
        d_xq.copy_from_host(&hx)?;
        let mut plan = PackPlan::default();
        let (so, sn) = plan.push(PackSeg::words(&d_sel, n_sel)?);
        let (eo, en) = plan.push(PackSeg::words(&d_ew, n_sel)?);
        let (xo, _) = plan.push(PackSeg::bytes(&d_xq, xq_bytes)?);
        e.rb_pack.launch(&cs, &mut pack, plan.segs())?;
        ready.record(&cs)?;

        // Pull: iGPU waits the dGPU event, then three async copies from the pinned pack.
        ig.set_current()?;
        is.wait_event(&ready)?;
        z0.record(&is)?;
        let words = pack.as_slice();
        let sel_i: &[i32] = unsafe { std::slice::from_raw_parts(words[so as usize..].as_ptr() as *const i32, sn as usize) };
        let ew_f: &[f32] = unsafe { std::slice::from_raw_parts(words[eo as usize..].as_ptr() as *const f32, en as usize) };
        let xq_u8: &[u8] = unsafe { std::slice::from_raw_parts(words[xo as usize..].as_ptr() as *const u8, xq_bytes) };
        let t = Instant::now();
        i_sel.copy_from_host_async(sel_i, &is)?;
        i_ew.copy_from_host_async(ew_f, &is)?;
        i_xq.copy_from_host_async(xq_u8, &is)?;
        host_us.push(t.elapsed().as_secs_f64() * 1e6);
        z1.record(&is)?;

        // Today's path for comparison: three peer pushes on a dGPU stream after the same event.
        dg.set_current()?;
        xfer.wait_event(&ready)?;
        p0.record(&xfer)?;
        peer_push_i32(&d_sel.slice_view(0, n_sel), &mut i_sel.slice_view_mut(0, n_sel), &xfer)?;
        peer_push_f32(&d_ew.slice_view(0, n_sel), &mut i_ew.slice_view_mut(0, n_sel), &xfer)?;
        peer_push_u8(&d_xq.slice_view(0, xq_bytes), &mut i_xq.slice_view_mut(0, xq_bytes), &xfer)?;
        p1.record(&xfer)?;

        // Check the PULL's result (before the push lands: synchronize the iGPU pull first and read).
        ig.set_current()?;
        is.synchronize()?;
        let mut gs = vec![0i32; n_sel];
        let mut ge = vec![0f32; n_sel];
        let mut gx = vec![0u8; xq_bytes];
        i_sel.copy_to_host(&mut gs)?;
        i_ew.copy_to_host(&mut ge)?;
        i_xq.copy_to_host(&mut gx)?;
        // (The push may have overwritten them with the SAME values meanwhile: still a valid check.)
        if gs != hs || ge.iter().zip(&he).any(|(a, b)| a.to_bits() != b.to_bits()) || gx != hx {
            mismatches += 1;
        }
        pull_us.push(Event::elapsed_ms(&z0, &z1)? as f64 * 1e3);
        dg.set_current()?;
        xfer.synchronize()?;
        push_us.push(Event::elapsed_ms(&p0, &p1)? as f64 * 1e3);
    }
    let rep = |name: &str, v: &mut Vec<f64>| {
        eprintln!("{name}: p50 {:.1} us  p90 {:.1}  p99 {:.1}  max {:.1}", pctl(v, 0.5), pctl(v, 0.9), pctl(v, 0.99), pctl(v, 1.0));
    };
    eprintln!("zc_pull_bench: rounds {rounds}, b {b}, {} B per lane-layer, mismatches {mismatches}", 8 * n_sel + xq_bytes);
    rep("host time of the 3 async pull calls", &mut host_us);
    rep("iGPU device time of the pull", &mut pull_us);
    rep("dGPU device time of the 3 peer pushes", &mut push_us);
    if mismatches > 0 {
        return Err(eyre!("{mismatches} of {rounds} rounds read a stale / wrong pack on the iGPU"));
    }
    Ok(())
}
