//! dGPU bundle Step 0c (docs/v41/DGPU_BUNDLE_DESIGN.md 4): what does a stage-graph REPLAY cost, and
//! does merging adjacent graphs pay? Seven small kernels (vec_add over `b * N_EMBD` -- the size of the
//! q/kv chain's nodes at decode rows) four ways:
//!   direct  7 direct launches
//!   split   a 5-node graph + a 2-node graph (today's g.q_chain + g.kv_chain)
//!   merged  one 7-node graph (the q+kv merge)
//!   single  7 one-node graphs (GRAPH_KEYS_DESIGN 2.11's +6.7 us case, for reference)
//! in two regimes (cold = idle stream: host submission shows; queued = behind a ~50 us spin: device
//! time only), every round all four in RANDOM order; paired differences vs `split` (positive = faster)
//! with a bootstrap CI. A live hub shares the GPU: only the paired differences transfer.
//!
//!   BENCH_ROUNDS=1500 BENCH_B=4 CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release \
//!     --features v41 -p v4flash-kernels --test bench_graph_replay -- --ignored --nocapture
//! VRAM: < 10 MB.
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, GraphExec, Stream};
use v4flash_kernels::config::N_EMBD;
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::wmma_probe::WmmaProbe;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
}

fn ci(d: &[f64], rng: &mut Lcg) -> (f64, f64) {
    let mut m: Vec<f64> = (0..2000)
        .map(|_| median(&(0..d.len()).map(|_| d[(rng.next() as usize) % d.len()]).collect::<Vec<_>>()))
        .collect();
    m.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (m[50], m[1949])
}

#[test]
#[ignore]
fn bench_graph_replay() -> eyre::Result<()> {
    install_panic_handler()?;
    let rounds: usize = std::env::var("BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1500);
    let b: usize = std::env::var("BENCH_B").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201"))?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let s = Stream::new(id)?;
    let (ea, eb) = (Event::new()?, Event::new()?);
    let n = b * N_EMBD as usize;
    let mut bufs: Vec<DeviceBuffer<f32>> = (0..8).map(|_| DeviceBuffer::new(id, n)).collect::<Result<_, _>>()?;
    for x in bufs.iter_mut() {
        x.copy_from_host(&vec![0.0f32; n])?;
    }
    let addend = {
        let mut a = DeviceBuffer::new(id, n)?;
        a.copy_from_host(&vec![1e-6f32; n])?;
        a
    };
    // The seven "nodes": buf[i] += addend.
    let body = |st: &Stream, bufs: &mut [DeviceBuffer<f32>], range: std::ops::Range<usize>| -> eyre::Result<()> {
        for i in range {
            e.vec_add.launch(st, &mut bufs[i], &addend, n as u32)?;
        }
        Ok(())
    };
    let cap = |bufs: &mut [DeviceBuffer<f32>], range: std::ops::Range<usize>| -> eyre::Result<GraphExec> {
        s.begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let r = body(&s, bufs, range);
        let g = s.end_capture()?;
        r?;
        Ok(g.instantiate()?)
    };
    let g5 = cap(&mut bufs, 0..5)?;
    let g2 = cap(&mut bufs, 5..7)?;
    let g7 = cap(&mut bufs, 0..7)?;
    let g1: Vec<GraphExec> = (0..7).map(|i| cap(&mut bufs, i..i + 1)).collect::<Result<_, _>>()?;
    // Spin kernel for the queued regime (~50 us, calibrated).
    let probe = WmmaProbe::for_arch(&arch)?;
    let mut spin_out = DeviceBuffer::<f32>::new(id, 32)?;
    let mut sa = DeviceBuffer::<f32>::new(id, 1)?;
    sa.copy_from_host(&[1.0])?;
    let mut sb = DeviceBuffer::<f32>::new(id, 1)?;
    sb.copy_from_host(&[0.999])?;
    let mut iters = 700u32;
    for _ in 0..3 {
        ea.record(&s)?;
        probe.launch_fma_f32(&s, &mut spin_out, &sa, &sb, iters, 1, 32)?;
        eb.record(&s)?;
        s.synchronize()?;
        let us = Event::elapsed_ms(&ea, &eb)? as f64 * 1e3;
        iters = ((iters as f64) * 50.0 / us.max(1.0)).clamp(100.0, 200000.0) as u32;
    }
    let names = ["direct", "split", "merged", "single"];
    let mut t: [[Vec<f64>; 4]; 2] = Default::default(); // [regime][arm]
    let mut host: [Vec<f64>; 4] = Default::default();
    let mut rng = Lcg(7);
    for _ in 0..rounds {
        for queued in [false, true] {
            let mut order = [0usize, 1, 2, 3];
            for i in (1..4).rev() {
                order.swap(i, (rng.next() as usize) % (i + 1));
            }
            for &a in &order {
                if queued {
                    probe.launch_fma_f32(&s, &mut spin_out, &sa, &sb, iters, 1, 32)?;
                }
                ea.record(&s)?;
                let h0 = std::time::Instant::now();
                match a {
                    0 => body(&s, &mut bufs, 0..7)?,
                    1 => {
                        g5.launch(&s)?;
                        g2.launch(&s)?;
                    }
                    2 => g7.launch(&s)?,
                    _ => {
                        for g in &g1 {
                            g.launch(&s)?;
                        }
                    }
                }
                let hus = h0.elapsed().as_secs_f64() * 1e6;
                eb.record(&s)?;
                s.synchronize()?;
                t[queued as usize][a].push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
                if !queued {
                    host[a].push(hus);
                }
            }
        }
    }
    eprintln!("bench_graph_replay: b {b}, {rounds} rounds, 7 nodes of vec_add over {n} floats");
    for (ri, rname) in ["cold (idle stream)", "queued (device only)"].iter().enumerate() {
        eprintln!("== {rname}: event time per 7 nodes (us), median; paired diff vs split (positive = faster) [95% CI]");
        for a in 0..4 {
            let d: Vec<f64> = t[ri][1].iter().zip(&t[ri][a]).map(|(x, y)| x - y).collect();
            let (lo, hi) = ci(&d, &mut rng);
            eprintln!("  {:7}  {:8.2}   vs split {:+7.2} [{:+.2}, {:+.2}]", names[a], median(&t[ri][a]), median(&d), lo, hi);
        }
    }
    eprintln!("== host submission time (us), cold regime");
    for a in 0..4 {
        eprintln!("  {:7}  {:8.2}", names[a], median(&host[a]));
    }
    Ok(())
}
