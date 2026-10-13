//! G1-optimal floor probes (dGPU only, bench-only): a hand-compiled code object (`G1O_HSACO`, from
//! scratch/g1opt/probe.hip) with stream kernels `probe_m{mode}_r{rows per wave}` that read the router's
//! two f16 gates (768 rows x 10240 B) with every load in flight at once. Per kernel: the in-kernel
//! span (first WG start -> last WG end, 100 MHz steady counter) and the event-pair time, COLD (96 MB
//! flush first), interleaved in a random order every round, beside the live hub.
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, launch_kernel, Device, DeviceBuffer, Event, LaunchConfig, Module, Stream};
use v4flash_kernels::wmma_probe::WmmaProbe;

fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() {
        return f64::NAN;
    }
    s[((s.len() - 1) as f64 * p).round() as usize]
}

#[test]
#[ignore]
fn g1o_probe() -> eyre::Result<()> {
    install_panic_handler();
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201 visible"))?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let id = dev.id;
    let path = std::env::var("G1O_HSACO").map_err(|_| eyre!("set G1O_HSACO"))?;
    let module = Module::load_data(&std::fs::read(&path)?)?;
    let probe = WmmaProbe::for_arch(&arch)?;
    let s = Stream::new(id)?;
    let (ea, eb) = (Event::new()?, Event::new()?);
    let nw = 768 * 10240usize;
    let mut w = DeviceBuffer::<u8>::new(id, nw)?;
    w.copy_from_host(&(0..nw).map(|i| (i * 7 + 3) as u8).collect::<Vec<_>>())?;
    let flush = DeviceBuffer::<u8>::new(id, 96 << 20)?;
    let sink = DeviceBuffer::<u32>::new(id, 1024)?;
    let ts = DeviceBuffer::<u64>::new(id, 4096)?;
    let mut spin_out = DeviceBuffer::<f32>::new(id, 32)?;
    let mut spin_a = DeviceBuffer::<f32>::new(id, 1)?;
    spin_a.copy_from_host(&[1.0])?;
    let mut spin_b = DeviceBuffer::<f32>::new(id, 1)?;
    spin_b.copy_from_host(&[0.999])?;
    let rounds: usize = std::env::var("BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let names: Vec<String> = std::env::var("G1O_KERNELS")
        .unwrap_or_else(|_| "probe_m0_r1,probe_m1_r1,probe_m2_r1,probe_m3_r1,probe_m1_r2,probe_m3_r2,probe_m3_r3,probe_empty".into())
        .split(',')
        .map(|x| x.to_string())
        .collect();
    let cold = std::env::var("BENCH_WARM").as_deref() != Ok("1");
    if std::env::var("G1O_QUEUE").as_deref() == Ok("1") {
        // queue probe: probe_q0 (no stream) vs probe_q1 (8 waves per WG streaming their gate row)
        let hot = DeviceBuffer::<u32>::new(id, 32768)?;
        let qn = ["probe_q0", "probe_q10", "probe_q20", "probe_q40", "probe_q80"];
        let mut res: Vec<Vec<[f64; 3]>> = vec![Vec::new(); qn.len()];
        for round in 0..rounds + 3 {
            for _ in 0..60 {
                probe.launch_fma_f32(&s, &mut spin_out, &spin_a, &spin_b, 2000, 1, 32)?;
            }
            for kk in 0..qn.len() {
                let k = (kk + round) % qn.len();
                let f = module.get_function("probe_flush")?;
                let cfg = LaunchConfig { grid: (512, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
                launch_kernel!(f, cfg, &s, [flush.raw() as u64, ((96u32 << 20) / 16), sink.raw() as u64])?;
                let fh = module.get_function("probe_hot")?;
                launch_kernel!(fh, LaunchConfig { grid: (128, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &s, [hot.raw() as u64])?;
                let fq = module.get_function(qn[k])?;
                launch_kernel!(fq, LaunchConfig { grid: (96, 1, 1), block: (320, 1, 1), shared_mem_bytes: 0 }, &s,
                    [w.raw() as u64, hot.raw() as u64, sink.raw() as u64, ts.raw() as u64])?;
                s.synchronize()?;
                if round < 3 {
                    continue;
                }
                let mut t = vec![0u64; ts.len()];
                ts.copy_to_host(&mut t)?;
                for g in 0..96 {
                    let r = &t[g * 8..g * 8 + 8];
                    res[k].push([(r[2] - r[1]) as f64 / 100.0, (r[3] - r[2]) as f64 / 800.0, (r[4] - r[3]) as f64 / 100.0]);
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        for (k, n) in qn.iter().enumerate() {
            let col = |i: usize| res[k].iter().map(|v| v[i]).collect::<Vec<_>>();
            eprintln!("{n} (K u32 words/lane prefetched by 8 waves): store+fence p50 {:.2} p90 {:.2} us | scalar load p50 {:.2} p90 {:.2} us | 8 dependent vector loads total p50 {:.2} p90 {:.2} us",
                pct(&col(0), 0.5), pct(&col(0), 0.9), pct(&col(1), 0.5), pct(&col(1), 0.9), pct(&col(2), 0.5), pct(&col(2), 0.9));
        }
        return Ok(());
    }
    let mut span: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    let mut evt: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    let mut pz: Vec<Vec<(f64, f64)>> = vec![Vec::new(); names.len()];
    let mut rng = 0x1234_5678u64;
    let launch = |name: &str| -> eyre::Result<u32> {
        let f = module.get_function(name)?;
        if name == "probe_empty" {
            let cfg = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 };
            launch_kernel!(f, cfg, &s, [sink.raw() as u64])?;
            return Ok(0);
        }
        let r: u32 = name.rsplit("_r").next().and_then(|x| x.parse().ok()).unwrap_or(1);
        let nwg = (768 / r).div_ceil(8);
        let cfg = LaunchConfig { grid: (nwg, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &s, [w.raw() as u64, sink.raw() as u64, ts.raw() as u64, nwg])?;
        Ok(nwg)
    };
    let do_flush = || -> eyre::Result<()> {
        let f = module.get_function("probe_flush")?;
        let cfg = LaunchConfig { grid: (512, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &s, [flush.raw() as u64, ((96u32 << 20) / 16), sink.raw() as u64])
    };
    for round in 0..rounds + 3 {
        // clock ramp (~3 ms of one-wave spin)
        for _ in 0..60 {
            probe.launch_fma_f32(&s, &mut spin_out, &spin_a, &spin_b, 2000, 1, 32)?;
        }
        s.synchronize()?;
        let mut order: Vec<usize> = (0..names.len()).collect();
        for i in (1..order.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            order.swap(i, (rng >> 33) as usize % (i + 1));
        }
        for &k in &order {
            if cold {
                do_flush()?;
            }
            probe.launch_fma_f32(&s, &mut spin_out, &spin_a, &spin_b, 200, 1, 32)?;
            ea.record(&s)?;
            let nwg = launch(&names[k])?;
            eb.record(&s)?;
            s.synchronize()?;
            if round < 3 {
                continue;
            }
            evt[k].push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
            if nwg > 0 {
                let mut t = vec![0u64; ts.len()];
                ts.copy_to_host(&mut t)?;
                if names[k].contains("_pr") {
                    let st: Vec<f64> = (0..nwg as usize).map(|g| t[2 * g] as f64).collect();
                    let pe: Vec<f64> = (0..nwg as usize).map(|g| t[2048 + g] as f64).collect();
                    let en = (0..nwg as usize).map(|g| t[2 * g + 1] as f64).fold(0f64, f64::max);
                    let t0 = st.iter().cloned().fold(f64::INFINITY, f64::min);
                    let _ = en;
                    let rest: Vec<f64> = (0..nwg as usize).map(|g| (t[2 * g + 1] as f64 - pe[g]) / 100.0).collect();
                    pz[k].push(((pct(&pe, 0.1) - t0) / 100.0, (pct(&pe, 0.9) - t0) / 100.0));
                    pz[k].push((pct(&rest, 0.5), pct(&rest, 0.9)));
                }
                t.truncate(2 * nwg as usize);
                let t0 = t.iter().step_by(2).min().copied().unwrap_or(0);
                let t1 = t.iter().skip(1).step_by(2).max().copied().unwrap_or(0);
                span[k].push((t1 - t0) as f64 / 100.0);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(15));
    }
    eprintln!("{} rounds, {}", rounds, if cold { "COLD" } else { "WARM" });
    for (k, n) in names.iter().enumerate() {
        if !pz[k].is_empty() {
            let a: Vec<(f64, f64)> = pz[k].iter().step_by(2).cloned().collect();
            let r: Vec<(f64, f64)> = pz[k].iter().skip(1).step_by(2).cloned().collect();
            eprintln!("  {:<24} pause ends p10..p90 {:>6.2}..{:>6.2} us; per-WG rest p50 {:>6.2} p90 {:>6.2} us",
                n, pct(&a.iter().map(|v| v.0).collect::<Vec<_>>(), 0.5), pct(&a.iter().map(|v| v.1).collect::<Vec<_>>(), 0.5),
                pct(&r.iter().map(|v| v.0).collect::<Vec<_>>(), 0.5), pct(&r.iter().map(|v| v.1).collect::<Vec<_>>(), 0.5));
        }
    }
    for (k, n) in names.iter().enumerate() {
        let sp = pct(&span[k], 0.5);
        eprintln!(
            "  {:<16} span p10 {:>6.2} p50 {:>6.2} p90 {:>6.2} us ({:>4.0} GB/s at p50)   event p50 {:>6.2} p90 {:>6.2}",
            n,
            pct(&span[k], 0.1),
            sp,
            pct(&span[k], 0.9),
            7.86432e6 / sp / 1e3,
            pct(&evt[k], 0.5),
            pct(&evt[k], 0.9)
        );
    }
    Ok(())
}
