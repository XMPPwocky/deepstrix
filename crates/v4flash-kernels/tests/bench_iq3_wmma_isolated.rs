//! Isolated A/B: IQ3_XXS down projection, kwide2 (Q8_K mid) vs f16 WMMA.
//! BENCH_B tokens x 6 slots; BENCH_DIST=uniform (default: BENCH_WI work
//! items of BENCH_CHUNK members each) or zipf (real prefill routing from
//! BENCH_STATS=expert_stats.json, layer BENCH_LAYER, top BENCH_HOT_K
//! experts removed). Both kernels run the same work items; times include
//! q2_k_reduce_partials.
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_DOWN_IN, N_EMBD, N_EXPERT, N_EXPERT_USED};
use v4flash_kernels::iq3_xxs::{Iq3XxsMatvec, BLOCK_IQ3_XXS_BYTES};
use v4flash_kernels::q2_k::Q2KAccumulateMatvec;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") { return Ok(d); }
    }
    Err(eyre!("no gfx1151"))
}

#[test]
#[ignore]
fn bench_iq3_wmma_isolated() -> eyre::Result<()> {
    install_panic_handler()?;
    let b: usize = std::env::var("BENCH_B").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let chunk: u32 = std::env::var("BENCH_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let wi_target: usize = std::env::var("BENCH_WI").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let dist = std::env::var("BENCH_DIST").unwrap_or_else(|_| "uniform".into());
    let n_used = N_EXPERT_USED as usize;
    let n_rows = N_EMBD as usize;
    let nb = BLOCKS_Q8K_DOWN_IN as usize;
    let k_dim = nb * 256;
    let dbpe = n_rows * nb * BLOCK_IQ3_XXS_BYTES;
    let xq_stride = nb * BLOCK_Q8_K_BYTES;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq3 = Iq3XxsMatvec::for_arch(&arch)?;
    let q2k = Q2KAccumulateMatvec::for_arch(&arch)?;

    // weights: random bytes (realistic dequant work)
    let mut w_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, (N_EXPERT as usize) * dbpe)?;
    {
        let mut h = vec![0u8; (N_EXPERT as usize) * dbpe];
        let mut st: u64 = 0x9E3779B97F4A7C15;
        for c in h.chunks_exact_mut(8) { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); c.copy_from_slice(&st.to_le_bytes()); }
        w_d.copy_from_host(&h)?;
    }
    let n_mem = b * n_used;
    let mut x16_d: DeviceBuffer<u16> = DeviceBuffer::new(igpu.id, n_mem * k_dim)?;
    x16_d.copy_from_host(&vec![0x2C00u16; n_mem * k_dim])?;   // 1/16
    let mut xq_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, n_mem * xq_stride)?;
    xq_d.fill_zero()?;

    // work items
    let max_per_expert = b;
    let mut gc = vec![0i32; N_EXPERT as usize];
    let mut em = vec![0i32; (N_EXPERT as usize) * max_per_expert];
    let mut wi: Vec<i32> = Vec::new();
    if dist == "zipf" {
        let stats_path = std::env::var("BENCH_STATS").map_err(|_| eyre!("BENCH_DIST=zipf needs BENCH_STATS"))?;
        let layer: usize = std::env::var("BENCH_LAYER").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
        let hot_k: usize = std::env::var("BENCH_HOT_K").ok().and_then(|s| s.parse().ok()).unwrap_or(17);
        let js: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&stats_path)?)?;
        let counts: Vec<f64> = js["prefill"]["counts"].as_array().ok_or_else(|| eyre!("no prefill.counts"))?
            .iter().skip(layer * 256).take(256).map(|v| v.as_f64().unwrap_or(0.0)).collect();
        let mut order: Vec<usize> = (0..256).collect();
        order.sort_by(|&x, &y| counts[y].partial_cmp(&counts[x]).unwrap());
        let hot: std::collections::HashSet<usize> = order.iter().take(hot_k).cloned().collect();
        let mut cdf = Vec::with_capacity(256); let mut acc = 0f64;
        for e in 0..256 { if !hot.contains(&e) { acc += counts[e]; } cdf.push(acc); }
        let mut rng: u64 = 0xC0FFEE_2026_0908;
        let mut per: Vec<Vec<i32>> = vec![Vec::new(); 256];
        for tok in 0..b {
            let mut picked = std::collections::HashSet::new();
            let mut slot = 0usize;
            while slot < n_used {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((rng >> 11) as f64) / ((1u64 << 53) as f64) * acc;
                let e = cdf.partition_point(|&c| c <= u).min(255);
                if hot.contains(&e) || !picked.insert(e) { continue; }
                per[e].push(((tok as i32) << 16) | (slot as i32));
                slot += 1;
            }
        }
        let mut hist = std::collections::BTreeMap::new();
        for e in 0..256 {
            let n = per[e].len();
            gc[e] = n as i32;
            for (i, &m) in per[e].iter().enumerate() { em[e * max_per_expert + i] = m; }
            let mut s = 0usize;
            while s < n { wi.push(((e as i32) << 16) | (s as i32)); s += chunk as usize; }
            if n > 0 { *hist.entry((n + 15) / 16).or_insert(0usize) += n; }
        }
        eprintln!("zipf dist: layer={layer} hot_k={hot_k} members={} work items={} by tile count {:?}", n_mem, wi.len(), hist);
    } else {
        let n_distinct = wi_target.min(N_EXPERT as usize);
        for i in 0..wi_target { wi.push((((i % n_distinct) as i32) << 16) | 0); }
        for e in 0..n_distinct {
            gc[e] = chunk as i32;
            for i in 0..(chunk as usize) {
                let bi = i % b; let sl = i % n_used;
                em[e * max_per_expert + i] = ((bi as i32) << 16) | (sl as i32);
            }
        }
        eprintln!("uniform dist: {} work items x {chunk} members", wi.len());
    }
    let mut gc_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, gc.len())?; gc_d.copy_from_host(&gc)?;
    let mut em_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, em.len())?; em_d.copy_from_host(&em)?;
    let mut wi_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, wi.len())?; wi_d.copy_from_host(&wi)?;
    let n_wi = wi.len() as u32;
    let mut part_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_mem * n_rows)?;
    let mut out_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, b * n_rows)?;

    for pass in 0..2 {
        let name = if pass == 0 { "iq3 kwide2 (Q8_K)" } else { "iq3 wmma (f16)" };
        let mut run = |part_d: &mut DeviceBuffer<f32>, out_d: &mut DeviceBuffer<f32>| -> eyre::Result<()> {
            if pass == 0 {
                iq3.launch_by_expert_kwide2(&stream, part_d, &w_d, &xq_d, &gc_d, &em_d, &wi_d, n_wi,
                    dbpe as u32, xq_stride as u32, n_used as u32, max_per_expert as u32, chunk, n_rows as u32, nb as u32)?;
            } else {
                iq3.launch_by_expert_wmma(&stream, part_d, &w_d, &x16_d, &gc_d, &em_d, &wi_d, n_wi,
                    dbpe as u32, k_dim as u32, n_used as u32, max_per_expert as u32, chunk, n_rows as u32, nb as u32)?;
            }
            q2k.launch_reduce_partials(&stream, out_d, part_d, n_used as u32, n_rows as u32, b as u32)
        };
        run(&mut part_d, &mut out_d)?;
        stream.synchronize()?;
        let mut walls: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let s = Event::new()?; let e = Event::new()?;
            s.record(&stream)?;
            run(&mut part_d, &mut out_d)?;
            e.record(&stream)?;
            stream.synchronize()?;
            walls.push(Event::elapsed_ms(&s, &e)?);
        }
        walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("{name}: min={:.3} ms  median={:.3} ms  max={:.3} ms", walls[0], walls[walls.len() / 2], walls[walls.len() - 1]);
    }
    Ok(())
}
