//! The decode indexer's top-512 (`IndexerTopkBitonic::launch_batched`, no bitmap, `done` = the threshold select with
//! the bitonic chain as fallback) at decode shapes: n = compressed rows per stream (~ctx / 4), b = rows per launch.
//! Scores like the indexer's: relu-dot sums (non-negative, heavy-tailed) with a slice masked to -inf (the candidate
//! mask) and the past-n tail -inf (the batched scorer's stamp). Per-call device time of the select path vs the chain
//! alone, the give-up count, and (when built) the radix top-K, which must equal the chain row for row.
//! dGPU, beside a live hub.
//!
//!   cargo test --release --features v41 -p v4flash-kernels --test indexer_topk_decode -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::het::engine::DeviceEngine;

const K: u32 = 512;
const STRIDE: u32 = 65536;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        ((self.next() & 0xFFFFFF) as f32 + 0.5) / 16777216.0
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (6.2831853 * v).cos()
    }
}

fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len())?;
    d.copy_from_host(host)?;
    Ok(d)
}

#[test]
#[ignore]
fn decode_indexer_top512() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201"))?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x1D_E512);
    let (ea, eb) = (Event::new()?, Event::new()?);
    let q = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * p) as usize]
    };
    for &n in &[8192u32, 16384, 32768, 65536] {
        for &b in &[1u32, 2, 4] {
            let scratch_len: usize = v4flash_kernels::indexer::topk_merge_levels(n, K).iter().map(|&x| x as usize).sum::<usize>().max(1);
            let mut scratch = DeviceBuffer::<u32>::new(id, scratch_len * b as usize + 4096)?;
            let mut done = upload(id, &vec![7u32; b as usize])?;
            let mut sel_a = DeviceBuffer::<i32>::new(id, (b * K) as usize)?;
            let mut sel_b = DeviceBuffer::<i32>::new(id, (b * K) as usize)?;
            let n_per = upload(id, &vec![n; b as usize])?;
            let mut rx = v4flash_kernels::topk_radix::TopkRadixScratch::alloc(id, b)?;
            let mut sel_c = DeviceBuffer::<i32>::new(id, (b * K) as usize)?;
            let (mut t_sel, mut t_chain, mut t_rx, mut giveups) = (Vec::new(), Vec::new(), Vec::new(), 0u32);
            for _rep in 0..20 {
                let mut host = vec![f32::NEG_INFINITY; (b * STRIDE) as usize];
                for r in 0..b as usize {
                    for i in 0..n as usize {
                        // relu-dot sums: mostly small, a heavy tail; ~10% masked out by the candidate mask
                        host[r * STRIDE as usize + i] = if rng.next() % 10 == 0 {
                            f32::NEG_INFINITY
                        } else {
                            (rng.normal().max(0.0) * (rng.normal() * 0.7).exp() * 4.0).max(0.0)
                        };
                    }
                }
                let scores = upload(id, &host)?;
                ea.record(&s)?;
                e.indexer_topk_bitonic.launch_batched(&s, &mut sel_a, None, &mut scratch, &scores, &n_per, n, STRIDE, 0, K, b, Some(&mut done))?;
                eb.record(&s)?;
                s.synchronize()?;
                t_sel.push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
                let mut d = vec![0u32; b as usize];
                done.copy_to_host(&mut d)?;
                giveups += d.iter().filter(|&&x| x == 0).count() as u32;
                ea.record(&s)?;
                e.indexer_topk_bitonic.launch_batched(&s, &mut sel_b, None, &mut scratch, &scores, &n_per, n, STRIDE, 0, K, b, None)?;
                eb.record(&s)?;
                s.synchronize()?;
                t_chain.push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
                ea.record(&s)?;
                e.topk_radix.launch(&s, &mut sel_c, &mut rx, &scores, &n_per, STRIDE, K, b)?;
                eb.record(&s)?;
                s.synchronize()?;
                t_rx.push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
                let (mut a, mut c, mut r) = (vec![0i32; (b * K) as usize], vec![0i32; (b * K) as usize], vec![0i32; (b * K) as usize]);
                sel_a.copy_to_host(&mut a)?;
                sel_b.copy_to_host(&mut c)?;
                sel_c.copy_to_host(&mut r)?;
                assert_eq!(a, c, "n {n} b {b}: select and chain disagree");
                assert_eq!(r, c, "n {n} b {b}: radix and chain disagree");
            }
            eprintln!(
                "n {n:6} b {b}: select p50 {:6.1} p90 {:6.1} us (give-ups {giveups}) | chain p50 {:6.1} p90 {:6.1} | RADIX p50 {:6.1} p90 {:6.1} us",
                q(&mut t_sel, 0.5), q(&mut t_sel, 0.9), q(&mut t_chain, 0.5), q(&mut t_chain, 0.9), q(&mut t_rx, 0.5), q(&mut t_rx, 0.9)
            );
        }
    }
    // Edge rows (exactness only): ragged n per row incl. 0, < K and > K; all-equal scores; signed zeros.
    for case in 0..3u32 {
        let b = 4u32;
        let ns: Vec<u32> = match case { 0 => vec![0, 100, 600, 40000], 1 => vec![513, 512, 511, 1], _ => vec![3000, 3000, 3000, 3000] };
        let mut host = vec![f32::NEG_INFINITY; (b * STRIDE) as usize];
        for r in 0..b as usize {
            for i in 0..ns[r] as usize {
                host[r * STRIDE as usize + i] = match case {
                    0 => rng.normal().max(0.0),
                    1 => 1.0,
                    _ => if i % 3 == 0 { 0.0 } else if i % 3 == 1 { -0.0 } else { -rng.unit() },
                };
            }
        }
        let scores = upload(id, &host)?;
        let n_per = upload(id, &ns)?;
        let n_max = *ns.iter().max().unwrap();
        let scratch_len: usize = v4flash_kernels::indexer::topk_merge_levels(n_max, K).iter().map(|&x| x as usize).sum::<usize>().max(1);
        let mut scratch = DeviceBuffer::<u32>::new(id, scratch_len * b as usize + 4096)?;
        let mut sel_b = DeviceBuffer::<i32>::new(id, (b * K) as usize)?;
        let mut sel_c = DeviceBuffer::<i32>::new(id, (b * K) as usize)?;
        let mut rx = v4flash_kernels::topk_radix::TopkRadixScratch::alloc(id, b)?;
        e.indexer_topk_bitonic.launch_batched(&s, &mut sel_b, None, &mut scratch, &scores, &n_per, n_max, STRIDE, 0, K, b, None)?;
        e.topk_radix.launch(&s, &mut sel_c, &mut rx, &scores, &n_per, STRIDE, K, b)?;
        s.synchronize()?;
        let (mut c, mut r) = (vec![0i32; (b * K) as usize], vec![0i32; (b * K) as usize]);
        sel_b.copy_to_host(&mut c)?;
        sel_c.copy_to_host(&mut r)?;
        assert_eq!(r, c, "edge case {case} (n {ns:?}): radix and chain disagree");
        eprintln!("edge case {case} (n {ns:?}): radix == chain");
    }
    Ok(())
}
