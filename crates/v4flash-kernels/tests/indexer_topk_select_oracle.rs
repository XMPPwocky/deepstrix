//! `indexer_topk_select_batched` (threshold select + guarded bitonic
//! fallback) vs the bitonic chain alone: `selected` must be bit-identical
//! (same set AND order). Cases: random scores; heavy exact ties at the
//! boundary; all-equal; n < top_k; n <= 4096 (single sort path); the 96K and
//! 192K production shapes; -inf-masked entries. Reports fallback counts and
//! timing. dGPU only.
//! Run: `cargo test --release -p v4flash-kernels --test indexer_topk_select_oracle -- --ignored --nocapture`
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::{IndexerTopkBitonic, INDEXER_TOP_K};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") { return Ok(d); }
    }
    Err(eyre!("no gfx1201"))
}
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 32) as u32 }
    fn unit(&mut self) -> f32 { (self.next() as f32) / (u32::MAX as f32) }
}

struct Case { name: &'static str, batch: u32, n_idx_max: u32, n_idx: Vec<u32>, scores: Vec<f32>, time: bool }

fn gen(rng: &mut Lcg, name: &'static str, batch: u32, n_idx_max: u32, n_idx: Vec<u32>, kind: u32, time: bool) -> Case {
    let stride = n_idx_max as usize;
    let mut scores = vec![-3.4028235e38f32; batch as usize * stride];
    for bi in 0..batch as usize {
        let n = n_idx[bi] as usize;
        for c in 0..n {
            scores[bi * stride + c] = match kind {
                0 => (rng.unit() - 0.5) * 20.0,                       // random
                1 => ((rng.next() % 64) as f32) * 0.125 - 4.0,        // 64 distinct values: massive ties
                2 => 1.0,                                             // all equal
                3 => if rng.next() % 3 == 0 { -3.4028235e38 } else { (rng.unit() - 0.5) * 20.0 }, // masked entries
                _ => { let v = (rng.unit() - 0.5) * 20.0; if c % 7 == 0 { 3.0 } else { v } }   // a tie cluster near the top
            };
        }
    }
    Case { name, batch, n_idx_max, n_idx, scores, time }
}

#[test]
#[ignore]
fn indexer_topk_select_matches_chain() -> eyre::Result<()> {
    install_panic_handler()?;
    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let k = IndexerTopkBitonic::for_arch(&arch)?;
    let top_k = INDEXER_TOP_K;
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let mut rng = Lcg(0x70CC_2026_0908);
    let ragged = |n_max: u32, b: u32| -> Vec<u32> { (0..b).map(|i| match i % 7 { 0 => n_max, 1 => n_max - 1, 2 => 4096, 3 => 4097, 4 => 511, 5 => 1, _ => n_max / 2 + i }).collect() };
    let causal = |n_max: u32, b: u32| -> Vec<u32> { (0..b).map(|i| n_max - 128 + i / 4).collect() };
    let cases = vec![
        gen(&mut rng, "random ragged 9K", 40, 9000, ragged(9000, 40), 0, false),
        gen(&mut rng, "64-value ties 9K", 40, 9000, ragged(9000, 40), 1, false),
        gen(&mut rng, "all-equal 9K", 24, 9000, ragged(9000, 24), 2, false),
        gen(&mut rng, "masked -inf 9K", 40, 9000, ragged(9000, 40), 3, false),
        gen(&mut rng, "tie cluster 9K", 40, 9000, ragged(9000, 40), 4, false),
        gen(&mut rng, "random 96K", 512, 24576, causal(24576, 512), 0, true),
        gen(&mut rng, "random 192K", 512, 49152, causal(49152, 512), 0, true),
        gen(&mut rng, "ties 192K", 128, 49152, causal(49152, 128), 1, false),
    ];
    let mut total_fallback = 0usize;
    for case in &cases {
        let (b, nmax, stride) = (case.batch, case.n_idx_max, case.n_idx_max);
        let mut d_scores: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, case.scores.len())?; d_scores.copy_from_host(&case.scores)?;
        let mut d_n: DeviceBuffer<u32> = DeviceBuffer::new(dgpu.id, b as usize)?; d_n.copy_from_host(&case.n_idx)?;
        let n_chunks = nmax.div_ceil(4096);
        let scratch_len = (b as usize) * ((n_chunks * top_k) as usize + 4096);
        let mut scratch: DeviceBuffer<u32> = DeviceBuffer::new(dgpu.id, scratch_len)?;
        let mut sel_ref: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, (b * top_k) as usize)?;
        let mut sel_new: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, (b * top_k) as usize)?;
        let mut done: DeviceBuffer<u32> = DeviceBuffer::new(dgpu.id, b as usize)?;
        sel_ref.fill_zero()?; sel_new.fill_zero()?; done.fill_zero()?;
        k.launch_batched(&stream, &mut sel_ref, None, &mut scratch, &d_scores, &d_n, nmax, stride, 0, top_k, b, None)?;
        k.launch_batched(&stream, &mut sel_new, None, &mut scratch, &d_scores, &d_n, nmax, stride, 0, top_k, b, Some(&mut done))?;
        stream.synchronize()?;
        let mut h_ref = vec![0i32; (b * top_k) as usize]; sel_ref.copy_to_host(&mut h_ref)?;
        let mut h_new = vec![0i32; (b * top_k) as usize]; sel_new.copy_to_host(&mut h_new)?;
        let mut h_done = vec![0u32; b as usize]; done.copy_to_host(&mut h_done)?;
        let fallbacks = h_done.iter().filter(|&&d| d == 0).count();
        total_fallback += fallbacks;
        let mut bad_tokens = 0usize; let mut first = None;
        for bi in 0..b as usize {
            let (r, n) = (&h_ref[bi * top_k as usize..(bi + 1) * top_k as usize], &h_new[bi * top_k as usize..(bi + 1) * top_k as usize]);
            if r != n { bad_tokens += 1; if first.is_none() { let p = r.iter().zip(n).position(|(a, b)| a != b).unwrap(); first = Some((bi, p, r[p], n[p])); } }
        }
        eprintln!("{:20} B={b:3} n_idx_max={nmax:5}: mismatched tokens {bad_tokens}, fallbacks {fallbacks}{}", case.name,
            first.map(|(bi, p, a, c)| format!("  first diff tok {bi} pos {p}: chain={a} select={c}")).unwrap_or_default());
        if bad_tokens > 0 { return Err(eyre!("select diverges from chain on {}", case.name)); }
        if case.time {
            let mut time = |use_fast: bool| -> eyre::Result<f32> {
                let mut ws = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let s = Event::new()?; let e = Event::new()?;
                    s.record(&stream)?;
                    k.launch_batched(&stream, &mut sel_new, None, &mut scratch, &d_scores, &d_n, nmax, stride, 0, top_k, b, if use_fast { Some(&mut done) } else { None })?;
                    e.record(&stream)?; stream.synchronize()?;
                    ws.push(Event::elapsed_ms(&s, &e)?);
                }
                ws.sort_by(|a, b| a.partial_cmp(b).unwrap()); Ok(ws[ws.len() / 2])
            };
            let t_chain = time(false)?; let t_fast = time(true)?;
            eprintln!("    chain {t_chain:7.3} ms | select(+guarded chain) {t_fast:7.3} ms | {:.2}x", t_chain / t_fast);
        }
    }
    eprintln!("PASS (total fallbacks across cases: {total_fallback})");
    Ok(())
}
