//! `indexer_score_wmma_gemm` (8 tokens/WG, K tile shared) vs the 1-wave
//! `indexer_score_wmma_batched` reference: bit-exact on valid cols, -INF on
//! [n_comp, stride), over ragged per-token n_comp (incl. tokens whose whole
//! range is empty) and a tail that is not a multiple of the 64-row tile.
//! Then a timing at the 192K production shape (B=512, n_idx 49152) for the
//! mw kernel vs gemm at a few split counts. dGPU only.
//! Run: `cargo test --release -p v4flash-kernels --test indexer_score_gemm_oracle -- --ignored --nocapture`
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::q8_k::Q8KQuantize;
use v4flash_kernels::IndexerScoreWmma;

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
fn f16_bits(x: f32) -> u16 { v4flash_kernels::weight_contract::f32_to_f16_bits(x) }

fn run_case(dgpu: &Device, stream: &Stream, k: &IndexerScoreWmma, cast: &Q8KQuantize,
            batch: u32, n_idx_max: u32, n_idx_stride: u32, n_idx_host: &[u32], rng: &mut Lcg,
            splits: &[u32], time_iters: usize) -> eyre::Result<()> {
    let (nh, hd) = (64usize, 128usize);
    let q_host: Vec<f32> = (0..batch as usize * nh * hd).map(|_| (rng.unit() - 0.5) * 2.0).collect();
    let hw_host: Vec<f32> = (0..batch as usize * nh).map(|_| rng.unit() * 0.25).collect();
    let kv_host: Vec<u16> = (0..n_idx_max as usize * hd).map(|_| f16_bits((rng.unit() - 0.5) * 2.0)).collect();
    let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, q_host.len())?; d_q.copy_from_host(&q_host)?;
    let mut d_q16: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, q_host.len())?;
    cast.launch_cast_f16(stream, &mut d_q16, &d_q, q_host.len() as u32)?;
    let mut d_hw: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, hw_host.len())?; d_hw.copy_from_host(&hw_host)?;
    let mut d_kv: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, kv_host.len())?; d_kv.copy_from_host(&kv_host)?;
    let mut d_n: DeviceBuffer<u32> = DeviceBuffer::new(dgpu.id, batch as usize)?; d_n.copy_from_host(n_idx_host)?;
    let n_out = batch as usize * n_idx_stride as usize;
    let mut s_ref: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_out)?;
    let mut s_new: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_out)?;
    s_ref.fill_zero()?;
    k.launch_batched(stream, &mut s_ref, &d_q, &d_hw, &d_kv, &d_n, n_idx_max, n_idx_stride, batch)?;
    stream.synchronize()?;
    let mut h_ref = vec![0f32; n_out]; s_ref.copy_to_host(&mut h_ref)?;
    for &sp in splits {
        s_new.fill_zero()?;
        // poison: any col the kernel forgets to write shows up as 0 vs -inf/finite
        k.launch_batched_gemm(stream, &mut s_new, &d_q16, &d_hw, &d_kv, &d_n, n_idx_max, n_idx_stride, batch, sp)?;
        stream.synchronize()?;
        let mut h_new = vec![0f32; n_out]; s_new.copy_to_host(&mut h_new)?;
        let (mut n_diff, mut n_tail_bad, mut max_abs) = (0usize, 0usize, 0f32);
        for bi in 0..batch as usize {
            let nc = n_idx_host[bi] as usize;
            for c in 0..n_idx_stride as usize {
                let (a, b) = (h_ref[bi * n_idx_stride as usize + c], h_new[bi * n_idx_stride as usize + c]);
                if c < nc {
                    if a.to_bits() != b.to_bits() { n_diff += 1; max_abs = max_abs.max((a - b).abs()); }
                } else if b != -3.4028235e38f32 { n_tail_bad += 1; }
            }
        }
        eprintln!("  B={batch} n_idx_max={n_idx_max} stride={n_idx_stride} splits={sp}: bit-diffs={n_diff} (max_abs {max_abs:.3e}) tail-bad={n_tail_bad}");
        if n_diff != 0 || n_tail_bad != 0 { return Err(eyre!("gemm diverges from batched (splits={sp})")); }
    }
    if time_iters > 0 {
        let mut time = |f: &mut dyn FnMut() -> eyre::Result<()>| -> eyre::Result<f32> {
            f()?; stream.synchronize()?;
            let mut ws = Vec::with_capacity(time_iters);
            for _ in 0..time_iters { let s = Event::new()?; let e = Event::new()?; s.record(stream)?; f()?; e.record(stream)?; stream.synchronize()?; ws.push(Event::elapsed_ms(&s, &e)?); }
            ws.sort_by(|a, b| a.partial_cmp(b).unwrap()); Ok(ws[ws.len() / 2])
        };
        let flops = 2.0 * batch as f64 * n_idx_max as f64 * 64.0 * 128.0;
        let t_mw = time(&mut || k.launch_batched_mw(stream, &mut s_ref, &d_q, &d_hw, &d_kv, &d_n, n_idx_max, n_idx_stride, batch))?;
        eprintln!("  mw   : {t_mw:7.3} ms  {:6.1} TF ({:4.1}% of 194.6)", flops / (t_mw as f64 * 1e-3) / 1e12, 100.0 * flops / (t_mw as f64 * 1e-3) / 1e12 / 194.6);
        for &sp in splits {
            let t = time(&mut || k.launch_batched_gemm(stream, &mut s_new, &d_q16, &d_hw, &d_kv, &d_n, n_idx_max, n_idx_stride, batch, sp))?;
            eprintln!("  gemm splits={sp:2}: {t:7.3} ms  {:6.1} TF ({:4.1}% of 194.6)  {:.2}x", flops / (t as f64 * 1e-3) / 1e12, 100.0 * flops / (t as f64 * 1e-3) / 1e12 / 194.6, t_mw / t);
        }
    }
    Ok(())
}

#[test]
#[ignore]
fn indexer_score_gemm_matches_batched() -> eyre::Result<()> {
    install_panic_handler()?;
    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let k = IndexerScoreWmma::for_arch(&arch)?;
    let cast = Q8KQuantize::for_arch(&arch)?;
    let mut rng = Lcg(0x1DE7_2026_0908);
    // 1. ragged small case (the mw oracle's shape + odd batch)
    let batch = 35u32; let n_idx_max = 4096 + 17; let stride = 4352;
    let n_idx: Vec<u32> = (0..batch).map(|i| match i % 6 { 0 => n_idx_max, 1 => n_idx_max - 7, 2 => 1024, 3 => 33, 4 => 1, _ => 2048 + (i * 97) % 1000 }).collect();
    run_case(&dgpu, &stream, &k, &cast, batch, n_idx_max, stride, &n_idx, &mut rng, &[0, 1, 3], 0)?;
    // 2. causal-like: consecutive tokens, n_comp grows by 1 every 4 tokens, split boundary inside a token's range
    let batch = 64u32; let n_idx_max = 2000; let stride = 2048;
    let n_idx: Vec<u32> = (0..batch).map(|i| 1900 + i / 4).collect();
    run_case(&dgpu, &stream, &k, &cast, batch, n_idx_max, stride, &n_idx, &mut rng, &[0, 5], 0)?;
    // 3. production shape @192K: B=512, n_idx 49152 (+timing)
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let batch = 512u32; let n_idx_max = 49152; let stride = 49152;
    let n_idx: Vec<u32> = (0..batch).map(|i| n_idx_max - 128 + i / 4).collect();
    run_case(&dgpu, &stream, &k, &cast, batch, n_idx_max, stride, &n_idx, &mut rng, &[0, 2, 4, 8], iters)?;
    // 4. @96K
    let batch = 512u32; let n_idx_max = 24576; let stride = 24576;
    let n_idx: Vec<u32> = (0..batch).map(|i| n_idx_max - 128 + i / 4).collect();
    run_case(&dgpu, &stream, &k, &cast, batch, n_idx_max, stride, &n_idx, &mut rng, &[0, 4], iters)?;
    eprintln!("PASS");
    Ok(())
}
