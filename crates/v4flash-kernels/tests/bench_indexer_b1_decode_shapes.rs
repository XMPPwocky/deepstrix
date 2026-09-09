//! Decode-shape (B=1) timing of the indexer kernels: does the 2026-09-08
//! prefill work (GEMM-shaped score, exact threshold top-k) carry over to
//! the single-token path?  Kernels only — no model load, ~1 MB of VRAM —
//! so it can run next to a live server.
//!
//!   cargo test --release -p v4flash-kernels --test bench_indexer_b1_decode_shapes -- --ignored --nocapture
//!   BENCH_N_COMPS=1024,24576,49152 BENCH_ITERS=40

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::{IndexerScoreWmma, IndexerTopkBitonic, INDEXER_TOP_K};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 device"))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f32) / (u32::MAX as f32)
    }
}

/// Round-to-nearest f32 -> f16 bits (values here are small and normal).
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

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time<F: FnMut() -> eyre::Result<()>>(stream: &Stream, iters: usize, mut f: F) -> eyre::Result<f32> {
    for _ in 0..3 {
        f()?;
    }
    stream.synchronize()?;
    let mut ws = Vec::with_capacity(iters);
    for _ in 0..iters {
        let s = Event::new()?;
        let e = Event::new()?;
        s.record(stream)?;
        f()?;
        e.record(stream)?;
        stream.synchronize()?;
        ws.push(Event::elapsed_ms(&s, &e)? * 1000.0);
    }
    Ok(median(ws))
}

#[test]
#[ignore]
fn bench_indexer_b1_decode_shapes() -> eyre::Result<()> {
    install_panic_handler()?;
    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let score = IndexerScoreWmma::for_arch(&arch)?;
    let topk = IndexerTopkBitonic::for_arch(&arch)?;
    let top_k = INDEXER_TOP_K;
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let n_comps: Vec<u32> = std::env::var("BENCH_N_COMPS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1024, 8192, 24576, 49152]);
    let mut rng = Lcg(0x70CC_2026_0908);

    eprintln!("B=1 indexer kernels on {arch}, top_k={top_k}, median of {iters} (us)");
    eprintln!(
        "{:>7} | {:>9} {:>9} | {:>11} {:>11} {:>11}",
        "n_comp", "score_mw", "score_gemm", "topk_decode", "topk_chainB1", "topk_select"
    );
    for &n in &n_comps {
        // ---- score inputs: q [64x128] f32 + f16 twin, hw [64], K [n x 128] f16
        let q: Vec<f32> = (0..64 * 128).map(|_| (rng.unit() - 0.5) * 0.2).collect();
        let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, q.len())?;
        d_q.copy_from_host(&q)?;
        let q16: Vec<u16> = q.iter().map(|&v| f16_bits(v)).collect();
        let mut d_q16: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, q16.len())?;
        d_q16.copy_from_host(&q16)?;
        let mut d_hw: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, 64)?;
        d_hw.copy_from_host(&vec![0.05f32; 64])?;
        let kv: Vec<u16> = (0..(n as usize) * 128).map(|_| f16_bits((rng.unit() - 0.5) * 0.5)).collect();
        let mut d_kv: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, kv.len())?;
        d_kv.copy_from_host(&kv)?;
        let mut d_scores: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n as usize)?;
        d_scores.fill_zero()?;
        let mut d_n: DeviceBuffer<u32> = DeviceBuffer::new(dgpu.id, 1)?;
        d_n.copy_from_host(&[n])?;

        let t_mw = time(&stream, iters, || score.launch_mw(&stream, &mut d_scores, &d_q, &d_hw, &d_kv, n))?;
        let t_gemm = time(&stream, iters, || {
            score.launch_batched_gemm(&stream, &mut d_scores, &d_q16, &d_hw, &d_kv, &d_n, n, n, 1, 0)
        })?;

        // ---- top-k on the real (mw) scores
        score.launch_mw(&stream, &mut d_scores, &d_q, &d_hw, &d_kv, n)?;
        stream.synchronize()?;
        let n_chunks = n.div_ceil(4096);
        let mut d_sel: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, top_k as usize)?;
        let mut d_bits: DeviceBuffer<u32> = DeviceBuffer::new(dgpu.id, n.div_ceil(32) as usize)?;
        let mut d_scratch: DeviceBuffer<u32> =
            DeviceBuffer::new(dgpu.id, ((n_chunks * top_k) as usize + 4096).max(1))?;
        let mut d_done: DeviceBuffer<u32> = DeviceBuffer::new(dgpu.id, 1)?;
        d_sel.fill_zero()?;
        d_bits.fill_zero()?;
        d_done.fill_zero()?;

        let t_dec = time(&stream, iters, || {
            topk.launch(&stream, &mut d_sel, &mut d_bits, &mut d_scratch, &d_scores, n, top_k)
        })?;
        let mut sel_dec = vec![0i32; top_k as usize];
        d_sel.copy_to_host(&mut sel_dec)?;

        let t_chain = time(&stream, iters, || {
            topk.launch_batched(&stream, &mut d_sel, None, &mut d_scratch, &d_scores, &d_n, n, n, 0, top_k, 1, None)
        })?;
        let t_sel = time(&stream, iters, || {
            topk.launch_batched(
                &stream, &mut d_sel, None, &mut d_scratch, &d_scores, &d_n, n, n, 0, top_k, 1, Some(&mut d_done),
            )
        })?;
        let mut sel_new = vec![0i32; top_k as usize];
        d_sel.copy_to_host(&mut sel_new)?;
        let mut done = [0u32; 1];
        d_done.copy_to_host(&mut done)?;
        let same = sel_dec == sel_new;

        eprintln!(
            "{:>7} | {:>9.1} {:>9.1} | {:>11.1} {:>11.1} {:>11.1}   select==decode: {} done={}",
            n, t_mw, t_gemm, t_dec, t_chain, t_sel, same, done[0]
        );
        if !same {
            return Err(eyre!("select result differs from the decode chain at n={n}"));
        }
    }
    Ok(())
}
