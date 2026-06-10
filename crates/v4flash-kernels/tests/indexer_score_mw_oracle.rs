//! Oracle + isolated A/B: `indexer_score_wmma_batched_mw` (M52 multi-wave)
//! vs production `indexer_score_wmma_batched`.
//!
//! Both kernels compute identical WMMA tiles with identical accumulation
//! order — only the staging differs (cooperative-Q + global-B-frag vs
//! per-WG-Q + LDS-K). Expect BIT-EXACT score equality, including the
//! -INF stamping for cols ∈ [n_comp, n_idx_stride).
//!
//! n_idx_per varies per token to exercise: full chunks, partial tiles
//! (not %16), tiny n_comp (early-out WGs), and zero-tail stamping.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test indexer_score_mw_oracle -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use std::time::Instant;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{IndexerScoreWmma, INDEXER_HEAD_DIM, INDEXER_N_HEAD};

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
    fn new(seed: u64) -> Self { Lcg(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        // [-1, 1)
        (self.next() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    /// Random small f16 bit pattern: finite values in (-2, 2).
    fn next_f16_bits(&mut self) -> u16 {
        let v = self.next_f32();
        let bits = (v.to_bits() >> 16) as u16;
        // crude f32→bf16-ish; instead build a real f16 from a small float:
        let _ = bits;
        half_from_f32(v)
    }
}

fn half_from_f32(x: f32) -> u16 {
    // minimal f32→f16 (round-to-nearest-even not needed for test data)
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 { return sign; }
    if exp >= 31 { return sign | 0x7800; }
    let mant = ((b >> 13) & 0x3ff) as u16;
    sign | ((exp as u16) << 10) | mant
}

#[test]
#[ignore]
fn indexer_score_mw_matches_batched() -> eyre::Result<()> {
    install_panic_handler()?;
    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let kernel = IndexerScoreWmma::for_arch(&arch)?;

    let n_head = INDEXER_N_HEAD as usize;
    let head_dim = INDEXER_HEAD_DIM as usize;
    let batch: u32 = 32;
    let n_idx_max: u32 = 4096 + 17; // partial tail tile
    let n_idx_stride: u32 = 4352;   // > n_idx_max, exercises -INF stamping

    let mut rng = Lcg::new(0xabad1dea);
    let mut q_host = vec![0f32; (batch as usize) * n_head * head_dim];
    for v in q_host.iter_mut() { *v = rng.next_f32(); }
    let mut hw_host = vec![0f32; (batch as usize) * n_head];
    for v in hw_host.iter_mut() { *v = rng.next_f32() * 0.25; }
    let mut kv_host = vec![0u16; (n_idx_max as usize) * head_dim];
    for v in kv_host.iter_mut() { *v = rng.next_f16_bits(); }
    // Varied per-token n_comp: full, partial tiles, tiny, 1, mid.
    let n_idx_host: Vec<u32> = (0..batch)
        .map(|i| match i % 6 {
            0 => n_idx_max,
            1 => n_idx_max - 7,
            2 => 1024,
            3 => 33,
            4 => 1,
            _ => 2048 + (i * 97) % 1000,
        })
        .collect();

    let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, q_host.len())?;
    d_q.copy_from_host(&q_host)?;
    let mut d_hw: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, hw_host.len())?;
    d_hw.copy_from_host(&hw_host)?;
    let mut d_kv: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, kv_host.len())?;
    d_kv.copy_from_host(&kv_host)?;
    let mut d_n_idx: DeviceBuffer<u32> = DeviceBuffer::new(dgpu.id, batch as usize)?;
    d_n_idx.copy_from_host(&n_idx_host)?;

    let score_elems = (batch as usize) * (n_idx_stride as usize);
    let mut d_scores_ref: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, score_elems)?;
    let mut d_scores_mw: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, score_elems)?;
    d_scores_ref.fill_zero()?;
    d_scores_mw.fill_zero()?;

    kernel.launch_batched(
        &stream, &mut d_scores_ref, &d_q, &d_hw, &d_kv, &d_n_idx,
        n_idx_max, n_idx_stride, batch,
    )?;
    kernel.launch_batched_mw(
        &stream, &mut d_scores_mw, &d_q, &d_hw, &d_kv, &d_n_idx,
        n_idx_max, n_idx_stride, batch,
    )?;
    stream.synchronize()?;

    let mut ref_host = vec![0f32; score_elems];
    let mut mw_host = vec![0f32; score_elems];
    d_scores_ref.copy_to_host(&mut ref_host)?;
    d_scores_mw.copy_to_host(&mut mw_host)?;

    // Contract: bit-exact on [0, n_comp) (the only range topk reads); on
    // [n_comp, n_idx_stride) mw must stamp -INF (the 1-wave kernel's grid is
    // sized by n_idx_max and may leave cols past its coverage unwritten).
    let mut n_diff = 0usize;
    let mut n_tail_bad = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    for bi in 0..(batch as usize) {
        let n_comp = n_idx_host[bi] as usize;
        for c in 0..(n_idx_stride as usize) {
            let i = bi * (n_idx_stride as usize) + c;
            let (a, b) = (ref_host[i], mw_host[i]);
            if c < n_comp {
                if a.to_bits() != b.to_bits() {
                    n_diff += 1;
                    if first.is_none() { first = Some((i, a, b)); }
                }
            } else if b != f32::MIN.min(-3.4028235e38f32) && b != -3.4028235e38f32 {
                n_tail_bad += 1;
                if first.is_none() { first = Some((i, a, b)); }
            }
        }
    }
    eprintln!("mw vs batched: n={score_elems} n_bit_diff(valid)={n_diff} n_tail_not_neginf={n_tail_bad} first={first:?}");
    assert_eq!(n_diff, 0, "mw diverges from batched on valid cols");
    assert_eq!(n_tail_bad, 0, "mw left non--INF in [n_comp, stride)");

    // --- M58: non-batched (decode) mw vs 1-wave — bit-exact expected ---
    {
        let n_comp: u32 = 4096 + 17;
        let kv1 = d_kv.slice_view(0, (n_comp as usize) * head_dim);
        let mut s_ref: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_comp as usize)?;
        let mut s_mw: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_comp as usize)?;
        s_ref.fill_zero()?;
        s_mw.fill_zero()?;
        let q1 = d_q.slice_view(0, n_head * head_dim);
        let hw1 = d_hw.slice_view(0, n_head);
        kernel.launch(&stream, &mut s_ref, &q1, &hw1, &kv1, n_comp)?;
        kernel.launch_mw(&stream, &mut s_mw, &q1, &hw1, &kv1, n_comp)?;
        stream.synchronize()?;
        let mut a = vec![0f32; n_comp as usize];
        let mut b = vec![0f32; n_comp as usize];
        s_ref.copy_to_host(&mut a)?;
        s_mw.copy_to_host(&mut b)?;
        let bad = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        eprintln!("decode mw vs 1-wave: n={} n_bit_diff={}", n_comp, bad);
        assert_eq!(bad, 0, "decode mw diverges");
    }

    // --- Isolated timing A/B at the 96K production shape ---
    let b96: u32 = 512;
    let n96: u32 = 24576;
    let mut d_q2: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, (b96 as usize) * n_head * head_dim)?;
    d_q2.copy_from_host(&vec![0.1f32; (b96 as usize) * n_head * head_dim])?;
    let mut d_hw2: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, (b96 as usize) * n_head)?;
    d_hw2.copy_from_host(&vec![0.05f32; (b96 as usize) * n_head])?;
    let mut d_kv2: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, (n96 as usize) * head_dim)?;
    d_kv2.copy_from_host(&kv_host.iter().cycle().take((n96 as usize) * head_dim).copied().collect::<Vec<_>>())?;
    let mut d_n2: DeviceBuffer<u32> = DeviceBuffer::new(dgpu.id, b96 as usize)?;
    d_n2.copy_from_host(&vec![n96; b96 as usize])?;
    let mut d_s2: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, (b96 as usize) * (n96 as usize))?;
    d_s2.fill_zero()?;

    for name in ["batched(1-wave)", "mw(8-wave)"] {
        let mw = name.starts_with("mw");
        for _ in 0..2 {
            if mw {
                kernel.launch_batched_mw(&stream, &mut d_s2, &d_q2, &d_hw2, &d_kv2, &d_n2, n96, n96, b96)?;
            } else {
                kernel.launch_batched(&stream, &mut d_s2, &d_q2, &d_hw2, &d_kv2, &d_n2, n96, n96, b96)?;
            }
        }
        stream.synchronize()?;
        let t0 = Instant::now();
        let iters = 10;
        for _ in 0..iters {
            if mw {
                kernel.launch_batched_mw(&stream, &mut d_s2, &d_q2, &d_hw2, &d_kv2, &d_n2, n96, n96, b96)?;
            } else {
                kernel.launch_batched(&stream, &mut d_s2, &d_q2, &d_hw2, &d_kv2, &d_n2, n96, n96, b96)?;
            }
        }
        stream.synchronize()?;
        eprintln!(
            "{name}: B={b96} n_idx={n96}: {:.2} ms/call",
            t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
        );
    }
    Ok(())
}
