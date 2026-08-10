//! Oracle: every iq3_xxs GPU kernel vs the scalar CPU reference
//! (`cpu_dot_iq3_xxs_q8_k`, mirroring llama.cpp) on RANDOM blocks.
//!
//! - decode: accumulate (single expert) + batched (n_used experts) +
//!   hetsplit partition identity (mode0 + mode1 == batched)
//! - prefill: by_expert_kwide2 partials, full chunk (32) + partial (19)
//!
//! Random data hardening as q2k_kwide_oracle: d varies per (row, block),
//! gas u32 fully random (signs + 4-bit scale), grid index bytes fully
//! random (all 256 codebook entries valid), xq d varies per (token, block),
//! bsums correct (unused by iq3 but keeps the harness shared-shape).
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test iq3_xxs_oracle -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_DOWN_IN, N_EMBD, N_EXPERT, N_EXPERT_USED};
use v4flash_kernels::iq3_xxs::{Iq3XxsMatvec, BLOCK_IQ3_XXS_BYTES};
use v4flash_kernels::iq3_xxs_tables::cpu_dot_iq3_xxs_q8_k;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 device"))
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_byte(&mut self) -> u8 { (self.next() & 0xff) as u8 }
}

/// Small f16 scales: 1/64, 1/32, 1/16, 1/8.
const F16_SCALES: [u16; 4] = [0x2400, 0x2800, 0x2c00, 0x3000];

fn gen_iq3_expert(rng: &mut Lcg, w: &mut [u8], e: usize, dbpe: usize, n_rows: usize, nb: usize) {
    for r in 0..n_rows {
        for bi in 0..nb {
            let o = e * dbpe + (r * nb + bi) * BLOCK_IQ3_XXS_BYTES;
            let d = F16_SCALES[(rng.next() & 3) as usize].to_le_bytes();
            w[o..o + 2].copy_from_slice(&d);
            for k in 2..BLOCK_IQ3_XXS_BYTES {
                w[o + k] = rng.next_byte();
            }
        }
    }
}

fn gen_xq(rng: &mut Lcg, xq: &mut [u8], n_tokens: usize, stride: usize, nb: usize) {
    for t in 0..n_tokens {
        for bi in 0..nb {
            let o = t * stride + bi * BLOCK_Q8_K_BYTES;
            let d_val = 0.02f32 + ((rng.next() & 0xff) as f32) * 0.0008f32;
            xq[o..o + 4].copy_from_slice(&d_val.to_le_bytes());
            let mut bsums = [0i16; 16];
            for k in 0..256 {
                let v = rng.next_byte() as i8;
                xq[o + 4 + k] = v as u8;
                bsums[k / 16] += v as i16;
            }
            for (j, s) in bsums.iter().enumerate() {
                xq[o + 260 + 2 * j..o + 262 + 2 * j].copy_from_slice(&s.to_le_bytes());
            }
        }
    }
}

fn check(name: &str, got: &[f32], want: &[f32]) -> eyre::Result<()> {
    assert_eq!(got.len(), want.len());
    let mut max_abs_diff = 0f32;
    let mut max_abs_ref = 0f32;
    for (g, w) in got.iter().zip(want) {
        max_abs_diff = max_abs_diff.max((g - w).abs());
        max_abs_ref = max_abs_ref.max(w.abs());
    }
    if max_abs_ref < 1e-6 {
        return Err(eyre!("{name}: reference near-zero — degenerate test"));
    }
    let rel = max_abs_diff / max_abs_ref;
    eprintln!("iq3 {name}: n={} max|ref|={max_abs_ref:.4} max_diff={max_abs_diff:.6} rel={rel:.6}",
        got.len());
    if rel >= 5e-3 {
        return Err(eyre!("iq3 {name} diverges: rel={rel}"));
    }
    Ok(())
}

#[test]
#[ignore]
fn iq3_xxs_decode_kernels_match_cpu() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq3 = Iq3XxsMatvec::for_arch(&arch)?;

    let n_used = N_EXPERT_USED as usize; // 6
    let n_rows = N_EMBD as usize;        // 4096 (down out-dim)
    let nb = BLOCKS_Q8K_DOWN_IN as usize; // 8 blocks of 256 = 2048 in-dim
    let dbpe = n_rows * nb * BLOCK_IQ3_XXS_BYTES;
    let stride = nb * BLOCK_Q8_K_BYTES;

    let mut rng = Lcg::new(0x1930c0de);
    let sel: [i32; 6] = [3, 250, 17, 3, 99, 42]; // includes a repeat
    let mut w_host = vec![0u8; (N_EXPERT as usize) * dbpe];
    for e in [3usize, 250, 17, 99, 42] {
        gen_iq3_expert(&mut rng, &mut w_host, e, dbpe, n_rows, nb);
    }
    let mut xq_host = vec![0u8; n_used * stride];
    gen_xq(&mut rng, &mut xq_host, n_used, stride, nb);

    let mut w_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, w_host.len())?;
    w_d.copy_from_host(&w_host)?;
    let mut xq_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, xq_host.len())?;
    xq_d.copy_from_host(&xq_host)?;
    let mut sel_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_used)?;
    sel_d.copy_from_host(&sel)?;

    // CPU reference: out[row] = Σ_s dot(expert sel[s] row, xq slot s)
    let mut want = vec![0f32; n_rows];
    for (s, &e) in sel.iter().enumerate() {
        let xq_s = &xq_host[s * stride..(s + 1) * stride];
        for row in 0..n_rows {
            let wo = (e as usize) * dbpe + row * nb * BLOCK_IQ3_XXS_BYTES;
            want[row] += cpu_dot_iq3_xxs_q8_k(nb, &w_host[wo..wo + nb * BLOCK_IQ3_XXS_BYTES], xq_s);
        }
    }

    // accumulate path (slot by slot)
    let mut out_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_rows)?;
    for (s, &e) in sel.iter().enumerate() {
        iq3.launch_accumulate(
            &stream, &mut out_d, &w_d, (e as usize) * dbpe,
            // slot s activation: pass an offset view via a fresh buffer slice
            &xq_d_slice(&xq_d, igpu.id, &xq_host, s * stride, stride)?,
            n_rows as u32, nb as u32, s == 0,
        )?;
    }
    stream.synchronize()?;
    let mut got = vec![0f32; n_rows];
    out_d.copy_to_host(&mut got)?;
    check("accumulate", &got, &want)?;

    // batched path
    let mut out_b: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_rows)?;
    iq3.launch_batched(
        &stream, &mut out_b, &w_d, &xq_d, &sel_d,
        dbpe as u32, stride as u32, n_used as u32, n_rows as u32, nb as u32,
    )?;
    stream.synchronize()?;
    out_b.copy_to_host(&mut got)?;
    check("batched", &got, &want)?;

    // hetsplit partition identity: mode0(cap=2) + mode1(cap=2) == batched.
    // remap: experts 250 and 99 "resident" at dense slots 0/1.
    let mut remap_h = vec![-1i32; 256];
    remap_h[250] = 0;
    remap_h[99] = 1;
    let mut remap_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, 256)?;
    remap_d.copy_from_host(&remap_h)?;
    // dense-packed hot weights: slot 0 = expert 250, slot 1 = expert 99
    let mut hot_host = vec![0u8; 2 * dbpe];
    hot_host[..dbpe].copy_from_slice(&w_host[250 * dbpe..251 * dbpe]);
    hot_host[dbpe..].copy_from_slice(&w_host[99 * dbpe..100 * dbpe]);
    let mut hot_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, hot_host.len())?;
    hot_d.copy_from_host(&hot_host)?;

    let mut out_m0: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_rows)?;
    let mut out_m1: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_rows)?;
    iq3.launch_batched_hetsplit(
        &stream, &mut out_m0, &w_d, &xq_d, &sel_d, &remap_d, 0, 2,
        dbpe as u32, stride as u32, n_used as u32, n_rows as u32, nb as u32,
    )?;
    iq3.launch_batched_hetsplit(
        &stream, &mut out_m1, &hot_d, &xq_d, &sel_d, &remap_d, 1, 2,
        dbpe as u32, stride as u32, n_used as u32, n_rows as u32, nb as u32,
    )?;
    stream.synchronize()?;
    let mut g0 = vec![0f32; n_rows];
    let mut g1 = vec![0f32; n_rows];
    out_m0.copy_to_host(&mut g0)?;
    out_m1.copy_to_host(&mut g1)?;
    let sum: Vec<f32> = g0.iter().zip(&g1).map(|(a, b)| a + b).collect();
    check("hetsplit m0+m1", &sum, &want)?;
    Ok(())
}

/// Upload a sub-range of xq as its own buffer (launch_accumulate takes the
/// whole buffer as slot activation).
fn xq_d_slice(
    _xq_d: &DeviceBuffer<u8>,
    device_id: i32,
    xq_host: &[u8],
    off: usize,
    len: usize,
) -> eyre::Result<DeviceBuffer<u8>> {
    let mut b: DeviceBuffer<u8> = DeviceBuffer::new(device_id, len)?;
    b.copy_from_host(&xq_host[off..off + len])?;
    Ok(b)
}

#[test]
#[ignore]
fn iq3_xxs_kwide2_matches_cpu() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq3 = Iq3XxsMatvec::for_arch(&arch)?;

    let b: u32 = 48;
    let n_used = N_EXPERT_USED as usize;
    let n_rows = N_EMBD as usize;
    let nb = BLOCKS_Q8K_DOWN_IN as usize;
    let dbpe = n_rows * nb * BLOCK_IQ3_XXS_BYTES;
    let stride = nb * BLOCK_Q8_K_BYTES;
    let max_per_expert = b as usize;
    let chunk_size: u32 = 32;
    // Full chunk + partial chunk (unroll tail).
    let experts: [(usize, i32); 2] = [(1, 32), (7, 19)];

    let mut rng = Lcg::new(0xabad1dea);
    let mut w_host = vec![0u8; (N_EXPERT as usize) * dbpe];
    for (e, _) in experts {
        gen_iq3_expert(&mut rng, &mut w_host, e, dbpe, n_rows, nb);
    }
    let n_tokens = (b as usize) * n_used;
    let mut xq_host = vec![0u8; n_tokens * stride];
    gen_xq(&mut rng, &mut xq_host, n_tokens, stride, nb);

    let mut gc_h = vec![0i32; N_EXPERT as usize];
    let mut em_h = vec![0i32; (N_EXPERT as usize) * max_per_expert];
    let mut wi_h: Vec<i32> = Vec::new();
    let mut touched: Vec<(usize, usize, usize)> = Vec::new(); // (e, b, slot)
    let mut next_pair = 0usize;
    for (e, group_n) in experts {
        gc_h[e] = group_n;
        for i in 0..(group_n as usize) {
            let b_idx = next_pair % (b as usize);
            let slot = (next_pair / (b as usize)) % n_used;
            em_h[e * max_per_expert + i] = ((b_idx as i32) << 16) | (slot as i32);
            touched.push((e, b_idx, slot));
            next_pair += 1;
        }
        wi_h.push(((e as i32) << 16) | 0i32);
    }
    let n_work_items = wi_h.len() as u32;

    let mut w_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, w_host.len())?;
    w_d.copy_from_host(&w_host)?;
    let mut xq_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, xq_host.len())?;
    xq_d.copy_from_host(&xq_host)?;
    let mut gc_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, gc_h.len())?;
    gc_d.copy_from_host(&gc_h)?;
    let mut em_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, em_h.len())?;
    em_d.copy_from_host(&em_h)?;
    let mut wi_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, wi_h.len())?;
    wi_d.copy_from_host(&wi_h)?;

    let part_elems = (b as usize) * n_used * n_rows;
    let mut part_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, part_elems)?;
    part_d.fill_zero()?;

    iq3.launch_by_expert_kwide2(
        &stream, &mut part_d, &w_d, &xq_d, &gc_d, &em_d, &wi_d, n_work_items,
        dbpe as u32, stride as u32, n_used as u32, max_per_expert as u32, chunk_size,
        n_rows as u32, nb as u32,
    )?;
    stream.synchronize()?;
    let mut got = vec![0f32; part_elems];
    part_d.copy_to_host(&mut got)?;

    // CPU reference for touched (b, slot) pairs.
    let mut got_t = Vec::new();
    let mut want_t = Vec::new();
    for &(e, b_idx, slot) in &touched {
        let xo = (b_idx * n_used + slot) * stride;
        let xq_s = &xq_host[xo..xo + stride];
        for row in 0..n_rows {
            let wo = e * dbpe + row * nb * BLOCK_IQ3_XXS_BYTES;
            want_t.push(cpu_dot_iq3_xxs_q8_k(
                nb,
                &w_host[wo..wo + nb * BLOCK_IQ3_XXS_BYTES],
                xq_s,
            ));
            got_t.push(got[(b_idx * n_used + slot) * n_rows + row]);
        }
    }
    check("kwide2 partials", &got_t, &want_t)?;
    Ok(())
}
