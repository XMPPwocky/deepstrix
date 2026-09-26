//! `AttentionDec::launch_score_f16s_rows` (kernels/attention_dec.hip) must
//! write BIT-IDENTICAL f16 scores to
//! `AttentionMixed::launch_score_batched_htiled_wmma_f16s_rows`, on gfx1201
//! (the kernels are gfx12 WMMA; the dGPU is where attention runs), for:
//! B = 1..8 rows; decode windows (128 raw + 512 gathered, per-row stride) and
//! ragged / partial-tile shapes; the CSA bitmask on and off; per-row comp
//! bases (arena dense path); repeated launches. Untouched slots must stay
//! untouched (both write exactly the same key range).
//!
//! The same for the 4-warp `DecScoreKernel::Blk128` kernel (2026-09-26), and
//! for the fused pair `AttentionDec::launch_fused_f16s_rows`
//! (kernels/attention_dec_fused.hip, 2026-09-26): its `out` must be
//! BIT-IDENTICAL to score + `_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows`
//! for B = 1..16, gathered / dense-base / NULL comp stores, ragged tails,
//! n_raw = 0 rows and empty rows, a flat and a peaked softmax, and every
//! output slot written (NaN canary).
//!
//!   cargo test -p v4flash-kernels --release --features v41 --test attention_dec_bitexact -- --ignored --nocapture
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::attention_dec::{AttentionDec, DecScoreKernel, ATTN_DEC_FUSED_MAX_KEYS};
use v4flash_kernels::AttentionMixed;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

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

fn up<T: Copy>(id: i32, h: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}

fn gfx1201() -> eyre::Result<Device> {
    install_panic_handler()?;
    Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201"))
}

fn score_is_bit_identical(kernel: DecScoreKernel) -> eyre::Result<()> {
    let dev = gfx1201()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let mixed = AttentionMixed::for_arch(&arch)?;
    let dec = AttentionDec::for_arch(&arch)?;
    let (nh, hd, bmax) = (64usize, 512usize, 8usize);
    let mut rng = Lcg(0xa77e_dec0_2026);
    let raw_slots = 256usize;
    let store_rows = 4096usize;
    let q = up(id, &(0..bmax * nh * hd).map(|_| rng.unit() * 0.1).collect::<Vec<f32>>())?;
    let raw_kv = up(id, &(0..bmax * raw_slots * hd).map(|_| f16_bits(rng.unit())).collect::<Vec<u16>>())?;
    let comp = up(id, &(0..store_rows * hd).map(|_| f16_bits(rng.unit())).collect::<Vec<u16>>())?;
    let stride = 3072u32;
    let words = ((v4flash_kernels::ATTN_MIXED_MAX_KEYS + 31) / 32) as usize;
    let mask_h: Vec<u32> = (0..bmax * words).map(|_| rng.next() | rng.next()).collect();
    let mask = up(id, &mask_h)?;
    let sentinel = vec![0x7e7e_7e7eu32 as f32; bmax * nh * stride as usize / 2];
    let (mut sa, mut sb) = (up(id, &sentinel)?, up(id, &sentinel)?);
    let mut cases = 0;
    // (n_raw, n_comp, stride mode: 0 per-row gathered, 1 per-row comp base).
    // n_total_max of 129 / 145 / 9 are not multiples of 64 or 256 (the two
    // kernels' keys per WG), so their grid.x differ.
    let shapes: [(u32, u32, u32); 10] = [
        (128, 512, 0),
        (128, 512, 1),
        (1, 1, 0),
        (17, 300, 0),
        (128, 0, 0),
        (100, 1000, 1),
        (128, 511, 0),
        (128, 1, 0),
        (17, 128, 1),
        (9, 0, 0),
    ];
    for b in 1..=bmax as u32 {
        for &(n_raw_max, n_comp_max, mode) in &shapes {
            for masked in [false, true] {
                let bu = b as usize;
                // Ragged rows below the shape's maximum.
                let nr: Vec<i32> = (0..bu).map(|r| (n_raw_max as i32 - (r as i32 * 3)).max(1)).collect();
                let nc: Vec<i32> = (0..bu).map(|r| (n_comp_max as i32 - (r as i32 * 5)).max(0)).collect();
                let off: Vec<i32> = (0..bu).map(|r| (r * raw_slots + (r % 3)) as i32).collect();
                let n_total_max = nr.iter().zip(&nc).map(|(a, c)| (a + c) as u32).max().unwrap();
                let (bstride, base): (u32, Option<Vec<i32>>) = if mode == 0 {
                    (n_comp_max.max(1), None)
                } else {
                    (0, Some((0..bu).map(|r| (r * 311 % 2000) as i32).collect()))
                };
                if mode == 0 && (bu * bstride as usize) > store_rows {
                    continue;
                }
                let (d_nr, d_nc, d_off) = (up(id, &nr)?, up(id, &nc)?, up(id, &off)?);
                let d_base = match &base { Some(v) => Some(up(id, v)?), None => None };
                sa.copy_from_host(&sentinel)?;
                sb.copy_from_host(&sentinel)?;
                let m = if masked { Some(&mask) } else { None };
                mixed.launch_score_batched_htiled_wmma_f16s_rows(
                    &s, &mut sa, &q, &raw_kv, Some(&comp), &d_nr, &d_off, &d_nc, m, 64, 512, n_total_max, b, bstride, stride, d_base.as_ref(),
                )?;
                for _ in 0..2 {
                    dec.launch_score_f16s_rows_with(
                        kernel, &s, &mut sb, &q, &raw_kv, Some(&comp), &d_nr, &d_off, &d_nc, m, 64, 512, n_total_max, b, bstride, stride, d_base.as_ref(),
                    )?;
                }
                s.synchronize()?;
                let n = bu * nh * stride as usize / 2;
                let (mut ha, mut hb) = (vec![0f32; n], vec![0f32; n]);
                sa.slice_view(0, n).copy_to_host(&mut ha)?;
                sb.slice_view(0, n).copy_to_host(&mut hb)?;
                if let Some(i) = ha.iter().zip(&hb).position(|(x, y)| x.to_bits() != y.to_bits()) {
                    return Err(eyre!(
                        "{kernel:?} b={b} shape=({n_raw_max},{n_comp_max},{mode}) masked={masked}: first f16-pair difference at {i}: {:#010x} vs {:#010x}",
                        ha[i].to_bits(), hb[i].to_bits()
                    ));
                }
                cases += 1;
            }
        }
    }
    eprintln!("{arch}: attention_dec score {kernel:?} == batched htiled f16s score in {cases} cases (B = 1..8, ragged, mask on/off, per-row base), x2 launches");
    Ok(())
}

#[test]
#[ignore]
fn attention_dec_score_is_bit_identical() -> eyre::Result<()> {
    score_is_bit_identical(DecScoreKernel::Blk256)
}

/// 2026-09-26 sweep: the 4-warp WG (`attention_dec_score_blk128`).
#[test]
#[ignore]
fn attention_dec_score_blk128_is_bit_identical() -> eyre::Result<()> {
    score_is_bit_identical(DecScoreKernel::Blk128)
}

/// 2026-09-26 sweep: `attention_dec_fused_vt_qreg_sp_d4` vs the pair
/// (dec score `Blk256` + `_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows`).
#[test]
#[ignore]
fn attention_dec_fused_is_bit_identical_to_the_pair() -> eyre::Result<()> {
    let dev = gfx1201()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let mixed = AttentionMixed::for_arch(&arch)?;
    let dec = AttentionDec::for_arch(&arch)?;
    let (nh, hd, bmax) = (64usize, 512usize, 16usize);
    let mut rng = Lcg(0xf05e_dec0_2026);
    let raw_slots = 256usize;
    let store_rows = 8192usize + 512;
    let q_flat = up(id, &(0..bmax * nh * hd).map(|_| rng.unit() * 0.1).collect::<Vec<f32>>())?;
    // Peaked softmax: weights underflow to f16 zero away from the top keys.
    let q_peak = up(id, &(0..bmax * nh * hd).map(|_| rng.unit() * 8.0).collect::<Vec<f32>>())?;
    let raw_kv = up(id, &(0..bmax * raw_slots * hd).map(|_| f16_bits(rng.unit())).collect::<Vec<u16>>())?;
    let comp = up(id, &(0..store_rows * hd).map(|_| f16_bits(rng.unit())).collect::<Vec<u16>>())?;
    let sinks = up(id, &(0..nh).map(|_| rng.unit() * 2.0).collect::<Vec<f32>>())?;
    let stride = 3072u32;
    let mut scores = DeviceBuffer::<f32>::new(id, bmax * nh * stride as usize / 2)?;
    let canary = vec![f32::from_bits(0x7fc0_dead); bmax * nh * hd];
    let (mut oa, mut ob) = (up(id, &canary)?, up(id, &canary)?);
    let mut cases = 0;
    // (n_raw_max, n_comp_max, mode: 0 gathered per-row stride, 1 per-row comp
    // base, 2 comp_kv NULL (ratio-0 layers)). Rows are ragged below the max
    // and may reach n_raw = 0 / n_total = 0.
    let shapes: [(u32, u32, u32); 9] = [
        (128, 512, 0),
        (128, 512, 1),
        (128, 512, 2),
        (1, 1, 0),
        (17, 300, 0),
        (128, 0, 0),
        (128, 511, 0),
        (5, 16, 1),
        (0, 512, 0),
    ];
    for b in 1..=bmax as u32 {
        for &(n_raw_max, n_comp_max, mode) in &shapes {
            for peaked in [false, true] {
                let bu = b as usize;
                let nr: Vec<i32> = (0..bu).map(|r| (n_raw_max as i32 - (r as i32 * 3)).max(0)).collect();
                // A NULL comp store comes with all-zero n_comp_per (ratio-0
                // layers): the pair dereferences comp_kv for any comp key.
                let nc: Vec<i32> = (0..bu)
                    .map(|r| if mode == 2 { 0 } else { (n_comp_max as i32 - (r as i32 * 5)).max(0) })
                    .collect();
                let off: Vec<i32> = (0..bu).map(|r| (r * raw_slots + (r % 3)) as i32).collect();
                let n_total_max = nr.iter().zip(&nc).map(|(a, c)| (a + c) as u32).max().unwrap();
                assert!(n_total_max <= ATTN_DEC_FUSED_MAX_KEYS);
                let (bstride, base): (u32, Option<Vec<i32>>) = match mode {
                    0 => (n_comp_max.max(1), None),
                    1 => (0, Some((0..bu).map(|r| (r * 311 % 2000) as i32).collect())),
                    _ => (0, None),
                };
                if mode == 0 && (bu * bstride as usize) > store_rows {
                    continue;
                }
                let comp_opt = if mode == 2 { None } else { Some(&comp) };
                let q = if peaked { &q_peak } else { &q_flat };
                let (d_nr, d_nc, d_off) = (up(id, &nr)?, up(id, &nc)?, up(id, &off)?);
                let d_base = match &base { Some(v) => Some(up(id, v)?), None => None };
                oa.copy_from_host(&canary)?;
                ob.copy_from_host(&canary)?;
                // The pair (the smwsum runs even at n_total_max == 0: it
                // writes zeros for an empty row).
                dec.launch_score_f16s_rows(
                    &s, &mut scores, q, &raw_kv, comp_opt, &d_nr, &d_off, &d_nc, None, 64, 512, n_total_max, b, bstride, stride, d_base.as_ref(),
                )?;
                mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows(
                    &s, &mut oa, &mut scores, &sinks, &raw_kv, comp_opt, &d_nr, &d_off, &d_nc, 64, 512, b, bstride, stride, d_base.as_ref(),
                )?;
                for _ in 0..2 {
                    dec.launch_fused_f16s_rows(
                        &s, &mut ob, &sinks, q, &raw_kv, comp_opt, &d_nr, &d_off, &d_nc, 64, 512, n_total_max, b, bstride, d_base.as_ref(),
                    )?;
                }
                s.synchronize()?;
                let n = bu * nh * hd;
                let (mut ha, mut hb) = (vec![0f32; n], vec![0f32; n]);
                oa.slice_view(0, n).copy_to_host(&mut ha)?;
                ob.slice_view(0, n).copy_to_host(&mut hb)?;
                if let Some(i) = ha.iter().position(|x| x.to_bits() == 0x7fc0_dead) {
                    return Err(eyre!("b={b} shape=({n_raw_max},{n_comp_max},{mode}): the PAIR left slot {i} unwritten"));
                }
                if let Some(i) = hb.iter().position(|x| x.to_bits() == 0x7fc0_dead) {
                    return Err(eyre!("b={b} shape=({n_raw_max},{n_comp_max},{mode}): the fused kernel left slot {i} unwritten"));
                }
                if let Some(i) = ha.iter().zip(&hb).position(|(x, y)| x.to_bits() != y.to_bits()) {
                    return Err(eyre!(
                        "b={b} shape=({n_raw_max},{n_comp_max},{mode}) peaked={peaked}: first difference at {i} (row {}, head {}, dim {}): pair {:#010x} ({}) vs fused {:#010x} ({})",
                        i / (nh * hd), (i / hd) % nh, i % hd, ha[i].to_bits(), ha[i], hb[i].to_bits(), hb[i]
                    ));
                }
                cases += 1;
            }
        }
    }
    // The launcher must refuse rows the kernel cannot hold.
    let nr = vec![128i32];
    let nc = vec![(ATTN_DEC_FUSED_MAX_KEYS - 128 + 1) as i32];
    let (d_nr, d_nc, d_off) = (up(id, &nr)?, up(id, &nc)?, up(id, &[0i32])?);
    if dec
        .launch_fused_f16s_rows(&s, &mut ob, &sinks, &q_flat, &raw_kv, Some(&comp), &d_nr, &d_off, &d_nc, 64, 512, ATTN_DEC_FUSED_MAX_KEYS + 1, 1, 1, None)
        .is_ok()
    {
        return Err(eyre!("fused launcher accepted n_total_max = {}", ATTN_DEC_FUSED_MAX_KEYS + 1));
    }
    eprintln!("{arch}: attention_dec fused == score + smwsum pair in {cases} cases (B = 1..16, gathered / dense base / NULL comp, ragged, n_raw = 0, empty rows, flat + peaked), x2 launches, all slots written");
    Ok(())
}
