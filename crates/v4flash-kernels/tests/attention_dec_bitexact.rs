//! `AttentionDec::launch_score_f16s_rows` (kernels/attention_dec.hip) must
//! write BIT-IDENTICAL f16 scores to
//! `AttentionMixed::launch_score_batched_htiled_wmma_f16s_rows`, on gfx1201
//! (the kernels are gfx12 WMMA; the dGPU is where attention runs), for:
//! B = 1..8 rows; decode windows (128 raw + 512 gathered, per-row stride) and
//! ragged / partial-tile shapes; the CSA bitmask on and off; per-row comp
//! bases (arena dense path); repeated launches. Untouched slots must stay
//! untouched (both write exactly the same key range).
//!
//!   cargo test -p v4flash-kernels --release --features v41 --test attention_dec_bitexact -- --ignored --nocapture
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::attention_dec::AttentionDec;
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

#[test]
#[ignore]
fn attention_dec_score_is_bit_identical() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201"))?;
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
    // (n_raw, n_comp, stride mode: 0 per-row gathered, 1 per-row comp base)
    let shapes: [(u32, u32, u32); 7] = [(128, 512, 0), (128, 512, 1), (1, 1, 0), (17, 300, 0), (128, 0, 0), (100, 1000, 1), (128, 511, 0)];
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
                    dec.launch_score_f16s_rows(
                        &s, &mut sb, &q, &raw_kv, Some(&comp), &d_nr, &d_off, &d_nc, m, 64, 512, n_total_max, b, bstride, stride, d_base.as_ref(),
                    )?;
                }
                s.synchronize()?;
                let n = bu * nh * stride as usize / 2;
                let (mut ha, mut hb) = (vec![0f32; n], vec![0f32; n]);
                sa.slice_view(0, n).copy_to_host(&mut ha)?;
                sb.slice_view(0, n).copy_to_host(&mut hb)?;
                if let Some(i) = ha.iter().zip(&hb).position(|(x, y)| x.to_bits() != y.to_bits()) {
                    return Err(eyre!(
                        "b={b} shape=({n_raw_max},{n_comp_max},{mode}) masked={masked}: first f16-pair difference at {i}: {:#010x} vs {:#010x}",
                        ha[i].to_bits(), hb[i].to_bits()
                    ));
                }
                cases += 1;
            }
        }
    }
    eprintln!("{arch}: attention_dec score == batched htiled f16s score in {cases} cases (B = 1..8, ragged, mask on/off, per-row base), x2 launches");
    Ok(())
}
