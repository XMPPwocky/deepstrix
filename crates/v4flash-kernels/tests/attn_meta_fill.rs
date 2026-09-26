//! `AttnMetaFill` (kernels/attn_meta.hip) must leave exactly the device bytes
//! the per-array `copy_from_host_async` uploads it replaces leave, for 1..=16
//! rows, with any subset of destinations (i32 and u32 buffers), repeated
//! launches, and neighbours past `n` untouched. gfx1201 + gfx1151.
//!
//!   cargo test -p v4flash-kernels --release --features v41 --test attn_meta_fill -- --ignored --nocapture
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::attn_meta::{AttnMetaFill, MetaDst, ATTN_META_MAX_ROWS};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn run_on(dev: Device) -> eyre::Result<()> {
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let k = AttnMetaFill::for_arch(&arch)?;
    let mut rng = Lcg(0xa77e_2026_0926);
    let cap = 64usize;
    let sentinel = vec![0x5a5a_5a5au32 as i32; cap];
    let mk = |v: &[i32]| -> eyre::Result<DeviceBuffer<i32>> {
        let mut d = DeviceBuffer::new(id, v.len())?;
        d.copy_from_host(v)?;
        Ok(d)
    };
    for n in 1..=ATTN_META_MAX_ROWS {
        for mask in 1u32..16 {
            let vals: Vec<Vec<i32>> = (0..4)
                .map(|_| (0..n).map(|_| (rng.next() % 400_000) as i32 - if rng.next() % 5 == 0 { 7 } else { 0 }).collect())
                .collect();
            // Reference: the copies.
            let mut refs: Vec<DeviceBuffer<i32>> = (0..4).map(|_| mk(&sentinel)).collect::<eyre::Result<_>>()?;
            let mut outs: Vec<DeviceBuffer<i32>> = (0..4).map(|_| mk(&sentinel)).collect::<eyre::Result<_>>()?;
            let mut nidx_ref: DeviceBuffer<u32> = DeviceBuffer::new(id, cap)?;
            nidx_ref.copy_from_host(&vec![0x5a5a_5a5au32; cap])?;
            let mut nidx_out: DeviceBuffer<u32> = DeviceBuffer::new(id, cap)?;
            nidx_out.copy_from_host(&vec![0x5a5a_5a5au32; cap])?;
            let nidx_vals: Vec<u32> = vals[3].iter().map(|&v| v as u32).collect();
            for (j, r) in refs.iter_mut().enumerate().take(3) {
                if mask & (1 << j) != 0 {
                    r.slice_view_mut(0, n).copy_from_host_async(&vals[j], &s)?;
                }
            }
            if mask & 8 != 0 {
                nidx_ref.slice_view_mut(0, n).copy_from_host_async(&nidx_vals, &s)?;
            }
            for _rep in 0..2 {
                let (o0, rest) = outs.split_at_mut(1);
                let (o1, rest) = rest.split_at_mut(1);
                let o2 = &mut rest[0];
                k.launch(
                    &s,
                    [
                        if mask & 1 != 0 { Some(MetaDst::new(&mut o0[0], &vals[0])?) } else { None },
                        if mask & 2 != 0 { Some(MetaDst::new(&mut o1[0], &vals[1])?) } else { None },
                        if mask & 4 != 0 { Some(MetaDst::new(o2, &vals[2])?) } else { None },
                        if mask & 8 != 0 { Some(MetaDst::new(&mut nidx_out, &vals[3])?) } else { None },
                    ],
                )?;
            }
            s.synchronize()?;
            for j in 0..3 {
                let (mut a, mut b) = (vec![0i32; cap], vec![0i32; cap]);
                refs[j].copy_to_host(&mut a)?;
                outs[j].copy_to_host(&mut b)?;
                if a != b {
                    return Err(eyre!("{arch} n={n} mask={mask:#x} dst {j}: {a:?} vs {b:?}"));
                }
            }
            let (mut a, mut b) = (vec![0u32; cap], vec![0u32; cap]);
            nidx_ref.copy_to_host(&mut a)?;
            nidx_out.copy_to_host(&mut b)?;
            if a != b {
                return Err(eyre!("{arch} n={n} mask={mask:#x} u32 dst: {a:?} vs {b:?}"));
            }
        }
    }
    // Rejections: too many rows, mismatched lengths, no destination.
    let mut d = mk(&sentinel)?;
    let mut e2 = mk(&sentinel)?;
    let big = vec![1i32; ATTN_META_MAX_ROWS + 1];
    if k.launch(&s, [Some(MetaDst::new(&mut d, &big)?), None, None, None]).is_ok() {
        return Err(eyre!("accepted {} rows", big.len()));
    }
    if k.launch(&s, [Some(MetaDst::new(&mut d, &[1, 2])?), Some(MetaDst::new(&mut e2, &[1])?), None, None]).is_ok() {
        return Err(eyre!("accepted mismatched lengths"));
    }
    if k.launch(&s, [None, None, None, None]).is_ok() {
        return Err(eyre!("accepted no destination"));
    }
    eprintln!("{arch}: attn_meta_fill == copies for n = 1..={ATTN_META_MAX_ROWS}, all 15 destination subsets, x2");
    Ok(())
}

#[test]
#[ignore]
fn attn_meta_fill_matches_copies() -> eyre::Result<()> {
    install_panic_handler()?;
    let mut ran = 0;
    for d in Device::all()? {
        let arch = d.properties()?.gcn_arch_name;
        if arch.starts_with("gfx1201") || arch.starts_with("gfx1151") {
            run_on(d)?;
            ran += 1;
        }
    }
    if ran == 0 {
        return Err(eyre!("no gfx1201/gfx1151 device"));
    }
    Ok(())
}
