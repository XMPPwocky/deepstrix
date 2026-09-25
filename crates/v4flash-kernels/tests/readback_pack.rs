//! `ReadbackPack` gathers device segments back to back into pinned host
//! memory, visible to the host after an event wait (no stream sync, no copy).
//! Both archs; segment counts 1..=10 including empty ones, odd sizes, a
//! multi-block tail, and the error paths.
//!
//!   cargo test -p v4flash-kernels --release --features v41 --test readback_pack -- --ignored --nocapture
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, PinnedBuffer, Stream};
use v4flash_kernels::readback_pack::{PackSeg, RB_PACK_MAX_SEG};
use v4flash_kernels::ReadbackPack;

#[test]
#[ignore]
fn readback_pack_gathers_segments() -> eyre::Result<()> {
    install_panic_handler()?;
    let mut archs_seen = 0;
    for dev in Device::all()? {
        let arch = dev.properties()?.gcn_arch_name;
        if !(arch.starts_with("gfx1201") || arch.starts_with("gfx1151")) {
            continue;
        }
        dev.set_current()?;
        let id = dev.id;
        let pack = ReadbackPack::for_arch(&arch)?;
        let s = Stream::new(id)?;
        // As production's `selected_ready`: timing off, system fence on.
        let ev = Event::new_no_timing()?;
        // Sizes in words; 0 = empty segment. The last case spans several blocks.
        let cases: [&[usize]; 5] = [
            &[6],
            &[6, 0, 6, 1, 6],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 13],
            &[48, 48, 48, 384, 384, 48, 8, 48, 9344, 5],
        ];
        for (ci, sizes) in cases.iter().enumerate() {
            let mut want: Vec<u32> = Vec::new();
            let mut bufs_u32: Vec<DeviceBuffer<u32>> = Vec::new();
            for (k, &n) in sizes.iter().enumerate() {
                // Longer than used: only the first n words may be gathered.
                let h: Vec<u32> = (0..n + 3).map(|i| ((ci * 1000 + k * 100 + i) as u32).wrapping_mul(2654435761)).collect();
                let mut d = DeviceBuffer::<u32>::new(id, h.len())?;
                d.copy_from_host(&h)?;
                want.extend_from_slice(&h[..n]);
                bufs_u32.push(d);
            }
            let segs: Vec<PackSeg> = bufs_u32.iter().zip(sizes.iter()).map(|(d, &n)| PackSeg::words(d, n)).collect::<eyre::Result<_>>()?;
            let mut dst = PinnedBuffer::<u32>::new(want.len() + 7)?;
            // Poison, so a missed word shows.
            dst.as_mut_slice().fill(0xdead_beef);
            pack.launch(&s, &mut dst, &segs)?;
            ev.record(&s)?;
            ev.synchronize()?;
            if dst.as_slice()[..want.len()] != want[..] {
                return Err(eyre!("{arch} case {ci}: gathered words differ"));
            }
            if dst.as_slice()[want.len()..].iter().any(|&w| w != 0xdead_beef) {
                return Err(eyre!("{arch} case {ci}: wrote past the gathered words"));
            }
        }
        // Byte segments (box 2's xq) land as native-endian words.
        let hb: Vec<u8> = (0..292u32 * 2).map(|i| (i * 13 % 251) as u8).collect();
        let mut db = DeviceBuffer::<u8>::new(id, hb.len())?;
        db.copy_from_host(&hb)?;
        let mut dst = PinnedBuffer::<u32>::new(hb.len() / 4)?;
        pack.launch(&s, &mut dst, &[PackSeg::bytes(&db, hb.len())?])?;
        ev.record(&s)?;
        ev.synchronize()?;
        let got: Vec<u8> = dst.as_slice().iter().flat_map(|w| w.to_ne_bytes()).collect();
        if got != hb {
            return Err(eyre!("{arch}: byte segment differs"));
        }
        // ONE destination reused across launches, new data each time (as a
        // lane's `rb_pack` is reused every layer), checked after each wait.
        let sizes = [6usize, 6, 6, 1, 6, 1460];
        let mut srcs: Vec<DeviceBuffer<u32>> = Vec::new();
        for &n in &sizes {
            srcs.push(DeviceBuffer::<u32>::new(id, n)?);
        }
        let total: usize = sizes.iter().sum();
        let mut dst = PinnedBuffer::<u32>::new(total)?;
        for it in 0..200u32 {
            let mut want: Vec<u32> = Vec::with_capacity(total);
            for (k, (d, &n)) in srcs.iter_mut().zip(sizes.iter()).enumerate() {
                let h: Vec<u32> = (0..n as u32).map(|i| it.wrapping_mul(7919) ^ (k as u32) << 24 ^ i).collect();
                d.copy_from_host(&h)?;
                want.extend_from_slice(&h);
            }
            let segs: Vec<PackSeg> = srcs.iter().zip(sizes.iter()).map(|(d, &n)| PackSeg::words(d, n)).collect::<eyre::Result<_>>()?;
            pack.launch(&s, &mut dst, &segs)?;
            ev.record(&s)?;
            ev.synchronize()?;
            if dst.as_slice() != &want[..] {
                return Err(eyre!("{arch}: reused destination, iteration {it}: words differ"));
            }
        }
        // Error paths: too many segments, staging too small, bad byte count, overlong, wrong element size.
        let one = DeviceBuffer::<u32>::new(id, 4)?;
        let segs = vec![PackSeg::words(&one, 1)?; RB_PACK_MAX_SEG + 1];
        let mut big = PinnedBuffer::<u32>::new(64)?;
        if pack.launch(&s, &mut big, &segs).is_ok() {
            return Err(eyre!("{arch}: accepted {} segments", RB_PACK_MAX_SEG + 1));
        }
        let mut small = PinnedBuffer::<u32>::new(3)?;
        if pack.launch(&s, &mut small, &[PackSeg::words(&one, 4)?]).is_ok() {
            return Err(eyre!("{arch}: overflowed the staging"));
        }
        if PackSeg::bytes(&db, 6).is_ok() || PackSeg::words(&one, 5).is_ok() {
            return Err(eyre!("{arch}: accepted a bad segment"));
        }
        let halves = DeviceBuffer::<u16>::new(id, 8)?;
        if PackSeg::words(&halves, 2).is_ok() {
            return Err(eyre!("{arch}: accepted 2-byte elements"));
        }
        println!("{arch}: readback_pack OK");
        archs_seen += 1;
    }
    if archs_seen < 2 {
        return Err(eyre!("expected gfx1201 and gfx1151, saw {archs_seen}"));
    }
    Ok(())
}
