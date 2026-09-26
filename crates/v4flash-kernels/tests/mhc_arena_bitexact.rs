//! `MhcArena` (kernels/mhc_arena.hip) must be BIT-IDENTICAL to the kernel
//! chains the arena path ran before, at V4.1 shape, for 1..=8 rows:
//!   pre-attn  : per row {rms_nw_mw inv-only, f16 matvec_pre_scaled} + sinkhorn
//!               vs launch_mix(MIX_PRE_SCALED)
//!   pre-ffn   : rms_nw batched + f16 matvec_narrow_batched + sinkhorn
//!               vs launch_mix(MIX_NORMED)
//! and stay so on repeated launches and on a captured-graph replay (the
//! last-WG counter resets itself).
//!
//!   cargo test -p v4flash-kernels --release --features v41 --test mhc_arena_bitexact -- --ignored --nocapture
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, N_HC, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS};
use v4flash_kernels::mhc_arena::{MIX_KSPLIT, MIX_NORMED, MIX_PRE_SCALED};
use v4flash_kernels::{F16Matvec, HcSinkhorn, MhcArena, RmsNormNoWeight, RmsNormNoWeightMultiWG};

/// The tests run on parallel threads of one process; a graph capture on one
/// test's blocking stream and a legacy-stream memset / copy on another
/// (`fill_zero`, `copy_to_host`) invalidate each other
/// (hipErrorStreamCaptureImplicit). Serialize them.
static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn up<T: Copy>(id: i32, h: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}

fn down(d: &DeviceBuffer<f32>, n: usize) -> eyre::Result<Vec<u32>> {
    let mut h = vec![0f32; n];
    d.slice_view(0, n).copy_to_host(&mut h)?;
    Ok(h.iter().map(|v| v.to_bits()).collect())
}

fn same(name: &str, a: &[u32], b: &[u32]) -> eyre::Result<()> {
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        return Err(eyre!("{name}: first difference at {i}: {} vs {}", f32::from_bits(a[i]), f32::from_bits(b[i])));
    }
    Ok(())
}

fn run_on(dev: Device) -> eyre::Result<()> {
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let rms_nw = RmsNormNoWeight::for_arch(&arch)?;
    let rms_nw_mw = RmsNormNoWeightMultiWG::for_arch(&arch)?;
    let f16 = F16Matvec::for_arch(&arch)?;
    let sink = HcSinkhorn::for_arch(&arch)?;
    let arena = MhcArena::for_arch(&arch)?;
    let (hcd, m, bmax) = (HC_DIM as usize, HC_MIX_DIM as usize, 8usize);
    let mut rng = Lcg(0x4d48_4341_7265_6e61);

    // f16 weights with magnitudes 2^-5..2^-1, random signs.
    let w_bits: Vec<u16> = (0..m * hcd)
        .map(|_| {
            let r = rng.next();
            (((r >> 31) as u16) << 15) | ((10 + (r % 5) as u16) << 10) | ((r >> 8) as u16 & 0x3ff)
        })
        .collect();
    let w_bytes: Vec<u8> = w_bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w = up(id, &w_bytes)?;
    let x = up(id, &(0..bmax * hcd).map(|_| rng.unit() * 2.0).collect::<Vec<f32>>())?;
    let scale = up(id, &[0.8f32, 1.3, 0.6])?;
    let base = up(id, &(0..m).map(|_| rng.unit()).collect::<Vec<f32>>())?;
    let z = |n: usize| -> eyre::Result<DeviceBuffer<f32>> {
        let mut d = DeviceBuffer::new(id, n)?;
        d.fill_zero()?;
        Ok(d)
    };
    let (mut mix_o, mut split_o, mut mix_n, mut split_n) = (z(bmax * m)?, z(bmax * m)?, z(bmax * m)?, z(bmax * m)?);
    let mut flat = z(bmax * hcd)?;
    let (mut inv, mut part) = (z(1)?, z(16)?);
    let mut counters: DeviceBuffer<u32> = DeviceBuffer::new(id, bmax)?;
    counters.fill_zero()?;

    for b in 1..=bmax as u32 {
        let bu = b as usize;
        for mode in [MIX_PRE_SCALED, MIX_NORMED] {
            // old chain
            if mode == MIX_PRE_SCALED {
                for r in 0..bu {
                    let row = x.slice_view(r * hcd, hcd);
                    rms_nw_mw.launch_inv_only(&s, &mut inv, &row, &mut part, HC_DIM, 16, RMS_EPS)?;
                    let mut mr = mix_o.slice_view_mut(r * m, m);
                    f16.matvec_pre_scaled(&s, &mut mr, &w, &row, &inv, HC_MIX_DIM, HC_DIM)?;
                }
            } else {
                rms_nw.launch_batched(&s, &mut flat, &x, 1, HC_DIM, RMS_EPS, b)?;
                f16.matvec_narrow_batched(&s, &mut mix_o, &w, &flat, HC_MIX_DIM, HC_DIM, b)?;
            }
            sink.launch_batched(&s, &mut split_o, &mix_o, &scale, &base, N_HC, SINKHORN_ITERS, SINKHORN_EPS, b)?;
            // new, three times: fresh, repeated, and from a captured graph
            for rep in 0..3 {
                mix_n.fill_zero()?;
                split_n.fill_zero()?;
                if rep < 2 {
                    arena.launch_mix(&s, &mut split_n, &mut mix_n, &mut counters, &w, &x, &scale, &base, HC_DIM, mode, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b)?;
                } else {
                    s.begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    let enq = arena.launch_mix(&s, &mut split_n, &mut mix_n, &mut counters, &w, &x, &scale, &base, HC_DIM, mode, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b);
                    let g = s.end_capture()?;
                    enq?;
                    let exec = g.instantiate()?;
                    exec.launch(&s)?;
                    exec.launch(&s)?;
                }
                s.synchronize()?;
                let tag = format!("{arch} b={b} mode={mode} rep={rep}");
                same(&format!("mix {tag}"), &down(&mix_o, bu * m)?, &down(&mix_n, bu * m)?)?;
                same(&format!("split {tag}"), &down(&split_o, bu * m)?, &down(&split_n, bu * m)?)?;
                let mut c = vec![0u32; bmax];
                counters.copy_to_host(&mut c)?;
                if c.iter().any(|&v| v != 0) {
                    return Err(eyre!("counters not reset after {tag}: {c:?}"));
                }
            }
        }
        eprintln!("{arch} b={b}: mix/split (both modes, x3 incl. graph replay) bit-identical");
    }
    Ok(())
}

#[test]
#[ignore]
fn mhc_arena_is_bit_identical() -> eyre::Result<()> {
    let _gpu = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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

/// TIER 2 (`launch_mix_ksplit`) is NOT bit-identical: report how far it lands
/// from the exact chains (pre-attn pre-scaled; pre-ffn normalize-then-dot).
#[test]
#[ignore]
fn mhc_arena_ksplit_error() -> eyre::Result<()> {
    let _gpu = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    install_panic_handler()?;
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201"))?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let rms_nw = RmsNormNoWeight::for_arch(&arch)?;
    let rms_nw_mw = RmsNormNoWeightMultiWG::for_arch(&arch)?;
    let f16 = F16Matvec::for_arch(&arch)?;
    let sink = HcSinkhorn::for_arch(&arch)?;
    let arena = MhcArena::for_arch(&arch)?;
    let (hcd, m, bmax, sp) = (HC_DIM as usize, HC_MIX_DIM as usize, 4usize, MIX_KSPLIT as usize);
    let mut rng = Lcg(0x6b73_706c_6974);
    let w_bits: Vec<u16> = (0..m * hcd)
        .map(|_| {
            let r = rng.next();
            (((r >> 31) as u16) << 15) | ((10 + (r % 5) as u16) << 10) | ((r >> 8) as u16 & 0x3ff)
        })
        .collect();
    let w_bytes: Vec<u8> = w_bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w = up(id, &w_bytes)?;
    let x = up(id, &(0..bmax * hcd).map(|_| rng.unit() * 2.0).collect::<Vec<f32>>())?;
    let scale = up(id, &[0.8f32, 1.3, 0.6])?;
    let base = up(id, &(0..m).map(|_| rng.unit()).collect::<Vec<f32>>())?;
    let z = |n: usize| -> eyre::Result<DeviceBuffer<f32>> {
        let mut d = DeviceBuffer::new(id, n)?;
        d.fill_zero()?;
        Ok(d)
    };
    let (mut mix_o, mut split_o, mut mix_k, mut split_k) = (z(bmax * m)?, z(bmax * m)?, z(bmax * m)?, z(bmax * m)?);
    let (mut flat, mut inv, mut part, mut dotp, mut sqp) = (z(bmax * hcd)?, z(1)?, z(16)?, z(bmax * m * sp)?, z(bmax * sp)?);
    let mut counters: DeviceBuffer<u32> = DeviceBuffer::new(id, bmax)?;
    counters.fill_zero()?;
    let b = bmax as u32;
    for mode in [MIX_PRE_SCALED, MIX_NORMED] {
        if mode == MIX_PRE_SCALED {
            for r in 0..bmax {
                let row = x.slice_view(r * hcd, hcd);
                rms_nw_mw.launch_inv_only(&s, &mut inv, &row, &mut part, HC_DIM, 16, RMS_EPS)?;
                let mut mr = mix_o.slice_view_mut(r * m, m);
                f16.matvec_pre_scaled(&s, &mut mr, &w, &row, &inv, HC_MIX_DIM, HC_DIM)?;
            }
        } else {
            rms_nw.launch_batched(&s, &mut flat, &x, 1, HC_DIM, RMS_EPS, b)?;
            f16.matvec_narrow_batched(&s, &mut mix_o, &w, &flat, HC_MIX_DIM, HC_DIM, b)?;
        }
        sink.launch_batched(&s, &mut split_o, &mix_o, &scale, &base, N_HC, SINKHORN_ITERS, SINKHORN_EPS, b)?;
        arena.launch_mix_ksplit(&s, &mut split_k, &mut mix_k, &mut dotp, &mut sqp, &mut counters, &w, &x, &scale, &base, HC_DIM, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b)?;
        s.synchronize()?;
        let f = |d: &DeviceBuffer<f32>| -> eyre::Result<Vec<f32>> {
            let mut h = vec![0f32; bmax * m];
            d.slice_view(0, bmax * m).copy_to_host(&mut h)?;
            Ok(h)
        };
        let (mo, mk, so, sk) = (f(&mix_o)?, f(&mix_k)?, f(&split_o)?, f(&split_k)?);
        let err = |a: &[f32], c: &[f32]| -> (f32, f32, usize) {
            let mut mx = 0f32;
            let mut rel = 0f32;
            let mut n_diff = 0;
            for (u, v) in a.iter().zip(c) {
                let d = (u - v).abs();
                if u.to_bits() != v.to_bits() { n_diff += 1; }
                mx = mx.max(d);
                rel = rel.max(d / u.abs().max(1e-6));
            }
            (mx, rel, n_diff)
        };
        let (ma, mr, mn) = err(&mo, &mk);
        let (sa, srl, sn) = err(&so, &sk);
        eprintln!("ksplit vs exact, mode {mode}: mix max|d| {ma:.3e} max rel {mr:.3e} ({mn}/{} differ); split max|d| {sa:.3e} max rel {srl:.3e} ({sn}/{} differ)", bmax * m, bmax * m);
        if !(mr < 1e-3 && sa < 1e-3) {
            return Err(eyre!("ksplit error too large in mode {mode}"));
        }
    }
    Ok(())
}

/// `MhcArena::launch_fast` (kernels/mhc_fast.hip) must be BIT-IDENTICAL to the
/// arena sequences it replaces, for every production combination, 1..=8 rows,
/// fresh / repeated / captured-graph replay (twice):
///   pre-attn      : launch_mix(PRE_SCALED); hc_weighted(x, carry); carry := split; rms_w
///   pre-attn CED  : the same without the carry copy (KvSourceOnly)
///   pre-ffn inline: launch_mix(NORMED); hc_weighted; carry := split; rms_w
///   pre-ffn late  : hc_weighted; rms_w                       (collapse only)
///   mix_ffn_late  : launch_mix(NORMED); carry := split       (mix only)
/// Compared: split, mix, cur, norm and the carry after the launch.
/// (`launch_mix` itself is pinned to the pre-arena chains by
/// `mhc_arena_is_bit_identical` above.)
fn fast_run_on(dev: Device) -> eyre::Result<()> {
    use v4flash_kernels::mhc_arena::{FastCollapse, FastMix};
    use v4flash_kernels::{HcWeightedSum, RmsNorm};
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let arena = MhcArena::for_arch(&arch)?;
    let wsum = HcWeightedSum::for_arch(&arch)?;
    let rms_w = RmsNorm::for_arch(&arch)?;
    let n_embd = v4flash_kernels::config::N_EMBD;
    let (hcd, m, ne, bmax) = (HC_DIM as usize, HC_MIX_DIM as usize, n_embd as usize, 8usize);
    let mut rng = Lcg(0x6661_7374_6d68_6321);
    let w_bits: Vec<u16> = (0..m * hcd)
        .map(|_| {
            let r = rng.next();
            (((r >> 31) as u16) << 15) | ((10 + (r % 5) as u16) << 10) | ((r >> 8) as u16 & 0x3ff)
        })
        .collect();
    let w_bytes: Vec<u8> = w_bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w = up(id, &w_bytes)?;
    let x = up(id, &(0..bmax * hcd).map(|_| rng.unit() * 2.0).collect::<Vec<f32>>())?;
    let scale = up(id, &[0.8f32, 1.3, 0.6])?;
    let base = up(id, &(0..m).map(|_| rng.unit()).collect::<Vec<f32>>())?;
    let norm_w = up(id, &(0..ne).map(|_| rng.unit() + 1.0).collect::<Vec<f32>>())?;
    // A realistic carry: positive pre weights, then post / comb.
    let carry0: Vec<f32> =
        (0..bmax * m).map(|i| if i % m < 4 { 0.1 + 0.5 * (rng.unit() + 1.0) } else { rng.unit() }).collect();
    let z = |n: usize| -> eyre::Result<DeviceBuffer<f32>> {
        let mut d = DeviceBuffer::new(id, n)?;
        d.fill_zero()?;
        Ok(d)
    };
    let (mut mix_o, mut split_o, mut cur_o, mut norm_o) = (z(bmax * m)?, z(bmax * m)?, z(bmax * ne)?, z(bmax * ne)?);
    let (mut mix_n, mut split_n, mut cur_n, mut norm_n) = (z(bmax * m)?, z(bmax * m)?, z(bmax * ne)?, z(bmax * ne)?);
    let (mut carry_o, mut carry_n) = (up(id, &carry0)?, up(id, &carry0)?);
    let mut inv_rows = z(bmax)?;
    let mut counters: DeviceBuffer<u32> = DeviceBuffer::new(id, bmax)?;
    counters.fill_zero()?;
    // (name, mix mode, collapse, write_carry)
    let cases: [(&str, Option<u32>, bool, bool); 5] = [
        ("pre_attn", Some(MIX_PRE_SCALED), true, true),
        ("pre_attn_ced_source", Some(MIX_PRE_SCALED), true, false),
        ("pre_ffn_inline", Some(MIX_NORMED), true, true),
        ("pre_ffn_late_collapse", None, true, false),
        ("mix_ffn_late", Some(MIX_NORMED), false, true),
    ];
    for b in 1..=bmax as u32 {
        let bu = b as usize;
        for &(name, mode, collapse, write_carry) in &cases {
            // Reference: the arena chain, in production order.
            carry_o.copy_from_host(&carry0)?;
            for d in [&mut mix_o, &mut split_o, &mut cur_o, &mut norm_o] {
                d.fill_zero()?;
            }
            if let Some(md) = mode {
                arena.launch_mix(&s, &mut split_o, &mut mix_o, &mut counters, &w, &x, &scale, &base, HC_DIM, md, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b)?;
            }
            if collapse {
                wsum.launch_batched(&s, &mut cur_o, &x, &carry_o, n_embd, N_HC, HC_MIX_DIM, b)?;
            }
            if write_carry {
                let src = split_o.slice_view(0, bu * m);
                carry_o.slice_view_mut(0, bu * m).copy_from_buffer_async(&src, &s)?;
            }
            if collapse {
                rms_w.launch_weighted_batched(&s, &mut norm_o, &cur_o, &norm_w, n_embd, RMS_EPS, b)?;
            }
            s.synchronize()?;
            for rep in 0..3 {
                carry_n.copy_from_host(&carry0)?;
                for d in [&mut mix_n, &mut split_n, &mut cur_n, &mut norm_n] {
                    d.fill_zero()?;
                }
                let go = |s: &Stream,
                          split_n: &mut DeviceBuffer<f32>,
                          mix_n: &mut DeviceBuffer<f32>,
                          cur_n: &mut DeviceBuffer<f32>,
                          norm_n: &mut DeviceBuffer<f32>,
                          carry_n: &mut DeviceBuffer<f32>,
                          counters: &mut DeviceBuffer<u32>,
                          inv_rows: &mut DeviceBuffer<f32>|
                 -> eyre::Result<()> {
                    let mx = mode.map(|md| FastMix {
                        weight: &w,
                        x: &x,
                        scale: &scale,
                        base: &base,
                        mode: md,
                        split_out: split_n,
                        mix_out: mix_n,
                        counters,
                        inv_rows,
                    });
                    let cl = if collapse { Some(FastCollapse { x: &x, cur_out: cur_n, norm_out: norm_n, norm_w: &norm_w }) } else { None };
                    arena.launch_fast(s, mx, cl, carry_n, write_carry, HC_DIM, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b)
                };
                if rep < 2 {
                    go(&s, &mut split_n, &mut mix_n, &mut cur_n, &mut norm_n, &mut carry_n, &mut counters, &mut inv_rows)?;
                } else {
                    s.begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    let enq = go(&s, &mut split_n, &mut mix_n, &mut cur_n, &mut norm_n, &mut carry_n, &mut counters, &mut inv_rows);
                    let g = s.end_capture()?;
                    enq?;
                    let exec = g.instantiate()?;
                    exec.launch(&s)?;
                    s.synchronize()?;
                    // Replay again from the same starting carry.
                    carry_n.copy_from_host(&carry0)?;
                    exec.launch(&s)?;
                }
                s.synchronize()?;
                let tag = format!("{arch} b={b} {name} rep={rep}");
                if mode.is_some() {
                    same(&format!("mix {tag}"), &down(&mix_o, bu * m)?, &down(&mix_n, bu * m)?)?;
                    same(&format!("split {tag}"), &down(&split_o, bu * m)?, &down(&split_n, bu * m)?)?;
                }
                if collapse {
                    same(&format!("cur {tag}"), &down(&cur_o, bu * ne)?, &down(&cur_n, bu * ne)?)?;
                    same(&format!("norm {tag}"), &down(&norm_o, bu * ne)?, &down(&norm_n, bu * ne)?)?;
                }
                same(&format!("carry {tag}"), &down(&carry_o, bmax * m)?, &down(&carry_n, bmax * m)?)?;
                let mut c = vec![0u32; bmax];
                counters.copy_to_host(&mut c)?;
                if c.iter().any(|&v| v != 0) {
                    return Err(eyre!("counters not reset after {tag}: {c:?}"));
                }
            }
        }
        eprintln!("{arch} b={b}: launch_fast (5 production cases, x3 incl. graph replay) bit-identical");
    }
    Ok(())
}

#[test]
#[ignore]
fn mhc_fast_is_bit_identical() -> eyre::Result<()> {
    let _gpu = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    install_panic_handler()?;
    let mut ran = 0;
    for d in Device::all()? {
        let arch = d.properties()?.gcn_arch_name;
        if arch.starts_with("gfx1201") || arch.starts_with("gfx1151") {
            fast_run_on(d)?;
            ran += 1;
        }
    }
    if ran == 0 {
        return Err(eyre!("no gfx1201/gfx1151 device"));
    }
    Ok(())
}
