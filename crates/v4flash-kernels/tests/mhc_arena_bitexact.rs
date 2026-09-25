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
use v4flash_kernels::mhc_arena::{MIX_NORMED, MIX_PRE_SCALED};
use v4flash_kernels::{F16Matvec, HcSinkhorn, MhcArena, RmsNormNoWeight, RmsNormNoWeightMultiWG};

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
