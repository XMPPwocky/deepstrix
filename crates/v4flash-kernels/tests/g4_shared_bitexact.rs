//! Bit-exactness of fusion group G4 (`V41_DGPU_G4_SHARED`, src/g4_shared.rs) against the production
//! decode chain it replaces (forward_prefill.rs `issue_shared_expert_prefill`, direct dp4a arm, +
//! the `mhc_mix_ffn_late` launch under V41_MHC_FAST):
//!
//!   q8.quantize_input_batched(ffn_input_norm) -> xq, xscale
//!   rows 1..5: shared_fused.launch (gate/up/swiglu/quant); 6..8: gate, up, swiglu, quantize mid
//!   q8.matvec_batched(down) -> ffn_shared; mhc_arena.launch_fast(mix NORMED on after_attn_hc,
//!   carry := split)
//!
//! vs `G4Shared::launch_qmix` (the quantize + the mixes, ticketed RMS) followed by the same GEMVs.
//! Compared bit for bit, WHOLE buffers: xq / xscale, mid_xq, mid_xs, ffn_shared, split, mix, carry,
//! the mHC counters back at zero, the G4 sync words reset and its sticky error word 0. Rows 1..16 (the
//! GEMVs only to 8, the dp4a arm's range), TWO lanes interleaved on one stream with no host syncs (the mix scratch shared as in production), two passes per case,
//! adversarial activations (exact zeros, -0, all-zero 32-blocks, spikes, tiny values), synthetic Q8_0
//! weights with extreme block scales. Negative control: one changed input value must show.
//!
//! Run: `cargo test --release --features v41 -p v4flash-kernels --test g4_shared_bitexact --
//! --ignored --test-threads=1 --nocapture` (HIP_VISIBLE_DEVICES=0 = the dGPU on box 1). ~60 MB VRAM.
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, N_EMBD, N_FF_SHARED, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS, SWIGLU_CLAMP_EXP};
use v4flash_kernels::g4_shared::{G4Mix, G4Quant, G4_MAX_B};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mhc_arena::{FastMix, MIX_NORMED};

const BMAX: usize = G4_MAX_B as usize;
const GEMV_MAX_B: u32 = 8;
const FUSED_MAX_B: u32 = 5;

fn dgpu() -> eyre::Result<(Device, String)> {
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201 visible (set HIP_VISIBLE_DEVICES to the dGPU)"))?;
    let arch = dev.properties()?.gcn_arch_name;
    Ok((dev, arch))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn sym(&mut self, a: f32) -> f32 {
        ((self.next() & 0xFFFFFF) as f32 / 16777216.0 * 2.0 - 1.0) * a
    }
    fn val(&mut self, a: f32) -> f32 {
        match self.next() % 257 {
            0 => 0.0,
            1 => -0.0,
            2 => 50.0 * a,
            3 => 1e-30 * a,
            _ => self.sym(a),
        }
    }
}
fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len())?;
    d.copy_from_host(host)?;
    Ok(d)
}
fn download<T: Copy + Default>(d: &DeviceBuffer<T>) -> eyre::Result<Vec<T>> {
    let mut v = vec![T::default(); d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if e <= 0 {
        return sign;
    }
    sign | (((e as u32) << 10) | (m >> 13)) as u16
}
/// Q8_0 weight [rows, k] in the repacked layout: per row [scales f16 x blocks | quants 32 x blocks].
fn q8_weight(id: i32, rng: &mut Lcg, rows: usize, k: usize) -> eyre::Result<DeviceBuffer<u8>> {
    let blocks = k / 32;
    let mut h = vec![0u8; rows * blocks * 34];
    for r in 0..rows {
        let row = &mut h[r * blocks * 34..(r + 1) * blocks * 34];
        for bl in 0..blocks {
            let s = match rng.next() % 97 {
                0 => 0.0,
                1 => 3.0,
                _ => 0.002 + 0.01 * (rng.next() % 100) as f32 / 100.0,
            };
            row[2 * bl..2 * bl + 2].copy_from_slice(&f16_bits(s).to_le_bytes());
        }
        for q in row[blocks * 2..].iter_mut() {
            *q = match rng.next() % 61 {
                0 => 0x80,
                1 => 0x7f,
                _ => (rng.next() & 0xff) as u8,
            };
        }
    }
    upload(id, &h)
}

struct Lane {
    ffn_in: DeviceBuffer<f32>,
    after_attn: DeviceBuffer<f32>,
    split: DeviceBuffer<f32>,
    carry: DeviceBuffer<f32>,
    ffn_shared: DeviceBuffer<f32>,
}
struct Shared {
    xq: DeviceBuffer<i8>,
    xs: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    mid: DeviceBuffer<f32>,
    mid_xq: DeviceBuffer<i8>,
    mid_xs: DeviceBuffer<f32>,
    mix: DeviceBuffer<f32>,
    counters: DeviceBuffer<u32>,
    inv_rows: DeviceBuffer<f32>,
}
struct W {
    wg: DeviceBuffer<u8>,
    wu: DeviceBuffer<u8>,
    wd: DeviceBuffer<u8>,
    hc: DeviceBuffer<u8>,
    scale: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
}

fn lane(id: i32, seed: u64) -> eyre::Result<Lane> {
    let mut rng = Lcg(seed);
    let (ne, k, m) = (N_EMBD as usize, HC_DIM as usize, HC_MIX_DIM as usize);
    let mut ffn_in: Vec<f32> = (0..BMAX * ne).map(|_| rng.val(2.0)).collect();
    // an all-zero 32-block and an all -0 block (quantize: d = 0 -> id = 0)
    ffn_in[64..96].fill(0.0);
    ffn_in[ne + 160..ne + 192].fill(-0.0);
    let carry: Vec<f32> = (0..BMAX * m).map(|i| if i % m < 4 { 0.25 } else { 0.1 }).collect();
    Ok(Lane {
        ffn_in: upload(id, &ffn_in)?,
        after_attn: upload(id, &(0..BMAX * k).map(|_| rng.val(1.0)).collect::<Vec<_>>())?,
        split: upload(id, &carry)?,
        carry: upload(id, &carry)?,
        ffn_shared: upload(id, &vec![3.5f32; BMAX * ne])?,
    })
}
fn shared(id: i32) -> eyre::Result<Shared> {
    let (ne, nf, m) = (N_EMBD as usize, N_FF_SHARED as usize, HC_MIX_DIM as usize);
    Ok(Shared {
        xq: upload(id, &vec![5i8; BMAX * ne])?,
        xs: upload(id, &vec![0.5f32; BMAX * ne / 32])?,
        gate: upload(id, &vec![0f32; BMAX * nf])?,
        up: upload(id, &vec![0f32; BMAX * nf])?,
        mid: upload(id, &vec![0f32; BMAX * nf])?,
        mid_xq: upload(id, &vec![0i8; BMAX * nf])?,
        mid_xs: upload(id, &vec![0f32; BMAX * nf / 32])?,
        mix: upload(id, &vec![0f32; BMAX * m])?,
        counters: upload(id, &vec![0u32; BMAX])?,
        inv_rows: upload(id, &vec![0f32; BMAX])?,
    })
}

/// The shared expert's GEMVs after its input quantize (production's dp4a arm).
fn gemvs(e: &DeviceEngine, s: &Stream, l: &mut Lane, sh: &mut Shared, w: &W, b: u32) -> eyre::Result<()> {
    first_half(e, s, sh, w, b)?;
    e.q8.matvec_batched(s, &mut l.ffn_shared, &w.wd, &sh.mid_xq, &sh.mid_xs, N_EMBD, N_FF_SHARED, b)
}

fn first_half(e: &DeviceEngine, s: &Stream, sh: &mut Shared, w: &W, b: u32) -> eyre::Result<()> {
    if b <= FUSED_MAX_B {
        e.shared_fused.launch(s, &mut sh.mid_xq, &mut sh.mid_xs, &w.wg, &w.wu, &sh.xq, &sh.xs, N_EMBD, N_FF_SHARED, b, SWIGLU_CLAMP_EXP)
    } else {
        e.q8.matvec_batched(s, &mut sh.gate, &w.wg, &sh.xq, &sh.xs, N_FF_SHARED, N_EMBD, b)?;
        e.q8.matvec_batched(s, &mut sh.up, &w.wu, &sh.xq, &sh.xs, N_FF_SHARED, N_EMBD, b)?;
        e.swiglu.launch_clamped(s, &mut sh.mid, &sh.gate, &sh.up, b * N_FF_SHARED, SWIGLU_CLAMP_EXP)?;
        e.q8.quantize_input_batched(s, &mut sh.mid_xq, &mut sh.mid_xs, &sh.mid, N_FF_SHARED, b)
    }
}

fn run_chain(e: &DeviceEngine, s: &Stream, l: &mut Lane, sh: &mut Shared, w: &W, b: u32) -> eyre::Result<()> {
    e.q8.quantize_input_batched(s, &mut sh.xq, &mut sh.xs, &l.ffn_in, N_EMBD, b)?;
    if b <= GEMV_MAX_B {
        gemvs(e, s, l, sh, w, b)?;
    }
    e.mhc_arena.launch_fast(
        s,
        Some(FastMix {
            weight: &w.hc,
            x: &l.after_attn,
            scale: &w.scale,
            base: &w.base,
            mode: MIX_NORMED,
            split_out: &mut l.split,
            mix_out: &mut sh.mix,
            counters: &mut sh.counters,
            inv_rows: &mut sh.inv_rows,
        }),
        None,
        &mut l.carry,
        true,
        HC_DIM,
        RMS_EPS,
        SINKHORN_ITERS,
        SINKHORN_EPS,
        b,
    )
}

fn run_g4(e: &DeviceEngine, s: &Stream, l: &mut Lane, sh: &mut Shared, w: &W, b: u32) -> eyre::Result<()> {
    e.g4.launch_qmix(
        s,
        G4Quant { xq: &mut sh.xq, xscale: &mut sh.xs, x: &l.ffn_in, n: N_EMBD },
        G4Mix {
            weight: &w.hc,
            x: &l.after_attn,
            scale: &w.scale,
            base: &w.base,
            split_out: &mut l.split,
            mix_out: &mut sh.mix,
            counters: &mut sh.counters,
            carry: &mut l.carry,
            write_carry: true,
            rms_eps: RMS_EPS,
            sinkhorn_iters: SINKHORN_ITERS,
            sinkhorn_eps: SINKHORN_EPS,
        },
        b,
    )?;
    if b <= GEMV_MAX_B {
        gemvs(e, s, l, sh, w, b)?;
    }
    Ok(())
}

type Snap = Vec<(&'static str, Vec<u8>)>;
fn bytes<T: Copy + Default>(d: &DeviceBuffer<T>) -> eyre::Result<Vec<u8>> {
    let h = download(d)?;
    // SAFETY: plain numeric element types read as bytes.
    Ok(unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * std::mem::size_of::<T>()) }.to_vec())
}
fn snap(l: &Lane, sh: &Shared, with_xq: bool) -> eyre::Result<Snap> {
    let mut v = vec![
        ("ffn_shared", bytes(&l.ffn_shared)?),
        ("split", bytes(&l.split)?),
        ("carry", bytes(&l.carry)?),
        ("mid_xq", bytes(&sh.mid_xq)?),
        ("mid_xs", bytes(&sh.mid_xs)?),
        ("mix", bytes(&sh.mix)?),
        ("counters", bytes(&sh.counters)?),
    ];
    if with_xq {
        v.push(("xq_n_embd", bytes(&sh.xq)?));
        v.push(("xscale_n_embd", bytes(&sh.xs)?));
    }
    Ok(v)
}
fn diff(tag: &str, a: &Snap, b: &Snap) -> Vec<String> {
    a.iter()
        .zip(b)
        .filter(|((_, x), (_, y))| x != y)
        .map(|((n, x), (_, y))| {
            let first = x.iter().zip(y).position(|(p, q)| p != q).unwrap_or(0);
            format!("{tag}: {n} {} bytes differ (first at {first})", x.iter().zip(y).filter(|(p, q)| p != q).count())
        })
        .collect()
}

#[test]
#[ignore]
fn g4_shared_is_bit_identical_to_the_chain() -> eyre::Result<()> {
    install_panic_handler()?;
    let (dev, arch) = dgpu()?;
    dev.set_current()?;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let id = e.device.id;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x6a4_5e11);
    let (ne, nf, k, m) = (N_EMBD as usize, N_FF_SHARED as usize, HC_DIM as usize, HC_MIX_DIM as usize);
    let w = W {
        wg: q8_weight(id, &mut rng, nf, ne)?,
        wu: q8_weight(id, &mut rng, nf, ne)?,
        wd: q8_weight(id, &mut rng, ne, nf)?,
        hc: upload(id, &(0..m * k).flat_map(|_| f16_bits(rng.sym(0.05)).to_le_bytes()).collect::<Vec<u8>>())?,
        scale: upload(id, &[0.7f32, 0.45, 1.3])?,
        base: upload(id, &(0..m).map(|_| rng.sym(0.6)).collect::<Vec<f32>>())?,
    };
    let mut fails: Vec<String> = Vec::new();
    let mut n_cmp = 0usize;
    for b in 1..=G4_MAX_B {
        let b2 = G4_MAX_B + 1 - b;
        let (seed_a, seed_b) = (0x1000 + b as u64, 0x2000 + b as u64);
        let (mut ra, mut rb, mut na, mut nb) = (lane(id, seed_a)?, lane(id, seed_b)?, lane(id, seed_a)?, lane(id, seed_b)?);
        let (mut sr, mut sn) = (shared(id)?, shared(id)?);
        for pass in 0..2 {
            run_chain(&e, &s, &mut ra, &mut sr, &w, b)?;
            let snap_ra = {
                s.synchronize()?;
                snap(&ra, &sr, true)?
            };
            run_g4(&e, &s, &mut na, &mut sn, &w, b)?;
            s.synchronize()?;
            let f = diff(&format!("b={b} lane A pass {pass}"), &snap_ra, &snap(&na, &sn, true)?);
            fails.extend(f);
            // lane B interleaved without host syncs between its two launches and the next lane's
            run_chain(&e, &s, &mut rb, &mut sr, &w, b2)?;
            run_g4(&e, &s, &mut nb, &mut sn, &w, b2)?;
            s.synchronize()?;
            n_cmp += 2;
        }
        let f = diff(&format!("b={b} lane B ({b2} rows)"), &snap(&rb, &sr, true)?, &snap(&nb, &sn, true)?);
        if !f.is_empty() {
            eprintln!("FAIL {}", f.join("\n     "));
        }
        fails.extend(f);
        eprintln!("b={b} (lane B {b2}): compared");
    }
    // negative control: one input value changed in the G4 arm must show
    {
        let b = 4u32;
        let (mut r, mut n) = (lane(id, 77)?, lane(id, 77)?);
        let (mut sr, mut sn) = (shared(id)?, shared(id)?);
        let mut h = download(&n.ffn_in)?;
        h[1000] = h[1000] * 1.5 + 0.25;
        n.ffn_in.copy_from_host(&h)?;
        run_chain(&e, &s, &mut r, &mut sr, &w, b)?;
        run_g4(&e, &s, &mut n, &mut sn, &w, b)?;
        s.synchronize()?;
        let d = diff("negative control", &snap(&r, &sr, true)?, &snap(&n, &sn, true)?);
        eprintln!("negative control (ffn_in[1000] changed, b=4): {} buffers differ", d.len());
        if d.is_empty() {
            return Err(eyre!("negative control not detected"));
        }
    }
    let sw = e.g4.sync_words()?;
    if e.g4.error_word()? != 0 || sw.iter().enumerate().any(|(i, &v)| v != 0 && i != sw.len() - 32) {
        fails.push("g4 sync words not reset / error word set".into());
    }
    eprintln!("G4: {n_cmp} lane comparisons, {} failures", fails.len());
    for f in &fails {
        eprintln!("  {f}");
    }
    if !fails.is_empty() {
        return Err(eyre!("{} failures", fails.len()));
    }
    Ok(())
}
