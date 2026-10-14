//! Bit-exactness of the decode layer BOUNDARY in one launch (`V41_DGPU_G6_BOUNDARY`, fusion group
//! G6, src/g6_boundary.rs) against the production chain it replaces (forward_prefill.rs
//! `forward_layer_post_moe_v2` under V41_DGPU_COMBINE3 + V41_DEC_FUSE + V41_REMOTE_PARTIAL_ASYNC, then
//! the next layer's `pre_moe_chain` mhc_pre_attn under V41_MHC_FAST):
//!
//!   remote_ffn_moe <- pinned partial (async H2D)                  [only with a box-2 partial]
//!   hc_post.launch_from_split_batched_add2(residual_next; moe, shared, remote, after_attn_hc, split)
//!     (or `_add` with the shared expert alone when the lane-layer has no box-2 ticket)
//!   mhc_arena.launch_fast(mix PRE_SCALED on residual_next + collapse on residual_next, carry)
//!
//! vs `G6Boundary::launch` (the partial read zero-copy from the same pinned staging). Compared bit for
//! bit, WHOLE buffers (rows past b included): residual_next, split, carry, cur, attn_input_norm, mix,
//! inv_rows; the shared mHC counters and the G6 sync words must be zero after every launch, the
//! sticky error word must stay 0. Rows b = 1..16, with and without a box-2 partial, write_carry on
//! and off, TWO lanes' buffer sets interleaved on one stream with no host syncs (production shares
//! the mHC counters / mix / inv_rows scratch between lanes, so does this test), two passes per case
//! (the second runs on the first's carry / split). Adversarial values: exact zeros, -0, spikes, tiny
//! magnitudes. Plus a negative control (a 1-element change of the partial must show) and a mixed
//! back-to-back sequence (b changes every launch, no syncs).
//!
//! Run: `cargo test --release --features v41 -p v4flash-kernels --test g6_boundary_bitexact --
//! --ignored --test-threads=1 --nocapture` (HIP_VISIBLE_DEVICES=0 = the dGPU on box 1).
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, PinnedBuffer, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, N_EMBD, N_HC, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS};
use v4flash_kernels::g6_boundary::{G6Combine, G6Mhc, G6Remote, G6_MAX_B};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mhc_arena::{FastCollapse, FastMix, MIX_PRE_SCALED};

const BMAX: usize = G6_MAX_B as usize;

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
    /// Mostly uniform, with exact zeros, negative zeros, spikes and tiny values.
    fn val(&mut self, a: f32) -> f32 {
        match self.next() % 211 {
            0 => 0.0,
            1 => -0.0,
            2 => 300.0 * a,
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
    let mut h = ((e as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// The per-lane inputs (host copies, so the ref and the new set start identical).
struct Inputs {
    moe: Vec<f32>,
    shared: Vec<f32>,
    remote: Vec<f32>,
    after_attn: Vec<f32>,
    split: Vec<f32>,
    carry: Vec<f32>,
}

fn inputs(rng: &mut Lcg) -> Inputs {
    let (ne, k, m) = (N_EMBD as usize, HC_DIM as usize, HC_MIX_DIM as usize);
    let mut split = vec![0f32; BMAX * m];
    let mut carry = vec![0f32; BMAX * m];
    for r in 0..BMAX {
        for i in 0..m {
            // pre in (0, 1], post in (0, 2), comb ~ doubly stochastic-ish
            split[r * m + i] = if i < 4 { 0.1 + 0.9 * (rng.next() % 1000) as f32 / 1000.0 } else if i < 8 { 2.0 * (rng.next() % 1000) as f32 / 1000.0 } else { 0.05 + 0.4 * (rng.next() % 1000) as f32 / 1000.0 };
            carry[r * m + i] = if i < 4 { 0.05 + 0.5 * (rng.next() % 1000) as f32 / 1000.0 } else { rng.sym(1.0) };
        }
    }
    Inputs {
        moe: (0..BMAX * ne).map(|_| rng.val(0.2)).collect(),
        shared: (0..BMAX * ne).map(|_| rng.val(0.2)).collect(),
        remote: (0..BMAX * ne).map(|_| rng.val(0.2)).collect(),
        after_attn: (0..BMAX * k).map(|_| rng.val(1.5)).collect(),
        split,
        carry,
    }
}

/// One lane's device buffers (inputs + outputs). `remote_pin` is the pinned staging.
struct Lane {
    moe: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    remote_dev: DeviceBuffer<f32>,
    remote_pin: PinnedBuffer<f32>,
    after_attn: DeviceBuffer<f32>,
    split: DeviceBuffer<f32>,
    carry: DeviceBuffer<f32>,
    resid: DeviceBuffer<f32>,
    cur: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
}

impl Lane {
    fn new(id: i32, x: &Inputs) -> eyre::Result<Self> {
        let (ne, k) = (N_EMBD as usize, HC_DIM as usize);
        let mut remote_pin = PinnedBuffer::<f32>::new(BMAX * ne)?;
        remote_pin.as_mut_slice().copy_from_slice(&x.remote);
        // outputs start from a recognisable non-zero pattern (rows past b must stay untouched)
        let fill = |n: usize, v: f32| -> eyre::Result<DeviceBuffer<f32>> { upload(id, &vec![v; n]) };
        Ok(Self {
            moe: upload(id, &x.moe)?,
            shared: upload(id, &x.shared)?,
            remote_dev: fill(BMAX * ne, 7.0)?,
            remote_pin,
            after_attn: upload(id, &x.after_attn)?,
            split: upload(id, &x.split)?,
            carry: upload(id, &x.carry)?,
            resid: fill(BMAX * k, 3.25)?,
            cur: fill(BMAX * ne, -1.5)?,
            norm: fill(BMAX * ne, 9.5)?,
        })
    }
    fn snap(&self) -> eyre::Result<Vec<(&'static str, Vec<u32>)>> {
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
        Ok(vec![
            ("residual_next", bits(download(&self.resid)?)),
            ("split", bits(download(&self.split)?)),
            ("carry", bits(download(&self.carry)?)),
            ("cur", bits(download(&self.cur)?)),
            ("attn_input_norm", bits(download(&self.norm)?)),
        ])
    }
}

/// The mHC scratch production shares between lanes (BatchDgpuShared).
struct Shared {
    mix: DeviceBuffer<f32>,
    counters: DeviceBuffer<u32>,
    inv_rows: DeviceBuffer<f32>,
}

struct Weights {
    w: DeviceBuffer<u8>,
    scale: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
    norm_w: DeviceBuffer<f32>,
}

fn weights(id: i32, rng: &mut Lcg) -> eyre::Result<Weights> {
    let (k, m, ne) = (HC_DIM as usize, HC_MIX_DIM as usize, N_EMBD as usize);
    let wh: Vec<u16> = (0..m * k).map(|_| f16_bits(rng.sym(0.05))).collect();
    let mut wb = vec![0u8; m * k * 2];
    for (i, h) in wh.iter().enumerate() {
        wb[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
    }
    Ok(Weights {
        w: upload(id, &wb)?,
        scale: upload(id, &[0.7f32, 0.45, 1.3])?,
        base: upload(id, &(0..m).map(|_| rng.sym(0.6)).collect::<Vec<f32>>())?,
        norm_w: upload(id, &(0..ne).map(|_| 0.5 + rng.sym(0.4)).collect::<Vec<f32>>())?,
    })
}

/// The production chain for one lane.
fn run_chain(e: &DeviceEngine, s: &Stream, l: &mut Lane, sh: &mut Shared, w: &Weights, b: u32, remote: bool, write_carry: bool) -> eyre::Result<()> {
    let n = b as usize * N_EMBD as usize;
    if remote {
        let src = &l.remote_pin.as_slice()[..n];
        l.remote_dev.slice_view_mut(0, n).copy_from_host_async(src, s)?;
        e.hc_post.launch_from_split_batched_add2(s, &mut l.resid, &l.moe, &l.shared, &l.remote_dev, &l.after_attn, &l.split, N_HC, N_EMBD, N_HC, b)?;
    } else {
        e.hc_post.launch_from_split_batched_add(s, &mut l.resid, &l.moe, &l.shared, &l.after_attn, &l.split, N_HC, N_EMBD, N_HC, b)?;
    }
    e.mhc_arena.launch_fast(
        s,
        Some(FastMix {
            weight: &w.w,
            x: &l.resid,
            scale: &w.scale,
            base: &w.base,
            mode: MIX_PRE_SCALED,
            split_out: &mut l.split,
            mix_out: &mut sh.mix,
            counters: &mut sh.counters,
            inv_rows: &mut sh.inv_rows,
        }),
        Some(FastCollapse { x: &l.resid, cur_out: &mut l.cur, norm_out: &mut l.norm, norm_w: &w.norm_w }),
        &mut l.carry,
        write_carry,
        HC_DIM,
        RMS_EPS,
        SINKHORN_ITERS,
        SINKHORN_EPS,
        b,
    )
}

fn run_g6(e: &DeviceEngine, s: &Stream, l: &mut Lane, sh: &mut Shared, w: &Weights, b: u32, remote: bool, write_carry: bool) -> eyre::Result<()> {
    run_g6m(e, s, l, sh, w, b, if remote { 1 } else { 0 }, write_carry)
}

/// `mode`: 0 no partial, 1 the partial zero-copy from the pinned staging, 2 uploaded first (H2D on
/// `s`, the early side-stream copy's result) and read from the device buffer.
fn run_g6m(e: &DeviceEngine, s: &Stream, l: &mut Lane, sh: &mut Shared, w: &Weights, b: u32, mode: u32, write_carry: bool) -> eyre::Result<()> {
    if mode == 2 {
        let n = b as usize * N_EMBD as usize;
        let src = &l.remote_pin.as_slice()[..n];
        l.remote_dev.slice_view_mut(0, n).copy_from_host_async(src, s)?;
    }
    let remote = match mode {
        1 => Some(G6Remote::Pinned(&l.remote_pin, 0usize)),
        2 => Some(G6Remote::Device(&l.remote_dev)),
        _ => None,
    };
    e.g6.launch(
        s,
        G6Combine {
            moe: &l.moe,
            shared: &l.shared,
            remote,
            after_attn: &l.after_attn,
            resid_out: &mut l.resid,
        },
        G6Mhc {
            weight: &w.w,
            scale: &w.scale,
            base: &w.base,
            split: &mut l.split,
            mix_out: &mut sh.mix,
            counters: &mut sh.counters,
            inv_rows: &mut sh.inv_rows,
            carry: &mut l.carry,
            write_carry,
            cur_out: &mut l.cur,
            norm_out: &mut l.norm,
            norm_w: &w.norm_w,
            rms_eps: RMS_EPS,
            sinkhorn_iters: SINKHORN_ITERS,
            sinkhorn_eps: SINKHORN_EPS,
        },
        b,
    )
}

fn shared(id: i32) -> eyre::Result<Shared> {
    let m = HC_MIX_DIM as usize;
    Ok(Shared { mix: upload(id, &vec![0.5f32; BMAX * m])?, counters: upload(id, &vec![0u32; BMAX])?, inv_rows: upload(id, &vec![0.25f32; BMAX])? })
}

fn diff(tag: &str, a: &[(&'static str, Vec<u32>)], b: &[(&'static str, Vec<u32>)]) -> Vec<String> {
    let mut out = Vec::new();
    for ((na, va), (_, vb)) in a.iter().zip(b) {
        let nd = va.iter().zip(vb).filter(|(x, y)| x != y).count();
        if nd > 0 {
            let first = va.iter().zip(vb).position(|(x, y)| x != y).unwrap();
            out.push(format!(
                "{tag}: {na} {nd} diffs, first at {first}: ref {:e} ({:#x}) new {:e} ({:#x})",
                f32::from_bits(va[first]), va[first], f32::from_bits(vb[first]), vb[first]
            ));
        }
    }
    out
}

#[test]
#[ignore]
fn g6_boundary_is_bit_identical_to_the_chain() -> eyre::Result<()> {
    install_panic_handler()?;
    let (dev, arch) = dgpu()?;
    dev.set_current()?;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let id = e.device.id;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x96_b0_0d_a7);
    let w = weights(id, &mut rng)?;
    let mut fails: Vec<String> = Vec::new();
    let mut n_cmp = 0usize;
    for b in 1..=G6_MAX_B {
        let b2 = (G6_MAX_B + 1 - b).clamp(1, G6_MAX_B);
        for (mode, write_carry) in [(1u32, true), (0, true), (2, true), (1, false)] {
            let remote = mode != 0;
            let (xa, xb) = (inputs(&mut rng), inputs(&mut rng));
            let (mut ra, mut rb) = (Lane::new(id, &xa)?, Lane::new(id, &xb)?);
            let (mut na, mut nb) = (Lane::new(id, &xa)?, Lane::new(id, &xb)?);
            let (mut sr, mut sn) = (shared(id)?, shared(id)?);
            for _pass in 0..2 {
                // lanes interleaved on one stream, no host syncs (lane B: no partial on odd b)
                let rem_b = remote && b % 2 == 0;
                run_chain(&e, &s, &mut ra, &mut sr, &w, b, remote, write_carry)?;
                run_chain(&e, &s, &mut rb, &mut sr, &w, b2, rem_b, write_carry)?;
                run_g6m(&e, &s, &mut na, &mut sn, &w, b, mode, write_carry)?;
                run_g6m(&e, &s, &mut nb, &mut sn, &w, b2, if rem_b { mode } else { 0 }, write_carry)?;
            }
            s.synchronize()?;
            let tag = format!("b={b} (lane B {b2}) remote mode {mode} write_carry={write_carry}");
            let mut f = diff(&format!("{tag} lane A"), &ra.snap()?, &na.snap()?);
            f.extend(diff(&format!("{tag} lane B"), &rb.snap()?, &nb.snap()?));
            // the shared scratch: last writer is lane B in both
            let sb = |x: &Shared| -> eyre::Result<Vec<(&'static str, Vec<u32>)>> {
                Ok(vec![
                    ("mix", download(&x.mix)?.into_iter().map(f32::to_bits).collect()),
                    ("inv_rows", download(&x.inv_rows)?.into_iter().map(f32::to_bits).collect()),
                    ("counters", download(&x.counters)?),
                ])
            };
            f.extend(diff(&format!("{tag} shared"), &sb(&sr)?, &sb(&sn)?));
            if download(&sn.counters)?.iter().any(|&c| c != 0) {
                f.push(format!("{tag}: mHC counters not zero after the launches"));
            }
            let sw = e.g6.sync_words()?;
            let err = e.g6.error_word()?;
            if err != 0 || sw.iter().enumerate().any(|(i, &v)| v != 0 && i != sw.len() - 32) {
                f.push(format!("{tag}: g6 sync words not reset (err {err})"));
            }
            n_cmp += 2;
            if !f.is_empty() {
                eprintln!("FAIL {}", f.join("\n     "));
            }
            fails.extend(f);
        }
    }
    eprintln!("G6: {n_cmp} lane comparisons (x2 passes), {} failures", fails.len());

    // ---- negative control: a 1-element change of the partial must show in residual_next
    {
        let b = 4u32;
        let x = inputs(&mut rng);
        let (mut r, mut n) = (Lane::new(id, &x)?, Lane::new(id, &x)?);
        let (mut sr, mut sn) = (shared(id)?, shared(id)?);
        n.remote_pin.as_mut_slice()[N_EMBD as usize + 17] += 0.03125;
        run_chain(&e, &s, &mut r, &mut sr, &w, b, true, true)?;
        run_g6(&e, &s, &mut n, &mut sn, &w, b, true, true)?;
        s.synchronize()?;
        let d = diff("negative control", &r.snap()?, &n.snap()?);
        eprintln!("negative control: {} buffers differ (expect >= 1): {:?}", d.len(), d.iter().map(|x| x.split(':').nth(1).unwrap_or("").to_string()).collect::<Vec<_>>());
        if d.is_empty() {
            fails.push("negative control: a changed partial did not change any output".into());
        }
    }

    // ---- mixed back-to-back sequence: b and the partial change every launch, no host syncs
    {
        let x = inputs(&mut rng);
        let (mut r, mut n) = (Lane::new(id, &x)?, Lane::new(id, &x)?);
        let (mut sr, mut sn) = (shared(id)?, shared(id)?);
        for i in 0..200u32 {
            let b = 1 + (rng.next() % G6_MAX_B);
            let rem = i % 3 != 0;
            run_chain(&e, &s, &mut r, &mut sr, &w, b, rem, true)?;
            run_g6(&e, &s, &mut n, &mut sn, &w, b, rem, true)?;
            if i % 50 == 49 {
                s.synchronize()?;
                let d = diff(&format!("mixed seq @ {i}"), &r.snap()?, &n.snap()?);
                if !d.is_empty() {
                    eprintln!("FAIL {}", d.join("\n     "));
                }
                fails.extend(d);
            }
            // the residual chains forward: the next combine reads after_attn = this residual
            // (keeps values bounded: the same input buffers are reused otherwise)
        }
        s.synchronize()?;
        if e.g6.error_word()? != 0 {
            fails.push("mixed seq: g6 error word set".into());
        }
        eprintln!("mixed sequence: 200 launches each, compared every 50");
    }
    if !fails.is_empty() {
        return Err(eyre!("{} failures", fails.len()));
    }
    Ok(())
}
