//! `attention_swa_batched` with a NON-TRAILING raw window — the Vision-Exp
//! image-row case, where a row inside an `[IMAGE_START .. IMAGE_END]` span
//! attends to keys AHEAD of itself (`het::image_spans::raw_window`).
//!
//! The kernel has always taken an arbitrary `(n_raw_offset_per[b],
//! n_raw_per[b])` slice of the cache, but production only ever handed it
//! trailing causal windows of at most `SWA_WINDOW` (128) keys. This test
//! pins the two things vision changes:
//!   1. windows that extend PAST the row (offset + count > b's own slot),
//!   2. windows WIDER than 128, up to `ATTN_SWA_BATCHED_MAX_KV` (512),
//!      which is what the kernel's dynamic-LDS `max_n_kv` argument sizes.
//! and checks a text-shaped launch (`max_n_kv = 128`) is still exact.
//!
//! Tiny + synthetic: no model, no dump, ~0.5 MB of device buffers, run on
//! the iGPU (gfx1151) when present. Freed on drop at end of test.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!         --test attention_swa_visible_window -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::SWA_WINDOW;
use v4flash_kernels::het::image_spans;
use v4flash_kernels::{AttentionSwa, ATTN_SWA_BATCHED_MAX_KV};

const N_HEAD: u32 = 8;
const HEAD_DIM: u32 = 64;
const CACHE_ROWS: usize = 1024;

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    exp = exp - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x800000) >> (1 - exp);
        return sign | (((m + 0x1000 + ((m >> 13) & 1)) >> 13) as u16);
    }
    let rm = (mant + 0x1000 + ((mant >> 13) & 1)) >> 13;
    if rm & 0x400 != 0 {
        return sign | ((exp as u16 + 1) << 10);
    }
    sign | ((exp as u16) << 10) | rm as u16
}

fn f16_bits_to_f32(b: u16) -> f32 {
    let sign = ((b as u32) & 0x8000) << 16;
    let exp = ((b >> 10) & 0x1f) as u32;
    let mant = ((b as u32) & 0x3ff) << 13;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x800000 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | (m & 0x7fffff)
        }
    } else if exp == 0x1f {
        sign | 0x7f800000 | mant
    } else {
        sign | ((exp + 127 - 15) << 23) | mant
    };
    f32::from_bits(bits)
}

fn pick_device() -> eyre::Result<Device> {
    let devices = Device::all()?;
    for d in &devices {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(*d);
        }
    }
    devices.first().copied().ok_or_else(|| eyre!("no HIP devices"))
}

/// Deterministic pseudo-random in [-1, 1).
fn rnd(seed: u64) -> f32 {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    ((x >> 40) as f32 / 8388608.0) - 1.0
}

/// Host reference for one (row, head): softmax over the EXACT slot slice
/// `[off, off+cnt)` with the attention sink, weights applied to the same
/// f16 cache values the kernel reads.
fn reference(
    q: &[f32],
    kv16: &[u16],
    sink: f32,
    off: usize,
    cnt: usize,
    kq_scale: f32,
) -> Vec<f32> {
    let mut scores = Vec::with_capacity(cnt);
    for r in 0..cnt {
        let row = &kv16[(off + r) * HEAD_DIM as usize..(off + r + 1) * HEAD_DIM as usize];
        // Same accumulation order the kernel's tree reduction ends at:
        // f32 sum over head_dim; tolerance below covers the reassociation.
        let mut s = 0.0f32;
        for i in 0..HEAD_DIM as usize {
            s += q[i] * f16_bits_to_f32(row[i]);
        }
        scores.push(s * kq_scale);
    }
    let mut max_score = sink;
    for &s in &scores {
        if s > max_score {
            max_score = s;
        }
    }
    let mut denom = (sink - max_score).exp();
    let mut w = Vec::with_capacity(cnt);
    for &s in &scores {
        let e = (s - max_score).exp();
        w.push(e);
        denom += e;
    }
    let inv = 1.0 / denom;
    let mut out = vec![0.0f32; HEAD_DIM as usize];
    for (r, &wr) in w.iter().enumerate() {
        let row = &kv16[(off + r) * HEAD_DIM as usize..(off + r + 1) * HEAD_DIM as usize];
        for d in 0..HEAD_DIM as usize {
            out[d] += wr * f16_bits_to_f32(row[d]);
        }
    }
    for v in out.iter_mut() {
        *v *= inv;
    }
    out
}

/// One launch: `windows[b] = (offset, count)`.
#[allow(clippy::too_many_arguments)]
fn run_case(
    label: &str,
    dev: Device,
    swa: &AttentionSwa,
    stream: &Stream,
    kv16: &[u16],
    windows: &[(u32, u32)],
    max_n_kv: u32,
) -> eyre::Result<f32> {
    let b = windows.len();
    let kq_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let q_host: Vec<f32> = (0..b * N_HEAD as usize * HEAD_DIM as usize)
        .map(|i| rnd(0x9e37 + i as u64) * 0.5)
        .collect();
    let sinks_host: Vec<f32> = (0..N_HEAD as usize).map(|h| rnd(0x5151 + h as u64)).collect();
    let nrp: Vec<i32> = windows.iter().map(|w| w.1 as i32).collect();
    let nrop: Vec<i32> = windows.iter().map(|w| w.0 as i32).collect();

    dev.set_current()?;
    let mut d_kv: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, kv16.len())?;
    d_kv.copy_from_host(kv16)?;
    let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, q_host.len())?;
    d_q.copy_from_host(&q_host)?;
    let mut d_sinks: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, sinks_host.len())?;
    d_sinks.copy_from_host(&sinks_host)?;
    let mut d_nrp: DeviceBuffer<i32> = DeviceBuffer::new(dev.id, b)?;
    d_nrp.copy_from_host(&nrp)?;
    let mut d_nrop: DeviceBuffer<i32> = DeviceBuffer::new(dev.id, b)?;
    d_nrop.copy_from_host(&nrop)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, b * N_HEAD as usize * HEAD_DIM as usize)?;

    swa.launch_batched(
        stream, &mut d_out, &d_q, &d_kv, &d_sinks, &d_nrp, &d_nrop, N_HEAD, HEAD_DIM, b as u32,
        max_n_kv,
    )?;
    stream.synchronize()?;
    let mut got = vec![0.0f32; b * N_HEAD as usize * HEAD_DIM as usize];
    d_out.copy_to_host(&mut got)?;

    let mut max_abs = 0.0f32;
    let hd = HEAD_DIM as usize;
    for (bi, &(off, cnt)) in windows.iter().enumerate() {
        for h in 0..N_HEAD as usize {
            let base = (bi * N_HEAD as usize + h) * hd;
            let want = reference(
                &q_host[base..base + hd],
                kv16,
                sinks_host[h],
                off as usize,
                cnt as usize,
                kq_scale,
            );
            for d in 0..hd {
                let diff = (got[base + d] - want[d]).abs();
                if diff > max_abs {
                    max_abs = diff;
                }
            }
        }
    }
    println!("  {label}: B={b} max_n_kv={max_n_kv} max_abs={max_abs:.3e}");
    Ok(max_abs)
}

#[test]
#[ignore]
fn attention_swa_batched_non_trailing_window() -> eyre::Result<()> {
    let _ = install_panic_handler();
    let dev = pick_device()?;
    println!(
        "device {} ({})",
        dev.id,
        dev.properties()?.gcn_arch_name.trim_end_matches('\0')
    );
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let swa = AttentionSwa::for_arch(arch.trim_end_matches('\0'))?;
    let stream = Stream::new(dev.id)?;

    // Synthetic cache: every slot distinguishable, so a window that is off
    // by one row cannot pass.
    let kv16: Vec<u16> = (0..CACHE_ROWS * HEAD_DIM as usize)
        .map(|i| {
            let r = i / HEAD_DIM as usize;
            let d = i % HEAD_DIM as usize;
            f32_to_f16_bits(rnd(0xabc0 + i as u64) * 0.4 + (r as f32) * 1e-3 + (d as f32) * 1e-4)
        })
        .collect();
    const TOL: f32 = 2.0e-3;

    // --- Case 1: the historical text window. max_n_kv = SWA_WINDOW, so the
    // kernel's dynamic LDS is exactly the 1 KiB the old static arrays used.
    let n_raw_before = SWA_WINDOW;
    let text: Vec<(u32, u32)> = (0..256u32)
        .map(|i| image_spans::raw_window(n_raw_before, i, 0, 0))
        .collect();
    for &(_off, cnt) in &text {
        assert!(cnt <= SWA_WINDOW);
    }
    let e = run_case("text causal (trailing, <=128)", dev, &swa, &stream, &kv16, &text, SWA_WINDOW)?;
    assert!(e < TOL, "text window max_abs {e} >= {TOL}");

    // --- Case 2: a real Vision-Exp image block. 366-token span starting at
    // row 40 of the chunk; every row inside it looks BOTH ways.
    let (span_off, span_len) = (40u32, 366u32);
    let b = 512usize;
    let vis = image_spans::rows_visibility(0, b, &[(span_off, span_len)])?;
    let img: Vec<(u32, u32)> = (0..b)
        .map(|i| image_spans::raw_window(n_raw_before, i as u32, vis[i].0, vis[i].1))
        .collect();
    // Sanity: the span's rows really are non-trailing and really are wider
    // than the old cap somewhere.
    let mut n_forward = 0usize;
    let mut widest = 0u32;
    for (i, &(off, cnt)) in img.iter().enumerate() {
        let self_slot = n_raw_before + i as u32;
        if off + cnt > self_slot + 1 {
            n_forward += 1;
        }
        widest = widest.max(cnt);
        assert!(
            (off + cnt) as usize <= n_raw_before as usize + b,
            "row {i} window past appended K/V"
        );
    }
    println!("  span rows looking forward: {n_forward}, widest window: {widest}");
    assert!(n_forward >= 300, "expected the span's rows to look forward, got {n_forward}");
    assert!(widest > SWA_WINDOW, "expected a window wider than {SWA_WINDOW}, got {widest}");
    assert!(widest <= ATTN_SWA_BATCHED_MAX_KV);
    let e = run_case("image span 366 (non-trailing)", dev, &swa, &stream, &kv16, &img, widest)?;
    assert!(e < TOL, "image window max_abs {e} >= {TOL}");

    // --- Case 3: the widest window the cap allows (512 keys), forced.
    let wide: Vec<(u32, u32)> = (0..64u32)
        .map(|i| (i * 4, ATTN_SWA_BATCHED_MAX_KV))
        .collect();
    let e = run_case("max cap (512 keys)", dev, &swa, &stream, &kv16, &wide, ATTN_SWA_BATCHED_MAX_KV)?;
    assert!(e < TOL, "512-key window max_abs {e} >= {TOL}");

    // --- Case 4: the launcher rejects an out-of-range LDS stride instead of
    // corrupting LDS.
    let mut d_dummy: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, 8)?;
    let d_q: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, 8)?;
    let d_kv2: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, 8)?;
    let d_s: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, 8)?;
    let d_i: DeviceBuffer<i32> = DeviceBuffer::new(dev.id, 8)?;
    for bad in [0u32, ATTN_SWA_BATCHED_MAX_KV + 1] {
        let r = swa.launch_batched(
            &stream, &mut d_dummy, &d_q, &d_kv2, &d_s, &d_i, &d_i, 1, 8, 1, bad,
        );
        assert!(r.is_err(), "max_n_kv={bad} should be rejected");
    }
    stream.synchronize()?;
    println!("attention_swa_batched non-trailing window: OK");
    Ok(())
}
