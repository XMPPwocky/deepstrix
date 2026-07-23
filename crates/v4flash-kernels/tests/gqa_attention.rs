//! Single-query GQA attention unit test — validates the HIP
//! `gqa_attn_single_query` kernel against an independent CPU f32 softmax
//! attention with the SAME grouping.
//!
//! Strategy (self-contained, no GGUF / no activation dump):
//!   1. Build random Q `[n_head, head_dim]` and K/V `[n_kv, n_kv_head, head_dim]`.
//!   2. Round-trip every value through f16 (the caller stores an f16 cache),
//!      so CPU and GPU decode the *same* half-precision inputs — the only
//!      divergence is f32 accumulation order + expf ULPs.
//!   3. CPU reference: per query head h, kv_head = h / (n_head/n_kv_head),
//!      score[j] = dot(q[h], k[j,kv_head]) * scale, stable softmax, weighted
//!      sum of v[j, kv_head]. Compare max-abs error vs the GPU kernel.
//!
//! Tolerance: 2e-3 max-abs. Both sides consume identical f16-decoded values,
//! so the gap is purely f32 reduction order (kernel tree-reduce + online
//! softmax rescale vs. CPU sequential) — empirically ~1e-4. 2e-3 is a
//! comfortable ceiling; a real decode/grouping/softmax bug blows past it by
//! orders of magnitude.
//!
//! NOTE: this test drives the GPU. It is `#[ignore]`-gated and must be run
//! explicitly (and only when the production server is not using the GPUs):
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!       --test gqa_attention -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::iq2_xxs_tables::f16_to_f32;
use v4flash_kernels::GqaAttention;

// ---------------------------------------------------------------------------
// f16 helper (round-to-nearest f32 -> IEEE-754 half bits). Copied from the
// q4_k_dense test; inputs here are modest magnitudes (no inf/subnormal edges).
// ---------------------------------------------------------------------------
fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mant = x & 0x007f_ffff;
    let exp = ((x >> 23) & 0xff) as i32;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    } else if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half_mant = (m >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        let mut result = sign | half_mant;
        if (m & round_bit) != 0 && ((m & (round_bit - 1)) != 0 || (half_mant & 1) != 0) {
            result += 1;
        }
        return result;
    }
    let half_mant = (mant >> 13) as u16;
    let mut result = sign | ((e as u16) << 10) | half_mant;
    if (mant & 0x0000_1000) != 0 && ((mant & 0x0000_0fff) != 0 || (half_mant & 1) != 0) {
        result += 1;
    }
    result
}

/// f32 -> f16 bits -> f32 (the value both CPU and GPU actually see).
fn round_f16(f: f32) -> (u16, f32) {
    let bits = f32_to_f16(f);
    (bits, f16_to_f32(bits))
}

// Deterministic pseudo-random (xorshift*), no external rng dep.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let u = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as u32; // 24 bits
        (u as f32 / (1u32 << 24) as f32) * 2.0 - 1.0 // [-1, 1)
    }
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

/// One test case: build data, run kernel, compare against CPU reference.
fn run_case(
    kernel: &GqaAttention,
    device: &Device,
    stream: &Stream,
    seed: u64,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_kv: usize,
) -> eyre::Result<f32> {
    assert_eq!(n_head % n_kv_head, 0, "n_head must divide n_kv_head");
    let kv_group = n_head / n_kv_head;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut rng = Lcg(seed);

    // Q [n_head, head_dim] — store f16 bits + keep the decoded f32.
    let mut q_bits = vec![0u16; n_head * head_dim];
    let mut q_f = vec![0f32; n_head * head_dim];
    for i in 0..n_head * head_dim {
        let (b, v) = round_f16(rng.next_f32());
        q_bits[i] = b;
        q_f[i] = v;
    }

    // K, V [n_kv, n_kv_head, head_dim].
    let kv_len = n_kv * n_kv_head * head_dim;
    let mut k_bits = vec![0u16; kv_len];
    let mut k_f = vec![0f32; kv_len];
    let mut v_bits = vec![0u16; kv_len];
    let mut v_f = vec![0f32; kv_len];
    for i in 0..kv_len {
        let (kb, kv) = round_f16(rng.next_f32());
        k_bits[i] = kb;
        k_f[i] = kv;
        let (vb, vv) = round_f16(rng.next_f32());
        v_bits[i] = vb;
        v_f[i] = vv;
    }

    // ----- CPU reference (f32 softmax attention, same grouping) -----
    let mut expect = vec![0f32; n_head * head_dim];
    for h in 0..n_head {
        let kv_head = h / kv_group;
        let qh = &q_f[h * head_dim..h * head_dim + head_dim];

        // scores
        let mut scores = vec![0f32; n_kv];
        for (j, s) in scores.iter_mut().enumerate() {
            let base = (j * n_kv_head + kv_head) * head_dim;
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += qh[d] * k_f[base + d];
            }
            *s = dot * scale;
        }
        // stable softmax
        let mut m = f32::NEG_INFINITY;
        for &s in &scores {
            if s > m {
                m = s;
            }
        }
        let mut denom = 0f32;
        for s in &mut scores {
            *s = (*s - m).exp();
            denom += *s;
        }
        let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        // weighted sum
        let oh = &mut expect[h * head_dim..h * head_dim + head_dim];
        for (j, &w) in scores.iter().enumerate() {
            let base = (j * n_kv_head + kv_head) * head_dim;
            let ww = w * inv;
            for d in 0..head_dim {
                oh[d] += ww * v_f[base + d];
            }
        }
    }

    // ----- GPU -----
    let mut d_q: DeviceBuffer<u16> = DeviceBuffer::new(device.id, q_bits.len())?;
    d_q.copy_from_host(&q_bits)?;
    let mut d_k: DeviceBuffer<u16> = DeviceBuffer::new(device.id, k_bits.len())?;
    d_k.copy_from_host(&k_bits)?;
    let mut d_v: DeviceBuffer<u16> = DeviceBuffer::new(device.id, v_bits.len())?;
    d_v.copy_from_host(&v_bits)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * head_dim)?;

    kernel.single_query(
        stream,
        &mut d_out,
        &d_q,
        &d_k,
        &d_v,
        n_head as u32,
        n_kv_head as u32,
        head_dim as u32,
        n_kv as u32,
        scale,
    )?;
    stream.synchronize()?;

    let mut got = vec![0f32; n_head * head_dim];
    d_out.copy_to_host(&mut got)?;

    let mut max_abs = 0f32;
    for i in 0..n_head * head_dim {
        max_abs = max_abs.max((got[i] - expect[i]).abs());
    }
    eprintln!(
        "gqa case: n_head={n_head}, n_kv_head={n_kv_head} (kv_group={kv_group}), head_dim={head_dim}, n_kv={n_kv} -> max_abs={max_abs:.3e}"
    );
    Ok(max_abs)
}

/// Batched-prefill case: build B query rows + a KV cache holding all B keys,
/// run the single ONE-launch `prefill` kernel, and compare each row against a
/// CPU reference that applies the SAME causal mask (row i attends keys 0..=i).
/// Returns the max-abs error over the whole [B, n_head, head_dim] output.
#[allow(clippy::too_many_arguments)]
fn run_prefill_case(
    kernel: &GqaAttention,
    device: &Device,
    stream: &Stream,
    seed: u64,
    batch: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> eyre::Result<f32> {
    assert_eq!(n_head % n_kv_head, 0);
    let kv_group = n_head / n_kv_head;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut rng = Lcg(seed);

    // Q [B, n_head, head_dim]
    let mut q_bits = vec![0u16; batch * n_head * head_dim];
    let mut q_f = vec![0f32; batch * n_head * head_dim];
    for i in 0..q_bits.len() {
        let (b, v) = round_f16(rng.next_f32());
        q_bits[i] = b;
        q_f[i] = v;
    }
    // KV cache [B, n_kv_head, head_dim] — all B chunk positions present.
    let kv_len = batch * n_kv_head * head_dim;
    let mut k_bits = vec![0u16; kv_len];
    let mut k_f = vec![0f32; kv_len];
    let mut v_bits = vec![0u16; kv_len];
    let mut v_f = vec![0f32; kv_len];
    for i in 0..kv_len {
        let (kb, kv) = round_f16(rng.next_f32());
        k_bits[i] = kb;
        k_f[i] = kv;
        let (vb, vv) = round_f16(rng.next_f32());
        v_bits[i] = vb;
        v_f[i] = vv;
    }

    // CPU reference with causal mask: query row i attends keys [0..=i].
    let mut expect = vec![0f32; batch * n_head * head_dim];
    for i in 0..batch {
        let n_kv_i = i + 1;
        for h in 0..n_head {
            let kv_head = h / kv_group;
            let qh = &q_f[(i * n_head + h) * head_dim..(i * n_head + h) * head_dim + head_dim];
            let mut scores = vec![0f32; n_kv_i];
            for (j, s) in scores.iter_mut().enumerate() {
                let base = (j * n_kv_head + kv_head) * head_dim;
                let mut dot = 0f32;
                for d in 0..head_dim {
                    dot += qh[d] * k_f[base + d];
                }
                *s = dot * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                denom += *s;
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            let oh = &mut expect[(i * n_head + h) * head_dim..(i * n_head + h) * head_dim + head_dim];
            for (j, &w) in scores.iter().enumerate() {
                let base = (j * n_kv_head + kv_head) * head_dim;
                let ww = w * inv;
                for d in 0..head_dim {
                    oh[d] += ww * v_f[base + d];
                }
            }
        }
    }

    // GPU
    let mut d_q: DeviceBuffer<u16> = DeviceBuffer::new(device.id, q_bits.len())?;
    d_q.copy_from_host(&q_bits)?;
    let mut d_k: DeviceBuffer<u16> = DeviceBuffer::new(device.id, k_bits.len())?;
    d_k.copy_from_host(&k_bits)?;
    let mut d_v: DeviceBuffer<u16> = DeviceBuffer::new(device.id, v_bits.len())?;
    d_v.copy_from_host(&v_bits)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, batch * n_head * head_dim)?;

    kernel.prefill(
        stream, &mut d_out, &d_q, &d_k, &d_v,
        batch as u32, n_head as u32, n_kv_head as u32, head_dim as u32, 0, scale, 0,
    )?;
    stream.synchronize()?;

    let mut got = vec![0f32; batch * n_head * head_dim];
    d_out.copy_to_host(&mut got)?;
    let mut max_abs = 0f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expect[i]).abs());
    }
    eprintln!(
        "gqa prefill case: B={batch}, n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim} -> max_abs={max_abs:.3e}"
    );
    Ok(max_abs)
}

/// FLASH-tiled prefill case: same CPU reference as `run_prefill_case`, but with
/// a non-zero `q_offset` (the cache already holds `q_offset` prior keys, so
/// query row i attends keys `[0 ..= q_offset+i]`). Drives `prefill_flash`.
#[allow(clippy::too_many_arguments)]
fn run_prefill_flash_case(
    kernel: &GqaAttention,
    device: &Device,
    stream: &Stream,
    seed: u64,
    q_offset: usize,
    batch: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> eyre::Result<f32> {
    assert_eq!(n_head % n_kv_head, 0);
    let kv_group = n_head / n_kv_head;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut rng = Lcg(seed);
    let n_kv_total = q_offset + batch;

    // Q [B, n_head, head_dim]
    let mut q_bits = vec![0u16; batch * n_head * head_dim];
    let mut q_f = vec![0f32; batch * n_head * head_dim];
    for i in 0..q_bits.len() {
        let (b, v) = round_f16(rng.next_f32());
        q_bits[i] = b;
        q_f[i] = v;
    }
    // KV cache [n_kv_total, n_kv_head, head_dim] — all prior + chunk keys.
    let kv_len = n_kv_total * n_kv_head * head_dim;
    let mut k_bits = vec![0u16; kv_len];
    let mut k_f = vec![0f32; kv_len];
    let mut v_bits = vec![0u16; kv_len];
    let mut v_f = vec![0f32; kv_len];
    for i in 0..kv_len {
        let (kb, kv) = round_f16(rng.next_f32());
        k_bits[i] = kb;
        k_f[i] = kv;
        let (vb, vv) = round_f16(rng.next_f32());
        v_bits[i] = vb;
        v_f[i] = vv;
    }

    // CPU reference: query row i (abs pos q_offset+i) attends keys [0..=q_offset+i].
    let mut expect = vec![0f32; batch * n_head * head_dim];
    for i in 0..batch {
        let n_kv_i = q_offset + i + 1;
        for h in 0..n_head {
            let kv_head = h / kv_group;
            let qh = &q_f[(i * n_head + h) * head_dim..(i * n_head + h) * head_dim + head_dim];
            let mut scores = vec![0f32; n_kv_i];
            for (j, s) in scores.iter_mut().enumerate() {
                let base = (j * n_kv_head + kv_head) * head_dim;
                let mut dot = 0f32;
                for d in 0..head_dim {
                    dot += qh[d] * k_f[base + d];
                }
                *s = dot * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                denom += *s;
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            let oh = &mut expect[(i * n_head + h) * head_dim..(i * n_head + h) * head_dim + head_dim];
            for (j, &w) in scores.iter().enumerate() {
                let base = (j * n_kv_head + kv_head) * head_dim;
                let ww = w * inv;
                for d in 0..head_dim {
                    oh[d] += ww * v_f[base + d];
                }
            }
        }
    }

    // GPU
    let mut d_q: DeviceBuffer<u16> = DeviceBuffer::new(device.id, q_bits.len())?;
    d_q.copy_from_host(&q_bits)?;
    let mut d_k: DeviceBuffer<u16> = DeviceBuffer::new(device.id, k_bits.len())?;
    d_k.copy_from_host(&k_bits)?;
    let mut d_v: DeviceBuffer<u16> = DeviceBuffer::new(device.id, v_bits.len())?;
    d_v.copy_from_host(&v_bits)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, batch * n_head * head_dim)?;

    kernel.prefill_flash(
        stream, &mut d_out, &d_q, &d_k, &d_v,
        batch as u32, n_head as u32, n_kv_head as u32, head_dim as u32, q_offset as u32, scale, 0,
    )?;
    stream.synchronize()?;

    let mut got = vec![0f32; batch * n_head * head_dim];
    d_out.copy_to_host(&mut got)?;
    let mut max_abs = 0f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expect[i]).abs());
    }
    eprintln!(
        "gqa flash case: q_offset={q_offset}, B={batch}, n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim} -> max_abs={max_abs:.3e}"
    );
    Ok(max_abs)
}

/// WMMA FLASH-tiled prefill case: identical CPU reference as
/// `run_prefill_flash_case`, but drives `prefill_flash_wmma`. Returns max-abs
/// error over the whole [B, n_head, head_dim] output.
#[allow(clippy::too_many_arguments)]
fn run_prefill_wmma_case(
    kernel: &GqaAttention,
    device: &Device,
    stream: &Stream,
    seed: u64,
    q_offset: usize,
    batch: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> eyre::Result<f32> {
    assert_eq!(n_head % n_kv_head, 0);
    let kv_group = n_head / n_kv_head;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut rng = Lcg(seed);
    let n_kv_total = q_offset + batch;

    let mut q_bits = vec![0u16; batch * n_head * head_dim];
    let mut q_f = vec![0f32; batch * n_head * head_dim];
    for i in 0..q_bits.len() {
        let (b, v) = round_f16(rng.next_f32());
        q_bits[i] = b;
        q_f[i] = v;
    }
    let kv_len = n_kv_total * n_kv_head * head_dim;
    let mut k_bits = vec![0u16; kv_len];
    let mut k_f = vec![0f32; kv_len];
    let mut v_bits = vec![0u16; kv_len];
    let mut v_f = vec![0f32; kv_len];
    for i in 0..kv_len {
        let (kb, kv) = round_f16(rng.next_f32());
        k_bits[i] = kb;
        k_f[i] = kv;
        let (vb, vv) = round_f16(rng.next_f32());
        v_bits[i] = vb;
        v_f[i] = vv;
    }

    // CPU reference: query row i (abs pos q_offset+i) attends keys [0..=q_offset+i].
    let mut expect = vec![0f32; batch * n_head * head_dim];
    for i in 0..batch {
        let n_kv_i = q_offset + i + 1;
        for h in 0..n_head {
            let kv_head = h / kv_group;
            let qh = &q_f[(i * n_head + h) * head_dim..(i * n_head + h) * head_dim + head_dim];
            let mut scores = vec![0f32; n_kv_i];
            for (j, s) in scores.iter_mut().enumerate() {
                let base = (j * n_kv_head + kv_head) * head_dim;
                let mut dot = 0f32;
                for d in 0..head_dim {
                    dot += qh[d] * k_f[base + d];
                }
                *s = dot * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                denom += *s;
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            let oh = &mut expect[(i * n_head + h) * head_dim..(i * n_head + h) * head_dim + head_dim];
            for (j, &w) in scores.iter().enumerate() {
                let base = (j * n_kv_head + kv_head) * head_dim;
                let ww = w * inv;
                for d in 0..head_dim {
                    oh[d] += ww * v_f[base + d];
                }
            }
        }
    }

    let mut d_q: DeviceBuffer<u16> = DeviceBuffer::new(device.id, q_bits.len())?;
    d_q.copy_from_host(&q_bits)?;
    let mut d_k: DeviceBuffer<u16> = DeviceBuffer::new(device.id, k_bits.len())?;
    d_k.copy_from_host(&k_bits)?;
    let mut d_v: DeviceBuffer<u16> = DeviceBuffer::new(device.id, v_bits.len())?;
    d_v.copy_from_host(&v_bits)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, batch * n_head * head_dim)?;

    kernel.prefill_flash_wmma(
        stream, &mut d_out, &d_q, &d_k, &d_v,
        batch as u32, n_head as u32, n_kv_head as u32, head_dim as u32, q_offset as u32, scale, 0,
    )?;
    stream.synchronize()?;

    let mut got = vec![0f32; batch * n_head * head_dim];
    d_out.copy_to_host(&mut got)?;
    let mut max_abs = 0f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expect[i]).abs());
    }
    eprintln!(
        "gqa wmma case: q_offset={q_offset}, B={batch}, n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim} -> max_abs={max_abs:.3e}"
    );
    Ok(max_abs)
}

#[test]
#[ignore]
fn gqa_attn_prefill_flash_wmma_correctness() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_dgpu()?; // WMMA only meaningful on the dGPU (gfx1201)
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("gqa flash wmma: using device {} ({arch})", device.id);
    let kernel = GqaAttention::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;
    const TOL: f32 = 2.0e-3;

    // Laguna full layer (48 heads) and SWA layer (72 heads), q_offset=0.
    let e1 = run_prefill_wmma_case(&kernel, &device, &stream, 0xabcd_1234_5678_9f01, 0, 130, 48, 8, 128)?;
    assert!(e1 < TOL, "wmma case1 max_abs {e1:.3e} >= tol {TOL:.3e}");
    let e2 = run_prefill_wmma_case(&kernel, &device, &stream, 0x1111_2222_3333_4444, 0, 200, 72, 8, 128)?;
    assert!(e2 < TOL, "wmma case2 max_abs {e2:.3e} >= tol {TOL:.3e}");
    // Non-zero q_offset (a later chunk) + non-round B that straddles a tile.
    let e3 = run_prefill_wmma_case(&kernel, &device, &stream, 0x9999_8888_7777_6666, 100, 77, 48, 8, 128)?;
    assert!(e3 < TOL, "wmma case3 max_abs {e3:.3e} >= tol {TOL:.3e}");
    // Tiny odd B (< FBR) to exercise partial-block guards.
    let e4 = run_prefill_wmma_case(&kernel, &device, &stream, 0x2468_ace0_1357_9bdf, 5, 7, 72, 8, 128)?;
    assert!(e4 < TOL, "wmma case4 max_abs {e4:.3e} >= tol {TOL:.3e}");
    // Larger B (multiple full query tiles) + deeper q_offset.
    let e5 = run_prefill_wmma_case(&kernel, &device, &stream, 0x0f0f_0f0f_1234_5678, 512, 512, 48, 8, 128)?;
    assert!(e5 < TOL, "wmma case5 max_abs {e5:.3e} >= tol {TOL:.3e}");
    Ok(())
}

/// Isolated A/B prefill-attention bench + parity: times the scalar-ILP
/// `prefill_flash` vs the `prefill_flash_wmma` kernel at realistic chunked-
/// prefill depths (B chunk rows attending a `depth`-long causal history), and
/// asserts the two agree (same online-softmax math). Prints µs/call for each.
///
/// Run:
///   nix develop -c cargo test --release -p v4flash-kernels \
///       --test gqa_attention -- --ignored --nocapture prefill_wmma_bench
#[test]
#[ignore]
fn gqa_attn_prefill_wmma_bench() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_dgpu()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("gqa prefill wmma bench: device {} ({arch})", device.id);
    let kernel = GqaAttention::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let n_kv_head = 8usize;
    let head_dim = 128usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    const ITERS: usize = 30;
    const WARMUP: usize = 5;
    let batch = std::env::var("WMMA_BENCH_B").ok().and_then(|v| v.parse().ok()).unwrap_or(512usize);

    // (n_head, depth): both Laguna layer types at 4K and 32K context depth.
    for &(n_head, depth) in &[
        (48usize, 4096usize), (48, 32768),
        (72usize, 4096usize), (72, 32768),
    ] {
        assert!(depth >= batch);
        let q_offset = depth - batch;
        let n_kv_total = depth;
        let mut rng = Lcg(0xabu64 ^ ((n_head as u64) << 20) ^ depth as u64);

        let mut q_bits = vec![0u16; batch * n_head * head_dim];
        for v in q_bits.iter_mut() {
            let (b, _) = round_f16(rng.next_f32());
            *v = b;
        }
        let kv_len = n_kv_total * n_kv_head * head_dim;
        let mut k_bits = vec![0u16; kv_len];
        let mut v_bits = vec![0u16; kv_len];
        for i in 0..kv_len {
            let (kb, _) = round_f16(rng.next_f32());
            k_bits[i] = kb;
            let (vb, _) = round_f16(rng.next_f32());
            v_bits[i] = vb;
        }

        let mut d_q: DeviceBuffer<u16> = DeviceBuffer::new(device.id, q_bits.len())?;
        d_q.copy_from_host(&q_bits)?;
        let mut d_k: DeviceBuffer<u16> = DeviceBuffer::new(device.id, k_bits.len())?;
        d_k.copy_from_host(&k_bits)?;
        let mut d_v: DeviceBuffer<u16> = DeviceBuffer::new(device.id, v_bits.len())?;
        d_v.copy_from_host(&v_bits)?;
        let mut d_flash: DeviceBuffer<f32> = DeviceBuffer::new(device.id, batch * n_head * head_dim)?;
        let mut d_wmma: DeviceBuffer<f32> = DeviceBuffer::new(device.id, batch * n_head * head_dim)?;

        for _ in 0..WARMUP {
            kernel.prefill_flash(&stream, &mut d_flash, &d_q, &d_k, &d_v,
                batch as u32, n_head as u32, n_kv_head as u32, head_dim as u32, q_offset as u32, scale, 0)?;
            kernel.prefill_flash_wmma(&stream, &mut d_wmma, &d_q, &d_k, &d_v,
                batch as u32, n_head as u32, n_kv_head as u32, head_dim as u32, q_offset as u32, scale, 0)?;
        }
        stream.synchronize()?;

        let mut got_flash = vec![0f32; batch * n_head * head_dim];
        let mut got_wmma = vec![0f32; batch * n_head * head_dim];
        d_flash.copy_to_host(&mut got_flash)?;
        d_wmma.copy_to_host(&mut got_wmma)?;
        let mut wf_max = 0f32;
        for i in 0..got_flash.len() {
            wf_max = wf_max.max((got_flash[i] - got_wmma[i]).abs());
        }

        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            kernel.prefill_flash(&stream, &mut d_flash, &d_q, &d_k, &d_v,
                batch as u32, n_head as u32, n_kv_head as u32, head_dim as u32, q_offset as u32, scale, 0)?;
        }
        stream.synchronize()?;
        let flash_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            kernel.prefill_flash_wmma(&stream, &mut d_wmma, &d_q, &d_k, &d_v,
                batch as u32, n_head as u32, n_kv_head as u32, head_dim as u32, q_offset as u32, scale, 0)?;
        }
        stream.synchronize()?;
        let wmma_us = t1.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        eprintln!(
            "n_head={n_head:>3} depth={depth:>6} B={batch}  flash={flash_us:9.1}  wmma={wmma_us:9.1} us  speedup={:.2}x  wmma-vs-flash={wf_max:.3e}",
            flash_us / wmma_us
        );
        assert!(wf_max < 2.0e-3, "wmma vs flash diverged at n_head={n_head} depth={depth}: {wf_max:.3e}");
    }
    Ok(())
}

#[test]
#[ignore]
fn gqa_attn_prefill_flash_correctness() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("gqa flash: using device {} ({arch})", device.id);
    let kernel = GqaAttention::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;
    const TOL: f32 = 2.0e-3;

    // Laguna full layer (48 heads) and SWA layer (72 heads), q_offset=0.
    let e1 = run_prefill_flash_case(&kernel, &device, &stream, 0xabcd_1234_5678_9f01, 0, 130, 48, 8, 128)?;
    assert!(e1 < TOL, "flash case1 max_abs {e1:.3e} >= tol {TOL:.3e}");
    let e2 = run_prefill_flash_case(&kernel, &device, &stream, 0x1111_2222_3333_4444, 0, 200, 72, 8, 128)?;
    assert!(e2 < TOL, "flash case2 max_abs {e2:.3e} >= tol {TOL:.3e}");
    // Non-zero q_offset (a later chunk) + non-round B that straddles a tile.
    let e3 = run_prefill_flash_case(&kernel, &device, &stream, 0x9999_8888_7777_6666, 100, 77, 48, 8, 128)?;
    assert!(e3 < TOL, "flash case3 max_abs {e3:.3e} >= tol {TOL:.3e}");
    // Tiny odd B (< FBR) to exercise partial-block guards.
    let e4 = run_prefill_flash_case(&kernel, &device, &stream, 0x2468_ace0_1357_9bdf, 5, 7, 72, 8, 128)?;
    assert!(e4 < TOL, "flash case4 max_abs {e4:.3e} >= tol {TOL:.3e}");
    Ok(())
}

#[test]
#[ignore]
fn gqa_attn_prefill_correctness() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("gqa prefill: using device {} ({arch})", device.id);
    let kernel = GqaAttention::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;
    const TOL: f32 = 2.0e-3;

    // Laguna full layer (48 heads) and SWA layer (72 heads), B=64.
    let e1 = run_prefill_case(&kernel, &device, &stream, 0xabcd_1234_5678_9f01, 64, 48, 8, 128)?;
    assert!(e1 < TOL, "prefill case1 max_abs {e1:.3e} >= tol {TOL:.3e}");
    let e2 = run_prefill_case(&kernel, &device, &stream, 0x1111_2222_3333_4444, 64, 72, 8, 128)?;
    assert!(e2 < TOL, "prefill case2 max_abs {e2:.3e} >= tol {TOL:.3e}");
    // Small odd batch to exercise causal edges.
    let e3 = run_prefill_case(&kernel, &device, &stream, 0x9999_8888_7777_6666, 7, 6, 2, 128)?;
    assert!(e3 < TOL, "prefill case3 max_abs {e3:.3e} >= tol {TOL:.3e}");
    Ok(())
}

/// Prefer the dGPU (gfx1201) — decode attention runs there in the het path and
/// ATT/rocprofv3 works on it. Falls back to the first device.
fn pick_dgpu() -> eyre::Result<Device> {
    let devices = Device::all()?;
    for d in &devices {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(*d);
        }
    }
    devices.first().copied().ok_or_else(|| eyre!("no HIP devices"))
}

/// Isolated decode-attention benchmark + parity: times the naive per-key-barrier
/// `single_query` vs the FLASH-tiled `single_query_flash` at 4K/32K/96K context,
/// and asserts the two kernels agree (quality-safe: same online-softmax math).
///
/// Run:
///   nix develop -c cargo test --release -p v4flash-kernels \
///       --test gqa_attention -- --ignored --nocapture bench_decode
#[test]
#[ignore]
fn gqa_attn_decode_bench() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_dgpu()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("gqa decode bench: device {} ({arch})", device.id);
    let kernel = GqaAttention::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let n_kv_head = 8usize;
    let head_dim = 128usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    const ITERS: usize = 50;
    const WARMUP: usize = 5;

    // Cover BOTH Laguna layer types: full-attn (n_head=48, kv_group=6) and SWA
    // (n_head=72, kv_group=9). splitkv is head-count-agnostic but validating
    // both proves the real model's two exact configs.
    for &(n_head, n_kv) in &[
        (48usize, 4096usize), (48, 32768), (48, 98304),
        (72usize, 4096usize), (72, 32768),
    ] {
        let kv_group = n_head / n_kv_head;
        let mut rng = Lcg(0xdec0de00 ^ n_kv as u64);
        let mut q_bits = vec![0u16; n_head * head_dim];
        let mut q_f = vec![0f32; n_head * head_dim];
        for i in 0..q_bits.len() {
            let (b, v) = round_f16(rng.next_f32());
            q_bits[i] = b;
            q_f[i] = v;
        }
        let kv_len = n_kv * n_kv_head * head_dim;
        let mut k_bits = vec![0u16; kv_len];
        let mut v_bits = vec![0u16; kv_len];
        let mut k_f = vec![0f32; kv_len];
        let mut v_f = vec![0f32; kv_len];
        for i in 0..kv_len {
            let (kb, kv) = round_f16(rng.next_f32());
            k_bits[i] = kb;
            k_f[i] = kv;
            let (vb, vv) = round_f16(rng.next_f32());
            v_bits[i] = vb;
            v_f[i] = vv;
        }

        let mut d_q: DeviceBuffer<u16> = DeviceBuffer::new(device.id, q_bits.len())?;
        d_q.copy_from_host(&q_bits)?;
        let mut d_k: DeviceBuffer<u16> = DeviceBuffer::new(device.id, k_bits.len())?;
        d_k.copy_from_host(&k_bits)?;
        let mut d_v: DeviceBuffer<u16> = DeviceBuffer::new(device.id, v_bits.len())?;
        d_v.copy_from_host(&v_bits)?;
        let mut d_naive: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * head_dim)?;
        let mut d_flash: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * head_dim)?;
        let mut d_split: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * head_dim)?;
        // split-KV scratch (sized for the chosen n_splits).
        let n_splits = v4flash_kernels::gqa_attention::decode_kv_splits(n_kv as u32);
        let mut d_op: DeviceBuffer<f32> =
            DeviceBuffer::new(device.id, n_head * n_splits as usize * head_dim)?;
        let mut d_mp: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * n_splits as usize)?;
        let mut d_lp: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * n_splits as usize)?;
        // head-grouped split-KV scratch (own split count; sized to the max).
        let n_splits_hg = v4flash_kernels::gqa_attention::decode_kv_splits_hg(n_kv as u32);
        let smax = n_splits.max(n_splits_hg) as usize;
        let mut d_hg: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * head_dim)?;
        let mut d_hop: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * smax * head_dim)?;
        let mut d_hmp: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * smax)?;
        let mut d_hlp: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * smax)?;

        // --- warmup + parity ---
        for _ in 0..WARMUP {
            kernel.single_query(&stream, &mut d_naive, &d_q, &d_k, &d_v,
                n_head as u32, n_kv_head as u32, head_dim as u32, n_kv as u32, scale)?;
            kernel.single_query_flash(&stream, &mut d_flash, &d_q, &d_k, &d_v,
                n_head as u32, n_kv_head as u32, head_dim as u32, n_kv as u32, scale)?;
            kernel.single_query_splitkv(&stream, &mut d_split, &mut d_op, &mut d_mp, &mut d_lp,
                &d_q, &d_k, &d_v, n_head as u32, n_kv_head as u32, head_dim as u32,
                n_kv as u32, n_splits, scale)?;
            kernel.single_query_splitkv_hg(&stream, &mut d_hg, &mut d_hop, &mut d_hmp, &mut d_hlp,
                &d_q, &d_k, &d_v, n_head as u32, n_kv_head as u32, head_dim as u32,
                n_kv as u32, n_splits_hg, scale)?;
        }
        stream.synchronize()?;

        let mut got_naive = vec![0f32; n_head * head_dim];
        let mut got_flash = vec![0f32; n_head * head_dim];
        let mut got_split = vec![0f32; n_head * head_dim];
        let mut got_hg = vec![0f32; n_head * head_dim];
        d_naive.copy_to_host(&mut got_naive)?;
        d_flash.copy_to_host(&mut got_flash)?;
        d_split.copy_to_host(&mut got_split)?;
        d_hg.copy_to_host(&mut got_hg)?;
        // flash-vs-naive + splitkv-vs-naive + hg-vs-naive agreement (same math)
        let mut fn_max = 0f32;
        let mut sn_max = 0f32;
        let mut hn_max = 0f32;
        for i in 0..got_naive.len() {
            fn_max = fn_max.max((got_naive[i] - got_flash[i]).abs());
            sn_max = sn_max.max((got_naive[i] - got_split[i]).abs());
            hn_max = hn_max.max((got_naive[i] - got_hg[i]).abs());
        }

        // Anchor: CPU f32 softmax reference at the smallest ctx only (98K CPU
        // ref is slow and the flash-vs-naive delta already bounds correctness).
        let mut cpu_note = String::new();
        if n_kv <= 4096 {
            let mut expect = vec![0f32; n_head * head_dim];
            for h in 0..n_head {
                let kv_head = h / kv_group;
                let qh = &q_f[h * head_dim..h * head_dim + head_dim];
                let mut scores = vec![0f32; n_kv];
                for (j, s) in scores.iter_mut().enumerate() {
                    let base = (j * n_kv_head + kv_head) * head_dim;
                    let mut dot = 0f32;
                    for d in 0..head_dim {
                        dot += qh[d] * k_f[base + d];
                    }
                    *s = dot * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0f32;
                for s in &mut scores {
                    *s = (*s - m).exp();
                    denom += *s;
                }
                let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                let oh = &mut expect[h * head_dim..h * head_dim + head_dim];
                for (j, &w) in scores.iter().enumerate() {
                    let base = (j * n_kv_head + kv_head) * head_dim;
                    let ww = w * inv;
                    for d in 0..head_dim {
                        oh[d] += ww * v_f[base + d];
                    }
                }
            }
            let mut cn = 0f32;
            let mut cf = 0f32;
            for i in 0..expect.len() {
                cn = cn.max((got_naive[i] - expect[i]).abs());
                cf = cf.max((got_flash[i] - expect[i]).abs());
            }
            cpu_note = format!("  cpu_ref: naive={cn:.3e} flash={cf:.3e}");
        }

        // --- time naive ---
        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            kernel.single_query(&stream, &mut d_naive, &d_q, &d_k, &d_v,
                n_head as u32, n_kv_head as u32, head_dim as u32, n_kv as u32, scale)?;
        }
        stream.synchronize()?;
        let naive_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        // --- time flash ---
        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            kernel.single_query_flash(&stream, &mut d_flash, &d_q, &d_k, &d_v,
                n_head as u32, n_kv_head as u32, head_dim as u32, n_kv as u32, scale)?;
        }
        stream.synchronize()?;
        let flash_us = t1.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        // --- time split-KV ---
        let t2 = std::time::Instant::now();
        for _ in 0..ITERS {
            kernel.single_query_splitkv(&stream, &mut d_split, &mut d_op, &mut d_mp, &mut d_lp,
                &d_q, &d_k, &d_v, n_head as u32, n_kv_head as u32, head_dim as u32,
                n_kv as u32, n_splits, scale)?;
        }
        stream.synchronize()?;
        let split_us = t2.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        // --- time head-grouped split-KV ---
        let t3 = std::time::Instant::now();
        for _ in 0..ITERS {
            kernel.single_query_splitkv_hg(&stream, &mut d_hg, &mut d_hop, &mut d_hmp, &mut d_hlp,
                &d_q, &d_k, &d_v, n_head as u32, n_kv_head as u32, head_dim as u32,
                n_kv as u32, n_splits_hg, scale)?;
        }
        stream.synchronize()?;
        let hg_us = t3.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        eprintln!(
            "n_kv={n_kv:>6}  naive={naive_us:8.1}  flash={flash_us:8.1}  splitkv={split_us:8.1}  hg={hg_us:8.1} us (splits {n_splits}/{n_splits_hg})  split/hg={:.2}x  s-vs-n={sn_max:.3e} h-vs-n={hn_max:.3e}{cpu_note}",
            split_us / hg_us
        );
        // Quality gate: all kernels must agree (identical f32 softmax math).
        assert!(fn_max < 2.0e-3, "flash vs naive diverged at n_kv={n_kv}: {fn_max:.3e}");
        assert!(sn_max < 2.0e-3, "splitkv vs naive diverged at n_kv={n_kv}: {sn_max:.3e}");
        assert!(hn_max < 2.0e-3, "hg vs naive diverged at n_kv={n_kv}: {hn_max:.3e}");
    }
    Ok(())
}

#[test]
#[ignore]
fn gqa_attn_single_query_correctness() -> eyre::Result<()> {
    install_panic_handler()?;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("gqa attn: using device {} ({arch})", device.id);

    let kernel = GqaAttention::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    const TOL: f32 = 2.0e-3;

    // Case 1: small, kv_group=3, non-round n_kv=17.
    let e1 = run_case(&kernel, &device, &stream, 0x1234_5678_9abc_def0, 6, 2, 128, 17)?;
    assert!(e1 < TOL, "case1 max_abs {e1:.3e} >= tol {TOL:.3e}");

    // Case 2: Laguna-shaped, kv_group=9.
    let e2 = run_case(&kernel, &device, &stream, 0x0fed_cba9_8765_4321, 72, 8, 128, 33)?;
    assert!(e2 < TOL, "case2 max_abs {e2:.3e} >= tol {TOL:.3e}");

    Ok(())
}

/// SMALL-n_kv decode correctness: the shipped decode-parity bench only covers
/// n_kv >= 4096 (n_splits >> 1). Real decode starts at n_kv ~ prompt_len (tens),
/// where decode_kv_splits_hg == 1 and per-head/per-split edge cases live. This
/// drives naive + splitkv + hg with the MODEL's own split counts and compares
/// each against an independent CPU f32 softmax reference, for both Laguna layer
/// shapes (n_head 48/72). Signed [-1,1) inputs (real q/k are signed post-RoPE).
#[test]
#[ignore]
fn gqa_attn_decode_small_ctx_correctness() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_dgpu()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let kernel = GqaAttention::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let n_kv_head = 8usize;
    let head_dim = 128usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut worst_hg = 0f32;
    let mut worst_sp = 0f32;

    for &n_head in &[48usize, 72usize] {
        let kv_group = n_head / n_kv_head;
        for &n_kv in &[17usize, 20, 29, 40, 63, 100, 200, 257, 511, 600] {
            let mut rng = Lcg(0xbeef00 ^ ((n_head as u64) << 20) ^ n_kv as u64);
            let mut q_bits = vec![0u16; n_head * head_dim];
            let mut q_f = vec![0f32; n_head * head_dim];
            for i in 0..q_bits.len() {
                let (b, v) = round_f16(rng.next_f32());
                q_bits[i] = b;
                q_f[i] = v;
            }
            let kv_len = n_kv * n_kv_head * head_dim;
            let mut k_bits = vec![0u16; kv_len];
            let mut v_bits = vec![0u16; kv_len];
            let mut k_f = vec![0f32; kv_len];
            let mut v_f = vec![0f32; kv_len];
            for i in 0..kv_len {
                let (kb, kv) = round_f16(rng.next_f32());
                k_bits[i] = kb; k_f[i] = kv;
                let (vb, vv) = round_f16(rng.next_f32());
                v_bits[i] = vb; v_f[i] = vv;
            }

            let mut d_q: DeviceBuffer<u16> = DeviceBuffer::new(device.id, q_bits.len())?;
            d_q.copy_from_host(&q_bits)?;
            let mut d_k: DeviceBuffer<u16> = DeviceBuffer::new(device.id, k_bits.len())?;
            d_k.copy_from_host(&k_bits)?;
            let mut d_v: DeviceBuffer<u16> = DeviceBuffer::new(device.id, v_bits.len())?;
            d_v.copy_from_host(&v_bits)?;
            let mut d_naive: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * head_dim)?;
            let mut d_split: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * head_dim)?;
            let mut d_hg: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * head_dim)?;

            let n_splits = v4flash_kernels::gqa_attention::decode_kv_splits(n_kv as u32);
            let n_splits_hg = v4flash_kernels::gqa_attention::decode_kv_splits_hg(n_kv as u32);
            let smax = n_splits.max(n_splits_hg) as usize;
            let mut d_op: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * smax * head_dim)?;
            let mut d_mp: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * smax)?;
            let mut d_lp: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head * smax)?;

            kernel.single_query(&stream, &mut d_naive, &d_q, &d_k, &d_v,
                n_head as u32, n_kv_head as u32, head_dim as u32, n_kv as u32, scale)?;
            kernel.single_query_splitkv(&stream, &mut d_split, &mut d_op, &mut d_mp, &mut d_lp,
                &d_q, &d_k, &d_v, n_head as u32, n_kv_head as u32, head_dim as u32,
                n_kv as u32, n_splits, scale)?;
            kernel.single_query_splitkv_hg(&stream, &mut d_hg, &mut d_op, &mut d_mp, &mut d_lp,
                &d_q, &d_k, &d_v, n_head as u32, n_kv_head as u32, head_dim as u32,
                n_kv as u32, n_splits_hg, scale)?;
            stream.synchronize()?;

            let mut got_naive = vec![0f32; n_head * head_dim];
            let mut got_split = vec![0f32; n_head * head_dim];
            let mut got_hg = vec![0f32; n_head * head_dim];
            d_naive.copy_to_host(&mut got_naive)?;
            d_split.copy_to_host(&mut got_split)?;
            d_hg.copy_to_host(&mut got_hg)?;

            // CPU f32 softmax reference.
            let mut expect = vec![0f32; n_head * head_dim];
            for h in 0..n_head {
                let kv_head = h / kv_group;
                let qh = &q_f[h * head_dim..h * head_dim + head_dim];
                let mut scores = vec![0f32; n_kv];
                for (j, s) in scores.iter_mut().enumerate() {
                    let base = (j * n_kv_head + kv_head) * head_dim;
                    let mut dot = 0f32;
                    for d in 0..head_dim { dot += qh[d] * k_f[base + d]; }
                    *s = dot * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0f32;
                for s in &mut scores { *s = (*s - m).exp(); denom += *s; }
                let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                let oh = &mut expect[h * head_dim..h * head_dim + head_dim];
                for (j, &w) in scores.iter().enumerate() {
                    let base = (j * n_kv_head + kv_head) * head_dim;
                    let ww = w * inv;
                    for d in 0..head_dim { oh[d] += ww * v_f[base + d]; }
                }
            }
            let (mut cn, mut cs, mut ch) = (0f32, 0f32, 0f32);
            for i in 0..expect.len() {
                cn = cn.max((got_naive[i] - expect[i]).abs());
                cs = cs.max((got_split[i] - expect[i]).abs());
                ch = ch.max((got_hg[i] - expect[i]).abs());
            }
            worst_hg = worst_hg.max(ch);
            worst_sp = worst_sp.max(cs);
            eprintln!("n_head={n_head} n_kv={n_kv:>4} splits(sp/hg)={n_splits}/{n_splits_hg}  cpu_ref: naive={cn:.3e} splitkv={cs:.3e} hg={ch:.3e}");
        }
    }
    const TOL: f32 = 2.0e-3;
    assert!(worst_sp < TOL, "splitkv vs cpu worst {worst_sp:.3e} >= {TOL:.3e}");
    assert!(worst_hg < TOL, "hg vs cpu worst {worst_hg:.3e} >= {TOL:.3e}");
    Ok(())
}

/// SWA-WINDOWED prefill correctness: drives prefill / prefill_flash /
/// prefill_flash_wmma with a non-zero `swa_window` and compares each against a
/// CPU reference that applies the SAME sliding window — query row i (abs pos
/// q_offset+i) attends keys [max(0, abs-window+1) .. abs]. Validates the new
/// per-key window mask + the key-tile-start clamp. Uses a small window to force
/// the windowed regime at modest B, and a q_offset>window case.
#[test]
#[ignore]
fn gqa_attn_prefill_swa_window_correctness() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_dgpu()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let kernel = GqaAttention::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;
    let n_kv_head = 8usize;
    let head_dim = 128usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    const TOL_SCALAR: f32 = 2.0e-3;
    const TOL_WMMA: f32 = 3.0e-3;

    // (q_offset, batch, n_head, window)
    for &(q_offset, batch, n_head, window) in &[
        (0usize, 40usize, 48usize, 8u32),
        (0, 200, 72, 64),
        (600, 64, 48, 512),
        (500, 77, 72, 512),
    ] {
        let kv_group = n_head / n_kv_head;
        let n_kv_total = q_offset + batch;
        let mut rng = Lcg(0x5a5a00 ^ ((n_head as u64) << 24) ^ ((window as u64) << 8) ^ batch as u64);
        let mut q_bits = vec![0u16; batch * n_head * head_dim];
        let mut q_f = vec![0f32; batch * n_head * head_dim];
        for i in 0..q_bits.len() { let (b, v) = round_f16(rng.next_f32()); q_bits[i] = b; q_f[i] = v; }
        let kv_len = n_kv_total * n_kv_head * head_dim;
        let mut k_bits = vec![0u16; kv_len];
        let mut v_bits = vec![0u16; kv_len];
        let mut k_f = vec![0f32; kv_len];
        let mut v_f = vec![0f32; kv_len];
        for i in 0..kv_len {
            let (kb, kv) = round_f16(rng.next_f32()); k_bits[i] = kb; k_f[i] = kv;
            let (vb, vv) = round_f16(rng.next_f32()); v_bits[i] = vb; v_f[i] = vv;
        }
        let mut d_q: DeviceBuffer<u16> = DeviceBuffer::new(device.id, q_bits.len())?;
        d_q.copy_from_host(&q_bits)?;
        let mut d_k: DeviceBuffer<u16> = DeviceBuffer::new(device.id, k_bits.len())?;
        d_k.copy_from_host(&k_bits)?;
        let mut d_v: DeviceBuffer<u16> = DeviceBuffer::new(device.id, v_bits.len())?;
        d_v.copy_from_host(&v_bits)?;
        let ol = batch * n_head * head_dim;
        let mut d_naive: DeviceBuffer<f32> = DeviceBuffer::new(device.id, ol)?;
        let mut d_flash: DeviceBuffer<f32> = DeviceBuffer::new(device.id, ol)?;
        let mut d_wmma: DeviceBuffer<f32> = DeviceBuffer::new(device.id, ol)?;
        kernel.prefill(&stream, &mut d_naive, &d_q, &d_k, &d_v,
            batch as u32, n_head as u32, n_kv_head as u32, head_dim as u32, q_offset as u32, scale, window)?;
        kernel.prefill_flash(&stream, &mut d_flash, &d_q, &d_k, &d_v,
            batch as u32, n_head as u32, n_kv_head as u32, head_dim as u32, q_offset as u32, scale, window)?;
        kernel.prefill_flash_wmma(&stream, &mut d_wmma, &d_q, &d_k, &d_v,
            batch as u32, n_head as u32, n_kv_head as u32, head_dim as u32, q_offset as u32, scale, window)?;
        stream.synchronize()?;
        let mut got_naive = vec![0f32; ol];
        let mut got_flash = vec![0f32; ol];
        let mut got_wmma = vec![0f32; ol];
        d_naive.copy_to_host(&mut got_naive)?;
        d_flash.copy_to_host(&mut got_flash)?;
        d_wmma.copy_to_host(&mut got_wmma)?;

        // CPU windowed causal reference.
        let mut expect = vec![0f32; ol];
        for i in 0..batch {
            let abs = q_offset + i;
            let lo = abs.saturating_sub(window as usize - 1);
            for h in 0..n_head {
                let kv_head = h / kv_group;
                let qh = &q_f[(i * n_head + h) * head_dim..(i * n_head + h) * head_dim + head_dim];
                let mut scores = vec![f32::NEG_INFINITY; abs + 1];
                for j in lo..=abs {
                    let base = (j * n_kv_head + kv_head) * head_dim;
                    let mut dot = 0f32;
                    for d in 0..head_dim { dot += qh[d] * k_f[base + d]; }
                    scores[j] = dot * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0f32;
                let mut sw = vec![0f32; abs + 1];
                for j in lo..=abs { sw[j] = (scores[j] - m).exp(); denom += sw[j]; }
                let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                let oh = &mut expect[(i * n_head + h) * head_dim..(i * n_head + h) * head_dim + head_dim];
                for j in lo..=abs {
                    let base = (j * n_kv_head + kv_head) * head_dim;
                    let ww = sw[j] * inv;
                    for d in 0..head_dim { oh[d] += ww * v_f[base + d]; }
                }
            }
        }
        let (mut en, mut ef, mut ew) = (0f32, 0f32, 0f32);
        for i in 0..ol {
            en = en.max((got_naive[i] - expect[i]).abs());
            ef = ef.max((got_flash[i] - expect[i]).abs());
            ew = ew.max((got_wmma[i] - expect[i]).abs());
        }
        eprintln!("q_off={q_offset} B={batch} n_head={n_head} win={window}  cpu_ref: naive={en:.3e} flash={ef:.3e} wmma={ew:.3e}");
        assert!(en < TOL_SCALAR, "naive windowed vs cpu {en:.3e}");
        assert!(ef < TOL_SCALAR, "flash windowed vs cpu {ef:.3e}");
        assert!(ew < TOL_WMMA, "wmma windowed vs cpu {ew:.3e}");
    }
    Ok(())
}
