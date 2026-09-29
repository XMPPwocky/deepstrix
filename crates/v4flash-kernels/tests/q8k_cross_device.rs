//! `V41_PUSH_XQ` precondition: `q8_k_quantize` produces the SAME BYTES on the
//! dGPU (gfx1201) and the iGPU (gfx1151). With the knob on, the iGPU's MoE
//! consumes the dGPU's Q8_K of `ffn_input_norm` instead of quantising the f32
//! rows itself, so any byte difference is a numerics change on box 1's routed
//! experts. (Box 2 already consumes the dGPU's bytes -- `pre_moe_route` relies
//! on the same equality.)
//!
//! Synthetic inputs, a few MB per device, no model load. Rows 1..512 of
//! N_EMBD = 5120 (20 blocks of 256) with the awkward blocks mixed in: all-zero,
//! +a/-a ties at several positions, a lone spike, subnormals, huge magnitudes,
//! a negative maximum, constant blocks, and values that sit on rint's .5 edge.
//!
//! Run (server DOWN -- touches both GPUs):
//! `cargo test --release --features v41 -p v4flash-kernels --test
//! q8k_cross_device -- --ignored --test-threads=1 --nocapture`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::Q8KQuantize;

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 292;
const N_EMBD: usize = 5120;

fn pick(prefix: &str) -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with(prefix) {
            return Ok(d);
        }
    }
    Err(eyre!("no {prefix} device"))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFFFF) as f32 / 16777216.0
    }
}

/// One 256-value block of kind `k` (0 = ordinary activations).
fn block(rng: &mut Lcg, k: u32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..QK_K).map(|_| (rng.unit() * 2.0 - 1.0) * 3.0).collect();
    match k {
        1 => v.iter_mut().for_each(|x| *x = 0.0),
        2 => {
            // +a / -a ties for the max at several positions: the tree
            // reduction's tie-break decides the sign of `iscale`.
            let a = 7.25;
            for (i, &p) in [3usize, 64, 130, 255].iter().enumerate() {
                v[p] = if i % 2 == 0 { a } else { -a };
            }
        }
        3 => v[rng.next() as usize % QK_K] = 900.0,
        4 => v.iter_mut().for_each(|x| *x *= 1e-39), // subnormal
        5 => v.iter_mut().for_each(|x| *x *= 1e30),
        6 => {
            let p = rng.next() as usize % QK_K;
            v[p] = -40.0;
        }
        7 => v.iter_mut().for_each(|x| *x = 0.3),
        8 => {
            // Values that land near n + 0.5 after scaling by -127/max.
            let max = 127.0;
            v[0] = max;
            for (i, x) in v.iter_mut().enumerate().skip(1) {
                *x = -((i % 120) as f32 + 0.5);
            }
        }
        _ => {}
    }
    v
}

fn quantize(dev: &Device, x: &[f32]) -> eyre::Result<Vec<u8>> {
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let q = Q8KQuantize::for_arch(&arch)?;
    let stream = Stream::new(dev.id)?;
    let mut xd: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, x.len())?;
    xd.copy_from_host(x)?;
    let n_blocks = x.len() / QK_K;
    let mut out: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, n_blocks * BLOCK_BYTES)?;
    // Sentinel: a block the kernel failed to write shows as a mismatch.
    out.copy_from_host(&vec![0xA5u8; n_blocks * BLOCK_BYTES])?;
    q.launch(&stream, &mut out, &xd, n_blocks as u32)?;
    stream.synchronize()?;
    let mut h = vec![0u8; n_blocks * BLOCK_BYTES];
    out.copy_to_host(&mut h)?;
    Ok(h)
}

#[test]
#[ignore]
fn q8k_quantize_bytes_match_across_dgpu_and_igpu() -> eyre::Result<()> {
    install_panic_handler()?;
    let dgpu = pick("gfx1201")?;
    let igpu = pick("gfx1151")?;
    let mut rng = Lcg(0x0808_0808_5EED);
    let mut total_blocks = 0usize;
    for &rows in &[1usize, 2, 3, 4, 7, 8, 16, 64, 511, 512] {
        let mut x = Vec::with_capacity(rows * N_EMBD);
        for bi in 0..rows * (N_EMBD / QK_K) {
            // Mostly ordinary blocks, every edge kind at least once per row set.
            let kind = if bi < 9 { bi as u32 } else if rng.next() % 5 == 0 { rng.next() % 9 } else { 0 };
            x.extend(block(&mut rng, kind));
        }
        let a = quantize(&dgpu, &x)?;
        let b = quantize(&igpu, &x)?;
        let bad: Vec<usize> = (0..a.len() / BLOCK_BYTES)
            .filter(|&k| a[k * BLOCK_BYTES..(k + 1) * BLOCK_BYTES] != b[k * BLOCK_BYTES..(k + 1) * BLOCK_BYTES])
            .collect();
        println!("rows {rows:4}: {} blocks, {} differ", a.len() / BLOCK_BYTES, bad.len());
        if let Some(&k) = bad.first() {
            let (da, db) = (&a[k * BLOCK_BYTES..k * BLOCK_BYTES + 4], &b[k * BLOCK_BYTES..k * BLOCK_BYTES + 4]);
            return Err(eyre!("rows {rows}: block {k} differs (d dgpu {da:02x?} vs igpu {db:02x?}); {} blocks total", bad.len()));
        }
        total_blocks += a.len() / BLOCK_BYTES;
    }
    println!("OK: {total_blocks} Q8_K blocks byte-identical on gfx1201 and gfx1151");
    Ok(())
}
