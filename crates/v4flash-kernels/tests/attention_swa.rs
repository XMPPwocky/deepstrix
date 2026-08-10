//! SWA attention oracle — validates `attention_swa` against ds4's
//! `layer_attention_rows_one` (ds4.c:4955) on the pure-SWA layers L=0, L=1
//! (`ds4_layer_compress_ratio(il) == 0`).
//!
//! For each (L, T): build up the KV cache by accumulating `kv_cached_row`
//! tensors across T (the f16-roundtripped post-FP8 values that ds4's cache
//! stores), feed `q_post_rope` + the accumulated cache + `attn_sinks`,
//! compare against `attn_heads`. n_kv = T + 1 (window never overflows
//! since the M1 prompt only reaches T=56 < DS4_N_SWA=128).
//!
//! Pass: `max_abs_diff < 1e-3` over 2 layers × 51 tokens × 32768 ≈ 3.34M
//! element comparisons.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!                              --test attention_swa -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{oracle::ActivationDump, AttentionSwa, ATTN_SWA_MAX_KV};

const N_HEAD: u32 = 64;
const N_HEAD_DIM: u32 = 512;
const Q_FLAT: u32 = N_HEAD * N_HEAD_DIM; // 32768
const SWA_LAYERS: &[i32] = &[0, 1]; // ratio==0 in V4 Flash
const THRESHOLD: f32 = 1.0e-3;

fn dump_dir() -> PathBuf {
    std::env::var("DEEPSTRIX_DUMP_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("reference/v4flash-cpu-activations")
    })
}

/// f32 → f16 bits (RTNE). Matches GPU `(_Float16)` cast.
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32;
    let mant = (bits & 0x7fffff) as u32;
    if exp == 0xff {
        let m = if mant != 0 { 0x200 } else { 0 };
        return sign | 0x7c00 | m as u16;
    }
    exp = exp - 127 + 15;
    if exp >= 0x1f { return sign | 0x7c00; }
    if exp <= 0 {
        if exp < -10 { return sign; }
        let m = (mant | 0x800000) >> (1 - exp);
        let rounded = (m + 0x1000 + ((m >> 13) & 1)) >> 13;
        return sign | rounded as u16;
    }
    let rounded_mant = (mant + 0x1000 + ((mant >> 13) & 1)) >> 13;
    if rounded_mant & 0x400 != 0 {
        return sign | ((exp as u16 + 1) << 10);
    }
    sign | ((exp as u16) << 10) | rounded_mant as u16
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

#[derive(Default)]
struct DiffStats {
    max_abs: f32,
    sum_abs: f64,
    count: usize,
}
impl DiffStats {
    fn update(&mut self, a: &[f32], b: &[f32]) {
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > self.max_abs {
                self.max_abs = d;
            }
            self.sum_abs += d as f64;
            self.count += 1;
        }
    }
    fn mean_abs(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_abs / self.count as f64
        }
    }
}

#[test]
#[ignore]
fn attention_swa_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let n_tokens = dump.n_logit_rows as i32;
    eprintln!(
        "dump: n_tensors={}, n_tokens={}, SWA layers={:?}",
        dump.len(),
        n_tokens,
        SWA_LAYERS,
    );
    assert!(
        (n_tokens as u32) <= ATTN_SWA_MAX_KV,
        "test prompt has {} tokens, exceeds kernel cap {ATTN_SWA_MAX_KV}",
        n_tokens
    );

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let kernel = AttentionSwa::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(device.id, Q_FLAT as usize)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, Q_FLAT as usize)?;
    let mut d_sinks: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_HEAD as usize)?;
    // One contiguous device buffer for the KV cache (f16-stored); grows by
    // N_HEAD_DIM per T. Host-side staging stays f32 then converts at copy.
    let mut d_kv: DeviceBuffer<u16> =
        DeviceBuffer::new(device.id, (ATTN_SWA_MAX_KV as usize) * (N_HEAD_DIM as usize))?;
    let mut got = vec![0f32; Q_FLAT as usize];

    let mut stats = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    for &layer in SWA_LAYERS {
        // Per-layer sinks.
        let sinks_entry = dump
            .weight("attn_sinks", layer)
            .ok_or_else(|| eyre!("missing weight:attn_sinks for L{layer}"))?;
        let sinks = dump.read_f32(sinks_entry)?;
        assert_eq!(sinks.len(), N_HEAD as usize);
        d_sinks.copy_from_host(&sinks)?;

        // Host-side cache buffer; copy each row into d_kv at the right offset.
        let mut host_cache = vec![0f32; (ATTN_SWA_MAX_KV as usize) * (N_HEAD_DIM as usize)];
        let mut layer_stats = DiffStats::default();

        for token in 0..n_tokens {
            let kv_row_entry = dump
                .tensor("kv_cached_row", layer, token)
                .ok_or_else(|| eyre!("missing kv_cached_row at L{layer} T{token}"))?;
            let kv_row = dump.read_f32(kv_row_entry)?;
            assert_eq!(kv_row.len(), N_HEAD_DIM as usize);

            let row_offset = (token as usize) * (N_HEAD_DIM as usize);
            host_cache[row_offset..row_offset + (N_HEAD_DIM as usize)].copy_from_slice(&kv_row);
            // Host f32 cache → f16 bits to match the f16 V cache layout.
            let host_cache_f16: Vec<u16> =
                host_cache.iter().map(|&x| f32_to_f16_bits(x)).collect();
            d_kv.copy_from_host(&host_cache_f16)?;

            let q_entry = dump
                .tensor("q_post_rope", layer, token)
                .ok_or_else(|| eyre!("missing q_post_rope at L{layer} T{token}"))?;
            let q_host = dump.read_f32(q_entry)?;
            assert_eq!(q_host.len(), Q_FLAT as usize);
            d_q.copy_from_host(&q_host)?;

            let n_kv = (token as u32) + 1;
            kernel.launch(
                &stream,
                &mut d_out,
                &d_q,
                &d_kv,
                &d_sinks,
                N_HEAD,
                N_HEAD_DIM,
                n_kv,
            )?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;

            let expected_entry = dump
                .tensor("attn_heads", layer, token)
                .ok_or_else(|| eyre!("missing attn_heads at L{layer} T{token}"))?;
            let expected = dump.read_f32(expected_entry)?;

            let prev = stats.max_abs;
            stats.update(&got, &expected);
            layer_stats.update(&got, &expected);
            if stats.max_abs > prev {
                worst = (layer, token);
            }
        }

        eprintln!(
            "  L{layer}: max_abs={:.3e}, mean={:.3e}, n={}",
            layer_stats.max_abs,
            layer_stats.mean_abs(),
            layer_stats.count,
        );
    }

    eprintln!(
        "OVERALL: max_abs_diff={:.3e}, mean_abs_diff={:.3e}, n={}, worst at L{} T{}",
        stats.max_abs,
        stats.mean_abs(),
        stats.count,
        worst.0,
        worst.1,
    );

    assert!(
        stats.max_abs < THRESHOLD,
        "max_abs_diff {:.3e} exceeds threshold {:.3e}",
        stats.max_abs,
        THRESHOLD
    );

    Ok(())
}
