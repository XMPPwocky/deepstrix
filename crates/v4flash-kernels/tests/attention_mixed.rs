//! `attention_mixed` oracle — validates the generalised mixed-attention
//! kernel against ds4's `layer_attention_mixed_one_decode_scratch`
//! (ds4.c:6738) and (when n_comp=0, mask=None) against `layer_attention_rows_one`
//! (ds4.c:4955).
//!
//! Three sub-blocks:
//!   1. SWA regression (L=0, L=1): n_comp=0, mask=None — must match
//!      `attn_heads` at f32-ULP, identical to M5's attention_swa.
//!   2. ratio==128 layers (L=3,5,…,41): compressor doesn't fire in our
//!      57-tok prompt, so n_comp=0, mask=None — also must match attn_heads.
//!   3. ratio==4 layers (L=2,4,…,42): n_comp grows by 1 every 4 tokens
//!      starting at T=3; comp_allowed_mask is loaded from the dump.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!                              --test attention_mixed -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{ActivationDump, AttentionMixed, ATTN_MIXED_MAX_KEYS};

const N_HEAD: u32 = 64;
const N_HEAD_DIM: u32 = 512;
const Q_FLAT: u32 = N_HEAD * N_HEAD_DIM; // 32768
const N_LAYER: i32 = 43;
const COMP_RATIO_4: u32 = 4;

const SWA_THRESHOLD: f32 = 1.0e-5;
const RATIO128_THRESHOLD: f32 = 1.0e-5;
const RATIO4_THRESHOLD: f32 = 5.0e-4;

fn dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations")
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

#[derive(Default, Clone, Copy)]
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

fn layer_compress_ratio(il: i32) -> u32 {
    // Mirrors ds4_layer_compress_ratio (ds4.c:411).
    if il < 2 {
        0
    } else if (il & 1) == 0 {
        4
    } else {
        128
    }
}

#[test]
#[ignore]
fn attention_mixed_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let n_tokens = dump.n_logit_rows as i32;
    eprintln!("dump: n_tensors={}, n_tokens={}", dump.len(), n_tokens);

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let kernel = AttentionMixed::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // Reused buffers across blocks.
    let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(device.id, Q_FLAT as usize)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, Q_FLAT as usize)?;
    let mut d_sinks: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_HEAD as usize)?;
    let mut d_raw_kv: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, (ATTN_MIXED_MAX_KEYS as usize) * (N_HEAD_DIM as usize))?;
    let mut d_comp_kv: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, (ATTN_MIXED_MAX_KEYS as usize) * (N_HEAD_DIM as usize))?;
    let mut d_mask: DeviceBuffer<i32> = DeviceBuffer::new(device.id, ATTN_MIXED_MAX_KEYS as usize)?;
    let mut got = vec![0f32; Q_FLAT as usize];

    let mut block1 = DiffStats::default();
    let mut block2 = DiffStats::default();
    let mut block3 = DiffStats::default();

    let mut worst_block1 = (-1i32, -1i32);
    let mut worst_block2 = (-1i32, -1i32);
    let mut worst_block3 = (-1i32, -1i32);

    for layer in 0..N_LAYER {
        let ratio = layer_compress_ratio(layer);

        let sinks_entry = dump
            .weight("attn_sinks", layer)
            .ok_or_else(|| eyre!("missing weight:attn_sinks for L{layer}"))?;
        let sinks = dump.read_f32(sinks_entry)?;
        d_sinks.copy_from_host(&sinks)?;

        let mut host_raw =
            vec![0f32; (ATTN_MIXED_MAX_KEYS as usize) * (N_HEAD_DIM as usize)];
        let mut host_comp = vec![0f32; (ATTN_MIXED_MAX_KEYS as usize) * (N_HEAD_DIM as usize)];
        let mut n_comp: u32 = 0;

        for token in 0..n_tokens {
            // Raw KV cache row for this token.
            let kv_entry = dump
                .tensor("kv_cached_row", layer, token)
                .ok_or_else(|| eyre!("missing kv_cached_row at L{layer} T{token}"))?;
            let kv_row = dump.read_f32(kv_entry)?;
            let off = (token as usize) * (N_HEAD_DIM as usize);
            host_raw[off..off + (N_HEAD_DIM as usize)].copy_from_slice(&kv_row);
            d_raw_kv.copy_from_host(&host_raw)?;

            // Compressed KV row, if compressor fired at this (L, T).
            if let Some(comp_entry) = dump.tensor("comp_kv_row", layer, token) {
                let comp_row = dump.read_f32(comp_entry)?;
                let coff = (n_comp as usize) * (N_HEAD_DIM as usize);
                host_comp[coff..coff + (N_HEAD_DIM as usize)].copy_from_slice(&comp_row);
                n_comp += 1;
            }
            if n_comp > 0 {
                d_comp_kv.copy_from_host(&host_comp)?;
            }

            // Mask (only present for ratio==4 layers from T=3 onwards).
            let mask_opt: Option<&DeviceBuffer<i32>> = if ratio == COMP_RATIO_4 && n_comp > 0 {
                let mask_entry = dump
                    .tensor("comp_allowed_mask", layer, token)
                    .ok_or_else(|| eyre!("missing comp_allowed_mask at L{layer} T{token}"))?;
                if mask_entry.dtype != v4flash_kernels::Dtype::I32 {
                    return Err(eyre!(
                        "comp_allowed_mask at L{layer} T{token} has dtype {:?}, expected I32",
                        mask_entry.dtype
                    ));
                }
                let bytes = dump.read_bytes(mask_entry)?;
                if bytes.len() != (n_comp as usize) * 4 {
                    return Err(eyre!(
                        "comp_allowed_mask size mismatch at L{layer} T{token}: {} bytes, n_comp={n_comp}",
                        bytes.len()
                    ));
                }
                // copy_from_host requires exact length match; pad to the
                // full device-buffer length. Kernel only reads mask[0..n_comp].
                let mut mask_host = vec![0i32; ATTN_MIXED_MAX_KEYS as usize];
                for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                    mask_host[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                d_mask.copy_from_host(&mask_host)?;
                Some(&d_mask)
            } else {
                None
            };

            // Q post-RoPE.
            let q_entry = dump
                .tensor("q_post_rope", layer, token)
                .ok_or_else(|| eyre!("missing q_post_rope at L{layer} T{token}"))?;
            let q_host = dump.read_f32(q_entry)?;
            d_q.copy_from_host(&q_host)?;

            let comp_kv_opt = if n_comp > 0 { Some(&d_comp_kv) } else { None };
            let n_raw = (token as u32) + 1;
            kernel.launch(
                &stream,
                &mut d_out,
                &d_q,
                &d_raw_kv,
                comp_kv_opt,
                mask_opt,
                &d_sinks,
                N_HEAD,
                N_HEAD_DIM,
                n_raw,
                n_comp,
            )?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;

            let expected_entry = dump
                .tensor("attn_heads", layer, token)
                .ok_or_else(|| eyre!("missing attn_heads at L{layer} T{token}"))?;
            let expected = dump.read_f32(expected_entry)?;

            match ratio {
                0 => {
                    let prev = block1.max_abs;
                    block1.update(&got, &expected);
                    if block1.max_abs > prev {
                        worst_block1 = (layer, token);
                    }
                }
                128 => {
                    let prev = block2.max_abs;
                    block2.update(&got, &expected);
                    if block2.max_abs > prev {
                        worst_block2 = (layer, token);
                    }
                }
                4 => {
                    let prev = block3.max_abs;
                    block3.update(&got, &expected);
                    if block3.max_abs > prev {
                        worst_block3 = (layer, token);
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    eprintln!(
        "block 1 (L=0,1, SWA regression):     max_abs={:.3e}, mean={:.3e}, n={}, worst L{} T{}",
        block1.max_abs,
        block1.mean_abs(),
        block1.count,
        worst_block1.0,
        worst_block1.1,
    );
    eprintln!(
        "block 2 (ratio==128, n_comp=0):      max_abs={:.3e}, mean={:.3e}, n={}, worst L{} T{}",
        block2.max_abs,
        block2.mean_abs(),
        block2.count,
        worst_block2.0,
        worst_block2.1,
    );
    eprintln!(
        "block 3 (ratio==4, with mask):       max_abs={:.3e}, mean={:.3e}, n={}, worst L{} T{}",
        block3.max_abs,
        block3.mean_abs(),
        block3.count,
        worst_block3.0,
        worst_block3.1,
    );

    assert!(
        block1.max_abs < SWA_THRESHOLD,
        "block 1 (SWA regression) max_abs {:.3e} > {:.3e}",
        block1.max_abs,
        SWA_THRESHOLD
    );
    assert!(
        block2.max_abs < RATIO128_THRESHOLD,
        "block 2 (ratio==128) max_abs {:.3e} > {:.3e}",
        block2.max_abs,
        RATIO128_THRESHOLD
    );
    assert!(
        block3.max_abs < RATIO4_THRESHOLD,
        "block 3 (ratio==4 with mask) max_abs {:.3e} > {:.3e}",
        block3.max_abs,
        RATIO4_THRESHOLD
    );

    Ok(())
}
