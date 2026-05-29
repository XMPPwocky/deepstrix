//! Q8_K quantize oracle — pack `ffn_input_norm` rows from the dump into
//! Q8_K blocks and compare to a Rust CPU port of ds4_quantize_row_q8_K
//! (ds4.c:1655). Goal is **bit-exact** since both implementations use
//! the same formula (`iscale = -127/mxv`, `clamp(lrintf(iscale*x))`).
//!
//! Coverage: every L,T in the dump (43 × 51 vectors × 16 blocks each).

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{oracle::ActivationDump, Q8KQuantize, BLOCK_Q8_K_BYTES, QK_K};

const N_EMBD: usize = 4096;
const N_BLOCKS: usize = N_EMBD / 256; // 16
const N_LAYER: i32 = 43;

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

/// CPU port of `ds4_quantize_row_q8_K`. Same arithmetic order.
fn cpu_quantize_q8_k(x: &[f32], out: &mut [u8]) {
    assert_eq!(x.len() % QK_K as usize, 0);
    let nb = x.len() / QK_K as usize;
    assert_eq!(out.len(), nb * BLOCK_Q8_K_BYTES);

    for b in 0..nb {
        let xb = &x[b * QK_K as usize..(b + 1) * QK_K as usize];
        let blk_off = b * BLOCK_Q8_K_BYTES;

        let mut amax = 0.0f32;
        let mut mxv = 0.0f32;
        for &v in xb {
            let a = v.abs();
            if a > amax {
                amax = a;
                mxv = v;
            }
        }

        if amax == 0.0 {
            out[blk_off..blk_off + 4].copy_from_slice(&0.0f32.to_le_bytes());
            for i in 0..256 {
                out[blk_off + 4 + i] = 0;
            }
            for i in 0..16 {
                let o = blk_off + 260 + i * 2;
                out[o] = 0;
                out[o + 1] = 0;
            }
            continue;
        }

        let iscale = -127.0f32 / mxv;
        let d = 1.0f32 / iscale;
        out[blk_off..blk_off + 4].copy_from_slice(&d.to_le_bytes());

        let mut qs = [0i8; 256];
        for j in 0..256 {
            let mut v = (iscale * xb[j]).round_ties_even() as i32;
            // Match lrintf semantics: round half to even.
            // (Rust's `as i32` truncates toward zero; we need round-to-nearest-even
            // for bit-exact agreement. round_ties_even gives nearest-even.)
            let _ = (); // explanatory no-op
            if v > 127 {
                v = 127;
            }
            if v < -128 {
                v = -128;
            }
            qs[j] = v as i8;
        }
        for j in 0..256 {
            out[blk_off + 4 + j] = qs[j] as u8;
        }

        for j in 0..16 {
            let mut s: i32 = 0;
            for i in 0..16 {
                s += qs[j * 16 + i] as i32;
            }
            let bs = s as i16;
            let o = blk_off + 260 + j * 2;
            out[o..o + 2].copy_from_slice(&bs.to_le_bytes());
        }
    }
}

#[test]
#[ignore]
fn q8_k_quantize_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch}); n_tokens={n_tokens}", device.id);

    let kernel = Q8KQuantize::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD)?;
    let mut d_out: DeviceBuffer<u8> = DeviceBuffer::new(device.id, N_BLOCKS * BLOCK_Q8_K_BYTES)?;
    let mut got = vec![0u8; N_BLOCKS * BLOCK_Q8_K_BYTES];
    let mut expected = vec![0u8; N_BLOCKS * BLOCK_Q8_K_BYTES];

    let mut total_compared: u64 = 0;
    let mut total_diff: u64 = 0;
    let mut max_qs_diff: i32 = 0;
    let mut max_d_diff: f32 = 0.0;

    for layer in 0..N_LAYER {
        for token in 0..n_tokens {
            let e = match dump.tensor("ffn_input_norm", layer, token) {
                Some(e) => e,
                None => continue,
            };
            let x = dump.read_f32(e)?;
            assert_eq!(x.len(), N_EMBD);

            d_x.copy_from_host(&x)?;
            kernel.launch(&stream, &mut d_out, &d_x, N_BLOCKS as u32)?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;

            cpu_quantize_q8_k(&x, &mut expected);

            for b in 0..N_BLOCKS {
                let off = b * BLOCK_Q8_K_BYTES;
                let d_g = f32::from_le_bytes([
                    got[off],
                    got[off + 1],
                    got[off + 2],
                    got[off + 3],
                ]);
                let d_e = f32::from_le_bytes([
                    expected[off],
                    expected[off + 1],
                    expected[off + 2],
                    expected[off + 3],
                ]);
                let dd = (d_g - d_e).abs();
                if dd > max_d_diff {
                    max_d_diff = dd;
                }
                for j in 0..256 {
                    let g = got[off + 4 + j] as i8 as i32;
                    let e = expected[off + 4 + j] as i8 as i32;
                    let diff = (g - e).abs();
                    if diff > max_qs_diff {
                        max_qs_diff = diff;
                    }
                    if diff != 0 {
                        total_diff += 1;
                    }
                    total_compared += 1;
                }
                // bsums must match exactly
                for j in 0..16 {
                    let o = off + 260 + j * 2;
                    let g = i16::from_le_bytes([got[o], got[o + 1]]);
                    let e = i16::from_le_bytes([expected[o], expected[o + 1]]);
                    assert_eq!(g, e, "bsum mismatch L{layer} T{token} b{b} g{j}");
                }
            }
        }
    }

    eprintln!(
        "q8_k_quantize: max_d_diff={:.3e}, max_qs_diff={}, total_diff={}/{}",
        max_d_diff, max_qs_diff, total_diff, total_compared
    );
    assert_eq!(max_qs_diff, 0, "qs mismatch (max abs diff = {max_qs_diff})");
    assert_eq!(max_d_diff, 0.0, "d mismatch (max abs diff = {max_d_diff})");
    Ok(())
}
