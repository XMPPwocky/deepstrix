//! Q4_K batched MoE matvec oracle. Validates `q4_k_matvec_par_batched`
//! against the Rust CPU port `cpu_dot_q4_k_q8_k` for one expert per
//! launch (n_used=1). Mirrors the q2_k_accumulate test structure.
//!
//! Weights come from the MTP GGUF — V4-Flash's main model has no Q4_K
//! experts but the MTP variant (`*-MTP-Q4K-Q8_0-F32.gguf`) does. Each
//! down expert is [in_dim=2048, out_dim=4096] Q4_K.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use color_eyre::eyre::WrapErr;
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::q4_k::cpu_dot_q4_k_q8_k;
use v4flash_kernels::{
    ActivationDump, Q4KMatvec, Q8KQuantize, BLOCK_Q4_K_BYTES, BLOCK_Q8_K_BYTES,
};

const MTP_MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf";

const N_EMBD: u32 = 4096;
const N_FF_EXP: u32 = 2048;
const N_BLOCKS_IN: u32 = N_FF_EXP / 256; // 8 blocks per row (down's in_dim is 2048)
const EXPERTS_TO_TEST: usize = 4;

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

fn expert_bytes(gguf: &MappedGguf, name: &str, expert: u32) -> eyre::Result<Vec<u8>> {
    let t = gguf
        .gguf()
        .tensor(name)
        .ok_or_else(|| eyre!("tensor {name} missing"))?;
    if t.dtype != GgufType::Q4_K {
        return Err(eyre!("tensor {name} dtype {:?} != Q4_K", t.dtype));
    }
    if t.dims.len() != 3 {
        return Err(eyre!("tensor {name} dims {:?} != 3D", t.dims));
    }
    let in_dim = t.dims[0] as usize;
    let out_dim = t.dims[1] as usize;
    let n_experts = t.dims[2] as usize;
    let blocks_per_row = in_dim / 256;
    let row_bytes = blocks_per_row * BLOCK_Q4_K_BYTES;
    let bytes_per_expert = out_dim * row_bytes;
    let total_bytes = bytes_per_expert * n_experts;
    let all = gguf
        .read_tensor(t)
        .wrap_err("tensor {name} bytes missing")?;
    if all.len() != total_bytes {
        return Err(eyre!(
            "tensor {name}: have {} bytes, expected {}",
            all.len(),
            total_bytes
        ));
    }
    let off = (expert as usize) * bytes_per_expert;
    Ok(all[off..off + bytes_per_expert].to_vec())
}

#[test]
#[ignore]
fn q4_k_matvec_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MTP_MODEL_PATH)?;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let matvec = Q4KMatvec::for_arch(&arch)?;
    let q8k = Q8KQuantize::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // MTP down expert: in_dim=2048 (8 blocks), out_dim=4096.
    let row_bytes = (N_BLOCKS_IN as usize) * BLOCK_Q4_K_BYTES;
    let bytes_per_expert = (N_EMBD as usize) * row_bytes;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FF_EXP as usize)?;
    let mut d_xq: DeviceBuffer<u8> =
        DeviceBuffer::new(device.id, (N_BLOCKS_IN as usize) * BLOCK_Q8_K_BYTES)?;
    let mut d_w: DeviceBuffer<u8> = DeviceBuffer::new(device.id, bytes_per_expert)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_selected: DeviceBuffer<i32> = DeviceBuffer::new(device.id, 1)?;
    d_selected.copy_from_host(&[0i32])?;
    let mut got = vec![0f32; N_EMBD as usize];
    let mut xq_host = vec![0u8; (N_BLOCKS_IN as usize) * BLOCK_Q8_K_BYTES];

    let mut max_diff: f32 = 0.0;
    let mut max_rel: f32 = 0.0;

    // Take a few ffn_input_norm activations from the dump as synthetic mid input.
    // (Real `mid` would be the routed MoE intermediate; for per-row dot
    // validation the source doesn't matter.)
    let test_layers: &[i32] = &[3, 8, 20, 42];
    for &layer in test_layers {
        let x_entry = dump
            .tensor("ffn_input_norm", layer, 0)
            .ok_or_else(|| eyre!("missing ffn_input_norm L{layer} T0"))?;
        let mut x_full = dump.read_f32(x_entry)?;
        x_full.truncate(N_FF_EXP as usize);
        d_x.copy_from_host(&x_full)?;
        q8k.launch(&stream, &mut d_xq, &d_x, N_BLOCKS_IN)?;
        stream.synchronize()?;
        d_xq.copy_to_host(&mut xq_host)?;

        for expert in 0..EXPERTS_TO_TEST as u32 {
            let w_bytes = expert_bytes(
                &gguf,
                "mtp.0.ffn_down_exps.weight",
                expert,
            )?;
            d_w.copy_from_host(&w_bytes)?;

            // n_used=1: single-expert pass. Treat as a per-expert sanity check.
            matvec.launch_batched(
                &stream,
                &mut d_out,
                &d_w,
                &d_xq,
                &d_selected,
                bytes_per_expert as u32,
                ((N_BLOCKS_IN as usize) * BLOCK_Q8_K_BYTES) as u32,
                1,
                N_EMBD,
                N_BLOCKS_IN,
            )?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;

            for r in 0..(N_EMBD as usize) {
                let exp = cpu_dot_q4_k_q8_k(
                    N_BLOCKS_IN as usize,
                    &w_bytes[r * row_bytes..(r + 1) * row_bytes],
                    &xq_host,
                );
                let d = (got[r] - exp).abs();
                if d > max_diff {
                    max_diff = d;
                }
                let denom = exp.abs().max(1.0e-6);
                let rel = d / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            eprintln!(
                "L{layer} expert {expert}: max_abs_diff so far {:.3e}, max_rel {:.3e}",
                max_diff, max_rel
            );
        }
    }

    eprintln!(
        "q4_k_matvec: max_abs_diff={:.3e}, max_rel_diff={:.3e}",
        max_diff, max_rel
    );
    // Integer sums are bit-exact; f32 multiply + warp-tree reduction differs from
    // sequential CPU order. Allow ULP-level drift.
    const THRESHOLD: f32 = 1.0e-3;
    assert!(
        max_diff < THRESHOLD,
        "max_abs_diff {:.3e} >= threshold {:.3e}",
        max_diff,
        THRESHOLD
    );
    Ok(())
}
