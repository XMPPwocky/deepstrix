//! Q2_K accumulate-matvec oracle. Validates `q2_k_accumulate_matvec`
//! against the Rust CPU port `cpu_dot_q2_k_q8_k`. Setup mirrors the
//! IQ2_XXS test:
//!   - Load expert bytes from blk.{L}.ffn_down_exps.weight (Q2_K).
//!   - Take `ffn_input_norm` activation (or any 4096-wide vector; for
//!     the down step it's actually a 2048-wide `mid`, but the kernel is
//!     dim-agnostic so we use ffn_input_norm sliced to 2048).
//!   - Q8KQuantize the activation, run the kernel, compare row-by-row.
//!
//! Expected: f32-ULP match (same algorithm in different summation order).

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use color_eyre::eyre::WrapErr;
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::q2_k::cpu_dot_q2_k_q8_k;
use v4flash_kernels::{
    oracle::ActivationDump, Q2KAccumulateMatvec, Q8KQuantize, BLOCK_Q2_K_BYTES, BLOCK_Q8_K_BYTES,
};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";

const N_EMBD: u32 = 4096;
const N_FF_EXP: u32 = 2048;
const N_BLOCKS_IN: u32 = N_FF_EXP / 256; // 8 (down's input is the mid vector)
const EXPERTS_TO_TEST: usize = 4;
const LAYERS_TO_TEST: &[i32] = &[3, 8, 20, 42];

fn dump_dir() -> PathBuf {
    std::env::var("DEEPSTRIX_DUMP_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("reference/v4flash-cpu-activations")
    })
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

fn expert_bytes(
    gguf: &MappedGguf,
    name: &str,
    expert: u32,
) -> eyre::Result<Vec<u8>> {
    let t = gguf
        .gguf()
        .tensor(name)
        .ok_or_else(|| eyre!("tensor {name} missing"))?;
    if t.dtype != GgufType::Q2_K {
        return Err(eyre!("tensor {name} dtype {:?} != Q2_K", t.dtype));
    }
    if t.dims.len() != 3 {
        return Err(eyre!("tensor {name} dims {:?} != 3D", t.dims));
    }
    let in_dim = t.dims[0] as usize;
    let out_dim = t.dims[1] as usize;
    let n_experts = t.dims[2] as usize;
    let blocks_per_row = in_dim / 256;
    let row_bytes = blocks_per_row * BLOCK_Q2_K_BYTES;
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
fn q2_k_accumulate_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(std::env::var("DEEPSTRIX_GGUF").unwrap_or_else(|_| MODEL_PATH.to_string()))?;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let matvec = Q2KAccumulateMatvec::for_arch(&arch)?;
    let q8k = Q8KQuantize::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // Down expert: in_dim=2048 (8 blocks), out_dim=4096.
    let row_bytes = (N_BLOCKS_IN as usize) * BLOCK_Q2_K_BYTES;
    let bytes_per_expert = (N_EMBD as usize) * row_bytes;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FF_EXP as usize)?;
    let mut d_xq: DeviceBuffer<u8> =
        DeviceBuffer::new(device.id, (N_BLOCKS_IN as usize) * BLOCK_Q8_K_BYTES)?;
    let mut d_w: DeviceBuffer<u8> = DeviceBuffer::new(device.id, bytes_per_expert)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut got = vec![0f32; N_EMBD as usize];
    let mut xq_host = vec![0u8; (N_BLOCKS_IN as usize) * BLOCK_Q8_K_BYTES];

    let mut max_diff: f32 = 0.0;

    for &layer in LAYERS_TO_TEST {
        // Use the first 2048 elements of ffn_input_norm as synthetic mid input.
        // (Real `mid` is built from selected experts' gate+up; for the per-row
        // dot validation the source doesn't matter.)
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
                &format!("blk.{layer}.ffn_down_exps.weight"),
                expert,
            )?;
            d_w.copy_from_host(&w_bytes)?;

            // zero_init=true: first launch writes; subsequent launches would add.
            matvec.launch(
                &stream,
                &mut d_out,
                &d_w,
                &d_xq,
                N_EMBD,
                N_BLOCKS_IN,
                true,
            )?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;

            for r in 0..(N_EMBD as usize) {
                let exp = cpu_dot_q2_k_q8_k(
                    N_BLOCKS_IN as usize,
                    &w_bytes[r * row_bytes..(r + 1) * row_bytes],
                    &xq_host,
                );
                let d = (got[r] - exp).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            eprintln!("L{layer} expert {expert}: max_diff so far {:.3e}", max_diff);
        }
    }

    eprintln!("q2_k_accumulate: max_abs_diff={:.3e}", max_diff);
    // bsum (integer) bit-exact; final f32 sum across 8 blocks differs by
    // warp-reduction-tree vs sequential order. f32 ULP.
    const THRESHOLD: f32 = 1.0e-5;
    assert!(
        max_diff < THRESHOLD,
        "max_abs_diff {:.3e} >= threshold {:.3e}",
        max_diff,
        THRESHOLD
    );
    Ok(())
}
