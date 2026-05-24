//! IQ2_XXS paired matvec oracle. Validates `iq2_xxs_pair_matvec` against
//! a Rust port of ds4's scalar `ds4_vec_dot_iq2_xxs_q8_K` (ds4.c:1874).
//!
//! Setup per layer L (L=3..42 where routed experts exist):
//!   1. Load one expert's bytes from ffn_gate_exps + ffn_up_exps.
//!   2. Take ffn_input_norm[L, T=0] from the dump.
//!   3. Quantize the activation with our (already-validated) Q8KQuantize.
//!   4. Run iq2_xxs_pair_matvec for that expert (2048 rows × 16 blocks).
//!   5. Compare each row to cpu_dot_iq2_xxs_q8_k.
//!
//! Expected: bit-exact (same int-sum order, same f32 reduction order).
//! Coverage: a few layers × first 4 experts each is plenty — the algorithm
//! is data-independent.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::iq2_xxs_tables::cpu_dot_iq2_xxs_q8_k;
use v4flash_kernels::{
    ActivationDump, Iq2XxsPairMatvec, Q8KQuantize, BLOCK_IQ2_XXS_BYTES, BLOCK_Q8_K_BYTES,
};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_EMBD: u32 = 4096;
const N_FF_EXP: u32 = 2048;
const N_BLOCKS_IN: u32 = N_EMBD / 256; // 16
const EXPERTS_TO_TEST: usize = 4;
const LAYERS_TO_TEST: &[i32] = &[3, 8, 20, 42];

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

/// Slice raw bytes for one expert from a 3D GGUF tensor `[in_dim, out_dim, n_experts]`
/// of IQ2_XXS format (block_size=256, block_bytes=66).
fn expert_bytes<'a>(
    gguf: &'a MappedGguf,
    name: &str,
    expert: u32,
) -> eyre::Result<&'a [u8]> {
    let t = gguf
        .gguf()
        .tensor(name)
        .ok_or_else(|| eyre!("tensor {name} missing"))?;
    if t.dtype != GgufType::IQ2_XXS {
        return Err(eyre!("tensor {name} dtype {:?} != IQ2_XXS", t.dtype));
    }
    if t.dims.len() != 3 {
        return Err(eyre!("tensor {name} dims {:?} != 3D", t.dims));
    }
    let in_dim = t.dims[0] as usize;
    let out_dim = t.dims[1] as usize;
    let n_experts = t.dims[2] as usize;
    let blocks_per_row = in_dim / 256;
    let row_bytes = blocks_per_row * BLOCK_IQ2_XXS_BYTES;
    let bytes_per_expert = out_dim * row_bytes;
    let total_bytes = bytes_per_expert * n_experts;
    let all = gguf
        .tensor_bytes(t)
        .ok_or_else(|| eyre!("tensor {name} bytes missing"))?;
    if all.len() != total_bytes {
        return Err(eyre!(
            "tensor {name}: have {} bytes, expected {} ({} per expert × {} experts)",
            all.len(),
            total_bytes,
            bytes_per_expert,
            n_experts
        ));
    }
    let off = (expert as usize) * bytes_per_expert;
    Ok(&all[off..off + bytes_per_expert])
}

#[test]
#[ignore]
fn iq2_xxs_pair_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let matvec = Iq2XxsPairMatvec::for_arch(&arch)?;
    let q8k = Q8KQuantize::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let row_bytes = (N_BLOCKS_IN as usize) * BLOCK_IQ2_XXS_BYTES;
    let bytes_per_expert = (N_FF_EXP as usize) * row_bytes;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_xq: DeviceBuffer<u8> =
        DeviceBuffer::new(device.id, (N_BLOCKS_IN as usize) * BLOCK_Q8_K_BYTES)?;
    let mut d_gate_w: DeviceBuffer<u8> = DeviceBuffer::new(device.id, bytes_per_expert)?;
    let mut d_up_w: DeviceBuffer<u8> = DeviceBuffer::new(device.id, bytes_per_expert)?;
    let mut d_gate: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FF_EXP as usize)?;
    let mut d_up: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FF_EXP as usize)?;

    let mut got_gate = vec![0f32; N_FF_EXP as usize];
    let mut got_up = vec![0f32; N_FF_EXP as usize];
    let mut xq_host = vec![0u8; (N_BLOCKS_IN as usize) * BLOCK_Q8_K_BYTES];

    let mut max_diff: f32 = 0.0;
    let mut count_diff: u64 = 0;
    let mut total: u64 = 0;

    for &layer in LAYERS_TO_TEST {
        let x_entry = dump
            .tensor("ffn_input_norm", layer, 0)
            .ok_or_else(|| eyre!("missing ffn_input_norm at L{layer} T0"))?;
        let x = dump.read_f32(x_entry)?;
        d_x.copy_from_host(&x)?;
        q8k.launch(&stream, &mut d_xq, &d_x, N_BLOCKS_IN)?;
        stream.synchronize()?;
        d_xq.copy_to_host(&mut xq_host)?;

        for expert in 0..EXPERTS_TO_TEST as u32 {
            let g_bytes = expert_bytes(
                &gguf,
                &format!("blk.{layer}.ffn_gate_exps.weight"),
                expert,
            )?;
            let u_bytes = expert_bytes(
                &gguf,
                &format!("blk.{layer}.ffn_up_exps.weight"),
                expert,
            )?;
            d_gate_w.copy_from_host(g_bytes)?;
            d_up_w.copy_from_host(u_bytes)?;

            matvec.launch(
                &stream,
                &mut d_gate,
                &mut d_up,
                &d_gate_w,
                &d_up_w,
                &d_xq,
                N_FF_EXP,
                N_BLOCKS_IN,
            )?;
            stream.synchronize()?;
            d_gate.copy_to_host(&mut got_gate)?;
            d_up.copy_to_host(&mut got_up)?;

            for r in 0..(N_FF_EXP as usize) {
                let exp_g = cpu_dot_iq2_xxs_q8_k(
                    N_BLOCKS_IN as usize,
                    &g_bytes[r * row_bytes..(r + 1) * row_bytes],
                    &xq_host,
                );
                let exp_u = cpu_dot_iq2_xxs_q8_k(
                    N_BLOCKS_IN as usize,
                    &u_bytes[r * row_bytes..(r + 1) * row_bytes],
                    &xq_host,
                );
                let dg = (got_gate[r] - exp_g).abs();
                let du = (got_up[r] - exp_u).abs();
                if dg > max_diff {
                    max_diff = dg;
                }
                if du > max_diff {
                    max_diff = du;
                }
                if dg > 0.0 || du > 0.0 {
                    count_diff += 1;
                }
                total += 2;
            }
            eprintln!(
                "L{layer} expert {expert}: ran 2048 rows, max_diff so far {:.3e}",
                max_diff
            );
        }
    }

    eprintln!(
        "iq2_xxs_pair: max_abs_diff={:.3e}, n_nonzero={}, total={}",
        max_diff, count_diff, total
    );
    // bsum (integer) is bit-exact; the final f32 sum of `0.125 * d_b * bsum_b`
    // across 16 super-blocks accumulates in a different order on GPU
    // (warp_sum_f32 reduction tree) vs CPU (sequential). Difference is at
    // f32 ULP for these magnitudes.
    const THRESHOLD: f32 = 1.0e-5;
    assert!(
        max_diff < THRESHOLD,
        "max_abs_diff {:.3e} >= threshold {:.3e}",
        max_diff,
        THRESHOLD
    );
    Ok(())
}
