//! Routed-MoE end-to-end oracle. Composes the full pipeline:
//!   x = ffn_input_norm                                       [4096]
//!   xq = q8_k_quantize(x)                                    [16 blocks]
//!   for each of 6 selected experts e:
//!     gate[e], up[e] = iq2_xxs_pair_matvec(W_g[e], W_u[e], xq)   [2048 each]
//!     mid[e] = swiglu_clamp(gate, up, expert_w[e], clamp=10)     [2048]
//!     midq[e] = q8_k_quantize(mid[e])                            [8 blocks]
//!     out += q2_k_accumulate(W_d[e], midq[e])                    [4096]
//!   compare to `ffn_moe` dump tag.
//!
//! Coverage: one routed layer × all 51 tokens. One layer is enough to
//! validate pipeline composition; the constituent kernels are tested
//! individually for cross-layer coverage in the M11.1-3 oracles.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{
    ActivationDump, Iq2XxsPairMatvec, Q2KAccumulateMatvec, Q8KQuantize, SwigluClampWeighted,
    BLOCK_IQ2_XXS_BYTES, BLOCK_Q2_K_BYTES, BLOCK_Q8_K_BYTES,
};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_EMBD: u32 = 4096;
const N_FF_EXP: u32 = 2048;
const N_EXPERT_USED: usize = 6;
const N_BLOCKS_GATE_IN: u32 = N_EMBD / 256; // 16
const N_BLOCKS_DOWN_IN: u32 = N_FF_EXP / 256; // 8
const SWIGLU_CLAMP_EXP: f32 = 10.0;
const TEST_LAYER: i32 = 3;

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
fn routed_moe_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch}); n_tokens={n_tokens}", device.id);

    let q8k = Q8KQuantize::for_arch(&arch)?;
    let iq2 = Iq2XxsPairMatvec::for_arch(&arch)?;
    let swiglu_cw = SwigluClampWeighted::for_arch(&arch)?;
    let q2k = Q2KAccumulateMatvec::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // Verify dtypes.
    for (name, want) in [
        (format!("blk.{TEST_LAYER}.ffn_gate_exps.weight"), GgufType::IQ2_XXS),
        (format!("blk.{TEST_LAYER}.ffn_up_exps.weight"), GgufType::IQ2_XXS),
        (format!("blk.{TEST_LAYER}.ffn_down_exps.weight"), GgufType::Q2_K),
    ] {
        let t = gguf.gguf().tensor(&name).ok_or_else(|| eyre!("{name} missing"))?;
        if t.dtype != want {
            return Err(eyre!("{name} dtype {:?} != {:?}", t.dtype, want));
        }
    }

    let gate_t = gguf
        .gguf()
        .tensor(&format!("blk.{TEST_LAYER}.ffn_gate_exps.weight"))
        .unwrap();
    let down_t = gguf
        .gguf()
        .tensor(&format!("blk.{TEST_LAYER}.ffn_down_exps.weight"))
        .unwrap();
    let gate_all = gguf.tensor_bytes(gate_t).ok_or_else(|| eyre!("gate bytes"))?;
    let up_all = gguf
        .tensor_bytes(
            gguf.gguf()
                .tensor(&format!("blk.{TEST_LAYER}.ffn_up_exps.weight"))
                .unwrap(),
        )
        .ok_or_else(|| eyre!("up bytes"))?;
    let down_all = gguf
        .tensor_bytes(down_t)
        .ok_or_else(|| eyre!("down bytes"))?;

    let gate_bytes_per_expert =
        (N_FF_EXP as usize) * (N_BLOCKS_GATE_IN as usize) * BLOCK_IQ2_XXS_BYTES;
    let up_bytes_per_expert = gate_bytes_per_expert;
    let down_bytes_per_expert =
        (N_EMBD as usize) * (N_BLOCKS_DOWN_IN as usize) * BLOCK_Q2_K_BYTES;

    // Per-slot buffers (one per selected expert, reused token-to-token).
    let mut d_gw: Vec<DeviceBuffer<u8>> = (0..N_EXPERT_USED)
        .map(|_| DeviceBuffer::new(device.id, gate_bytes_per_expert).unwrap())
        .collect();
    let mut d_uw: Vec<DeviceBuffer<u8>> = (0..N_EXPERT_USED)
        .map(|_| DeviceBuffer::new(device.id, up_bytes_per_expert).unwrap())
        .collect();
    let mut d_dw: Vec<DeviceBuffer<u8>> = (0..N_EXPERT_USED)
        .map(|_| DeviceBuffer::new(device.id, down_bytes_per_expert).unwrap())
        .collect();

    // Per-token scratch.
    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_xq: DeviceBuffer<u8> =
        DeviceBuffer::new(device.id, (N_BLOCKS_GATE_IN as usize) * BLOCK_Q8_K_BYTES)?;
    // Gate/up cated across 6 experts; SwiGLU also writes a 6×ff_exp mid.
    let total_ff = N_EXPERT_USED * (N_FF_EXP as usize);
    let mut d_gate: DeviceBuffer<f32> = DeviceBuffer::new(device.id, total_ff)?;
    let mut d_up: DeviceBuffer<f32> = DeviceBuffer::new(device.id, total_ff)?;
    let mut d_mid: DeviceBuffer<f32> = DeviceBuffer::new(device.id, total_ff)?;
    let mut d_mid_e: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FF_EXP as usize)?;
    let mut d_midq: DeviceBuffer<u8> =
        DeviceBuffer::new(device.id, (N_BLOCKS_DOWN_IN as usize) * BLOCK_Q8_K_BYTES)?;
    let mut d_ew: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EXPERT_USED)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut got = vec![0f32; N_EMBD as usize];

    // Per-token gate/up output strides into d_gate / d_up.
    // We launch iq2 with separate output buffers per slot, then copy
    // into the concatenated d_gate / d_up.
    let mut d_gate_e: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FF_EXP as usize)?;
    let mut d_up_e: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FF_EXP as usize)?;
    let mut staging = vec![0f32; N_FF_EXP as usize];
    let mut staging_full = vec![0f32; total_ff];
    let mut staging_full2 = vec![0f32; total_ff];

    let mut stats = DiffStats::default();

    for token in 0..n_tokens {
        let sel_e = dump
            .tensor("expert_selected", TEST_LAYER, token)
            .ok_or_else(|| eyre!("missing expert_selected L{TEST_LAYER} T{token}"))?;
        let sel_e_bytes = dump.read_bytes(sel_e)?;
        assert_eq!(sel_e_bytes.len(), N_EXPERT_USED * 4);
        let mut sel_ids = [0i32; 6];
        for i in 0..N_EXPERT_USED {
            sel_ids[i] = i32::from_le_bytes([
                sel_e_bytes[i * 4],
                sel_e_bytes[i * 4 + 1],
                sel_e_bytes[i * 4 + 2],
                sel_e_bytes[i * 4 + 3],
            ]);
        }
        let sel_w = dump
            .tensor("expert_weight_out", TEST_LAYER, token)
            .ok_or_else(|| eyre!("missing expert_weight_out L{TEST_LAYER} T{token}"))?;
        let expert_w_host = dump.read_f32(sel_w)?;
        d_ew.copy_from_host(&expert_w_host)?;

        // Upload 6 selected experts' bytes to the per-slot device buffers.
        for slot in 0..N_EXPERT_USED {
            let e = sel_ids[slot] as usize;
            d_gw[slot].copy_from_host(
                &gate_all[e * gate_bytes_per_expert..(e + 1) * gate_bytes_per_expert],
            )?;
            d_uw[slot].copy_from_host(
                &up_all[e * up_bytes_per_expert..(e + 1) * up_bytes_per_expert],
            )?;
            d_dw[slot].copy_from_host(
                &down_all[e * down_bytes_per_expert..(e + 1) * down_bytes_per_expert],
            )?;
        }

        let x_entry = dump
            .tensor("ffn_input_norm", TEST_LAYER, token)
            .ok_or_else(|| eyre!("missing ffn_input_norm L{TEST_LAYER} T{token}"))?;
        let x = dump.read_f32(x_entry)?;
        d_x.copy_from_host(&x)?;
        q8k.launch(&stream, &mut d_xq, &d_x, N_BLOCKS_GATE_IN)?;

        // Per-slot gate+up matvec, then copy into the concatenated d_gate/d_up.
        for slot in 0..N_EXPERT_USED {
            iq2.launch(
                &stream,
                &mut d_gate_e,
                &mut d_up_e,
                &d_gw[slot],
                &d_uw[slot],
                &d_xq,
                N_FF_EXP,
                N_BLOCKS_GATE_IN,
            )?;
            stream.synchronize()?;
            d_gate_e.copy_to_host(&mut staging)?;
            staging_full[slot * (N_FF_EXP as usize)..(slot + 1) * (N_FF_EXP as usize)]
                .copy_from_slice(&staging);
            d_up_e.copy_to_host(&mut staging)?;
            staging_full2[slot * (N_FF_EXP as usize)..(slot + 1) * (N_FF_EXP as usize)]
                .copy_from_slice(&staging);
        }
        d_gate.copy_from_host(&staging_full)?;
        d_up.copy_from_host(&staging_full2)?;

        // SwiGLU + clamp + expert_weight.
        swiglu_cw.launch(
            &stream,
            &mut d_mid,
            &d_gate,
            &d_up,
            &d_ew,
            SWIGLU_CLAMP_EXP,
            N_FF_EXP,
            N_EXPERT_USED as u32,
        )?;

        // Per expert: quantize mid slice, then accumulate q2_k down into out.
        stream.synchronize()?;
        d_mid.copy_to_host(&mut staging_full)?;
        for slot in 0..N_EXPERT_USED {
            d_mid_e.copy_from_host(
                &staging_full[slot * (N_FF_EXP as usize)..(slot + 1) * (N_FF_EXP as usize)],
            )?;
            q8k.launch(&stream, &mut d_midq, &d_mid_e, N_BLOCKS_DOWN_IN)?;
            q2k.launch(
                &stream,
                &mut d_out,
                &d_dw[slot],
                &d_midq,
                N_EMBD,
                N_BLOCKS_DOWN_IN,
                slot == 0,
            )?;
        }

        stream.synchronize()?;
        d_out.copy_to_host(&mut got)?;

        let exp_entry = dump
            .tensor("ffn_moe", TEST_LAYER, token)
            .ok_or_else(|| eyre!("missing ffn_moe L{TEST_LAYER} T{token}"))?;
        let expected = dump.read_f32(exp_entry)?;
        stats.update(&got, &expected);
        if (token % 10) == 0 {
            eprintln!(
                "T{token}: max so far {:.3e}, mean {:.3e}",
                stats.max_abs,
                stats.mean_abs()
            );
        }
    }

    eprintln!(
        "routed_moe: max={:.3e} mean={:.3e} n={}",
        stats.max_abs,
        stats.mean_abs(),
        stats.count
    );
    // Quant noise accumulates across iq2 matvec + swiglu + q8k + q2k. Per-row
    // ULP noise is ~1e-5 each; product noise is ~1e-3 per (e,r); sum over 6
    // experts can reach low-1e-2. Threshold 5e-2 with mean<1e-3 gate.
    assert!(
        stats.max_abs < 5.0e-2,
        "max {:.3e} >= 5e-2",
        stats.max_abs
    );
    assert!(
        stats.mean_abs() < 1.0e-3,
        "mean {:.3e} >= 1e-3",
        stats.mean_abs()
    );
    Ok(())
}
