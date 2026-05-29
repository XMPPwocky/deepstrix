//! Shared expert oracle. Mirrors ds4's `layer_shared_ffn_one_decode_scratch`
//! (ds4.c:5124):
//!   gate = matvec_q8_0(ffn_gate_shexp, ffn_input_norm)  [2048]
//!   up   = matvec_q8_0(ffn_up_shexp,   ffn_input_norm)  [2048]
//!   mid  = silu(gate) * up                              [2048]
//!   out  = matvec_q8_0(ffn_down_shexp, mid)             [4096]
//!
//! Validates against `ffn_shared` for all 43 layers (the shared expert
//! runs for every token regardless of routing).

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{weights, oracle::ActivationDump, Q8_0Matvec, Swiglu};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_EMBD: u32 = 4096;
const N_FF: u32 = 2048; // shared expert FF dim
const N_LAYER: i32 = 43;
// Same Q8_0-noise pattern as M4/M5/M6 chains: mean at f32-ULP, max
// dominated by Q8_0 quantisation on spiky (L, T) inputs flowing through
// 3 matvecs + SwiGLU.
const THRESHOLD: f32 = 5.0e-2;

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
fn shared_expert_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let swiglu = Swiglu::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_xq: DeviceBuffer<i8> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_xscale: DeviceBuffer<f32> = DeviceBuffer::new(device.id, (N_EMBD / 32) as usize)?;
    let mut d_gate: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FF as usize)?;
    let mut d_up: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FF as usize)?;
    let mut d_mid: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FF as usize)?;
    let mut d_mid_xq: DeviceBuffer<i8> = DeviceBuffer::new(device.id, N_FF as usize)?;
    let mut d_mid_xscale: DeviceBuffer<f32> = DeviceBuffer::new(device.id, (N_FF / 32) as usize)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut got = vec![0f32; N_EMBD as usize];

    let mut stats = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    for layer in 0..N_LAYER {
        let w_gate = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.ffn_gate_shexp.weight"),
            device.id,
        )?;
        let w_up = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.ffn_up_shexp.weight"),
            device.id,
        )?;
        let w_down = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.ffn_down_shexp.weight"),
            device.id,
        )?;

        for token in 0..n_tokens {
            let x_entry = dump
                .tensor("ffn_input_norm", layer, token)
                .ok_or_else(|| eyre!("missing ffn_input_norm at L{layer} T{token}"))?;
            d_x.copy_from_host(&dump.read_f32(x_entry)?)?;

            q8.quantize_input(&stream, &mut d_xq, &mut d_xscale, &d_x, N_EMBD)?;
            q8.matvec(&stream, &mut d_gate, &w_gate.buffer, &d_xq, &d_xscale, N_FF, N_EMBD)?;
            q8.matvec(&stream, &mut d_up, &w_up.buffer, &d_xq, &d_xscale, N_FF, N_EMBD)?;
            swiglu.launch(&stream, &mut d_mid, &d_gate, &d_up, N_FF)?;

            q8.quantize_input(&stream, &mut d_mid_xq, &mut d_mid_xscale, &d_mid, N_FF)?;
            q8.matvec(&stream, &mut d_out, &w_down.buffer, &d_mid_xq, &d_mid_xscale, N_EMBD, N_FF)?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;

            let exp_entry = dump
                .tensor("ffn_shared", layer, token)
                .ok_or_else(|| eyre!("missing ffn_shared at L{layer} T{token}"))?;
            let expected = dump.read_f32(exp_entry)?;
            let prev = stats.max_abs;
            stats.update(&got, &expected);
            if stats.max_abs > prev {
                worst = (layer, token);
            }
        }

        drop(w_gate);
        drop(w_up);
        drop(w_down);
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
