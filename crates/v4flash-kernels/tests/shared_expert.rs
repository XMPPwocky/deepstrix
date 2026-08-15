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
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";

const N_EMBD: u32 = 4096;
const N_FF: u32 = 2048; // shared expert FF dim
const N_LAYER: i32 = 43;
// Same Q8_0-noise pattern as M4/M5/M6 chains: mean at f32-ULP, max
// dominated by Q8_0 quantisation on spiky (L, T) inputs flowing through
// 3 matvecs + SwiGLU.
const THRESHOLD: f32 = 5.0e-2;

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
    let gguf = MappedGguf::open(std::env::var("DEEPSTRIX_GGUF").unwrap_or_else(|_| MODEL_PATH.to_string()))?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let q5d = v4flash_kernels::q5_k_dense::Q5_KDenseMatvec::for_arch(&arch)?;
    let q6d = v4flash_kernels::q6_k_dense::Q6_KDenseMatvec::for_arch(&arch)?;
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

            let dm = |q8: &Q8_0Matvec, out: &mut v4flash_hip::DeviceBuffer<f32>,
                      w: &weights::DeviceWeight,
                      x: &v4flash_hip::DeviceBuffer<f32>,
                      xq: &v4flash_hip::DeviceBuffer<i8>,
                      xs: &v4flash_hip::DeviceBuffer<f32>,
                      rows: u32, k: u32,
                      stream: &Stream| -> eyre::Result<()> {
                match w.dtype {
                    v4flash_core::gguf::GgufType::Q8_0 => q8.matvec(stream, out, &w.buffer, xq, xs, rows, k),
                    v4flash_core::gguf::GgufType::Q5_K => q5d.matvec(stream, out, &w.buffer, x, rows, k),
                    v4flash_core::gguf::GgufType::Q6_K => q6d.matvec(stream, out, &w.buffer, x, rows, k),
                    other => Err(eyre!("shexp test: dtype {other:?}")),
                }
            };
            if w_gate.dtype == v4flash_core::gguf::GgufType::Q8_0
                || w_up.dtype == v4flash_core::gguf::GgufType::Q8_0
            {
                q8.quantize_input(&stream, &mut d_xq, &mut d_xscale, &d_x, N_EMBD)?;
            }
            dm(&q8, &mut d_gate, &w_gate, &d_x, &d_xq, &d_xscale, N_FF, N_EMBD, &stream)?;
            dm(&q8, &mut d_up, &w_up, &d_x, &d_xq, &d_xscale, N_FF, N_EMBD, &stream)?;
            // NOTE: deliberately UNCLAMPED to match the pre-5bc1e6d dump this
            // oracle was generated with. Production (het/forward_layer.rs,
            // forward_prefill.rs) now uses launch_clamped(SWIGLU_CLAMP_EXP)
            // per ds4 5bc1e6d ("shared experts use the same swiglu_limit
            // clamp as routed experts"). Measured post-fix divergence vs this
            // dump: max_abs 3.1e2, mean_abs 1.5e-2 (clamp genuinely fires on
            // real shared-expert activations). Flip to launch_clamped only
            // together with a dump regenerated from post-fix ds4.
            swiglu.launch(&stream, &mut d_mid, &d_gate, &d_up, N_FF)?;

            if w_down.dtype == v4flash_core::gguf::GgufType::Q8_0 {
                q8.quantize_input(&stream, &mut d_mid_xq, &mut d_mid_xscale, &d_mid, N_FF)?;
            }
            dm(&q8, &mut d_out, &w_down, &d_mid, &d_mid_xq, &d_mid_xscale, N_EMBD, N_FF, &stream)?;
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

// ============================================================================
// swiglu_limit clamp unit test (ds4 5bc1e6d "Flash graph correctness").
//
// The official V4-Flash graph applies the same swiglu_limit clamp to shared
// experts as to routed experts. CPU reference mirrors ds4.c swiglu()
// post-fix exactly:
//
//   if (clamp > 1e-6) {
//       if (g > clamp)  g = clamp;      // one-sided on gate
//       if (u > clamp)  u = clamp;      // two-sided on up
//       if (u < -clamp) u = -clamp;
//   }
//   out = silu(g) * u
//
// No model / dump files needed — random data including values well beyond
// the clamp threshold (±10).
// ============================================================================

fn swiglu_clamp_cpu_ref(gate: f32, up: f32, clamp: f32) -> f32 {
    let mut g = gate;
    let mut u = up;
    if clamp > 1.0e-6 {
        if g > clamp {
            g = clamp;
        }
        if u > clamp {
            u = clamp;
        }
        if u < -clamp {
            u = -clamp;
        }
    }
    let sig = 1.0f32 / (1.0f32 + (-g).exp());
    g * sig * u
}

#[test]
fn swiglu_clamp_matches_cpu_reference() -> eyre::Result<()> {
    install_panic_handler()?;

    const N: usize = 8192;
    const CLAMP: f32 = 10.0; // SWIGLU_CLAMP_EXP

    // Deterministic LCG; amplitudes swept so a large fraction of values
    // exceed the ±10 clamp threshold.
    let mut s: u64 = 0x5eed_cafe_f00d_0001;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // [-1, 1)
    };
    let mut gate = vec![0f32; N];
    let mut up = vec![0f32; N];
    for i in 0..N {
        // Amplitude cycles 1, 5, 20, 100 — mixes sub-clamp and super-clamp.
        let amp = [1.0f32, 5.0, 20.0, 100.0][i % 4];
        gate[i] = rnd() * amp;
        up[i] = rnd() * amp;
    }
    // Pin exact boundary values.
    gate[0] = 10.0;
    up[0] = 10.0;
    gate[1] = -10.0;
    up[1] = -10.0;
    gate[2] = 10.0000001;
    up[2] = -10.0000001;

    for device in Device::all()? {
        device.set_current()?;
        let arch = device.properties()?.gcn_arch_name;
        let stream = Stream::new(device.id)?;
        let swiglu = Swiglu::for_arch(&arch)?;

        let mut d_gate: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N)?;
        let mut d_up: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N)?;
        let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N)?;
        d_gate.copy_from_host(&gate)?;
        d_up.copy_from_host(&up)?;

        // Clamped launch vs CPU reference.
        swiglu.launch_clamped(&stream, &mut d_out, &d_gate, &d_up, N as u32, CLAMP)?;
        stream.synchronize()?;
        let mut got = vec![0f32; N];
        d_out.copy_to_host(&mut got)?;

        let mut max_abs = 0f32;
        for i in 0..N {
            let want = swiglu_clamp_cpu_ref(gate[i], up[i], CLAMP);
            let d = (got[i] - want).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!("[{arch}] clamped: max_abs_diff vs CPU ref = {max_abs:.3e}");
        // Only expf differs between GPU and host libm; outputs are bounded
        // by |silu(10) * 10| < 100, so a few f32 ulps ≈ 1e-5 absolute.
        assert!(max_abs < 1.0e-4, "[{arch}] clamped max_abs {max_abs:.3e}");

        // clamp = 0.0 must reproduce the historical unclamped behaviour.
        swiglu.launch(&stream, &mut d_out, &d_gate, &d_up, N as u32)?;
        stream.synchronize()?;
        d_out.copy_to_host(&mut got)?;
        let mut max_abs_unclamped = 0f32;
        let mut n_diverge = 0usize;
        for i in 0..N {
            let want = swiglu_clamp_cpu_ref(gate[i], up[i], 0.0);
            let d = (got[i] - want).abs();
            let rel = d / want.abs().max(1.0);
            if rel > max_abs_unclamped {
                max_abs_unclamped = rel;
            }
            // Sanity: clamped and unclamped must differ where inputs exceed
            // the threshold (proves the clamp is live in-kernel).
            if (gate[i] > CLAMP || up[i].abs() > CLAMP)
                && swiglu_clamp_cpu_ref(gate[i], up[i], CLAMP) != want
            {
                n_diverge += 1;
            }
        }
        eprintln!(
            "[{arch}] unclamped: max_rel_diff vs CPU ref = {max_abs_unclamped:.3e}, \
             {n_diverge} inputs where clamp changes the result"
        );
        assert!(max_abs_unclamped < 1.0e-5);
        assert!(n_diverge > 0, "test data never exercised the clamp");
    }
    Ok(())
}
