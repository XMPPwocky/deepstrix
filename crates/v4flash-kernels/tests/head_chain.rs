//! Head chain oracle — composes the L43 (synthetic) head ops from
//! `output_logits_one_decode_scratch` (ds4.c:8185-8225):
//!
//!   flat   = rms_norm_no_weight(inp_hc, hc_dim=16384, eps)        [16384]
//!   pre    = matvec_f16(output_hc_fn, flat)                        [n_hc=4]
//!   w      = sigmoid_stable(pre*scale + base) + DS4_HC_EPS         [n_hc=4]
//!   embd   = sum_h inp_hc[h*n_embd + d] * w[h]                     [n_embd=4096]
//!
//! Stops at `output_embd` — the subsequent `rms_norm_weight` + Q8_0 vocab
//! projection (`output_norm`) is the M3 input. Validating each of the four
//! head-tag rows here closes the full forward-pass chain up to M3.
//!
//! `inp_hc` comes from `layer_output_residual` at L=42 (the last layer's
//! post-FFN HC stream is the head's input). Head dump tags use the
//! synthetic layer index `DS4_N_LAYER = 43`.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!                              --test head_chain -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use color_eyre::eyre::WrapErr;
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{
    weights, oracle::ActivationDump, F16Matvec, HcSigmoidBias, HcWeightedSum, RmsNormNoWeight,
};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";

const N_EMBD: u32 = 4096;
const N_HC: u32 = 4;
const HC_DIM: u32 = N_EMBD * N_HC; // 16384
const HEAD_LAYER: i32 = 43;
const LAST_LAYER: i32 = 42;
const RMS_EPS: f32 = 1.0e-6;

// rms_norm_no_weight @ n=16384 with double-precision partial sums has
// ULP-scale noise vs ds4's CPU.
const THRESHOLD_FLAT: f32 = 1.0e-4;
// F16 matvec dequant noise (k=16384) on a 4-dim output: ~1e-3.
const THRESHOLD_PRE: f32 = 1.0e-3;
// sigmoid + bias is pointwise on 4 elements. Noise is dominated by the
// CPU/GPU `expf` implementation gap (~2e-6 on small inputs), not by the
// pre noise upstream.
const THRESHOLD_W: f32 = 1.0e-5;
// hc_weighted_sum: out[d] = sum_h inp_hc[h*N+d] * w[h]. inp_hc is from
// the dump (bit-equal), but w[h] carries ~2e-6 noise from sigmoid. With
// inp_hc magnitudes reaching the 1e2-1e3 range, that propagates to
// ~1e-2 per element. Observed max 1.03e-2, mean 2.07e-5.
const THRESHOLD_EMBD: f32 = 5.0e-2;

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

fn load_f32_weight(
    gguf: &MappedGguf,
    name: &str,
    device_id: i32,
    expected_len: usize,
) -> eyre::Result<DeviceBuffer<f32>> {
    let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("tensor {name} missing"))?;
    if t.dtype != GgufType::F32 {
        return Err(eyre!("tensor {name} has dtype {:?}, expected F32", t.dtype));
    }
    let bytes = gguf
        .read_tensor(t).wrap_err("tensor {name} bytes missing")?;
    if bytes.len() != expected_len * 4 {
        return Err(eyre!(
            "tensor {name}: have {} bytes, expected {}",
            bytes.len(),
            expected_len * 4
        ));
    }
    let mut v = vec![0f32; expected_len];
    for (i, c) in bytes.chunks_exact(4).enumerate() {
        v[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    let mut buf: DeviceBuffer<f32> = DeviceBuffer::new(device_id, expected_len)?;
    buf.copy_from_host(&v)?;
    Ok(buf)
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
fn head_chain_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(std::env::var("DEEPSTRIX_GGUF").unwrap_or_else(|_| MODEL_PATH.to_string()))?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch}); n_tokens={n_tokens}", device.id);

    let rms = RmsNormNoWeight::for_arch(&arch)?;
    let f16 = F16Matvec::for_arch(&arch)?;
    let sig = HcSigmoidBias::for_arch(&arch)?;
    let ws = HcWeightedSum::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // Head weights (loaded once — head has no per-layer variants).
    let w_fn = weights::load_to_device(&gguf, "output_hc_fn.weight", device.id)?;
    let w_scale = load_f32_weight(&gguf, "output_hc_scale.weight", device.id, 1)?;
    let w_base = load_f32_weight(&gguf, "output_hc_base.weight", device.id, N_HC as usize)?;

    let mut d_inp: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_DIM as usize)?;
    let mut d_flat: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_DIM as usize)?;
    let mut d_pre: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_HC as usize)?;
    let mut d_w: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_HC as usize)?;
    let mut d_embd: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;

    let mut got_flat = vec![0f32; HC_DIM as usize];
    let mut got_pre = vec![0f32; N_HC as usize];
    let mut got_w = vec![0f32; N_HC as usize];
    let mut got_embd = vec![0f32; N_EMBD as usize];

    let mut s_flat = DiffStats::default();
    let mut s_pre = DiffStats::default();
    let mut s_w = DiffStats::default();
    let mut s_embd = DiffStats::default();

    for token in 0..n_tokens {
        // inp_hc = last layer's output residual.
        let inp_entry = dump
            .tensor("layer_output_residual", LAST_LAYER, token)
            .ok_or_else(|| eyre!("missing layer_output_residual at L{LAST_LAYER} T{token}"))?;
        let inp_host = dump.read_f32(inp_entry)?;
        assert_eq!(inp_host.len(), HC_DIM as usize);
        d_inp.copy_from_host(&inp_host)?;

        // 1. rms_norm_no_weight over the full 16384-vector (n_rows=1).
        rms.launch(&stream, &mut d_flat, &d_inp, 1, HC_DIM, RMS_EPS)?;
        stream.synchronize()?;
        d_flat.copy_to_host(&mut got_flat)?;
        let exp_flat = dump.read_f32(
            dump.tensor("output_flat", HEAD_LAYER, token)
                .ok_or_else(|| eyre!("missing output_flat at T{token}"))?,
        )?;
        s_flat.update(&got_flat, &exp_flat);

        // 2. F16 matvec: output_hc_fn [hc_dim=16384, n_hc=4] × flat[16384] → pre[4].
        //    Convention matches compressor_end_to_end: n_rows=output_dim(N_HC),
        //    k=input_dim(HC_DIM).
        f16.matvec(&stream, &mut d_pre, &w_fn.buffer, &d_flat, N_HC, HC_DIM)?;
        stream.synchronize()?;
        d_pre.copy_to_host(&mut got_pre)?;
        let exp_pre = dump.read_f32(
            dump.tensor("output_pre", HEAD_LAYER, token)
                .ok_or_else(|| eyre!("missing output_pre at T{token}"))?,
        )?;
        s_pre.update(&got_pre, &exp_pre);

        // 3. sigmoid_stable(pre * scale + base) + DS4_HC_EPS.
        sig.launch(&stream, &mut d_w, &d_pre, &w_scale, &w_base, N_HC)?;
        stream.synchronize()?;
        d_w.copy_to_host(&mut got_w)?;
        let exp_w = dump.read_f32(
            dump.tensor("output_hc_weights", HEAD_LAYER, token)
                .ok_or_else(|| eyre!("missing output_hc_weights at T{token}"))?,
        )?;
        s_w.update(&got_w, &exp_w);

        // 4. hc_weighted_sum: out[d] = sum_h inp_hc[h*n_embd + d] * w[h].
        ws.launch(&stream, &mut d_embd, &d_inp, &d_w, N_EMBD, N_HC)?;
        stream.synchronize()?;
        d_embd.copy_to_host(&mut got_embd)?;
        let exp_embd = dump.read_f32(
            dump.tensor("output_embd", HEAD_LAYER, token)
                .ok_or_else(|| eyre!("missing output_embd at T{token}"))?,
        )?;
        s_embd.update(&got_embd, &exp_embd);
    }

    eprintln!(
        "flat: max={:.3e} mean={:.3e} n={}",
        s_flat.max_abs, s_flat.mean_abs(), s_flat.count
    );
    eprintln!(
        "pre:  max={:.3e} mean={:.3e} n={}",
        s_pre.max_abs, s_pre.mean_abs(), s_pre.count
    );
    eprintln!(
        "w:    max={:.3e} mean={:.3e} n={}",
        s_w.max_abs, s_w.mean_abs(), s_w.count
    );
    eprintln!(
        "embd: max={:.3e} mean={:.3e} n={}",
        s_embd.max_abs, s_embd.mean_abs(), s_embd.count
    );

    assert!(s_flat.max_abs < THRESHOLD_FLAT, "flat max {:.3e}", s_flat.max_abs);
    assert!(s_pre.max_abs < THRESHOLD_PRE, "pre max {:.3e}", s_pre.max_abs);
    assert!(s_w.max_abs < THRESHOLD_W, "w max {:.3e}", s_w.max_abs);
    assert!(s_embd.max_abs < THRESHOLD_EMBD, "embd max {:.3e}", s_embd.max_abs);

    Ok(())
}
